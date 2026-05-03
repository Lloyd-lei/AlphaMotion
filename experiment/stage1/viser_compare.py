"""Interactive viser: GT / masked / model-prediction — three side-by-side
skeletons running a live forward pass on a selectable sample.

Pick any index from the browser UI, change mask mode / ratio on the fly,
see GT (green, left), masked input (orange, middle; masked joints shown
at origin as "holes"), and the model's reconstruction (blue, right).

Usage:
    python viser_compare.py \
        --checkpoint runs/full__mixed_fixed30/final.pt \
        --model full
    # then open http://localhost:7860

Shortcuts in the UI:
    "→ Next Kimodo / AMASS / HY"  jumps to the next sample of that source.
    Changing mask ratio re-rolls the mask (useful for fresh random seed).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import viser

from data import MixedMotionDataset, make_masked_batch
from motionformer import MotionFormerConfig, MotionFormer
from soma_skeleton import build_parent_index
from train import MF_VARIANTS


def load_model(variant: str, ckpt: str, T: int, J: int, C: int, device: str):
    kw = MF_VARIANTS[variant]
    cfg = MotionFormerConfig(
        T=T, J=J, C=C,
        hidden=128, pair_hidden=32, depth=6, heads=4, pair_heads=4,
        opm_chunk=16, tri_hidden=16, **kw,
    )
    model = MotionFormer(cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Path to final.pt from train.py.")
    ap.add_argument("--model", required=True, choices=list(MF_VARIANTS.keys()),
                    help="Variant the checkpoint was trained as (full / opm_only / ...).")
    ap.add_argument("--mixed-data", default="../dataset/mixed_soma77.npz",
                    help="Path to mixed_soma77.npz (axis-fixed version).")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--sep", type=float, default=2.2,
                    help="X separation (meters) between the three skeletons.")
    ap.add_argument("--no-center-root", action="store_true",
                    help="By default each frame subtracts its root so the body stays in place.")
    args = ap.parse_args()
    center_root = not args.no_center_root

    device = "cuda" if torch.cuda.is_available() else "cpu"
    parents = build_parent_index()

    print(f"Loading dataset from {args.mixed_data} ...")
    ds = MixedMotionDataset(args.mixed_data, crop_frames=90)
    print(f"  N={ds.N}  T={ds.T}  J={ds.J}")

    print(f"Loading [{args.model}] from {args.checkpoint} ...")
    model = load_model(args.model, args.checkpoint, ds.T, ds.J, ds.C, device)

    pos_mean = ds.mean[0]    # [J, 3]
    pos_std  = ds.std[0]

    # Group indices per source so we can quickly jump to next Kimodo / AMASS / etc.
    source_to_indices: dict[str, list[int]] = {}
    for i in range(ds.N):
        source_to_indices.setdefault(str(ds.source_name[i]), []).append(i)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"\n>>> viser_compare  model={args.model}  →  http://localhost:{args.port}\n")

    # Layout: GT on left, masked input in the middle, prediction on the right.
    OFFSETS = {"GT": -args.sep, "Masked": 0.0, "Pred": args.sep}
    JOINT_COLOR = {"GT": (60, 220, 100), "Masked": (230, 150, 40), "Pred": (70, 130, 240)}
    BONE_COLOR  = {"GT": (40, 180, 80),  "Masked": (210, 130, 20), "Pred": (50, 100, 220)}

    for name, x in OFFSETS.items():
        server.scene.add_label(f"/lbl_{name.lower()}", text=name, position=(x, 0, 1.8))

    # ---- GUI ----
    with server.gui.add_folder("Sample"):
        idx_slider = server.gui.add_slider(
            "Global idx", min=0, max=ds.N - 1, step=1, initial_value=0,
        )
        kimodo_btn = server.gui.add_button("→ Next Kimodo")
        amass_btn  = server.gui.add_button("→ Next AMASS")
        hy_btn     = server.gui.add_button("→ Next HY")

    with server.gui.add_folder("Masking"):
        mask_dd = server.gui.add_dropdown(
            "Mode",
            options=["random", "joint", "time", "keyframe", "kinematic_chain", "none"],
            initial_value="random",
        )
        # Use integer 0..80 as "ratio x 100" because some viser versions balk at float step.
        ratio_pct_slider = server.gui.add_slider(
            "Ratio %", min=0, max=80, step=5, initial_value=30,
        )

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame", min=0, max=ds.T - 1, step=1, initial_value=0,
        )
        play_cb = server.gui.add_checkbox("Auto-play", initial_value=True)
        fps_slider = server.gui.add_slider(
            "FPS", min=5, max=60, step=1, initial_value=20,
        )

    with server.gui.add_folder("Info"):
        info_md = server.gui.add_markdown("Loading …")

    # ---- Scene state ----
    # Each is [T, J, 3] physical-space positions (root-relative), ready to draw.
    scene = {"gt": None, "masked": None, "pred": None, "mask": None}

    def denorm(x_nrm: np.ndarray) -> np.ndarray:
        return x_nrm * pos_std + pos_mean

    def recompute():
        idx = int(idx_slider.value)
        mode = mask_dd.value
        ratio = float(ratio_pct_slider.value) / 100.0

        sample = ds.pos_tensor[idx].unsqueeze(0).to(device)       # [1, T, J, 3]

        if mode == "none":
            mask = torch.zeros(1, ds.T, ds.J, dtype=torch.bool, device=device)
        else:
            _, mask, _ = make_masked_batch(sample, mask_ratio=ratio, mask_mode=mode)

        with torch.no_grad():
            masked_in = sample.clone()
            masked_in[mask] = 0
            out = model(masked_in, mask)
            pred_nrm = out["pos"][0].cpu().numpy()               # [T, J, 3]
        gt_nrm = sample[0].cpu().numpy()
        mask_np = mask[0].cpu().numpy()                           # [T, J]

        gt_phys = denorm(gt_nrm)
        pred_phys = denorm(pred_nrm)

        if center_root:
            gt_phys   = gt_phys   - gt_phys[:,   :1, :]
            pred_phys = pred_phys - pred_phys[:, :1, :]

        # Pred panel shows composite: GT where unmasked, pred where masked
        composite = gt_phys.copy()
        composite[mask_np] = pred_phys[mask_np]

        # Masked panel: mark missing joints as origin so the "hole" is obvious
        masked_viz = gt_phys.copy()
        masked_viz[mask_np] = 0.0

        scene["gt"]     = gt_phys
        scene["masked"] = masked_viz
        scene["pred"]   = composite
        scene["mask"]   = mask_np

        err = 0.0
        if mask_np.any():
            err = float(np.sqrt(
                ((pred_phys - gt_phys) ** 2 * mask_np[..., None]).sum()
                / (mask_np.sum() * 3 + 1e-8)
            ))

        info_md.content = (
            f"**idx** `{idx}` — **sample** `{ds.sample_name[idx]}`  \n"
            f"**source** `{ds.source_name[idx]}` — **subset** `{ds.subset[idx]}`  \n"
            f"**action** `{ds.action_label[idx]}`  \n"
            f"**mask** mode=`{mode}`  ratio={ratio:.2f}  frac={mask_np.mean():.2f}  \n"
            f"**L2 on masked (m)** `{err:.4f}`"
        )

    def bones_from_joints(joints: np.ndarray, skip: np.ndarray | None = None) -> np.ndarray:
        """Build [N_bones, 2, 3] segments. If `skip` is a [J] bool mask, any bone
        whose either endpoint is True (i.e. masked) is dropped entirely."""
        pts = []
        for j, p in enumerate(parents):
            if p < 0:
                continue
            if skip is not None and (skip[j] or skip[p]):
                continue
            pts.append(np.stack([joints[j], joints[p]], axis=0))
        return np.stack(pts, axis=0) if pts else np.zeros((0, 2, 3))

    def draw_frame(frame: int):
        if scene["gt"] is None:
            return
        mask_this_frame = scene["mask"][frame]   # [J] bool
        for name, key in (("GT", "gt"), ("Masked", "masked"), ("Pred", "pred")):
            pts = scene[key][frame].copy()
            pts[:, 0] += OFFSETS[name]
            cj = np.array([JOINT_COLOR[name]])
            cb = np.array([BONE_COLOR[name]])

            if name == "Masked":
                # Only draw the un-masked joints as spheres — so missing ones
                # visually disappear instead of collapsing to the origin.
                vis_idx = ~mask_this_frame
                vis_pts = pts[vis_idx]
                server.scene.add_point_cloud(
                    f"/{name.lower()}_joints",
                    points=vis_pts,
                    colors=np.tile(cj, (vis_pts.shape[0], 1)) if vis_pts.shape[0] > 0
                           else np.zeros((0, 3), dtype=int),
                    point_size=0.028,
                    point_shape="sparkle",
                )
                # Bones: drop any segment touching a masked joint
                b = bones_from_joints(pts, skip=mask_this_frame)
            else:
                server.scene.add_point_cloud(
                    f"/{name.lower()}_joints",
                    points=pts,
                    colors=np.tile(cj, (ds.J, 1)),
                    point_size=0.025,
                    point_shape="sparkle",
                )
                b = bones_from_joints(pts)

            server.scene.add_line_segments(
                f"/{name.lower()}_bones",
                points=b,
                colors=np.tile(cb, (b.shape[0], 2, 1)) if b.shape[0] > 0
                       else np.zeros((0, 2, 3), dtype=int),
                line_width=3.0,
            )

    # ---- Callbacks: jump buttons ----
    def jump_to(src_name: str):
        cur = int(idx_slider.value)
        idxs = source_to_indices.get(src_name, [])
        if not idxs:
            return
        nxt = next((i for i in idxs if i > cur), idxs[0])
        idx_slider.value = nxt

    @kimodo_btn.on_click
    def _(_): jump_to("kimodo")
    @amass_btn.on_click
    def _(_): jump_to("amass")
    @hy_btn.on_click
    def _(_): jump_to("hy_motion")

    # Initial render
    recompute()
    draw_frame(0)

    # ---- Main loop: poll GUI state, recompute/redraw on change ----
    last_idx    = int(idx_slider.value)
    last_mode   = mask_dd.value
    last_ratio  = int(ratio_pct_slider.value)
    last_frame  = -1
    last_tick   = time.time()

    try:
        while True:
            cur_idx   = int(idx_slider.value)
            cur_mode  = mask_dd.value
            cur_ratio = int(ratio_pct_slider.value)

            if (cur_idx != last_idx) or (cur_mode != last_mode) or (cur_ratio != last_ratio):
                recompute()
                last_idx   = cur_idx
                last_mode  = cur_mode
                last_ratio = cur_ratio
                last_frame = -1   # force redraw

            now = time.time()
            if play_cb.value and (now - last_tick) >= 1.0 / float(fps_slider.value):
                new_f = (int(frame_slider.value) + 1) % ds.T
                frame_slider.value = new_f
                last_tick = now

            cur_frame = int(frame_slider.value)
            if cur_frame != last_frame:
                draw_frame(cur_frame)
                last_frame = cur_frame

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Shutting down viser_compare.")


if __name__ == "__main__":
    main()
