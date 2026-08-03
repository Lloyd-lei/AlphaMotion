"""Render GT vs model output for BOTH embodiments of the same motion, with per-joint
orientation triads (twist shows as axes rotating about a bone). Replaces the lost
script that produced R3_REVIEW/CLEAN_soma_gt_vs_recon.png, parameterized.

Modes:
  recon      4 panels/frame: SOMA GT | SOMA recon | G1 GT | G1 recon
  translate  3 panels/frame: SRC GT | SRC->TGT translation | TGT GT
Paired cache layout: each source file contributes exactly frames_per_file (24)
contiguous, cross-embodiment-aligned rows -> --file_idx selects a motion block.

Usage:
  python render_pair.py --run /var/tmp/alphamotion/topo_exp/runs/FIX_translate
  python render_pair.py --run ... --mode translate --direction soma77:g1_30 --file_idx 3
Prints absolute output paths; judge with your eyes.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from common import rot6d_to_matrix
from dual_common import load_paired_data, build_model, geom


def draw(ax, pos, R, parents, title, box):
    """Stick figure + orientation triads. `box` = (lo[3], hi[3]) per-axis limits."""
    lo, hi = box
    ax.set_title(title, fontsize=9)
    axis_len = 0.05 * float((hi - lo).max())
    for j, p in enumerate(parents):
        if p >= 0:
            ax.plot([pos[j,0],pos[p,0]],[pos[j,1],pos[p,1]],[pos[j,2],pos[p,2]], c="#888", lw=1.2)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c="#222", s=5)
    for j in range(pos.shape[0]):
        o = pos[j]
        for k, col in enumerate(["r","g","b"]):
            d = R[j,:,k] * axis_len
            ax.plot([o[0],o[0]+d[0]],[o[1],o[1]+d[1]],[o[2],o[2]+d[2]], c=col, lw=0.9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.set_zlim(lo[2],hi[2])
    ax.view_init(elev=8, azim=-70)
    try: ax.set_box_aspect(tuple(hi - lo))
    except Exception: pass


def frame_rows(file_idx, n_frames, frames_per_file):
    base = file_idx * frames_per_file
    sub = np.linspace(0, frames_per_file - 1, n_frames).round().astype(int)
    return torch.as_tensor(base + sub, dtype=torch.long)


def panel_data(data, name, idx):
    """GT rotations/positions (cm) for rows idx of one embodiment."""
    pose = data[name]["pose"][idx]
    R = rot6d_to_matrix(pose[..., :6]).cpu().numpy()
    p = (pose[..., 6:9] * data[name]["reach"]).cpu().numpy()
    return R, p


def model_out(model, data, src, tgt, idx):
    """src rows -> decode on tgt skeleton. Returns (R, pos_cm)."""
    B = len(idx)
    with torch.no_grad():
        zp, zr = model.encode(data[src]["pose"][idx], geom(data, src, B))
        prot, ppos = model.decode(zp, zr, geom(data, tgt, B))
    return rot6d_to_matrix(prot).cpu().numpy(), (ppos * data[tgt]["reach"]).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", default=None, help="default: from run config")
    ap.add_argument("--mode", choices=["recon", "translate"], default="recon")
    ap.add_argument("--direction", default="soma77:g1_30", help="translate mode: src:tgt")
    ap.add_argument("--file_idx", type=int, default=0, help="motion block in the paired cache")
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--frames_per_file", type=int, default=24)
    ap.add_argument("--out", default=None, help="default: {run}/viz")
    args = ap.parse_args()
    device = "cuda"

    cfg = json.load(open(f"{args.run}/config.json"))
    cache = args.cache or cfg["cache"]
    out_dir = args.out or f"{args.run}/viz"
    os.makedirs(out_dir, exist_ok=True)

    data, names, N = load_paired_data(cache, device)
    model = build_model(cfg, device)
    missing, unexpected = model.load_state_dict(torch.load(f"{args.run}/model.pt", map_location=device), strict=False)
    if missing or unexpected:
        print(f"WARN state_dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()

    idx = frame_rows(args.file_idx, args.n_frames, args.frames_per_file)
    assert idx[-1] < N, f"file_idx {args.file_idx} out of range (N={N})"

    if args.mode == "recon":
        panels = []
        for name in names:
            gtR, gtp = panel_data(data, name, idx)
            prR, prp = model_out(model, data, name, name, idx)
            panels += [(f"{name} GT", gtR, gtp, data[name]["spec"].parents),
                       (f"{name} recon", prR, prp, data[name]["spec"].parents)]
        tag = f"recon_f{args.file_idx}"
    else:
        src, tgt = args.direction.split(":")
        gtR_s, gtp_s = panel_data(data, src, idx)
        trR, trp = model_out(model, data, src, tgt, idx)
        gtR_t, gtp_t = panel_data(data, tgt, idx)
        panels = [(f"{src} GT (source)", gtR_s, gtp_s, data[src]["spec"].parents),
                  (f"{src}->{tgt} translation", trR, trp, data[tgt]["spec"].parents),
                  (f"{tgt} GT (target)", gtR_t, gtp_t, data[tgt]["spec"].parents)]
        tag = f"trans_{src[:4]}2{tgt[:4]}_f{args.file_idx}"

    ncol = len(panels)
    # per-panel per-axis limits over all selected frames, 8% margin -> figure fills the box
    boxes = []
    for _, R, pos, _ in panels:
        flat = pos.reshape(-1, 3)
        lo, hi = flat.min(0), flat.max(0)
        m = 0.08 * float((hi - lo).max())
        boxes.append((lo - m, hi + m))
    frames = []
    for i in range(len(idx)):
        fig = plt.figure(figsize=(3.6 * ncol, 4.0))
        for c, (title, R, p, parents) in enumerate(panels):
            ax = fig.add_subplot(1, ncol, c + 1, projection="3d")
            draw(ax, p[i], R[i], parents, f"{title}  fr{i}", boxes[c])
        fig.suptitle("triads=joint orientation; twist shows as axes rotating about a bone", fontsize=8)
        path = f"{out_dir}/{tag}_{i:02d}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)
        frames.append(path)
    try:
        import imageio.v2 as imageio
        gif = f"{out_dir}/{tag}.gif"
        imageio.mimsave(gif, [imageio.imread(p) for p in frames], duration=0.6)
        print("GIF:", os.path.abspath(gif))
    except Exception as e:
        print("gif skip:", repr(e)[:80])
    print("PNG:", *[os.path.abspath(p) for p in frames], sep="\n  ")


if __name__ == "__main__":
    main()
