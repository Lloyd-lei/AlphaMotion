"""Solid-body (capsule-mesh) motion rendering via headless MuJoCo — GT vs model
output side by side, continuous 30fps clips from the temporal cache, saved as GIF.

Works for ANY topology: each bone becomes a free-floating capsule whose world pose
is set per frame from the global joint positions (so it renders SOMA, G1, and any
future robot identically, for recon or translation outputs).

Usage:
  python render_mesh.py --run .../runs/FIX_translate --mode translate --direction soma77:g1_30 --seq_idx 12
  python render_mesh.py --run .../runs/FIX_translate --mode recon --seq_idx 12
Prints the absolute GIF path. Judge with your eyes.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from dual_common import load_paired_data, build_model, geom

RADIUS = 0.02          # capsule radius (m)
COL_GT = "0.55 0.57 0.60 1"
COL_PRED = "0.95 0.55 0.15 1"


def build_xml(specs_colors):
    """One free capsule body per bone, for each (spec, color, x_offset) panel."""
    bodies = []
    for pi, (spec, col, _) in enumerate(specs_colors):
        for j, p in enumerate(spec.parents):
            if p < 0:
                continue
            L = max(float(np.linalg.norm(spec.rest_offsets[j])) / 100.0, 1e-3)  # cm -> m
            bodies.append(
                f'<body name="p{pi}b{j}"><freejoint/>'
                f'<geom type="capsule" size="{RADIUS}" fromto="0 0 0 0 0 {L:.4f}"'
                f' rgba="{col}" mass="0.1"/></body>')
    return f"""<mujoco>
  <visual><headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
          <global offwidth="1920" offheight="1080"/></visual>
  <worldbody>
    <light pos="0 -2 3" dir="0 0.5 -1"/>
    <geom type="plane" size="6 6 0.1" rgba="0.9 0.9 0.9 1" pos="0 0 -1.2"/>
    {''.join(bodies)}
  </worldbody>
</mujoco>"""


def quat_z_to(v):
    """Quaternion rotating +z onto direction v. [..., wxyz]"""
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    c = v @ z
    axis = np.cross(np.broadcast_to(z, v.shape), v)
    s = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = np.where(s > 1e-6, axis / (s + 1e-12), np.array([1.0, 0.0, 0.0]))
    ang = np.arctan2(s.squeeze(-1), c)
    hw = np.cos(ang / 2)[..., None]
    xyz = axis * np.sin(ang / 2)[..., None]
    return np.concatenate([hw, xyz], -1)


def set_frame(mdata, specs_colors, positions):
    """positions: list of [J,3] world joint positions (m) per panel."""
    qi = 0
    for pi, (spec, _, xoff) in enumerate(specs_colors):
        pos = positions[pi]
        for j, p in enumerate(spec.parents):
            if p < 0:
                continue
            a, b = pos[p].copy(), pos[j].copy()
            a[0] += xoff; b[0] += xoff
            d = b - a
            if np.linalg.norm(d) < 1e-8:
                d = np.array([0.0, 0.0, 1e-6])
            mdata.qpos[qi:qi + 3] = a
            mdata.qpos[qi + 3:qi + 7] = quat_z_to(d[None])[0]
            qi += 7


def to_zup(pos):
    """Graphics convention (Y up) -> MuJoCo (Z up): (x,y,z) -> (x,-z,y)."""
    return np.stack([pos[..., 0], -pos[..., 2], pos[..., 1]], -1)


def model_positions(model, data, src, tgt, idx, mode):
    B = len(idx)
    with torch.no_grad():
        zp, zr = model.encode(data[src]["pose"][idx], geom(data, src, B))
        prot, ppos = model.decode(zp, zr, geom(data, tgt, B))
    return to_zup((ppos * data[tgt]["reach"]).cpu().numpy() / 100.0)   # cm -> m


def gt_positions(data, name, idx):
    return to_zup((data[name]["pose"][idx][..., 6:9] * data[name]["reach"]).cpu().numpy() / 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", default="/var/tmp/alphamotion/topo_exp/cache_paired_seq")
    ap.add_argument("--mode", choices=["recon", "translate"], default="translate")
    ap.add_argument("--direction", default="soma77:g1_30")
    ap.add_argument("--seq_idx", type=int, default=0, help="sequence index in the temporal cache")
    ap.add_argument("--max_frames", type=int, default=120)
    ap.add_argument("--stride", type=int, default=2, help="30fps source -> gif at 30/stride fps")
    ap.add_argument("--size", type=int, nargs=2, default=[960, 480])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda"

    cfg = json.load(open(f"{args.run}/config.json"))
    out_dir = args.out or f"{args.run}/viz"
    os.makedirs(out_dir, exist_ok=True)
    data, names, N = load_paired_data(args.cache, device)
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(f"{args.run}/model.pt", map_location=device), strict=False)
    model.eval()

    seqs = json.load(open(f"{args.cache}/sequences.json"))
    s = seqs[args.seq_idx]
    rows = torch.arange(s["start"], s["start"] + min(s["len"], args.max_frames), args.stride)
    print(f"sequence '{s['name']}' len={s['len']} rendering {len(rows)} frames")

    gap = 1.6
    if args.mode == "translate":
        src, tgt = args.direction.split(":")
        panels = [(data[src]["spec"], COL_GT, -gap), (data[tgt]["spec"], COL_PRED, 0.0),
                  (data[tgt]["spec"], COL_GT, gap)]
        pos_seq = [gt_positions(data, src, rows),
                   model_positions(model, data, src, tgt, rows, args.mode),
                   gt_positions(data, tgt, rows)]
        labels = f"{src} GT | {src}->{tgt} PRED (orange) | {tgt} GT"
        tag = f"mesh_trans_{src}2{tgt}_s{args.seq_idx}"
    else:
        panels, pos_seq = [], []
        for i, name in enumerate(names):
            panels += [(data[name]["spec"], COL_GT, (2 * i - 1.5) * gap),
                       (data[name]["spec"], COL_PRED, (2 * i - 0.5) * gap)]
            pos_seq += [gt_positions(data, name, rows),
                        model_positions(model, data, name, name, rows, args.mode)]
        labels = " | ".join(f"{n} GT,recon" for n in names)
        tag = f"mesh_recon_s{args.seq_idx}"

    mmodel = mujoco.MjModel.from_xml_string(build_xml(panels))
    mdata = mujoco.MjData(mmodel)
    renderer = mujoco.Renderer(mmodel, height=args.size[1], width=args.size[0])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, -0.1]; cam.distance = 3.6 + gap * len(panels) * 0.35
    cam.elevation = -12; cam.azimuth = 90

    frames = []
    for t in range(len(rows)):
        set_frame(mdata, panels, [p[t] for p in pos_seq])
        mujoco.mj_forward(mmodel, mdata)
        renderer.update_scene(mdata, cam)
        frames.append(renderer.render().copy())
    import imageio.v2 as imageio
    gif = f"{out_dir}/{tag}.gif"
    imageio.mimsave(gif, frames, duration=args.stride / 30.0, loop=0)
    print("panels:", labels)
    print("GIF:", os.path.abspath(gif))


if __name__ == "__main__":
    main()
