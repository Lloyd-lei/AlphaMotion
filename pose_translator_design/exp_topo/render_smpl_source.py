"""Render the SOURCE input motions as true SMPL-X body meshes.

    render_smpl_source.py <params_dir>       (env: genmo-gem — has smplx)

Consumes smpl_params_{fam}.npz from export_smpl_params.py, forwards SMPL-X
(neutral betas, zero hands), renders with a painter's-algorithm cv2 rasterizer
(same approach as alpha_motion2/scripts/fit_smplx_mesh_preview.py).

Output: R4_VIZ/mp4/exam_c_mesh/source_{fam}_smpl.mp4  @20fps (matches speedtok)
"""
from __future__ import annotations
import os, sys, glob, subprocess
import numpy as np
import torch
import cv2
import smplx

W, H, BAND, FPS = 900, 700, 46, 20
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "R4_VIZ", "mp4", "exam_c_mesh")
SMPLX_NPZ = "/home/arenalabs/Downloads/SMPLX_NEUTRAL.npz"


def render_clip(verts, faces, fam, cname, path):
    """verts [T,V,3] m Y-up -> mp4 with a fixed camera and painter shading."""
    allv = verts.reshape(-1, 3)
    center = allv.mean(0); center[1] = allv[:, 1].mean()
    span = float(np.abs(allv - center).max())
    scale = 0.42 * min(W, H) / max(span, 1e-6)
    yaw = np.deg2rad(180.0)
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0],
                   [-np.sin(yaw), 0, np.cos(yaw)]])
    light = np.array([0.35, 0.75, 0.55]); light /= np.linalg.norm(light)
    tmp = path + ".raw.mp4"
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H + BAND))
    T = verts.shape[0]
    for t in range(T):
        v = (verts[t] - center) @ Ry.T
        pts = np.empty((len(v), 2), np.int32)
        pts[:, 0] = (W / 2 + v[:, 0] * scale).astype(np.int32)
        pts[:, 1] = (H * 0.54 - v[:, 1] * scale).astype(np.int32)
        depth = v[:, 2]
        frame = np.full((H + BAND, W, 3), 24, np.uint8)
        frame[:H] = 34
        tri = v[faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        shade = np.clip(n @ light, 0, 1)
        order = np.argsort(depth[faces].mean(1))
        for fi in order:
            c = 60 + 165 * shade[fi]
            cv2.fillConvexPoly(frame, pts[faces[fi]],
                               (int(c * 0.75), int(c * 0.85), int(c)))
        cv2.putText(frame, f"SOURCE input | {fam} | SMPL-X mesh | f{t}/{T-1}",
                    (10, H + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, cname, (10, H + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 205), 1, cv2.LINE_AA)
        wr.write(frame)
    wr.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-pix_fmt",
                    "yuv420p", "-movflags", "+faststart", path], check=True)
    os.remove(tmp)


def main():
    pdir = sys.argv[1]
    mdir = os.path.join(pdir, "_model", "smplx")
    os.makedirs(mdir, exist_ok=True)
    lnk = os.path.join(mdir, "SMPLX_NEUTRAL.npz")
    if not os.path.exists(lnk):
        os.symlink(SMPLX_NPZ, lnk)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for f in sorted(glob.glob(os.path.join(pdir, "smpl_params_*.npz"))):
        d = np.load(f, allow_pickle=True)
        fam, cname = str(d["fam"]), str(d["clip"])
        T = d["body_pose"].shape[0]
        bm = smplx.create(os.path.join(pdir, "_model"), model_type="smplx",
                          gender="neutral", use_pca=False, batch_size=T).to(dev)
        with torch.no_grad():
            out = bm(global_orient=torch.tensor(d["global_orient"], device=dev),
                     body_pose=torch.tensor(d["body_pose"], device=dev),
                     transl=torch.tensor(d["transl"], device=dev))
        verts = out.vertices.cpu().numpy()
        path = f"{OUT}/source_{fam}_smpl.mp4"
        render_clip(verts, bm.faces.astype(np.int64), fam, cname, path)
        print(f"[{fam}] -> {path}", flush=True)


if __name__ == "__main__":
    main()
