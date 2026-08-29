"""GENMO/GEM output -> AlphaMotion input.  The front door for both of lloyd's
paths: video->motion (a data engine) and text->motion (a prompt front-end).

    genmo_ingest.py --smpl <genmo>/smpl_params.pt --out /var/tmp/.../clip.pt

GENMO writes SMPL body params:
    global_orient [T,3]  body_pose [T,63]=21x3 axis-angle (LOCAL)  transl [T,3] m
Greenwich eats what the paired cache holds: **global** rot6d [T,22,6] (the file
is named *_local6d.pt but fk_pos documents and consumes global rotations —
verified 0814 by FKing the cache and getting sane bone lengths and a 1.33 m
head-to-foot), plus root world translation in cm, Y-up.

So the adapter is: axis-angle -> local R -> accumulate down the SMPL tree ->
global R -> rot6d; transl m -> cm.  Joint order is already identical (GENMO's
own alpha_motion_synergy script lists the same 22 names in the same order),
which is why this bridge is nearly free.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402

# SMPL22 kinematic tree, same order as human_smpl_spec.npz joint_names
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
                16, 17, 18, 19]


def aa_to_R(aa):
    """axis-angle [..,3] -> rotation matrices [..,3,3] (Rodrigues)."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = np.divide(aa, np.where(th < 1e-8, 1.0, th))
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    s, c = np.sin(th)[..., None], np.cos(th)[..., None]
    return I + s * K + (1 - c) * (K @ K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smpl", required=True, help="GENMO smpl_params.pt")
    ap.add_argument("--space", default="global",
                    choices=("global", "incam"),
                    help="body_params_global is the world-frame one; incam is "
                         "camera-relative and must NOT be used for motion")
    ap.add_argument("--out", required=True)
    ap.add_argument("--segment", default=None,
                    help="only this segment_info entry type, e.g. 'text';\n                         GEM's demo needs a video for intrinsics even when\n                         you only want the prompted part")
    args = ap.parse_args()

    d = torch.load(args.smpl, map_location="cpu", weights_only=False)
    key = f"body_params_{args.space}"
    if key not in d:
        raise SystemExit(f"{key} not in {list(d)}")
    p = d[key]
    sl = slice(None)
    if args.segment:
        segs = [x for x in d.get("segment_info", []) if x["type"] == args.segment]
        if not segs:
            raise SystemExit(f"no {args.segment} segment in {list(d)}")
        sl = slice(segs[0]["start"], segs[0]["end"])
        print(f"segment '{args.segment}' {sl.start}:{sl.stop}"
              + (f"  prompt: {segs[0].get('caption','')}" if segs[0].get("caption") else ""),
              flush=True)
    go = p["global_orient"].float().numpy().reshape(-1, 1, 3)[sl]
    bp = p["body_pose"].float().numpy().reshape(-1, 21, 3)[sl]
    aa = np.concatenate([go, bp], 1)                       # [T,22,3] LOCAL
    tr = p.get("transl", torch.zeros(len(p["body_pose"]), 3))\
        .float().numpy()[sl]
    Rl = aa_to_R(aa)                                       # [T,22,3,3] local
    Rg = np.zeros_like(Rl)
    for j, par in enumerate(SMPL_PARENTS):
        Rg[:, j] = Rl[:, j] if par < 0 else Rg[:, par] @ Rl[:, j]
    rot6d = Rg[..., :, :2].transpose(0, 1, 3, 2).reshape(len(aa), 22, 6)

    # sanity: FK the result and check it looks like a human
    sp = np.load(f"{os.environ['V5_AMASS']}/human_smpl_spec.npz",
                 allow_pickle=True)

    class S:
        parents = sp["parents"]
        rest_offsets = sp["rest_offsets"]
    names = [str(x) for x in sp["joint_names"]]
    pj = V.fk_pos(rot6d, S)                                # cm, root at origin
    h = float(np.mean(pj[:, names.index("Head"), 1] -
                      pj[:, names.index("L_Ankle"), 1]))
    ub = float(np.linalg.norm(pj[:, names.index("L_Elbow")] -
                              pj[:, names.index("L_Shoulder")], axis=-1).mean())
    print(f"frames {len(aa)}  头-脚 {h:.1f} cm  上臂 {ub:.1f} cm "
          f"(语料参考: ~133 cm / ~26 cm)", flush=True)
    if not (100 < h < 200):
        print("!! 身高不在合理范围,父子链或坐标轴可能不对", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"rot6d": torch.from_numpy(rot6d).float(),
                "root_world_cm": torch.from_numpy(tr * 100.0).float(),
                "src": args.smpl, "space": args.space, "fps": 30},
               args.out)
    print(f"-> {args.out}", flush=True)


if __name__ == "__main__":
    main()
