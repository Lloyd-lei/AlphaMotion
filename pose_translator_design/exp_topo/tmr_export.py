"""Stage 1 of the TMR bridge: our motion -> SOMA30 world joint positions.

    tmr_export.py --n 400 --out /var/tmp/alphamotion/tmr_bridge/real.npz

TMR (nvidia/TMR-SOMA-RP-v1) is the standard text-motion retrieval judge and the
one thing we do not have: a semantic scorer that is NOT part of our training
loss.  It wants global 3D joint positions of a 30-joint SOMA skeleton, 30 fps,
in METRES (its normalisation stats top out at 1.5).

Our side already matches on two of three axes: 30 fps, and the cached root
translation is recorded in the "ardy Y-up" basis (ROOT_WORLD_meta.json), which
is the same convention.  Only the unit (cm -> m) and the skeleton differ.

SMPL22 -> SOMA30 is almost a rename; SOMA carries 8 joints we never had (a
second neck link, jaw, two eyes, four finger ends).  Those are filled by
copying the parent's position, which is a real distortion — hence the whole
point of `--zero_root`: the same export is written twice, once with the true
world translation and once with it zeroed, so the degradation from the thing
our representation actually lacks (D6) can be measured rather than assumed.

Runs in the `ardy` env; stage 2 (tmr_score.py) runs in the `kimodo` env.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402

TA = "/var/tmp/alphamotion/topo_exp/text_align"
CACHE = os.environ["V5_AMASS"]

# SOMA30 order (kimodo.skeleton.SOMASkeleton30.bone_order_names_with_parents)
# -> the SMPL22 joint that supplies it.  None = copy the parent (we have no
#    jaw / eyes / finger ends and no second neck link).
SOMA30 = [
    ("Hips", "Pelvis"), ("Spine1", "Torso"), ("Spine2", "Spine"),
    ("Chest", "Chest"), ("Neck1", "Neck"), ("Neck2", None),
    ("Head", "Head"), ("Jaw", None), ("LeftEye", None), ("RightEye", None),
    ("LeftShoulder", "L_Thorax"), ("LeftArm", "L_Shoulder"),
    ("LeftForeArm", "L_Elbow"), ("LeftHand", "L_Wrist"),
    ("LeftHandThumbEnd", None), ("LeftHandMiddleEnd", None),
    ("RightShoulder", "R_Thorax"), ("RightArm", "R_Shoulder"),
    ("RightForeArm", "R_Elbow"), ("RightHand", "R_Wrist"),
    ("RightHandThumbEnd", None), ("RightHandMiddleEnd", None),
    ("LeftLeg", "L_Hip"), ("LeftShin", "L_Knee"), ("LeftFoot", "L_Ankle"),
    ("LeftToeBase", "L_Toe"),
    ("RightLeg", "R_Hip"), ("RightShin", "R_Knee"), ("RightFoot", "R_Ankle"),
    ("RightToeBase", "R_Toe"),
]
SOMA_PARENT = {  # only needed for the 8 filled joints
    "Neck2": "Neck1", "Jaw": "Head", "LeftEye": "Head", "RightEye": "Head",
    "LeftHandThumbEnd": "LeftHand", "LeftHandMiddleEnd": "LeftHand",
    "RightHandThumbEnd": "RightHand", "RightHandMiddleEnd": "RightHand",
}


# Rest offsets of the 8 joints we do not have, in the PARENT's frame, measured
# off kimodo's own SOMA demo clip (150 frames, per-frame std 0.0000 -> rigid).
# The first export copied the parent's position instead, which put jaw, eyes and
# both finger ends at zero distance and left 8 of 30 joints degenerate.
FILL_OFFSET = {
    "Neck2": (-0.0000, 0.0771, 0.0230), "Jaw": (0.0, 0.0048, 0.0309),
    "LeftEye": (0.0321, 0.0538, 0.0759), "RightEye": (-0.0322, 0.0536, 0.0756),
    "LeftHandThumbEnd": (0.1227, -0.0322, 0.0483),
    "LeftHandMiddleEnd": (0.1901, -0.0031, -0.0003),
    "RightHandThumbEnd": (-0.1226, -0.0321, 0.0480),
    "RightHandMiddleEnd": (-0.1900, -0.0031, -0.0003),
}
# which SMPL joint carries the global rotation of that SOMA parent
FILL_ROT_SRC = {"Neck2": "Neck", "Jaw": "Head", "LeftEye": "Head",
                "RightEye": "Head", "LeftHandThumbEnd": "L_Wrist",
                "LeftHandMiddleEnd": "L_Wrist",
                "RightHandThumbEnd": "R_Wrist",
                "RightHandMiddleEnd": "R_Wrist"}


def to_soma30(p_smpl, smpl_names, Rg=None):
    """[T,22,3] SMPL positions (cm) -> [T,30,3] SOMA30 order (cm).

    Rg [T,22,3,3] global rotations places the 8 missing joints along their real
    SOMA offsets instead of collapsing them onto the parent."""
    idx = {n: i for i, n in enumerate(smpl_names)}
    out = np.zeros((p_smpl.shape[0], 30, 3), np.float64)
    pos_of = {}
    for k, (sname, src) in enumerate(SOMA30):
        if src is not None:
            out[:, k] = p_smpl[:, idx[src]]
        elif Rg is not None:
            base = pos_of[SOMA_PARENT[sname]]
            R = Rg[:, idx[FILL_ROT_SRC[sname]]]
            off = np.asarray(FILL_OFFSET[sname]) * 100.0        # m -> cm
            out[:, k] = base + np.einsum("tij,j->ti", R, off)
        else:
            out[:, k] = pos_of[SOMA_PARENT[sname]]
        pos_of[sname] = out[:, k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val_amass")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--T", type=int, default=90)
    ap.add_argument("--out", default="/var/tmp/alphamotion/tmr_bridge/real.npz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sp = np.load(f"{CACHE}/human_smpl_spec.npz", allow_pickle=True)
    names = [str(x) for x in sp["joint_names"]]

    # The cache holds LOCAL rotations (build_global -> fk_global composes them
    # down the tree); fk_pos consumes GLOBAL ones. The first export fed the raw
    # cache straight to fk_pos, which silently kept every bone length and the
    # standing height correct while displacing joints by 11.6 cm median — a pose
    # that no retriever can read. Compose first, always.
    from eval_newbody import SkeletonSpec
    spec = SkeletonSpec(str(sp["name"]), sp["parents"], sp["rest_offsets"],
                        [str(x) for x in sp["joint_names"]])

    class S:  # minimal spec for fk_pos
        parents = sp["parents"]
        rest_offsets = sp["rest_offsets"]
    pose = torch.load(f"{CACHE}/human_smpl_paired_local6d.pt", map_location="cpu")
    if isinstance(pose, dict):
        pose = pose.get("pose", next(iter(pose.values())))
    root = torch.load(f"{CACHE}/human_smpl_root_world.pt", map_location="cpu")
    if isinstance(root, dict):
        root = root.get("root_world", next(iter(root.values())))
    print(f"pose {tuple(pose.shape)}  root {tuple(root.shape)}  "
          f"joints {len(names)}", flush=True)

    items = [p for p in json.load(open(f"{TA}/pairs_{args.split}.json"))
             if p["len"] >= args.T + 2]
    byclip = {}
    for p in items:
        byclip.setdefault(p["name"], p)
    ps = list(byclip.values())
    g = np.random.default_rng(args.seed)
    if len(ps) > args.n:
        ps = [ps[i] for i in g.choice(len(ps), args.n, replace=False)]
    print(f"clips {len(ps)}", flush=True)

    P, P0, caps = [], [], []
    for p in ps:
        s = p["start"] + int(g.integers(0, max(1, p["len"] - args.T + 1)))
        loc = pose[s:s + args.T]
        if loc.shape[-1] > 6:
            loc = loc[..., :6]
        gr6, _gp, _reach = V.build_global(loc, spec, "cpu")
        R = gr6.cpu().numpy()                        # GLOBAL rot6d
        pj = V.fk_pos(R, S)                          # [T,22,3] cm, root at 0
        rw = root[s:s + args.T].numpy().reshape(-1, 3)   # [T,3] cm world
        Rg = V.rot6d_to_matrix(torch.as_tensor(R, dtype=torch.float64)).numpy() \
            if hasattr(V, "rot6d_to_matrix") else None
        soma = to_soma30(pj, names, Rg) / 100.0      # -> metres
        # Recentre HORIZONTALLY only. The vertical must stay absolute: TMR's
        # features are ground-referenced (local_joints_positions + global_root_y),
        # and the first export subtracted y as well, which buried the skeleton
        # 0.82 m underground (official SOMA demo: root y ~0.96, lowest joint
        # ~0.03; ours came out 0.06 / -0.82) and drove retrieval to chance.
        rw_m = rw / 100.0
        rw_m = rw_m - np.array([rw_m[0, 0], 0.0, rw_m[0, 2]])
        full = soma + rw_m[:, None, :]
        floor = np.percentile(full[..., 1], 1.0)     # 1st pct of all joint heights
        full[..., 1] -= floor                        # put the floor at y = 0
        P.append(full)
        # root-zeroed twin: horizontal translation removed, floor still at 0 —
        # this is exactly what our representation can produce today
        z = full.copy()
        z[..., 0] -= z[:, 0:1, 0]
        z[..., 2] -= z[:, 0:1, 2]
        P0.append(z)
        caps.append(p["caption"])
    P, P0 = np.stack(P).astype(np.float32), np.stack(P0).astype(np.float32)
    np.savez_compressed(args.out, pos=P, pos_zero_root=P0,
                        captions=np.array(caps, dtype=object),
                        fps=30, units="m", skeleton="SOMASkeleton30")
    print(f"   根高度 y 中位 {np.median(P[:, :, 0, 1]):+.3f} m  "
          f"最低关节 y 中位 {np.median(P[:, :, :, 1].min(-1)):+.3f} m  "
          f"(官方 SOMA: +0.96 / +0.03)", flush=True)
    disp = np.linalg.norm(P[:, -1, 0] - P[:, 0, 0], axis=-1)
    print(f"-> {args.out}  pos {P.shape}  净位移 中位 {np.median(disp):.2f} m "
          f"p90 {np.percentile(disp, 90):.2f} m", flush=True)
    print(f"   关节坐标范围(相对根) |v| p99 = "
          f"{np.percentile(np.abs(P0), 99):.2f} m  (TMR 统计量上限 ~1.5)", flush=True)


if __name__ == "__main__":
    main()
