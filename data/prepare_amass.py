"""Convert AMASS SMPL+H .npz into SOMA77 format — FULL-LENGTH, NATIVE FPS.

Rewritten for Stage-2 (§3 self-segmentation):
  * NO downsampling      — keep the native mocap_framerate (60/120 fps)
  * NO fixed windowing   — keep each AMASS take as ONE full-length sequence
  * attach BABEL labels  — per-frame action boundaries (the §3 ground truth)
  * variable length      — sequences differ in T, so we write ONE .npz per
                           sequence + a manifest.json (no single stacked array)

The SMPL-H -> SOMA77 joint mapping, the SOMA77 forward kinematics, and the
Z-up -> Y-up axis fix are PRESERVED VERBATIM from the original (correct, and
validated by test_axis_fix.py — do not touch).

Usage:
    python data/prepare_amass.py \
        --in-dir data/amass --amass-root data/amass \
        --babel-labels data/babel_labels.json \
        --out-dir data/amass_seq
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


# --------- Joint mapping: SMPL+H (52) -> subset of SOMA77 (77) --------- #
SMPLH_TO_SOMA = {
    0: "Hips", 1: "LeftLeg", 2: "RightLeg", 3: "Spine1", 4: "LeftShin",
    5: "RightShin", 6: "Spine2", 7: "LeftFoot", 8: "RightFoot", 9: "Chest",
    10: "LeftToeBase", 11: "RightToeBase", 12: "Neck1", 13: "LeftShoulder",
    14: "RightShoulder", 15: "Head", 16: "LeftArm", 17: "RightArm",
    18: "LeftForeArm", 19: "RightForeArm", 20: "LeftHand", 21: "RightHand",
    22: "LeftHandIndex1", 23: "LeftHandIndex2", 24: "LeftHandIndex3",
    25: "LeftHandMiddle1", 26: "LeftHandMiddle2", 27: "LeftHandMiddle3",
    28: "LeftHandPinky1", 29: "LeftHandPinky2", 30: "LeftHandPinky3",
    31: "LeftHandRing1", 32: "LeftHandRing2", 33: "LeftHandRing3",
    34: "LeftHandThumb1", 35: "LeftHandThumb2", 36: "LeftHandThumb3",
    37: "RightHandIndex1", 38: "RightHandIndex2", 39: "RightHandIndex3",
    40: "RightHandMiddle1", 41: "RightHandMiddle2", 42: "RightHandMiddle3",
    43: "RightHandPinky1", 44: "RightHandPinky2", 45: "RightHandPinky3",
    46: "RightHandRing1", 47: "RightHandRing2", 48: "RightHandRing3",
    49: "RightHandThumb1", 50: "RightHandThumb2", 51: "RightHandThumb3",
}

# Z-up (AMASS/SMPL) -> Y-up (our pipeline). v' = C v ; R' = C R C^T.
# Rotation by -90° about X; det=+1 (handedness preserved). See original
# derivation + test_axis_fix.py. DO NOT MODIFY.
C_ZUP_TO_YUP = np.array(
    [[1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, -1.0, 0.0]], dtype=np.float32,
)


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Rodrigues: [...,3] axis-angle -> [...,3,3] rotation matrix."""
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    theta_safe = np.where(theta > 1e-8, theta, 1.0)
    axis = aa / theta_safe
    cos = np.cos(theta)
    sin = np.sin(theta)
    x, y, z = axis[..., 0:1], axis[..., 1:2], axis[..., 2:3]
    zero = np.zeros_like(x)
    K = np.stack([
        np.concatenate([zero, -z, y], axis=-1),
        np.concatenate([z, zero, -x], axis=-1),
        np.concatenate([-y, x, zero], axis=-1),
    ], axis=-2)
    I = np.broadcast_to(np.eye(3), K.shape).copy()
    R = I + sin[..., None] * K + (1.0 - cos[..., None]) * (K @ K)
    R[(theta.squeeze(-1) < 1e-8)] = np.eye(3)
    return R


def convert_one(
    amass_npz: Path,
    bone_offsets_soma77: np.ndarray,        # [77, 3] fixed rest offsets
    soma_joint_index: dict,
    soma_parents: list,
    dst_fps: float,                         # <=0 => keep native (no downsample)
    min_frames: int,
) -> Optional[dict]:
    """Return ONE full-length SOMA77 sequence dict from an AMASS file."""
    try:
        d = np.load(amass_npz)
    except Exception as e:
        print(f"  skip {amass_npz.name}: load error {e}")
        return None
    if "poses" not in d or "trans" not in d:
        print(f"  skip {amass_npz.name}: missing poses/trans")
        return None

    poses = d["poses"]
    trans = d["trans"]
    src_fps = float(d.get("mocap_framerate", 60.0))
    if poses.shape[1] < 156:
        print(f"  skip {amass_npz.name}: poses dim {poses.shape[1]} < 156")
        return None

    # NO downsample by default. Only stride if an explicit dst_fps is given.
    if dst_fps and dst_fps > 0 and src_fps > dst_fps:
        stride = max(1, int(round(src_fps / dst_fps)))
        poses = poses[::stride]
        trans = trans[::stride]
        out_fps = src_fps / stride
    else:
        out_fps = src_fps                                   # keep native

    T = poses.shape[0]
    if T < min_frames:
        return None

    aa_52 = poses[:, :156].reshape(T, 52, 3)
    R_52 = axis_angle_to_matrix(aa_52)                      # [T, 52, 3, 3]

    J77 = 77
    local_rot_77 = np.tile(np.eye(3), (T, J77, 1, 1)).astype(np.float32)
    for smpl_idx, soma_name in SMPLH_TO_SOMA.items():
        local_rot_77[:, soma_joint_index[soma_name]] = R_52[:, smpl_idx]

    # FK (unchanged)
    global_rot = np.zeros_like(local_rot_77)
    global_pos = np.zeros((T, J77, 3), dtype=np.float32)
    for j in range(J77):
        p = soma_parents[j]
        if p < 0:
            global_rot[:, j] = local_rot_77[:, j]
            global_pos[:, j] = trans.astype(np.float32)
        else:
            global_rot[:, j] = np.einsum("tij,tjk->tik",
                                         global_rot[:, p], local_rot_77[:, j])
            global_pos[:, j] = (
                global_pos[:, p]
                + np.einsum("tij,j->ti", global_rot[:, p],
                            bone_offsets_soma77[j])
            )

    # Z-up -> Y-up (unchanged; validated by test_axis_fix.py)
    C, CT = C_ZUP_TO_YUP, C_ZUP_TO_YUP.T
    global_pos = (global_pos @ CT).astype(np.float32)
    trans_y = (trans.astype(np.float32) @ CT)
    local_rot_77[:, 0] = np.einsum("ij,tjk->tik", C, local_rot_77[:, 0])
    global_rot = np.einsum("ij,tkjl->tkil", C, global_rot)

    return {
        "joints_world":    global_pos.astype(np.float32),       # [T,77,3]
        "local_rot_mats":  local_rot_77.astype(np.float32),     # [T,77,3,3]
        "global_rot_mats": global_rot.astype(np.float32),       # [T,77,3,3]
        "root_positions":  trans_y.astype(np.float32),          # [T,3]
        "fps":             np.float32(out_fps),
        "T":               T,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/amass",
                    help="Directory of AMASS .npz (recursive)")
    ap.add_argument("--amass-root", default="data/amass",
                    help="Root used to form the BABEL key (relpath under it)")
    ap.add_argument("--babel-labels", default="data/babel_labels.json")
    ap.add_argument("--out-dir", default="data/amass_seq",
                    help="One .npz per sequence is written here (mirrors subset)")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="<=0 keeps native fps (default). >0 downsamples.")
    ap.add_argument("--min-frames", type=int, default=30)
    ap.add_argument("--bone-offsets",
                    default="experiment/stage1/runs/kimodo_cache.npz")
    ap.add_argument("--skel-dirs", nargs="+", default=[
        "experiment/stage1", "notebook/01_resources/scripts"])
    ap.add_argument("--limit", type=int, default=0,
                    help="Process only first N files (smoke test).")
    args = ap.parse_args()

    import sys
    for sd in args.skel_dirs:
        if Path(sd).is_dir():
            sys.path.insert(0, sd)
    from soma_skeleton import (build_parent_index, JOINT_INDEX,
                               compute_bone_offsets)

    kd = np.load(args.bone_offsets, allow_pickle=True)
    bone_offsets = compute_bone_offsets(kd["joints_world"],
                                        kd["global_rot_mats"])
    parents = build_parent_index()

    babel = {}
    bp = Path(args.babel_labels)
    if bp.exists():
        babel = json.loads(bp.read_text())
        print(f"BABEL labels loaded: {len(babel)} sequences")
    else:
        print(f"[warn] {bp} not found — sequences will have no BABEL labels")

    amass_root = Path(args.amass_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.in_dir).rglob("*_poses.npz"))
    if args.limit:
        files = files[: args.limit]
    print(f"Found {len(files)} AMASS sequences under {args.in_dir}")

    manifest = []
    n_ok = n_babel = n_frame = 0
    for i, fp in enumerate(files):
        seq = convert_one(fp, bone_offsets, JOINT_INDEX, parents,
                          args.fps, args.min_frames)
        if seq is None:
            continue
        try:
            rel = fp.resolve().relative_to(amass_root).as_posix()
        except ValueError:
            rel = fp.name
        b = babel.get(rel, {})
        seq_labels = b.get("seq_labels", [])
        frame_labels = b.get("frame_labels", [])
        if b:
            n_babel += 1
        if frame_labels:
            n_frame += 1

        subset = rel.split("/")[0] if "/" in rel else "misc"
        stem = fp.stem
        # IMPORTANT: keep the FULL relpath (incl. subject dir). AMASS file
        # stems collide across subjects (e.g. every BioMotionLab subject has
        # '0030_scamper_poses.npz') — flattening to subset/stem overwrites.
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            out_path,
            joints_world=seq["joints_world"],
            local_rot_mats=seq["local_rot_mats"],
            global_rot_mats=seq["global_rot_mats"],
            root_positions=seq["root_positions"],
            fps=seq["fps"],
            babel_seq_labels=np.array(seq_labels, dtype=object),
            babel_frame_labels=np.array(frame_labels, dtype=object),
            relpath=rel,
            sample_name=stem,
            subset=subset,
        )
        manifest.append({
            "relpath": rel, "subset": subset, "sample_name": stem,
            "file": str(out_path),
            "T": int(seq["T"]), "fps": float(seq["fps"]),
            "dur_s": round(seq["T"] / float(seq["fps"]), 2),
            "has_babel": bool(b),
            "n_seq_labels": len(seq_labels),
            "n_frame_labels": len(frame_labels),
        })
        n_ok += 1
        if n_ok % 200 == 0:
            print(f"  {i+1}/{len(files)} files -> {n_ok} seqs "
                  f"({n_frame} with frame boundaries)")

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False))

    if manifest:
        Ts = np.array([m["T"] for m in manifest])
        durs = np.array([m["dur_s"] for m in manifest])
        print(f"\nDONE. {n_ok} sequences -> {out_dir}")
        print(f"  with any BABEL label   : {n_babel}")
        print(f"  with frame boundaries  : {n_frame}  (§3 ground truth)")
        print(f"  length frames  min/med/max: {Ts.min()}/{int(np.median(Ts))}"
              f"/{Ts.max()}")
        print(f"  duration sec   min/med/max: {durs.min():.1f}/"
              f"{np.median(durs):.1f}/{durs.max():.1f}")
        print(f"  total motion hours        : {durs.sum()/3600:.1f}")
        print(f"  manifest -> {out_dir/'manifest.json'}")
    else:
        print("No sequences produced — check --in-dir.")


if __name__ == "__main__":
    main()
