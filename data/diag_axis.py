"""Axis convention diagnostic.

Pulls a single AMASS raw file and a single Kimodo cached clip, then prints
the root-translation axis distributions so we can nail down exactly which
axis is "up" in each source.

Expected (per SMPL / AMASS docs):   Z-up, Y-forward, right-handed
Expected (per Kimodo docs):          Y-up, Z-forward, right-handed

We verify both by looking at which axis of trans[:, :] has a large positive
mean (~0.8-1.0m — standing human hip height) while the other two are small.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def describe(name, x):
    """x: [T, 3]. print mean/std/min/max per axis."""
    print(f"\n=== {name} ===  shape={x.shape}")
    for i, ax in enumerate("XYZ"):
        col = x[..., i]
        print(f"  {ax}: mean={col.mean():+.3f}  std={col.std():.3f}  "
              f"range=[{col.min():+.3f}, {col.max():+.3f}]")


def inspect_amass_raw():
    # Pick a CMU walking clip (07_01 is the "walking" subject in CMU MoCap)
    candidates = list((ROOT / "amass" / "CMU" / "07").glob("*_poses.npz"))
    if not candidates:
        candidates = list((ROOT / "amass" / "CMU").rglob("*_poses.npz"))[:1]
    if not candidates:
        print("No AMASS CMU file found")
        return

    fp = candidates[0]
    print(f"\nLoading raw AMASS: {fp.relative_to(ROOT)}")
    d = np.load(fp)
    print(f"  keys: {list(d.keys())}")
    trans = d["trans"]          # [T, 3]   root world translation
    describe("AMASS raw trans[:, :]", trans)

    # Also: the first body joint (pelvis) is at trans in SMPL, but we want to
    # know where FK places joints AFTER our prepare_amass.py runs. We load the
    # processed npz instead below.


def inspect_amass_processed():
    # Already-prepared file with the bug baked in
    fp = ROOT / "amass" / "cmu_soma77.npz"
    if not fp.exists():
        print(f"\nNo processed file {fp}")
        return
    print(f"\nLoading processed AMASS: {fp.name}")
    d = np.load(fp, allow_pickle=True)
    print(f"  keys: {list(d.keys())}")
    joints = d["joints_world"]          # [N_clips, T, 77, 3]
    root   = d["root_positions"]        # [N_clips, T, 3]
    print(f"  joints_world shape: {joints.shape}")

    hips = joints[:, :, 0, :]           # Hips joint over all frames [N, T, 3]
    describe("AMASS processed Hips joint", hips.reshape(-1, 3))
    describe("AMASS processed root_positions", root.reshape(-1, 3))

    # Also inspect a foot joint — in either convention, feet should be lower
    # than hips along the "up" axis.
    # SOMA index 56 = LeftFoot (in our standard ordering). Let's read it.
    stage1_dir = ROOT.parent / "stage1"
    sys.path.insert(0, str(stage1_dir))
    from soma_skeleton import JOINT_INDEX
    foot_idx = JOINT_INDEX["LeftFoot"]
    foot = joints[:, :, foot_idx, :]
    describe(f"AMASS processed LeftFoot (idx={foot_idx})", foot.reshape(-1, 3))


def inspect_kimodo():
    stage1_dir = ROOT.parent / "stage1"
    fp = stage1_dir / "runs" / "kimodo_cache.npz"
    if not fp.exists():
        # try alternates
        for alt in ("kimodo_cache_pos_only.npz", "kimodo_smoke.npz"):
            if (stage1_dir / "runs" / alt).exists():
                fp = stage1_dir / "runs" / alt
                break
    if not fp.exists():
        print(f"\nNo Kimodo cache found under {stage1_dir / 'runs'}")
        return
    print(f"\nLoading Kimodo cache: {fp.name}")
    d = np.load(fp, allow_pickle=True)
    print(f"  keys: {list(d.keys())}")
    joints = d["joints_world"]          # [N_clips, T, 77, 3]
    print(f"  joints_world shape: {joints.shape}")

    hips = joints[:, :, 0, :]
    describe("Kimodo Hips joint", hips.reshape(-1, 3))

    sys.path.insert(0, str(stage1_dir))
    from soma_skeleton import JOINT_INDEX
    foot_idx = JOINT_INDEX["LeftFoot"]
    foot = joints[:, :, foot_idx, :]
    describe(f"Kimodo LeftFoot (idx={foot_idx})", foot.reshape(-1, 3))


def main():
    print("=" * 70)
    print("Coordinate system diagnostic")
    print("=" * 70)
    inspect_amass_raw()
    inspect_amass_processed()
    inspect_kimodo()

    print("\n" + "=" * 70)
    print("EXPECTED:")
    print("  AMASS raw trans: Z-axis mean ~0.9m (Z-up, human standing)")
    print("  Kimodo Hips:     Y-axis mean ~0.9m (Y-up, human standing)")
    print("  Foot should be BELOW hip on the up-axis by ~0.9m")
    print("=" * 70)


if __name__ == "__main__":
    main()
