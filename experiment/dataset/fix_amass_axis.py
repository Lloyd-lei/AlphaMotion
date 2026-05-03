"""Fast post-hoc Z-up → Y-up fix for already-processed AMASS npz files.

The OLD prepare_amass.py produced Z-up joint positions (Hips Z≈0.9, Foot Z≈0).
All downstream Stage 1.5 training used that data with coord mismatch relative
to Kimodo's Y-up convention, causing the PCA-by-source separation seen in the
diagnostic images.

Rather than re-running SMPL-H FK on ~42k clips (slow, would take hours), we
apply the change-of-basis transform directly to the saved arrays:

  positions        v_new  = v_old @ C.T                 (C = rot(-90° around X))
  trans            same
  root local_rot   C @ R_old                              (only joint 0)
  all global_rot   C @ R_old
  non-root local_rot / bone_offsets: UNCHANGED
                                     (invariant under global frame change)

This preserves FK self-consistency with Kimodo's Y-up bone_offsets, verified
in test_axis_fix.py.

Usage:
    python fix_amass_axis.py --dry-run   # preview on cmu_soma77.npz
    python fix_amass_axis.py             # rewrite all amass/*_soma77.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prepare_amass import C_ZUP_TO_YUP


def fix_file(in_path: Path, out_path: Path, verify: bool = True):
    print(f"\nFixing {in_path.name}  →  {out_path.name}")
    d = np.load(in_path, allow_pickle=True)
    keys = list(d.keys())
    print(f"  keys: {keys}")

    joints = d["joints_world"].astype(np.float32)       # [N, T, 77, 3]
    local  = d["local_rot_mats"].astype(np.float32)     # [N, T, 77, 3, 3]
    glob_  = d["global_rot_mats"].astype(np.float32)    # [N, T, 77, 3, 3]
    root   = d["root_positions"].astype(np.float32)     # [N, T, 3]
    meta   = d["meta"]

    C  = C_ZUP_TO_YUP
    CT = C.T

    joints_y = joints @ CT                              # (... 3) @ (3,3)
    root_y   = root   @ CT

    # Root local rotation only (joint 0): left-multiply by C
    local_y = local.copy()
    local_y[:, :, 0] = np.einsum("ij,ntjk->ntik", C, local[:, :, 0])

    # All global rotations: left-multiply by C
    glob_y = np.einsum("ij,ntkjl->ntkil", C, glob_)

    # ---- Quick sanity ----
    # Hips Y should be ~0.9, LeftFoot Y should be ~0, on the first few clips.
    sys.path.insert(0, str(Path(__file__).parent.parent / "stage1"))
    from soma_skeleton import JOINT_INDEX
    foot_idx = JOINT_INDEX["LeftFoot"]
    n_show = min(3, joints_y.shape[0])
    hips_y = joints_y[:n_show, :, 0, 1].mean()
    foot_y = joints_y[:n_show, :, foot_idx, 1].mean()
    print(f"  Post-fix: Hips Y mean={hips_y:+.3f}   LeftFoot Y mean={foot_y:+.3f}")
    if not (0.5 < hips_y < 1.3 and foot_y < 0.3):
        print(f"  WARNING: unexpected Y values — please review")

    if verify:
        # FK self-consistency on the first 5 clips: re-FK with Kimodo bone_offsets
        # must reproduce the saved joints_y.
        from soma_skeleton import (
            build_parent_index, compute_bone_offsets,
        )
        kc = Path(__file__).parent.parent / "stage1" / "runs" / "kimodo_cache.npz"
        km = np.load(kc, allow_pickle=True)
        bo = compute_bone_offsets(km["joints_world"], km["global_rot_mats"])
        parents = np.array(build_parent_index())

        Nv = min(5, joints_y.shape[0])
        T = joints_y.shape[1]
        J = 77
        gr_check = np.zeros((Nv, T, J, 3, 3), dtype=np.float32)
        gp_check = np.zeros((Nv, T, J, 3), dtype=np.float32)
        for j in range(J):
            p = parents[j]
            if p < 0:
                gr_check[:, :, j] = local_y[:Nv, :, j]
                gp_check[:, :, j] = root_y[:Nv]
            else:
                gr_check[:, :, j] = np.einsum(
                    "ntij,ntjk->ntik", gr_check[:, :, p], local_y[:Nv, :, j])
                gp_check[:, :, j] = (
                    gp_check[:, :, p]
                    + np.einsum("ntij,j->nti", gr_check[:, :, p], bo[j])
                )
        err = np.abs(gp_check - joints_y[:Nv]).max()
        print(f"  FK self-consistency (5 clips): max err = {err:.6f}")
        if err > 1e-3:
            print(f"  ERROR: FK consistency broken. Not writing output.")
            return False

    # ---- Write ----
    np.savez(
        out_path,
        joints_world    = joints_y.astype(np.float32),
        local_rot_mats  = local_y.astype(np.float32),
        global_rot_mats = glob_y.astype(np.float32),
        root_positions  = root_y.astype(np.float32),
        meta            = meta,
    )
    print(f"  Saved {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--amass-dir",
                   default=str(Path(__file__).parent / "amass"))
    p.add_argument("--dry-run", action="store_true",
                   help="Only fix cmu and write to _yfixed variant.")
    p.add_argument("--suffix", default="_yfixed",
                   help="Output suffix instead of overwriting. Empty = overwrite.")
    args = p.parse_args()

    amass_dir = Path(args.amass_dir)
    npz_files = sorted(amass_dir.glob("*_soma77.npz"))
    npz_files = [f for f in npz_files if "_yfixed" not in f.stem]

    if args.dry_run:
        npz_files = [f for f in npz_files if f.stem.startswith("cmu")][:1]
        print(f"Dry run: only {npz_files}")

    print(f"Will fix {len(npz_files)} files:")
    for f in npz_files:
        print(f"  {f.name}")

    for fp in npz_files:
        if args.suffix:
            out = fp.with_name(fp.stem + args.suffix + ".npz")
        else:
            out = fp.with_name(fp.stem + ".tmp.npz")
        ok = fix_file(fp, out)
        if not ok:
            print(f"  aborting remaining files due to failure")
            return
        if not args.suffix:
            # Atomic overwrite: mv new to old
            out.replace(fp)
            print(f"  replaced {fp}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
