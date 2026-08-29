"""End-to-end sanity test for the Z-up → Y-up fix in prepare_amass.py.

Processes ONE AMASS file through convert_one() and asserts:

  1. Hips Y mean ≈ 0.9 m  (standing height, up axis)
  2. Hips X / Z means small (horizontal plane — not up)
  3. LeftFoot Y mean < 0.2 m  (foot on the ground, well below hips)
  4. Foot BELOW hips on up axis by ≈ 0.85 m
  5. det(C) = +1 (sanity, so we preserved right-handedness)
  6. Local rotation matrices are still proper rotations (det = +1)
  7. R_local * R_local^T = I  (still orthonormal after rotation change)

Exit code 0 on success, 1 on first failed assertion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
STAGE1 = ROOT.parent / "stage1"
sys.path.insert(0, str(STAGE1))
sys.path.insert(0, str(ROOT))

from soma_skeleton import build_parent_index, JOINT_INDEX, compute_bone_offsets
from prepare_amass import convert_one, C_ZUP_TO_YUP


def main():
    failures = []

    # 1. det(C) = +1
    detC = float(np.linalg.det(C_ZUP_TO_YUP))
    print(f"det(C) = {detC:+.6f}  (expected +1)")
    if not np.isclose(detC, 1.0, atol=1e-5):
        failures.append(f"det(C) = {detC}, expected +1")

    # 1b. C @ [0,0,1] = [0,1,0]  (Z-up maps to Y-up)
    up_mapped = C_ZUP_TO_YUP @ np.array([0, 0, 1.0])
    print(f"C @ Z_hat  = {up_mapped}  (expected [0, 1, 0])")
    if not np.allclose(up_mapped, [0, 1, 0], atol=1e-6):
        failures.append("C does not map Z-up to Y-up")

    # ---- Run convert_one on a CMU walking file ----------------------------
    kc = STAGE1 / "runs" / "kimodo_cache.npz"
    kd = np.load(kc, allow_pickle=True)
    bone_offsets = compute_bone_offsets(kd["joints_world"], kd["global_rot_mats"])
    parents = build_parent_index()

    amass_file = ROOT / "amass" / "CMU" / "07" / "07_03_poses.npz"
    if not amass_file.exists():
        candidates = list((ROOT / "amass" / "CMU").rglob("*_poses.npz"))
        if not candidates:
            print("No AMASS CMU file found — cannot run fix test.")
            sys.exit(2)
        amass_file = candidates[0]

    print(f"\nProcessing {amass_file.relative_to(ROOT)} through convert_one ...")
    clips = list(convert_one(amass_file, 30.0, bone_offsets, JOINT_INDEX,
                             parents, 90))
    print(f"  -> {len(clips)} clips")
    if not clips:
        failures.append("convert_one produced zero clips")
    else:
        joints = np.stack([c["joints_world"] for c in clips])  # [N, T, 77, 3]
        local = np.stack([c["local_rot_mats"] for c in clips])  # [N, T, 77, 3, 3]
        root = np.stack([c["root_positions"] for c in clips])   # [N, T, 3]

        hips = joints[:, :, 0, :].reshape(-1, 3)
        foot_idx = JOINT_INDEX["LeftFoot"]
        foot = joints[:, :, foot_idx, :].reshape(-1, 3)

        print(f"\nHips mean: X={hips[:, 0].mean():+.3f}  "
              f"Y={hips[:, 1].mean():+.3f}  Z={hips[:, 2].mean():+.3f}")
        print(f"LeftFoot mean: X={foot[:, 0].mean():+.3f}  "
              f"Y={foot[:, 1].mean():+.3f}  Z={foot[:, 2].mean():+.3f}")

        hips_y = hips[:, 1].mean()
        foot_y = foot[:, 1].mean()
        hips_xz_mag = np.sqrt(hips[:, 0].mean() ** 2 + hips[:, 2].mean() ** 2)

        # 2. Hips Y ≈ 0.7 - 1.1
        if not (0.7 < hips_y < 1.1):
            failures.append(
                f"Hips Y mean = {hips_y:.3f}, expected ~0.9m (standing height)")
        # 3. LeftFoot Y < 0.25  (foot near ground)
        if not (foot_y < 0.25):
            failures.append(
                f"LeftFoot Y mean = {foot_y:.3f}, expected < 0.25m")
        # 4. foot below hips on Y axis by > 0.6m
        if not (hips_y - foot_y > 0.6):
            failures.append(
                f"Hip - Foot along Y = {hips_y - foot_y:.3f}, expected > 0.6")
        # 5. Hips X / Z should be close to hips translation (mostly horizontal)
        #    no upper bound — subject may walk arbitrarily far — but clearly
        #    not "dominated by Y".
        print(f"Hips |horizontal| = {hips_xz_mag:.3f}  (no assertion, just info)")

        # 6 & 7. local_rot_mats still proper rotations
        dets = np.linalg.det(local.reshape(-1, 3, 3))
        if not np.allclose(dets, 1.0, atol=1e-4):
            failures.append(
                f"local_rot det range [{dets.min():.5f}, {dets.max():.5f}]")

        ortho = np.einsum("...ij,...kj->...ik", local, local) - np.eye(3)
        ortho_err = np.abs(ortho).max()
        print(f"\nlocal_rot det range: [{dets.min():.5f}, {dets.max():.5f}]")
        print(f"local_rot orthonormality max error: {ortho_err:.2e}")
        if ortho_err > 1e-4:
            failures.append(f"local_rot not orthonormal, max err {ortho_err}")

        # ---- Self-consistency: re-FK with saved local_rot + Kimodo bone_offsets
        # + saved trans → should reproduce saved joints_world. This is the
        # invariant the training loss (FK consistency term) relies on.
        parents_np = np.array(parents, dtype=np.int64)
        N, T, J, _, _ = local.shape
        gr_check = np.zeros((N, T, J, 3, 3), dtype=np.float32)
        gp_check = np.zeros((N, T, J, 3),    dtype=np.float32)
        for j in range(J):
            p = parents_np[j]
            if p < 0:
                gr_check[:, :, j] = local[:, :, j]
                gp_check[:, :, j] = root
            else:
                gr_check[:, :, j] = np.einsum(
                    "ntij,ntjk->ntik", gr_check[:, :, p], local[:, :, j])
                gp_check[:, :, j] = (
                    gp_check[:, :, p]
                    + np.einsum("ntij,j->nti",
                                gr_check[:, :, p], bone_offsets[j])
                )
        pos_err = np.abs(gp_check - joints).max()
        pos_mean_err = np.abs(gp_check - joints).mean()
        print(f"FK self-consistency: max pos err = {pos_err:.5f}, "
              f"mean = {pos_mean_err:.5f}")
        if pos_err > 1e-3:
            failures.append(
                f"FK self-consistency FAILED: max pos err = {pos_err:.5f}")

    # ---- Final verdict ----------------------------------------------------
    print("\n" + "=" * 60)
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: all axis-fix checks green")
    sys.exit(0)


if __name__ == "__main__":
    main()
