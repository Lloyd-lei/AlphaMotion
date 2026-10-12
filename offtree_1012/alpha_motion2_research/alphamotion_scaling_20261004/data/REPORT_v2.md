# Effective six-run data: full gripper v2

Use `data_manifest_v2.json` and `external_manifest_v2.json` for the six formal A/B × S/M/L runs. `REPORT.md`, `data_manifest.json`, `external_manifest.json` and `relay_franka_arm` describe the preserved v1 arm-only audit. **Formal training includes Adroit plus `relay_franka_full_v2`; it excludes v1 Franka to avoid double weighting.** No new raw downloads were needed for v2.

The full Franka skeleton has 11 nodes in source order: `panda0_link0..7`, `panda0_leftfinger`, `panda0_rightfinger`, `end_effector_site`. Parents are `[-1,0,1,2,3,4,5,6,7,7,7]`. It contains seven hinges, two real prismatic finger joints and two fixed nodes. The original MJCF has no finger mimic/equality definition, so no symmetry/coupling is invented. Native qpos, all 147,570 saved frames, exact timestamps, and the 19 whole-episode groups are byte-for-byte numerically identical to v1. Formal stride40 yields 3,699 frames: 2,900 train / 799 eval (15 / 4 episodes).

Prismatic slots are 8 and 9, original qpos addresses 7 and 8. Limits are 0..0.04 metres and reference q is zero. `slide_normalized=(q-lower)/(upper-lower)`, without clipping the actual recorded simulation state. Both native positions and slider lengths are metres; descriptor offsets are centimetres. When the shared model uses centimetres, multiply `slide_position_m`, `slide_lower/upper/reference` by100. Do not scale unit axes or normalized slider values.

The actual canonical rest-frame axis of the left finger is `[-0.00159265479664744, 0.000793967287038788, 0.999998416532069]`; the right axis is its negative. It follows the original relay Panda frames, not a substituted modern Panda model. Mixed FK is:

```
P[j] = P[parent] + G[parent] @ (rest_offset[j] + axis[j]*(q[j]-q_reference[j]))
G[j] = G[parent] @ C[j]   # prismatic/fixed nodes; verified C = I
```

All 147,570 frames independently agree with original-model MuJoCo geometry: maximum 1.713e-15 m in float64; the serialized float32 payload also passes all frames at 8.001e-8 m. Rotation inheritance differs by at most 6.67e-16. Root and all ancestors are mechanically fixed for both Adroit and Franka; every recorded root rotation differs from rebased identity by at most 1.06e-20 and world root translation never changes. `root_fixed_evidence.json` supplies this source-model evidence without modifying the frozen Adroit v1 manifest or requiring target GT at inference.

The true opening trace is saved as `aperture_m` (finger-origin distance) and `aperture_signed_m` (left-origin minus right-origin projected onto the actual left slide axis). The signed value equals q_left+q_right to 4.31e-16 m. This is **finger-origin separation, not contact-pad clearance or measured object contact**. Selected observed opening ranges from 0.02829344 to 0.08016369 m. Nineteen per-episode opening/closing summaries and original time series are available.

Recorded soft-limit excursions are retained: selected left/right sliders have 318/209 out-of-range samples; maximum excursion is 0.000738199 m. A strictly bounded sigmoid head has an unavoidable selected mean slider L1 floor of 0.000313443 in normalized units (1.25377e-5 m in joint travel). Report that floor; do not clip reference truth or fill predicted slider values from GT.

Valid selected labels:

| Body | Frames | Rotation / position joint-frames each | Recorded q scalars | Ground/contact/root-height-supervision |
|---|---:|---:|---:|---|
| Adroit | 5,025 | 125,625 | 120,600 true hinges | All invalid |
| Full Franka v2 | 3,699 | 40,689 | 25,893 hinges + 7,398 slides | All invalid |

`joint_type` codes are unknown0 / fixed1 / revolute2 / prismatic3 / root4. `mechanical_valid` is true for all 11 known nodes. The old 19D hinge-only descriptor remains explicitly invalid at the two slide nodes; their real mechanics are supplied through the new typed axis/limit/slider fields. Surface finger patches must use actual mixed FK and their source body binding.

The v2 external seal SHA256 is `5f83f6eb9771a4d3a5519cc21c67e8cacfe085f58824598b34e7619ced8292e3`. At sealing, owned data plus underlying linked HF bytes totaled 952,452,617 bytes, below the 2 GiB cap; free disk was 205,349,167,104 bytes. Both v1 payload/manifest hashes are verified unchanged. `all_required_qwen_names_v2.json` retains the original names and adds the two actual finger body names (36 distinct names total).

The same XWBC 4-body holdout, full-source-clip split, static geometry-only normalization, UniMate unit/DOF masks, recorded-simulation provenance, source license notices and dataset exclusions from the prior report remain in force. `trajectory_reader.py` reads true contiguous XWBC clips and complete unpaired episodes; v2 additionally returns slider masks, normalized values, axes and limits. It never invents cross-body mapping GT. Seven independent local contract tests pass.
