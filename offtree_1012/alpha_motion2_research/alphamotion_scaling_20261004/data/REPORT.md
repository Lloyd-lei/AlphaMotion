# Six-run data audit and executable manifest

The four requested categories now have genuine recorded motion coverage. New hand/arm examples are **recorded simulation demonstrations**, not physical robot recordings or randomly generated FK poses.

| Category | Actual source | Independent sequences | Scope |
|---|---|---:|---|
| Quadrupeds | UniMate two animated quadruped skeletons | 25 / 2,745 frames | One 31-joint body seen; one 35-joint body fully held out. Unknown metric/DOF; no bound mesh in these feature payloads. |
| Robot arm | Original Google relay-policy-learning Franka MJLs | 19 / 147,570 native frames | 15 train / 4 eval episodes. All six runs select frame_index % 40: 2,900 train + 799 eval frames. |
| Dexterous hand | Official D4RL/Minari Adroit pen demonstrations | 25 / 5,025 states | 20 train / 5 eval episodes; 4,020 + 1,005 states; 24 true hinges, including 22 finger hinges across five digit families. |
| Animated bipeds | UniMate Mixamo + two other biped skeletons | 168 / 20,166 frames | Canonical body units; same-body reconstruction/completion; no invented cross-body paired truth. |

The existing 25-joint animated hand has one 200-frame clip, with nine finger joints moving >1° locally; it stays whole-body held out. Its ring/little-finger joints are nearly static. A separate 39-joint character has only thumb/index chains (nine of 20 sequences have local digit movement), not a full dexterous hand. The Base/Arm1/Arm2/PushingBlock chain is one 100-frame sequence of unknown hardware identity. These facts are recorded individually in `coverage_audit.json`.

`external_manifest.json` seals the new payload and per-body metadata hashes. `data_manifest.json` freezes identities, rules, and 64 complete-clip evaluation examples for A/B × S/M/L. Existing sources remain on the remote host. A/B use identical sequence groups; B applies surface losses only where the independent actual-mesh binding is valid.

XWBC has 16 bodies × 30,322 sparse pilot rows, drawn from 397 complete source clips. These clips contain 655,875 original source frames. Main scaling initializes from scratch, trains 12 complete bodies, and reserves **kuavo_s45, fourier_gr3, unitree_h2, unitree_r1**. The human source remains in training. A held-out body on a train clip is **unseen target / seen motion**. A held-out body on an eval clip is **unseen target / unseen motion**. This is unseen-humanoid paired evaluation; cross-morphology UniMate evaluations are reconstruction/completion. The cached H2 anonymous 32-link tree is not the official 36-link H2 asset.

Normalization is **maximum static root-to-joint path length** with root offset excluded. It uses no motion or eval GT. Old full-cache fitted reaches and historical old-checkpoint eligibility flags are retained only as audit history, never the new normalizer/exposure truth.

The Franka derived MJCF preserves the original relay model’s real axes, origins, ranges and base transform, including its older joint-6/7 frames. Its training skeleton is seven hinges plus root and the source-defined fixed `end_effector` site (J=9). **Two real finger slides remain in recorded qpos but are excluded from rotation-only skeleton/surface loss.** No full-gripper execution claim is supported. The earlier Minari kitchen download contains noisy joint observations, so it is audit-only; the exact original MJL qpos replaces it.

Adroit/Franka geometry is metric: payload `position_native` and `rest_offsets` are metres; descriptor offsets are centimetres; pose9 divides position by static reach. `global_rot6` is own-link rest-relative global rotation: `AX @ R(q) @ R(rest).T @ AX.T`, with Z-up→Y-up axis permutation recorded in metadata. Ground/contact/root-height-supervision masks are false for both. The raw object/scene state is retained where present, but unverified object semantics are not promoted to supervised object pose/contact labels.

Full double-precision original FK maxima are 3.44e-16 m (Adroit) and 1.50e-15 m (Franka). Every serialized float32 frame was rechecked: maxima 4.03e-8 m and 6.90e-8 m. Five local contract tests pass. XWBC trajectory reader was tested on the full 3,065-frame G1 fallAndGetUp3_subject1 clip: all 96 sparse overlap frames agree within 1.92e-5 cm, actual hard contacts match exactly, and two world sole trajectories are returned. Existing multi-GB XWBC numeric arrays were not all hashed/scanned.

At external-data sealing: 846,868,528 new bytes including linked HF bytes, under the 2 GiB cap; free disk 205,462,917,120 bytes. The conservative ledger retains 160.99 GB for the existing download queue, 16 GiB reserve, and 6 GiB other independent budget. The 695,505,475-byte MJL ZIP remains compressed; selected members were streamed, never fully extracted. Original pilot and protected download processes were untouched.

New raw source identities and rights:
- Farama D4RL revision `a0fd465d1d9d253b075d532b270e03d8fb4f46e6`; dataset CC BY 4.0, [explicit D4RL data-license declaration](https://github.com/Farama-Foundation/D4RL#licenses). Attribution: DAPG / Rajeswaran et al. (RSS 2018), D4RL / Fu et al. (2020), Farama / Rodrigo de Lazcano.
- Google relay-policy-learning revision `cd70ac9334f584f86db281a1ffd3e5cbc3e5e293`; [official repository includes data and Apache-2.0 license](https://github.com/google-research/relay-policy-learning/tree/cd70ac9334f584f86db281a1ffd3e5cbc3e5e293). Franka source notices are preserved.
- Gymnasium-Robotics geometry revision `4d1ebecbc6436806cfbc0e42ebc36f594d05844e`; original top-level MIT and embedded asset notices are retained. Adroit geometry is not Allegro; no same-name model substitution was made.

Remote data root: `/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004/data`.

Read complete source clips without copying them:
```python
from trajectory_reader import load_xwbc_trajectory, load_unpaired_trajectory
case = load_xwbc_trajectory("data_manifest.json", "lafan::fallAndGetUp3_subject1", "unitree_g1_29dof")
# Metric outputs are Y-up cm; contact/sole are original source labels/geometry.
episode = load_unpaired_trajectory("data_manifest.json", "adroit_pen", 0)
# UniMate output retains canonical body units and explicitly reports metric=False.
```

`all_required_qwen_names.json` contains the 34 actual new skeleton names in original order. `mujoco_dataset_adapter.py` exposes exact recorded-qpos CPU geometry; it does not step the simulator or claim free-rotation output is executable. Stable patch bindings are independently produced in the sibling `surface/` directory (nine actual bodies; 96 patches/body). Acquisition/build scripts are retained for review, but **do not rerun extraction on the frozen payloads** for the current six-run experiment.
