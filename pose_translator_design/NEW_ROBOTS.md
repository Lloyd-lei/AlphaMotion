# New robot descriptions for zero-shot embodiment transfer

Acquired 2026-08-05. Everything lives under `/var/tmp/alphamotion/robot_models/`
(nothing written to `/home` except the render PNGs and this file).
Renders: `/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/R4_VIZ/new_robots/`

**Headline: both AgiBot targets exist and both work.** X2 Ultra and A3 are officially
open-sourced by AgibotTech with MJCF shipped in-repo, floating base already present,
zero closed kinematic loops in the body, clean 2-leg/2-arm topology. They are drop-in
for the pipeline.

---

## 1. What was found

### AgiBot X2 (灵犀 X2) — YES

* Repo: https://github.com/AgibotTech/agibot_x2_urdf
* Clone: `/var/tmp/alphamotion/robot_models/agibot_x2_urdf` (commit `77f43eb`, 2026-07-29, 278 MB)
* License: **Mulan PSL v2** (木兰宽松许可证第2版) — permissive, OSI-approved, allows use/modify/redistribute.
* Ships URDF **and** MuJoCo MJCF **and** STL meshes. Two hardware revisions:
  * `X2_URDF-v1.4.0/X2-Ultra.xml` — "X2 Ultra -N", upgraded motors, ~45 kg. **Use this one.**
  * `X2_URDF-v1.3.0/x2_ultra.xml` — original flagship, ~43.5 kg.
  * `X2_URDF-v1.3.0/x2_fist.xml` — same robot with dummy fist end-effectors.
* X2 EDU is listed as "coming soon", not in the repo yet.

### AgiBot A3 / A3 Ultra (远征 A3) — YES

* Repo: https://github.com/AgibotTech/A3-A3U-robot-model
* Clone: `/var/tmp/alphamotion/robot_models/agibot_a3` (commit `b8c8923`, 2026-08-04, 101 MB)
* License: **none declared** — no LICENSE file, README says nothing. Publicly published by
  AgiBot but legally unlicensed. Fine for internal research eval; do not redistribute.
* Four variants, all with URDF + MJCF + STL + rviz config:
  * `a3_t2d5/` (T2.5 base), `a3_t3d0/` (T3.0 base) — 31 DoF, simple sphere/box hands
  * `a3_ultra_t2d5/` (T2.5 Ultra) — 31 DoF, extra head sensors
  * `a3_ultra_t3d0/` (T3.0 Ultra) — 63 DoF, **includes 5-finger dexterous hands**
* One fix was needed: the `*_t3d0` MJCFs `<include>` a `../../terrain/mujoco/flat.xml`
  that AgiBot did not ship. I wrote it at
  `/var/tmp/alphamotion/robot_models/agibot_a3/terrain/mujoco/flat.xml`
  (skybox + checker ground plane + directional light). Without it the T3.0 models will not compile.

### AgiBot A2 (from vendored GMR) — partially usable, was broken

* Source: `'/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/external/GMR/assets/agibot_a2'`
* **GMR's shipped MuJoCo model does not load.** Two independent breakages:
  1. `mujoco/asset/*.xml` reference `../../meshes/public/*.stl`, but the meshes are in `mesh/`
     — wrong directory name, and there is no `public/` subdir.
  2. Hand assets reference `../../../s6_hand/meshes/original/` — that package is not in the
     GMR assets tree at all. Unrecoverable without the missing upstream files.
  That explains why `agibot_a2` was never registered in GMR's robot list.
* Workaround: I compiled GMR's **URDF** (`urdf/model.urdf`) instead, which uses correct
  relative mesh paths. Result at `/var/tmp/alphamotion/robot_models/agibot_a2_gmr/`.
* Caveat: that URDF is a **reduced** model — 20 hinges only. Legs are 6+6 (hip roll/yaw/pitch,
  tarsus, toe pitch, toe roll — note it is a Cassie-style tarsus/toe leg, not a knee/ankle leg),
  arms are only 4+4 (`arm_joint1..4`, so **no wrists**), and there is **no waist and no head**.
  The parallel linkage rods that the full MJCF closes with equality constraints are absent here,
  so the leg is an open chain but with non-standard joint semantics.
  Verdict: loadable and renderable, but a poor fit for a pose-retargeting benchmark. Use X2/A3.

### Bonus humanoids not already in our set

Sparse-checked out from `google-deepmind/mujoco_menagerie`
(`/var/tmp/alphamotion/robot_models/menagerie`, commit `c1a4eeb`, Apache-2.0 throughout):

* **`apptronik_apollo`** — full-size US humanoid, 1.80 m, 81 kg, 32 hinges, 2 legs 2 arms,
  no equality constraints. **Best bonus addition; drop-in clean.** Rendered.
* **`robotis_op3`** — 0.50 m, 3.1 kg hobby humanoid, 20 hinges, 2 legs 2 arms, clean. Rendered.
  Useful as an extreme-small-scale transfer probe.
* `berkeley_humanoid` — legs only (12 hinges, no arms). Not the "lite" variant we already have.
  Usable only for lower-body tests.
* `toddlerbot_2xm` — 0.50 m, 44 hinges but 13 equality constraints (4 `connect` = real closed
  loops) and 2 tendons. Closed loops will break naive FK/IK retargeting.
* `agility_cassie` — legs only, 2 **ball** joints + 4 `connect` equality constraints (closed
  four-bar linkages). Ball joints break any per-joint-angle representation. Avoid.
* `rainbow_robotics_rby1` — wheeled/torso mobile manipulator, not a legged humanoid. Skipped.

Menagerie contains no AgiBot models at all as of this commit.

---

## 2. Verified model table

All numbers from a real `mujoco.MjModel` load (mujoco 3.4.0). `height` is the true mesh
bounding box at the model's default pose.

| model | file | nbody | hinge | slide | ball | free | nu | equality | tendon | mass (kg) | height (m) | humanoid 2L/2A |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| agibot_x2_ultra (v1.4.0) | `agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml` | 33 | 31 | 0 | 0 | 1 | 31 | 0 | 0 | 44.8 | 1.32 | yes |
| agibot_x2_ultra_v130 | `agibot_x2_urdf/X2_URDF-v1.3.0/agibot_x2_ultra_v130_mjcf.xml` | 33 | 31 | 0 | 0 | 1 | 31 | 0 | 0 | 43.5 | 1.32 | yes |
| agibot_a3_t2d5 | `agibot_a3/a3_t2d5/mjcf/agibot_a3_t2d5_mjcf.xml` | 33 | 31 | 0 | 0 | 1 | 31 | 0 | 0 | 58.0 | 1.72 | yes |
| agibot_a3_t3d0 | `agibot_a3/a3_t3d0/mjcf/agibot_a3_t3d0_mjcf.xml` | 33 | 31 | 0 | 0 | 1 | 31 | 0 | 0 | 56.8 | 1.72 | yes |
| agibot_a3_ultra_t2d5 | `agibot_a3/a3_ultra_t2d5/mjcf/agibot_a3_ultra_t2d5_mjcf.xml` | 33 | 31 | 0 | 0 | 1 | 31 | 0 | 0 | 60.2 | 1.74 | yes |
| agibot_a3_ultra_t3d0 | `agibot_a3/a3_ultra_t3d0/mjcf/agibot_a3_ultra_t3d0_mjcf.xml` | 65 | 63 | 0 | 0 | 1 | 51 | 12 (joint) | 0 | 60.0 | 1.74 | yes + hands |
| agibot_a2_gmr | `agibot_a2_gmr/agibot_a2_gmr_mjcf.xml` | 51 | 20 | 0 | 0 | 1 | 0 | 0 | 0 | 69.7 | 1.69 | partial |
| apptronik_apollo | `menagerie/apptronik_apollo/apptronik_apollo.xml` | 37 | 32 | 0 | 0 | 1 | 32 | 0 | 0 | 80.9 | 1.80 | yes |
| robotis_op3 | `menagerie/robotis_op3/op3.xml` | 22 | 20 | 0 | 0 | 1 | 20 | 0 | 0 | 3.1 | 0.50 | yes |
| berkeley_humanoid | `menagerie/berkeley_humanoid/berkeley_humanoid.xml` | 14 | 12 | 0 | 0 | 1 | 12 | 0 | 0 | 16.1 | 0.68 | legs only |
| toddlerbot_2xm | `menagerie/toddlerbot_2xm/toddlerbot_2xm.xml` | 46 | 44 | 0 | 0 | 1 | 30 | 13 (4 connect) | 2 | 3.8 | 0.50 | yes, closed loops |
| agility_cassie | `menagerie/agility_cassie/cassie.xml` | 26 | 20 | 0 | **2** | 1 | 10 | 4 (connect) | 0 | 33.3 | 1.05 | legs only, closed loops |

**No sliders, no wheels, no missing meshes** in any AgiBot model. Every AgiBot MJCF already
carries a `<freejoint>` on the root body, so no free-joint injection was needed (unlike unitree
h2/r1). All meshes resolve.

---

## 3. Kinematic trees (AgiBot)

### X2 Ultra — 31 hinges, root `pelvis`, free joint `floating_base_joint`

```
pelvis
├── L/R hip_pitch → hip_roll → hip_yaw → knee → ankle_pitch → ankle_roll     (6 per leg)
└── waist_yaw → waist_pitch → torso                                          (2 waist)
    ├── L/R shoulder_pitch → shoulder_roll → shoulder_yaw → elbow
    │       → wrist_yaw → wrist_pitch → wrist_roll                           (7 per arm)
    └── head_yaw → head_pitch                                                (2 head)
```
6+6 legs, 7+7 arms, 2 waist (yaw+pitch), 2 head = 30... plus a third `waist_roll_joint`
present in the joint list = 31. Serial chain, no loops.

### A3 (T2.5 / T3.0 base) — 31 hinges, root `pelvis_link`, free joint `free_joint`

```
pelvis_link
├── waist_yaw → waist_roll → waist_pitch → torso_Link                        (3 waist)
│   ├── head_yaw → head_pitch                                                (2 head)
│   └── L/R shoulder_pitch → shoulder_roll → shoulder_yaw → elbow
│           → wrist_roll → wrist_pitch → wrist_yaw                           (7 per arm)
└── L/R hip_pitch → hip_roll → hip_yaw → knee → ankle_pitch → ankle_roll     (6 per leg)
```

Note the joint-ordering difference vs X2: A3 puts the waist chain first and uses
`wrist_roll → wrist_pitch → wrist_yaw` where X2 uses `wrist_yaw → wrist_pitch → wrist_roll`.
Body/joint names in A3 are `*_Link` (capital L) with `*_joint` (lowercase) — mixed case,
watch out for name-matching code.

### A3 Ultra T3.0 — 63 hinges

Same 31-DoF body as above **plus** two 16-DoF five-finger hands
(`{L,R}_{thumb_roll,thumb_abad,thumb_mcp,thumb_pip,thumb_dip,index_abad,index_pip,index_dip,middle_pip,middle_dip,ring_abad,ring_pip,ring_dip,pinky_abad,pinky_pip,pinky_dip}`).
Head joint is renamed `ultra_head_pitch_joint`.
The 12 equality constraints are **`mjEQ_JOINT` finger couplings only** (pip↔dip mimic) — they
are *not* closed kinematic loops in the torso/limbs. If the pipeline only drives the 31 body
joints, the hand couplings are harmless; just do not try to invert them.

---

## 4. Pipeline-compatibility notes

Good:
* All AgiBot models: single free joint at root, pure hinges elsewhere, no sliders, no ball
  joints, no tendons, no closed loops in the limbs. FK/IK and joint-angle representations work.
* Meshes are all STL and all resolve from the shipped `meshdir`.
* All have actuators (`nu == n_hinge` for the 31-DoF models), so PD-tracking rollouts work.

Watch out for:
1. **A3 T3.0 needs the terrain include I created.** If the repo is re-cloned fresh, recreate
   `agibot_a3/terrain/mujoco/flat.xml` or the model will not compile.
2. **A3 base MJCFs already contain a ground plane** (via that terrain include), so the
   `*_scene.xml` wrappers for A3 deliberately do **not** add a second floor. X2 wrappers do add one.
3. **Wrist ordering differs between X2 and A3** (yaw-pitch-roll vs roll-pitch-yaw). Any hardcoded
   wrist index mapping will silently produce wrong hand orientation.
4. **A3 uses `*_Link` capitalization** for body names; X2 uses `*_link`. Case-sensitive lookups break.
5. **A3 Ultra T3.0 has nu=51 < njnt-1=63** — 12 finger joints are driven by equality couplings,
   not actuators. Do not assume `nu == n_hinge`.
6. **A2 (GMR) has nu=0** (URDF import, no actuators) and a Cassie-style tarsus/toe leg. Skip it.
7. **X2 is short (1.32 m) and A3 is tall (1.72 m)** — a 1.3x scale spread within the same vendor.
   Good stress test for scale generalization; make sure root-height normalization is active.

---

## 5. Files created

Wrapper MJCFs + scenes (thin `<include>` of the vendor file, plus offscreen framebuffer size;
scene adds floor+lights only where the vendor model has none):

```
/var/tmp/alphamotion/robot_models/
├── agibot_x2_urdf/
│   ├── X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml       + agibot_x2_ultra_scene.xml
│   └── X2_URDF-v1.3.0/agibot_x2_ultra_v130_mjcf.xml  + agibot_x2_ultra_v130_scene.xml
├── agibot_a3/
│   ├── terrain/mujoco/flat.xml                        (written by us — required by T3.0)
│   ├── a3_t2d5/mjcf/agibot_a3_t2d5_mjcf.xml           + agibot_a3_t2d5_scene.xml
│   ├── a3_t3d0/mjcf/agibot_a3_t3d0_mjcf.xml           + agibot_a3_t3d0_scene.xml
│   ├── a3_ultra_t2d5/mjcf/agibot_a3_ultra_t2d5_mjcf.xml + agibot_a3_ultra_t2d5_scene.xml
│   └── a3_ultra_t3d0/mjcf/agibot_a3_ultra_t3d0_mjcf.xml + agibot_a3_ultra_t3d0_scene.xml
├── agibot_a2_gmr/                                     (rebuilt from GMR URDF)
│   ├── agibot_a2_gmr_mjcf.xml + agibot_a2_gmr_scene.xml
│   └── urdf/ mesh/ convex/                            (copied out of GMR)
└── menagerie/                                         (sparse checkout, Apache-2.0)
    ├── apptronik_apollo/ robotis_op3/ berkeley_humanoid/
    └── toddlerbot_2xm/ agility_cassie/ rainbow_robotics_rby1/
```

Renders (headless EGL, default pose, look at them yourself):
`/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/R4_VIZ/new_robots/`

* `agibot_x2_ultra.png` — clean humanoid, standing, all meshes present
* `agibot_x2_ultra_v130.png` — same, orange foot pads
* `agibot_a3_t2d5.png`, `agibot_a3_t3d0.png` — tall humanoid, black head shell
* `agibot_a3_ultra_t2d5.png` — fist end-effectors
* `agibot_a3_ultra_t3d0.png` — five-finger hands visible
* `agibot_a2_gmr.png` — T-pose, arms end in blocky stubs (no wrists) — shows the model's limits
* `apptronik_apollo.png`, `robotis_op3.png`

---

## 6. Recommendation

For the zero-shot embodiment-transfer test, use in priority order:

1. **`agibot_x2_ultra` (v1.4.0)** — 31 DoF, 1.32 m, clean tree, permissive license. Best target.
2. **`agibot_a3_t3d0`** — 31 DoF, 1.72 m, clean tree. Same DoF count as X2 but very different
   proportions and a different wrist ordering, which makes the X2/A3 pair a nice controlled
   contrast (same vendor, same DoF, 1.3x scale gap).
3. **`apptronik_apollo`** — 32 DoF, 1.80 m, Apache-2.0, a non-Chinese design entirely absent
   from our training distribution.
4. `agibot_a3_ultra_t3d0` if hand articulation matters; ignore the 12 finger equality couplings.
5. `robotis_op3` as an extreme-scale probe (0.50 m, 3 kg).

Skip `agibot_a2_gmr` (reduced kinematics), `agility_cassie` and `toddlerbot_2xm` (closed loops
/ ball joints), `berkeley_humanoid` and `rainbow_robotics_rby1` (not full humanoids).
