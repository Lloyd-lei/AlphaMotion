# OmniRetarget Inventory

Downloaded 2026-08-03. Workspace: `/var/tmp/alphamotion/omniretarget/`
(CPU/network only; nothing written to /home data dirs, no GPU touched.)

## What it is

OmniRetarget (Amazon FAR, arXiv 2509.26633) = interaction-preserving retargeting:
build an "interaction mesh" over human + object + terrain, minimize Laplacian
deformation under kinematic constraints, output kinematically feasible G1
trajectories with no foot-skate / no penetration. Their RL policies train on
these and transfer zero-shot to a real Unitree G1.

## Code repo — `/var/tmp/alphamotion/omniretarget/holosoma/` (509 MB)

No standalone OmniRetarget repo exists; the official release lives inside
`amazon-far/holosoma` (confirmed via omniretarget.github.io "Code" link).
License: **Apache 2.0**. Monorepo with three packages under `src/`:

- `holosoma/` — RL training (PPO/FastSAC; IsaacGym/IsaacSim/MJWarp). Locomotion + whole-body tracking.
- `holosoma_inference/` — sim-to-sim / sim-to-real deployment.
- `holosoma_retargeting/` — **the OmniRetarget code.** Key entry points:
  - `holosoma_retargeting/examples/robot_retarget.py` — single-sequence retarget
    (task types: `robot_only`, `object_interaction`, `climbing`; data formats: smplh / mocap / lafan).
  - `holosoma_retargeting/examples/parallel_robot_retarget.py` — batch version; `--augmentation` flag generates object/terrain-scaled + rot/trans variants.
  - `holosoma_retargeting/src/interaction_mesh_retargeter.py` — core optimizer (cvxpy + MuJoCo + trimesh; viser for viz).
  - `holosoma_retargeting/data_conversion/convert_data_format_mj.py` — converts retargeted npz to MuJoCo-order qpos + FK velocities for the tracking trainer (their equivalent of our bridge; good reference).
  - `holosoma_retargeting/data_utils/prep_amass_smplx_for_rt.py`, `extract_global_positions.py` — input prep (AMASS SMPL-X, LAFAN bvh -> world joint positions `(T, J, 3)`).

Robot support: **G1 29-DoF is the primary target** (`models/g1/g1_29dof.xml/.urdf`,
`g1_29dof_spherehand`, `g1_29dof_w_largebox.xml`); models also included for
**H1** and **Booster T1** (training framework officially supports G1 + T1).
Input requirement for custom data: world joint positions `(T, J, 3)` + edits to
`demo_joints`/`joints_mapping` in `config_types/data_type.py`.

Deps note: pins `numpy==2.3.5` (2.4 breaks yourdfpy); uses cvxpy, mujoco, drake not required for retargeting (drake only for the dataset viz script).

## Dataset — `/var/tmp/alphamotion/omniretarget/OmniRetarget_Dataset/` (772 MB unzipped)

HF: `omniretarget/OmniRetarget_Dataset` — license **MIT**, not gated.
Total remote size was only **0.43 GB**, so the **whole dataset is downloaded**
(3 zips, all extracted in place; zips kept). The paper's "8-9 h" includes
retargeted LAFAN1, which they cannot release (license); the public release is
**4.0 h**:

| Subset | files | qpos dim | duration | source | human_joints |
|---|---|---|---|---|---|
| `robot-object/` | 1826 | 43 | 3.00 h | OMOMO (box carrying) | all files, (T,52,3) |
| `robot-terrain/` | 145 | 36 | 0.51 h | in-house mocap (climbing) | 112/145, (T,53,3) |
| `robot-object-terrain/` | 148 | 43 | 0.48 h | in-house mocap (chair/box + terrain) | none |
| total | 2119 | | 3.99 h (~430k frames) | | |

Heavy augmentation baked into filenames: `*_original`, `*_rot_N`, `*_trans_N`,
`*_chair_scaled_X`, `*_z_scale_X`. Unique source clips are far fewer
(robot-object: ~312 `_original` clips; robot-object-terrain: ~15 base scenes).

### Trajectory format (per .npz)

- `fps`: scalar, **30.0 everywhere** (matches our cache fps).
- `qpos`: `[T, D]` float64, **Drake order, NOT MuJoCo order**:
  - `[0:4]` base quaternion `[qw,qx,qy,qz]` (verified: norm(q[0:4])==1)
  - `[4:7]` base position `[x,y,z]`
  - `[7:36]` 29 joint angles, standard Unitree G1 order
    (L hip p/r/y, L knee, L ankle p/r, R leg ..., waist y/r/p, L shoulder p/r/y,
    L elbow, L wrist r/p/y, R arm ...) — same 29-hinge order as our ardy G1 XML.
  - `[36:43]` (dim-43 files only) object pose `[qw,qx,qy,qz,x,y,z]`.
- `human_joints` (where present): `[T, J, 3]` world positions of the *source
  human* skeleton (52 joints for OMOMO/SMPL-H, 53 for their mocap) — free
  paired human<->robot data.
- **No contacts, no velocities** in the files (contacts must be re-derived;
  holosoma re-derives velocities by finite diff / FK in `convert_data_format_mj.py`).
- `models/`: URDF/SDF/OBJ for G1 (fake-hand + sphere-hand), largebox, chair
  (5 scales), 10+ climb terrains, ground box — viz-only, not needed for training.
- `visualize.py`: Drake + Meshcat player (confirms the quat-first convention —
  it feeds qpos straight into Drake `SetPositions`).

## Feeding into our qpos->skeleton bridge (exp_topo/prep_gmr_cache.py pattern)

`prep_gmr_cache.py` consumes `{robot}/{clip}.npz` with `qpos [T,nq]` @30 fps in
**MuJoCo order** and runs MuJoCo FK -> merge co-located bodies -> local rot6d +
SkeletonSpec with a <0.1 cm FK round-trip gate. To ingest OmniRetarget:

1. **Reorder the base**: `q_mj = concat(q[:,4:7], q[:,0:4], q[:,7:36])`
   (Drake quat-first -> MuJoCo pos-first). Joint block passes through unchanged —
   same 29-DoF order as the ardy G1 XML (`g1skel34/xml/g1.xml`, nq=36), so
   `load_robot("unitree_g1")` + `qpos_to_global` + `to_local6d` work as-is.
2. **Strip the object columns** for dim-43 files (`q[:, :36]` before reorder of
   base? no — object is the LAST 7: take `q[:, :36]`, then reorder base). Keep
   the object 7D `[36:43]` (quat,pos) in a sidecar array if we want
   object-conditioned motion later; `models/largebox|chair` URDFs give geometry.
3. **fps already 30** — no resampling needed; clips are variable length
   (~100-1500 frames), continuous, which suits the §3 long-sequence requirement.
4. **Manifest**: no accept/manifest.json here; build a sequences.json-style list
   from the file listing (subset, clip name, augmentation tag, T). Consider
   de-duplicating augmentations (train on `_original` first, augmented variants
   as an explicit augmentation axis).
5. **Bonus paired data**: `human_joints` (52/53-joint world positions, same T,
   same 30 fps) is a ready-made human<->G1 paired signal for the pose
   translator — no GMR pass needed for these clips. Would need a joint-name
   mapping for the 52/53-joint layouts (OMOMO/SMPL-H and their mocap skeleton;
   `config_types/data_type.py` in holosoma_retargeting defines the mappings).
6. Gate as usual: run the FK round-trip + bone-drift checks; visualize a few
   clips first (viser or their `visualize.py`) before trusting numbers.

Effort estimate: the bridge is a ~50-line adapter (base reorder + object strip +
manifest builder) in front of the existing prep_gmr_cache machinery.

## Sizes on disk

- repo `holosoma/`: 509 MB
- dataset (zips + extracted): 772 MB
- partition free space after download: ~349 GB
