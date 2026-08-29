# Per-joint DOF descriptors (`{emb}_dof.npz`)

The tokenizer's spec (`{emb}_spec.npz`) tells the decoder *where* a joint is
(parents + rest offsets) but nothing about **which axis it rotates about**. That is
the missing signal behind the leg / stacked-hinge failures: a knee and a hip-yaw
look identical to the decoder, so it has to guess the rotation subspace from data.

`{emb}_dof.npz` adds that signal. One file per embodiment, sitting next to the spec.

Generator: `pose_translator_design/exp_topo/prep_dof_features.py` (CPU, ~1 min).

## Files

```
/var/tmp/alphamotion/topo_exp/cache_xwbc_lafan/{emb}_dof.npz   18 embodiments
/var/tmp/alphamotion/topo_exp/cache_paired_fixed/{emb}_dof.npz  soma77, g1_30
```

Keys:

| key | dtype | shape | meaning |
|---|---|---|---|
| `dof` | float32 | `[J, 19]` | the descriptor (below) |
| `rest` | float32 | `[J, 3, 3]` | `R_rest`, the static rest rotation (below) — **required for angles -> rotations** |
| `rest6d` | float32 | `[J, 6]` | same thing as rot6d (`[column0, column1]`), for feeding a network |
| `joint_names` | object | `[J]` | copy of the spec's joint names, for alignment checks |
| `slot_joints` | object | `[J]` | comma-separated MuJoCo joint name(s) feeding each row (debug only) |

`J` and row order are **identical** to `{emb}_spec.npz`.

## Layout — `dof[j]` is 19 floats

3 DOF slots x 6 values, then 1 count:

```
[ 0: 3]  slot0 axis  (x, y, z)   unit vector, LOCAL (parent) frame, Y-up harmonized
[ 3]     slot0 limited_flag      1.0 if the hinge has a finite range, else 0.0
[ 4]     slot0 half_range / pi   0.0 if unlimited or slot absent
[ 5]     slot0 range_center / pi 0.0 if unlimited or slot absent
[ 6:12]  slot1  (same 6)
[12:18]  slot2  (same 6)
[18]     dof_count / 3.0         0.0, 1/3, 2/3 or 1.0
```

All values are O(1). Absent slots are exactly zero (axis norm 0), so a network can
tell "no DOF here" from "DOF along zero axis" by the norm.

### Axis semantics

A spec "joint" is a body/frame; the co-located-body merge in `prep_gmr_cache.build_merge`
may have collapsed 0, 1 or 2 stacked hinges into it. The slots reproduce the cached
local rotation exactly:

```
R_local(q) = rot(A_0, q_0) @ rot(A_1, q_1) @ rot(A_2, q_2) @ R_rest
A_k        = AX @ (B_1 ... B_k) @ jnt_axis_k          (hinges at 0)
R_rest     = AX @ (B_1 ... B_n) @ AX^T                (key `rest`)
AX         = MuJoCo Z-up -> ardy Y-up permutation, (x,y,z)_mj -> (y,z,x)_ardy
```

`B_i` are the fixed MJCF `<body quat>` rotations along the merged chain. **Slot order
is kinematic, parent-most first** — rotation composition does not commute, so slot 0
is the hinge closest to the parent.

### `R_rest` — the static rest rotation (key `rest`, `[J,3,3]`)

Same Y-up frame as the axes. It is what makes the map *angles -> rotations* invertible:
the axes alone give you `q` from a rotation, but rebuilding the rotation from `q`
without `R_rest` is wrong by exactly `R_rest` at every joint whose MJCF body carries a
non-identity `<body quat>`. Measured cost of assuming `R_rest = I` on GT frames
(fit angles -> FK -> compare to the stored globals): **10.5 deg** mean for
`unitree_g1_29dof`, **10.0** for `unitree_h1`, **6.7** for `fourier_n1`, **18.0** for
`unitree_h2`. With the stored `rest`: **~3e-6 deg** (float32 noise).

Number of joints with non-identity `R_rest`: `unitree_g1_29dof` 8/36 (max 16 deg),
`unitree_h1` 4/20 (max 25 deg), `fourier_n1` 7/27 (max 90 deg), `unitree_h2` 30/32
(max 30 deg), `g1_30` 8/30 (max 16 deg). `kuavo_s45`, `human_smpl` and `soma77` are
**verified** identity (not assumed): SMPL/BVH ball joints store the joint rotation
directly, and every merged kuavo chain happens to have identity body quats.

**Root joint**: `rest[root] = I` by convention. The root's "local" rotation *is* its
global orientation, which comes from the free joint / the data, not from hinge angles;
FK consumers must pass it through (`manifold_project.fk_from_angles(root_R=...)`).

In the Y-up frame the readable directions are: **X = lateral (pitch axis),
Y = up (yaw axis), Z = forward (roll axis)**.

### Conventions and special cases

* **0-DOF frames** (toe links, imu, head_mocap, camera, rubber hands, end effectors,
  R1's parallel-ankle rods): all 19 values zero. This is information — "this frame is
  welded to its parent" — not padding.
* **Ball joints** (`human_smpl` SMPL 22, `soma77` BVH 78): 3 slots filled with the
  orthonormal basis `(1,0,0),(0,1,0),(0,0,1)`, `limited_flag = 0`, ranges 0,
  `dof_count = 3`, `R_rest = I` — the three slots are then an intrinsic-XYZ Euler
  chart of SO(3), so the round trip is exact (measured 1e-14 deg).
* **Slider / free joints**: treated as 0-DOF for this rotation-oriented descriptor.
  In practice this only affects the floating base — there are **zero slide joints** in
  any of the 15 MJCF robots, and the only free joint is the root, so nothing is lost
  besides root translation (which the cache carries separately anyway).
* **Limits** come from `jnt_limited` / `jnt_range` of the same MJCF the cache was
  built from. `half_range = (hi-lo)/2/pi`, `range_center = (hi+lo)/2/pi`, both radians
  normalized by pi, and both 0 when the joint is unlimited.
* **Data-tree robots** (`unitree_h2`, `pal_talos`): their specs were built from stored
  body kinematics, not from an MJCF (see `prep_xwbc_cache.DATA_TREE`), so their frames
  are named `bodyNN`. Axes were recovered **directly from the cached motion** — each
  body has exactly one hinge, so `R_local(t) R_local(0)^T` is a pure rotation about the
  axis; the axis is the principal direction of the log map. Residual off-axis error
  `< 7e-7 rad`. Each recovered angle track was then matched (max abs error `< 1e-3 rad`)
  against a `joint_pos` column to name the joint and fix the sign, and the limits were
  looked up in the MJCF by that name. All 31 + 32 joints matched; nothing ambiguous
  survived. `R_rest = rot(a, -q(t)) L(t)`, averaged over the sampled frames and
  SVD-reprojected; its constancy over `t` holds to `1.4e-6`.

## Per-embodiment DOF counts

`hist` = number of spec joints with 0 / 1 / 2 / 3 DOF.

| embodiment | cache | J | 0 | 1 | 2 | 3 | total DOF | dataset ndof | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| human_smpl | xwbc | 22 | 0 | 0 | 0 | 22 | 66 | — | SMPL ball joints |
| booster_k1 | xwbc | 21 | 1 | 18 | 2 | 0 | 22 | 22 | |
| booster_t1 | xwbc | 24 | 1 | 23 | 0 | 0 | 23 | 23 | |
| booster_t1_29dof | xwbc | 32 | 5 | 27 | 0 | 0 | 27 | 27 | 2 head DOFs absent from the mocap XML (`DROP_JOINTS`) |
| engineai_pm01 | xwbc | 25 | 1 | 24 | 0 | 0 | 24 | 24 | |
| fourier_gr3 | xwbc | 32 | 6 | 21 | 5 | 0 | 31 | 31 | |
| fourier_n1 | xwbc | 27 | 6 | 19 | 2 | 0 | 23 | 23 | |
| hightorque_hi | xwbc | 26 | 1 | 25 | 0 | 0 | 25 | 25 | |
| kuavo_s45 | xwbc | 20 | 1 | 12 | 5 | 2 | 28 | 28 | two 3-DOF merged frames |
| pal_talos | xwbc | 33 | 1 | 32 | 0 | 0 | 32 | 32 | data-tree recovery; grippers kept |
| pnd_adam_lite | xwbc | 24 | 1 | 21 | 2 | 0 | 25 | 25 | |
| tienkung | xwbc | 22 | 4 | 16 | 2 | 0 | 20 | 20 | |
| unitree_g1_23dof | xwbc | 36 | 8 | 27 | 1 | 0 | 29 | 23 | runs on the 29dof XML; the 6 absent joints are zero-filled in the data but kept as hinges here (spec is byte-identical to g1_29dof) |
| unitree_g1_29dof | xwbc | 36 | 8 | 27 | 1 | 0 | 29 | 29 | |
| unitree_h1 | xwbc | 20 | 1 | 19 | 0 | 0 | 19 | 19 | |
| unitree_h1_2 | xwbc | 26 | 1 | 23 | 2 | 0 | 27 | 27 | |
| unitree_h2 | xwbc | 32 | 1 | 31 | 0 | 0 | 31 | 31 | data-tree recovery |
| unitree_r1 | xwbc | 38 | 14 | 22 | 2 | 0 | 26 | 26 | 12 of the 14 fixed frames are the parallel-ankle rod/constraint links |
| soma77 | paired_fixed | 78 | 0 | 0 | 0 | 78 | 234 | — | BVH ball joints |
| g1_30 | paired_fixed | 30 | 1 | 29 | 0 | 0 | 29 | — | ardy `g1skel34/xml/g1.xml`, one hinge per spec joint |

## Validation

1. **Hinge coverage** — for all 15 MJCF robots, `sum(dof_count)` equals the number of
   hinge joints in the model, and every hinge is used by exactly one spec joint
   (`missing = 0` everywhere). 0 slide joints, 1 free joint (the root) per model.
2. **Composition against MuJoCo FK** — random hinge angles, 4 trials per robot:
   `max |rot(A_0,q_0)...rot(A_n,q_n) R_rest − R_frame(parent)^T R_frame(joint)|`
   ranges `8.9e-16 … 2.0e-15` over all 15 robots. Axis directions *and* slot ordering
   are exact.
3. **Reconstruction of the cached stream** — for 9 robots (incl. all the 2- and 3-DOF
   cases) the descriptor plus the raw dataset hinge angles rebuilds the stored
   `{emb}_paired_local6d.pt` local rotations to `max |Δ| = 1.5e-7 … 1.8e-7`
   (float32 noise). This validates the Y-up harmonization end to end.
4. **1-DOF consistency on the cache** — every 1-DOF joint's local rotation across
   3000 sampled frames lies on `{rot(a,θ) R_rest}`: max off-axis residual
   `1e-5 deg` (MJCF robots), `6e-5 deg` (data-tree robots).
5. **Numerics** — no NaN anywhere; every axis is either exactly zero or unit-norm to
   `1e-5`.
5b. **Full angle <-> rotation round trip on real GT frames** (the check the missing
   `rest` used to fail). Take GT frames from `{emb}_paired_local6d.pt`, FK to global
   rotations, fit hinge angles (`manifold_project.fit_angles`), rebuild by FK
   (`fk_from_angles`), compare. float64, 400 frames, all 18 xwbc embodiments,
   **mean global geodesic 1.2e-14 … 1.9e-5 deg, max 9.7e-5 deg** — i.e. float32
   storage noise. Per-joint LOCAL rotation error by DOF count: 0-DOF `<= 7.5e-6`,
   1-DOF `<= 5.9e-5` (h2/talos) / `<= 7.7e-6` (MJCF), 2-DOF `<= 5.0e-6`,
   3-DOF `= 0.0` deg. Same numbers on `cache_xwbc_amass`; `cache_paired_fixed`
   gives `soma77` 2.0e-14 and `g1_30` 8.2e-7 deg.
   The 2- and 3-DOF fits are exact because `fit_angles` uses a closed-form
   generalized-Euler (Davenport) seed plus exact per-axis coordinate descent; plain
   sequential twist-peeling from `q = 0` lands in local minima (SMPL: 21 deg max).
6. **Known kinematics spot-check** (Y-up: X lateral, Y up, Z forward):
   * `unitree_g1_29dof` left leg — hip_pitch `(1,0,0)` pitch, hip_roll
     `(0,0.174,0.985)` roll tilted 10 deg (the real G1 hip-roll tilt), hip_yaw
     `(0,1,0)` yaw, **knee `(1,0,0)` pitch**, ankle_pitch `(1,0,0)`, ankle_roll
     `(0,0,1)`, toe link 0-DOF. Hip triple max `|dot| = 0.174` (= sin 10 deg).
   * `unitree_h1` left leg — hip_yaw `(0,1,0)`, hip_roll `(0,0,1)`, hip_pitch
     `(1,0,0)`, knee `(1,0,0)`, ankle `(1,0,0)`. Hip triple max `|dot| = 0.0000`
     (exactly orthogonal).
   * `fourier_n1` left leg — hip_pitch `(0.966,-0.259,0)` (15 deg tilt), hip_roll
     `(0,0,1)`, hip_yaw `(0,1,0)`, and a merged 2-DOF ankle frame
     `left_foot_roll_link = [roll (0,0,1), pitch (1,0,0)]`. Hip triple max
     `|dot| = 0.259` (= sin 15 deg).

### Known caveat: retargeted data can exceed the stored limits

The limits are the robot's real MJCF limits; the xwbc retargets do not always respect
them. Number of 1-DOF joints whose observed angle span exceeds `2*half_range` by more
than 2 deg: `unitree_g1_23dof` 10, `booster_k1` 2, `unitree_h1` 2, `tienkung` 1,
`unitree_h1_2` 1, `unitree_h2` 1, `unitree_r1` 1, everything else 0. The
`unitree_g1_23dof` cluster is the biggest — its retarget spins shoulder-yaw / wrist-roll
well past the 29dof XML range. The descriptor is still the correct hardware limit; treat
it as a soft prior, not a hard constraint, when using it in a loss.
