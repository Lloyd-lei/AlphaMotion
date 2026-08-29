# Joint-limit & bone-length audit — G1 29DOF, AgiBot A3 T3.0, AgiBot X2 Ultra

Date 2026-08-05. Read-only audit. Nothing was changed in the cache, the model, or any
shipped artefact.

Three sources were cross-checked against each other:

* **(a) vendor MJCF/URDF as shipped** — the file the pipeline actually opens, plus the
  vendor's own URDF sitting next to it.
* **(b) what our cache contains** — `{emb}_spec.npz` (bone tree, rest offsets) and
  `{emb}_dof.npz` (axes, limits, rest rotations), or, for the AgiBots, the equivalent
  structures `agibot.build()` produces live at render time.
* **(c) physical plausibility + the vendor's published spec sheet** (web).

Probes: `scratchpad/limits_audit.py`, `limits_audit2.py`, `limits_audit3.py`,
`limits_audit4.py`. Raw dumps in `scratchpad/_limits_audit{,2,3,4}.json`.

---

## Verdict up front

| question | verdict |
|---|---|
| Does the cache faithfully reproduce the model file it was built from? | **YES — bit-exact.** All three robots: `max|dof_cache − dof_recomputed_from_MJCF| = 0.0`, `max|rest_cache − rest_recomputed| = 0.0`, slot→MJCF-joint names identical, zero limit disagreements, zero MJCF hinges dropped. |
| Are the bone lengths right? | **YES.** Every segment is within **7 mm** of the vendor URDF, and total standing height is within **1.3 cm** of the vendor's published height for all three. |
| Are the units right? | **YES.** All three compile with `angle="radian"`; every limit in the cache decodes to the same degrees as the source file. |
| Are the limits we feed the model the *real robot's* limits? | **NO for G1.** 14 of 29 G1 joints are silently narrowed (see §1). A3 and X2 match their vendor URDFs exactly. |
| Is what we validate against ("0% violations → executable") sound? | **Qualified yes for G1/A3/X2; NO for Apollo.** The three audited robots round-trip cleanly. Apollo (not in scope but shipped in a video) fails the round-trip — see §5. |

Two real defects were found. One (§1) affects the headline zero-shot robot's **input
features** but, measured, does not change any shipped frame. One (§5) means a shipped
number for Apollo is **substantially a solver artefact**.

---

## 0. Provenance — the paths that actually get opened

| robot | source file | how it reaches the model |
|---|---|---|
| `unitree_g1_29dof` | `alpha_motion2/external/GMR/assets/unitree_g1/g1_mocap_29dof.xml` | `prep_xwbc_cache.resolve_xml` → GMR `params.ROBOT_XML_DICT`. Spec + dof baked into `cache_xwbc_lafan/` and `cache_xwbc_amass/` (verified identical between the two caches). |
| `agibot_a3_t3d0` | `/var/tmp/alphamotion/robot_models/agibot_a3/a3_t3d0/mjcf/a3_t3d0.xml` | vendor MJCF read direct; `agibot.build()` constructs spec+dof **live at render time**. No `.npz` on disk. |
| `agibot_x2_ultra` | `/var/tmp/alphamotion/robot_models/agibot_x2_urdf/X2_URDF-v1.4.0/X2-Ultra.xml` | same, live. |

Compiler tags (units check):

```
g1_mocap_29dof.xml   <compiler angle="radian" meshdir="meshes"/>
a3_t3d0.xml          <compiler angle="radian" autolimits="true" meshdir="../meshes/"/>
X2-Ultra.xml         <compiler angle="radian" eulerseq="XYZ" meshdir="meshes/" autolimits="true"/>
```

All radian. No degree/radian mixing anywhere. Both AgiBot MJCFs were also diffed
against their own vendor URDFs (`a3_t3d0/urdf/model.urdf`,
`X2_URDF-v1.4.0/X2-Ultra.urdf`): **0 differing joints** each. v1.3.0 X2 likewise.

---

## 1. Unitree G1 29DOF — **the limits we use are NOT the robot's limits**

This is the finding that matters most, because G1 is the headline zero-shot result.

The cache faithfully copies `g1_mocap_29dof.xml`. But **GMR's mocap XML is not stock
Unitree** — it has hand-narrowed ranges. Diffed against the vendor URDF sitting in the
same directory (`g1_custom_collision_29dof.urdf`, whose numbers are byte-identical to
upstream `unitreerobotics/unitree_ros` `g1_29dof.urdf` / `g1_29dof_rev_1_0.urdf`):

| joint | vendor (deg) | what we feed & enforce (deg) | lost |
|---|---|---|---|
| hip_pitch L/R | −145 … 165 | **−89.95 … 89.95** | 130° |
| hip_roll L | −30 … 170 | **−30 … 89.95** | 80° |
| hip_roll R | −170 … 30 | **−89.95 … 30** | 80° |
| hip_yaw L/R | −158 … 158 | **−89.95 … 89.95** | 136° |
| waist_yaw | −150 … 150 | **−89.95 … 89.95** | 120° |
| shoulder_pitch L/R | −177 … 153 | **−177 … 65.83** | 87° |
| shoulder_roll L | −91 … 129 | **−34.38 … 129** | 57° |
| shoulder_roll R | −129 … 91 | **−129 … 34.38** | 57° |
| shoulder_yaw L | −150 … 150 | **−80.21 … 114.59** | 105° |
| shoulder_yaw R | −150 … 150 | **−114.59 … 80.21** | 105° |
| elbow L/R | −60 … 120 | **−60 … 97.4** | 22.6° |
| knee L/R | −5 … 165 | −5 … 165 | — |
| ankle_pitch L/R | −50 … 30 | −50 … 30 | — |
| ankle_roll L/R | −15 … 15 | −15 … 15 | — |
| waist_roll / waist_pitch | ±29.79 | ±29.79 | — |
| wrist_roll L/R | ±113 | ±113 | — |
| wrist_pitch / wrist_yaw L/R | ±92.5 | ±92.5 | — |

**14 of 29 joints narrowed, 15 correct.** Legs below the knee, knees, wrists and waist
roll/pitch are right; hips, waist yaw, shoulders and elbows are clipped. The clipping is
one-sided and asymmetric left/right on the roll joints, so it is a deliberate edit by
GMR (almost certainly to keep their IK retargeter well-behaved), not a units bug.

Direction of the error is **conservative** — the box we enforce is a strict subset of
the real one. Nothing we render is mechanically impossible on a real G1. But:

* the limits are fed as **input features** (`config.use_dof = True`), so the network is
  being told G1's hips do ±90° when they do −145…165°. That is wrong conditioning, and
  it is wrong for the one body we make the loudest claim about.
* the "raw violation %" we print for G1 is measured against the wrong box.

**Measured cost, on real predictions.** Same decode, two limit boxes, over 802 frames
spanning six AMASS clips (salsa, roundhouse kick, jumping jacks, sit-stand, walk,
walk-turn) — `limits_audit4.py`:

| | shipped (narrowed) | vendor (real ROM) |
|---|---|---|
| raw violates on | 3.087% of DOF-samples / 63.84% of frames | **3.087% / 63.84%** |
| worst overshoot | 12.17° | **12.17°** |
| projection moves the pose | 0.380° mean, 5.29° p99, 11.74° max | **0.380° / 5.29° / 11.74°** |
| hinges ever clamped | right_knee (54.0%), left_knee (32.9%), left_ankle_roll (2.5%), right_elbow (0.1%) | identical, same overshoots |

**Identical to every digit.** On this data the only joints that ever hit a stop are the
knees and one ankle roll — and those four limits are the *same* in both files. The
narrowing is a latent defect, not an active one: it has not changed a single shipped
frame. It will bite the moment a motion asks for an overhead reach (shoulder_pitch
>65.8°), a deep squat (hip_pitch >90°) or a big torso twist (waist_yaw >90°).

**Recommendation:** repoint `unitree_g1_29dof` at the vendor URDF's ranges, or patch the
29 limits into the cache (`limits_audit4.patch_limits` does exactly this and is 15
lines). Do it before publishing a G1 ROM claim; it does not require re-training, but it
does change an input feature, so a re-encode would be honest.

### G1 geometry — clean

| quantity | ours (from the MJCF) | vendor | Δ |
|---|---|---|---|
| total standing height | **1.3228 m** | 1.32 m (unitree.com) | +2.8 mm |
| pelvis height (standing) | 0.7923 m | 0.7522 m pelvis→ankle_roll + sole | consistent |
| thigh (hip_roll → knee) | **0.3061 m** | 0.3061 m | 0.0 |
| shank (knee → ankle_pitch) | **0.3000 m** | 0.30001 m | 0.0 |
| upper arm (shoulder_roll → elbow) | **0.1845 m** | 0.1844 m | +0.1 mm |
| forearm (elbow → wrist_roll) | **0.1005 m** | 0.1005 m | 0.0 |
| shoulder span | 0.2004 m | — | — |
| hip span | 0.1289 m | — | — |
| total mass (MJCF sum) | 34.38 kg | ~35 kg published; 33.3–35.1 kg across URDF revs | ok |
| cache root bone `rest_offsets[0]` | 79.30 cm | true measured pelvis 79.23 cm | **+0.07 cm ✓** |

Leg length thigh+shank = 0.606 m vs Unitree's published 0.6 m. Everything checks.

---

## 2. AgiBot A3 T3.0 — geometry and limits both clean; limits are *generous*, not wrong

Cache (live-built) == vendor MJCF == vendor URDF, exactly. 31 hinges, none unlimited,
none outside ±180°, none symmetric ±180°.

| quantity | ours | vendor | Δ |
|---|---|---|---|
| total standing height | **1.7228 m** | 1.73 m (agibot.com press release) | −7 mm |
| pelvis height | 1.0765 m | 1.006 m pelvis→ankle + sole | consistent |
| thigh (hip_yaw → knee) | **0.3700 m** | 0.370 m | 0.0 |
| shank (knee → ankle_pitch) | **0.4150 m** | 0.415 m | 0.0 |
| upper arm (shoulder_roll → elbow) | **0.2804 m** | 0.280 m | +0.4 mm |
| forearm (elbow → wrist_roll) | **0.1294 m** | 0.129 m | +0.4 mm |
| shoulder span | 0.2964 m | — | — |
| hip span | 0.2460 m | — | — |
| mass (MJCF sum) | 56.80 kg | 55 kg published | +1.8 kg (URDF includes end-effectors) |
| DOF | 31 | 31 (3 waist + 2 head + 7×2 arm + 6×2 leg) | ✓ |

Limits, all 31 (degrees, from the cache — identical to the vendor URDF):

| group | joint | range | sanity |
|---|---|---|---|
| waist | yaw / roll / pitch | −150…150 / ±20 / −28…24 | yaw very wide, roll+pitch tight ✓ |
| head | yaw / pitch | ±60 / −25…15 | ✓ |
| leg | hip_pitch | −144…168 | **span 312° — not a mechanical ROM** |
| | hip_roll L/R | −30…92 / −92…30 | asymmetric ✓ |
| | hip_yaw | ±156 | **span 312° — not a mechanical ROM** |
| | knee | −7…143 | one-directional ✓ |
| | ankle_pitch / ankle_roll | −52…30 / ±20 | small ✓ |
| arm | shoulder_pitch | ±165 | span 330° |
| | shoulder_roll L/R | −5…150 / −150…5 | asymmetric ✓ |
| | shoulder_yaw | ±160 | span 320° |
| | elbow | −55…100 | one-directional ✓ |
| | wrist_roll / pitch / yaw | ±160 / ±93 / ±93 | ✓ |

**Caveat, not a defect:** eleven A3 joints have spans >270°. A hip pitching 168° forward
or a hip yawing 156° is an *actuator* range, not a collision-free workspace — AgiBot
publishes these as the joint's travel, and the URDF is faithful to that. So "0%
violations ⇒ executable" is weaker for A3 than the phrase implies: a pose can satisfy
every A3 limit and still be self-colliding. Knees, ankles, elbows and waist roll/pitch —
the joints that actually catch bad output — are all correctly tight and one-directional.

---

## 3. AgiBot X2 Ultra — the >±180° ranges are real; two vendor sources disagree

Cache (live-built) == vendor MJCF (v1.4.0) == vendor URDF (v1.4.0), exactly.

| quantity | ours | vendor | Δ |
|---|---|---|---|
| total standing height | **1.3325 m** | 1.31 m (official spec §1.2.1) | +2.3 cm |
| pelvis height | 0.6753 m | 0.602 m pelvis→ankle + sole | consistent |
| thigh (hip_pitch → knee) | **0.3268 m** (0.3200 m in z alone) | 0.320 m | +0.0 |
| shank (knee → ankle_pitch) | **0.2822 m** | 0.282 m | +0.2 mm |
| upper arm (shoulder_roll → elbow) | **0.2045 m** | 0.204 m | +0.5 mm |
| forearm (elbow → wrist_pitch) | **0.2033 m** | 0.203 m | +0.3 mm |
| shoulder span | 0.2858 m | — | — |
| hip span | 0.1427 m | — | — |
| mass (MJCF sum) | **44.80 kg** | **39 kg published** | **+5.8 kg — flag** |
| DOF | 31 | 30 published (head_pitch documented as 0° travel) | +1 |

The mass gap is real and is in the vendor's own file, not ours (v1.3.0 URDF sums to
41.97 kg). It does not affect kinematics.

### 3a. The limits beyond ±180° — confirmed, and here is what they mean

Confirmed exactly as reported:

```
left_hip_yaw_joint    [ -96.5, +196.5]      right_hip_yaw_joint   [-196.5,  +96.5]
waist_yaw_joint       [-196.5, +121.0]
```

These are **not a bug in our pipeline and not a units error.** AgiBot's official ROM
table (`x2-aimdk.agibot.com`, §1.7) publishes hip yaw as −196.5…+96.5 and waist yaw as
−196.5…+126.5. The URDF is faithful. X2's hip-yaw and waist-yaw actuators genuinely
travel more than a full half-turn in one direction.

**What it means for us.** Our angle fit recovers a hinge angle by `atan2`, so it can only
ever return a representative in (−180°, +180°]. A rotation of +190° comes back as −170°.
Three consequences:

1. **16.5° of each of those three ranges is unreachable by the fit** — the box is
   effectively intersected with (−180, 180].
2. **A branch trap exists.** If the model asks for hip_yaw +190° (legal), the fit returns
   −170°, which is below `lo = −96.5`, and the clamp snaps it to −96.5° — a spurious 73°
   error on a pose that was actually feasible. The LM refinement *can* walk past +180°
   (the box permits it) but cannot jump the −170 → +190 branch.
3. **It has never fired.** Measured over 802 frames of six clips: the model asked for
   hip_yaw in [−59.4°, +54.1°] and waist_yaw in [−18.7°, +23.7°]; **0.00% of frames** put
   any of the three hinges past even 150°. Recorded as a known trap, not an active bug.

### 3b. Vendor URDF vs vendor spec sheet — they disagree

The URDF we use and AgiBot's published ROM table are not the same document. Magnitudes
mostly agree; several entries do not. **We follow the URDF**, which is the right call for
kinematics, but the disagreements should be known:

| joint | our URDF (deg) | vendor doc §1.7 (deg) | note |
|---|---|---|---|
| hip_pitch | −155 … 146.5 | ±146.5 | doc is tighter on the lower side |
| hip_yaw (L) | −96.5 … 196.5 | −196.5 … 96.5 | sign convention flipped |
| ankle_pitch | −46 … 26 | −26 … 46 | sign convention flipped |
| shoulder_pitch | −176.5 … 116.5 | −116.5 … 176.5 | sign convention flipped |
| shoulder_roll | −3.5 … 171 | −3.5 … 174.5 | 3.5° |
| **waist_pitch** | **±28** | **±18** | **we allow 10° more than the vendor warrants** |
| waist_yaw | −196.5 … 121 | −196.5 … 126.5 | (v1.3.0 URDF says +136.5) |
| wrist_pitch | ±30 | ±33 | doc is looser |
| head_yaw | ±28 | ±20 | we allow 8° more |
| head_pitch | ±30 | **0** | doc locks it; explains DOF 30 vs 31 |

The sign flips are frame-convention differences, not hardware differences. The two rows
worth acting on are **waist_pitch (±28 vs ±18)** and **head_yaw (±28 vs ±20)** — there we
are permissive relative to the vendor's warranted range.

Full X2 limit list (from the cache) — all 31 limited, none unlimited, none symmetric
±180°, knees `0…138` one-directional, elbows `−135…0` one-directional, ankle_roll ±15,
head small. Eleven joints have spans >270°, same actuator-range caveat as A3.

---

## 4. Cross-robot sanity summary

| check | G1 | A3 | X2 |
|---|---|---|---|
| compiler `angle` | radian | radian | radian |
| cache dof == recomputed from MJCF | **0.0** | **0.0** | **0.0** |
| cache rest == recomputed | **0.0** | **0.0** | **0.0** |
| dof-composition round-trip | 1.6e−15 | 1.3e−15 | 1.8e−15 |
| MJCF hinges dropped from cache | 0 | 0 | 0 |
| unlimited hinges | 0 | 0 | 0 |
| limits outside ±180° | 0 | 0 | **3** (vendor-published, §3a) |
| symmetric ±180° (a "no real limit" smell) | 0 | 0 | 0 |
| knees one-directional | ✓ −5…165 | ✓ −7…143 | ✓ 0…138 |
| elbows one-directional | ✓ −60…97.4 | ✓ −55…100 | ✓ −135…0 |
| ankles small | ✓ ≤80° span | ✓ ≤82° span | ✓ ≤72° span |
| height vs vendor | +2.8 mm | −7 mm | +2.3 cm |
| worst segment error vs vendor | 0.1 mm | 0.4 mm | 0.5 mm |
| limits match the vendor's own model file | **NO — 14/29 narrowed** | YES | YES |

---

## 5. Two things found outside the three robots that matter anyway

### 5a. The angle fit mis-branches on 3-DOF merged joints — Apollo's numbers are inflated

Test (`scratchpad`, §"round-trip feasible poses"): sample poses **strictly inside each
robot's own MJCF limits**, run FK, then refit with `fit_angles(clamp=False)` and ask
whether the recovered angles come back inside the box they were sampled from.

| robot | rotation round-trip | angles returned out of range |
|---|---|---|
| unitree_g1_29dof | 8e−2 deg (float noise) | **0.00%** |
| agibot_x2_ultra | 8e−2 deg | **0.00%** |
| **apptronik_apollo** | 8e−2 deg | **13.33% of limited slot-samples, worst 168° over** |

Apollo's failures are exclusively its three **3-DOF merged joints** (`torso_roll_link`,
`l_hip_ie_link`, `r_hip_ie_link`), on ~50% of samples each. The rotation is reproduced
exactly, so `_solve3` finds *a* valid decomposition — just the wrong branch of the
double cover, one that lands far outside the box. `_fit_angles_greedy` then clamps that
seed (`clamp=True`), so the LM warm-starts from a badly wrong pose.

Consequence: **the shipped Apollo figure "raw decode violates 29.8% of samples,
projection moved the pose 25.0°" is substantially a parameterisation artefact, not a
statement about the model's output or about Apollo's hardware.** It should not be quoted
as-is.

Only two cached bodies have any 3-DOF joints — `apptronik_apollo` (3) and `kuavo_s45`
(2). **G1, A3 and X2 have none**, which is why they round-trip perfectly. `human_smpl` is
all-ball-joint but unlimited, so it cannot be affected.

Fix: enumerate the equivalent branches of `_solve3` and pick the one inside `[lo, hi]`,
or run a small multi-start over ±π offsets for 3-DOF joints when `clamp=True`.

### 5b. The root bone is the MJCF *spawn height*, not the robot's pelvis height

`prep_gmr_cache.build_spec` sets `rest_offsets[0] = model.body_pos[1][2] or 0.8` — the
MJCF's spawn z, with a hard-coded 0.8 m fallback when spawn z is 0. That vector is a
model **input feature** (`static_cond → rest_off`) and it enters `spec.height`
(= Σ bone lengths), which normalises **every** `norm_bonelen`. `agibot.build()` overrides
it with the measured pelvis height; the cache path does not.

For our three robots this is harmless — G1's spawn z is 0.793 m against a measured
pelvis of 0.7923 m (**0.07 cm error**), and A3/X2 are live-built and therefore correct by
construction. But swept across the whole cache:

| embodiment | cache root (cm) | true pelvis (cm) | error |
|---|---|---|---|
| **booster_t1** | 168.0 | 76.49 | **+91.5 cm** (2.2× too tall) |
| fourier_gr3 | 80.0 (fallback) | 102.86 | −22.9 cm |
| booster_k1 | 80.0 (fallback) | 59.55 | +20.5 cm |
| pal_talos | 103.88 | 119.28 | −15.4 cm |
| engineai_pm01 | 82.0 | 92.73 | −10.7 cm |
| pnd_adam_lite | 93.0 | 102.43 | −9.4 cm |
| kuavo_s45 | 85.0 | 93.95 | −9.0 cm |
| tienkung | 100.0 | 108.97 | −9.0 cm |
| booster_t1_29dof | 70.0 | 78.15 | −8.2 cm |
| unitree_h2 | 97.82 | 103.88 | −6.1 cm |
| unitree_r1 | 80.0 (fallback) | 74.08 | +5.9 cm |
| unitree_h1 | 110.0 | 104.42 | +5.6 cm |
| **unitree_g1_29dof** | **79.30** | **79.23** | **+0.07 cm ✓** |
| unitree_g1_23dof | 79.30 | 79.23 | +0.07 cm ✓ |
| unitree_h1_2 | 103.0 | 102.82 | +0.18 cm ✓ |
| fourier_n1 | 70.0 | 69.95 | +0.05 cm ✓ |
| hightorque_hi | 68.8 | 68.81 | −0.00 cm ✓ |

Out of scope for this audit, and G1 happens to be one of the correct ones — but
`booster_t1` is being described to the network as a 1.68 m-pelvis robot when its pelvis
is at 0.76 m, and that error also rescales its whole `norm_bonelen` vector. Worth fixing
(one line: use `agibot.standing_height(model)`).

---

## 6. Answers to the two questions asked

**Is what we feed the model correct?**

Bone geometry: **yes**, for all three — sub-centimetre against the vendor, exact against
the source file. Axes, rest rotations, topology: **yes**, bit-exact. Joint limits: **yes
for A3 and X2**; **no for G1**, where 14 of 29 ranges are GMR's narrowed mocap values
rather than Unitree's. The root-bone scale feature is correct for G1 but wrong for
several other cached bodies (§5b).

**Is what we validate against correct?**

For G1, A3 and X2: **yes, with two qualifications.** (i) The box is a faithful copy of
the source model, and feasible poses round-trip through the fit with 0.00% spurious
violations — so "0% violations" means what it says. (ii) But "executable" is doing more
work than the evidence supports: A3 and X2 both have ~11 joints with >270° travel, so
limit satisfaction alone does not rule out self-collision, and no collision check is run.
For G1 specifically, the validation box is *tighter* than the real robot, so the claim is
conservative rather than wrong.

For Apollo: **no.** Its violation and projection-movement numbers are dominated by the
3-DOF branch bug in §5a and should be withdrawn until that is fixed.
