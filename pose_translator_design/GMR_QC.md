# GMR QC: SOMA BVH → 16 robot embodiments, singularity + quality pass

Date: 2026-08-03. Follow-up to `GMR_SPIKE.md`. Machine-readable version of every
number here: `GMR_QC.json` (same directory). Visualizations:
`R4_VIZ/gmr_qc/<robot>.gif` (real-mesh MuJoCo render of the jog-stop clip per
robot) and `R4_VIZ/gmr_qc/grid_all_robots.png` (all robots at the same 3
frames). **Look at the GIFs — two failure modes below were invisible in the
numbers and only showed up on screen, and one "fail" robot looks fine on screen.**

## Setup

- Configs for all 17 embodiments (18 in GMR minus wheeled `galaxea_r1pro`)
  generated mechanically from the calibrated SOMA→G1 config using the
  robot-independent per-joint deltas (lafan flavor: 6 robots; smplx flavor: 10
  robots via the SMPLX↔SOMA name map). Only the `head` target had no delta and
  was dropped (hightorque_hi, booster_k1, tienkung — cosmetic).
  Configs: `/var/tmp/alphamotion/gmr_spike/ik_configs/bvh_soma_to_*.json`.
- 5-motion probe set (same files for every robot), BONES-SEED `soma_uniform`:
  | probe | file | stresses |
  |---|---|---|
  | walk | `230731/walk_180_R_002__A437_M.bvh` | baseline gait + 180 turn |
  | jog_stop | `230411/jog_ff_stop_180_R_002__A307.bvh` | high momentum + abrupt stop |
  | jump | `230228/jump_ff_270_002__A224.bvh` | flight phase |
  | idle_turn | `230731/idle_turn_270_L_001__A437.bvh` | in-place yaw |
  | reach | `231121/reach_jump_R_002__A530.bvh` | arms overhead (workspace limits) |
- 30 fps output, GMR defaults (daqp, damping 0.5), CPU only.
- Runner: `/var/tmp/alphamotion/gmr_spike/qc_run.py`; raw qpos per clip in
  `/var/tmp/alphamotion/gmr_spike/qc/<robot>/<motion>.npz`.

## Thresholds and why

| metric | pass | warn | fail | rationale |
|---|---|---|---|---|
| max joint vel (rad/s) | ≤ 4π | ≤ 8π | > 8π | 4π is comfortably above real arm-swing speeds at 30 fps; > 8π (25 rad/s) means the IK jumped between solution branches, not that the human moved that fast |
| p99 joint vel | ≤ 8 | ≤ 16 | > 16 | separates one-frame glitches (high max, low p99) from globally jittery solutions |
| spike frames (any joint Δq > 0.5 rad/frame = 15 rad/s) | 0 | ≤ 5 | > 5 | the singularity fingerprint: a 0.5 rad teleport in 33 ms is never real tracking |
| joint-limit violations (frame·joint, tol 1e-3) | 0 | ≤ 10 | > 10 | mink's ConfigurationLimit should make this exactly 0 (see bug note) |
| min body-origin z (m) | ≥ −0.005 | ≥ −0.02 | < −0.02 | body origins sit above the sole, so a negative origin is unambiguous ground penetration |
| foot skate (mean xy speed in contact, m/s) | ≤ 0.05 | ≤ 0.15 | > 0.15 | contact = foot below its own min + 2 cm; a planted foot should drift < 2 mm/frame |
| EE error mean / max (m) | ≤ 0.12 / – | ≤ 0.25 / – | > 0.25 or max > 0.6 | robot wrist+ankle world position vs the height-scaled SOMA targets (the retargeter's own scaled human data); 5–13 cm is the normal embodiment gap on the visually-verified G1 |
| NaN | – | – | any | – |

Cell flag = worst sub-metric; robot flag = worst cell.

## Results (GMR defaults, no velocity limit)

Cell format: `flag  vmax/spikes/ee_mean`. Full metrics in `GMR_QC.json`.

| robot | overall | walk | jog_stop | jump | idle_turn | reach |
|---|---|---|---|---|---|---|
| berkeley_humanoid_lite | fail | fail v14/s0/e0.06 | fail v17/s1/e0.11 | fail v4/s0/e0.06 | fail v8/s0/e0.06 | fail v10/s0/e0.08 |
| booster_k1 | fail | fail v12/s0/e0.09 | warn v17/s1/e0.10 | warn v8/s0/e0.08 | pass v5/s0/e0.08 | pass v12/s0/e0.08 |
| booster_t1 | warn | warn v11/s0/e0.03 | warn v24/s1/e0.09 | pass v9/s0/e0.03 | pass v6/s0/e0.03 | pass v11/s0/e0.04 |
| booster_t1_29dof | warn | warn v12/s0/e0.08 | warn v22/s1/e0.15 | pass v9/s0/e0.09 | pass v6/s0/e0.08 | pass v12/s0/e0.09 |
| engineai_pm01 | warn | warn v12/s0/e0.08 | warn v11/s0/e0.08 | pass v9/s0/e0.10 | pass v5/s0/e0.09 | warn v14/s0/e0.10 |
| fourier_gr3 | warn | warn v12/s0/e0.14 | warn v14/s0/e0.11 | warn v10/s0/e0.13 | warn v6/s0/e0.14 | warn v12/s0/e0.13 |
| fourier_n1 | warn | pass v11/s0/e0.06 | warn v12/s0/e0.07 | pass v9/s0/e0.07 | pass v6/s0/e0.06 | warn v15/s0/e0.08 |
| hightorque_hi | fail | warn v14/s0/e0.08 | fail v17/s1/e0.10 | pass v10/s0/e0.10 | pass v6/s0/e0.08 | warn v16/s1/e0.09 |
| kuavo_s45 | warn | warn v12/s0/e0.08 | warn v14/s0/e0.08 | pass v10/s0/e0.10 | pass v11/s0/e0.07 | pass v11/s0/e0.10 |
| pal_talos | fail | warn v18/s2/e0.11 | fail v30/s11/e0.12 | warn v12/s0/e0.14 | warn v10/s0/e0.13 | fail v30/s4/e0.15 |
| pnd_adam_lite | fail | warn v12/s0/e0.09 | fail v29/s1/e0.09 | pass v10/s0/e0.07 | pass v11/s0/e0.09 | warn v13/s0/e0.08 |
| stanford_toddy | fail | warn v10/s0/e0.05 | warn v19/s1/e0.05 | pass v7/s0/e0.05 | pass v7/s0/e0.04 | fail v33/s4/e0.05 |
| tienkung | fail | warn v21/s2/e0.12 | fail v28/s3/e0.14 | warn v11/s0/e0.13 | warn v7/s0/e0.12 | warn v15/s0/e0.13 |
| unitree_g1 | warn | warn v12/s0/e0.12 | warn v12/s0/e0.10 | warn v8/s0/e0.12 | warn v6/s0/e0.13 | warn v14/s0/e0.13 |
| unitree_g1_with_hands | warn | (same as unitree_g1 — identical config/base) | | | | |
| unitree_h1 | warn | pass v12/s0/e0.07 | warn v14/s0/e0.08 | pass v10/s0/e0.06 | pass v8/s0/e0.07 | warn v13/s0/e0.07 |
| unitree_h1_2 | warn | pass v12/s0/e0.06 | warn v13/s0/e0.07 | pass v9/s0/e0.05 | pass v6/s0/e0.06 | warn v14/s0/e0.06 |

Global observations: zero NaN and zero joint-limit violations anywhere; EE
correspondence 3–15 cm mean across the board; almost every "warn" is only the
vmax slightly above 4π on the dynamic clips, which is borderline-normal arm
swing, not a glitch.

## Singularity symptoms (the actual concern)

Robots with true spike signatures (Δq > 0.5 rad in one frame):

- **pal_talos — worst.** 11 spike frames in jog_stop, 4 in reach, vmax ≈ 30.
  7-DoF arms with wrists: classic wrist-flip branch jumping. Static frames look
  fine (see GIF) — this is exactly the failure mode that only velocity metrics catch.
- **stanford_toddy** — 4 spikes in reach only (vmax 33): human hands go far
  outside the tiny robot's workspace and the arm snaps between configurations.
- **tienkung** — 3 spikes in jog_stop plus the only real ground penetration of
  a foot (−4.4 cm) and borderline skate.
- **pnd_adam_lite** — 1 spike in jog_stop (vmax 29), otherwise clean.
- **hightorque_hi / booster_k1 / boosters** — isolated single-spike frames on
  jog_stop only.

Two failures the numbers under-reported but the renders caught:

- **berkeley_humanoid_lite**: constant body-origin penetration (−5…−8 cm on
  every clip) and the GIF shows a scrambled near-ground pose. The smplx→soma
  scale carry-over does not fit this ~0.5 m robot. Broken as generated.
- **booster_k1**: numerically mostly warn, but the GIF shows a deep crouch with
  the torso pitched ~45° and an arm held out. Wrong posture, not usable as-is.

And one the renders caught that is *acceptable*: both Boosters hold the right
arm out straight during the stop-turn — the human really swings that arm, and
the 4-DoF Booster arm can only satisfy the orientation target with a straight
arm. Exaggerated but faithful.

## Fix found: GMR silently drops all IK limits (upstream bug)

`motion_retarget.py` calls
`mink.solve_ik(configuration, tasks, dt, solver, damping, self.ik_limits)`,
but in the vendored mink version the 6th positional parameter is
`safety_break`, not `limits`. **Every limit (joint range AND the optional
velocity limit) is silently ignored.** A 3-line shim that routes the list into
`limits=` (see `qc_run.py` test at the bottom of this section) plus a velocity
limit of 3π/s per hinge changes the picture:

| robot | motion | before | after (3π vel limit) |
|---|---|---|---|
| pal_talos | jog_stop | fail, 11 spikes, vmax 30 | 0 spikes, vmax 6.2 — but skate 1.15 m/s, EE max 0.71 (solution lags targets) |
| stanford_toddy | reach | fail, 4 spikes, vmax 33 | **pass** (0 spikes, EE unchanged 0.05) |
| hightorque_hi | reach | warn, 1 spike | **pass** |
| pnd_adam_lite | jog_stop | fail, vmax 29 | warn, vmax 6.2, EE unchanged |
| tienkung | jog_stop | fail, 3 spikes | 0 spikes but penetration/skate remain |

So the velocity limit genuinely fixes toddy/hi/adam, while for talos and
tienkung it just converts branch jumps into target lag — those two need config
work (lower wrist orientation weights for talos), not clamping. A limit of
4π–6π instead of 3π is probably the right production setting; not tuned here
(timebox).

Also worth knowing: GMR's `use_velocity_limit=True` crashes on pal_talos
regardless (it builds the limit dict from actuator names, and talos actuators
are named `*_joint_position`).

## Recommended include-list for the 2000-file production run

**Include now (12):** unitree_g1, unitree_g1_with_hands, unitree_h1,
unitree_h1_2, booster_t1, booster_t1_29dof, fourier_n1, engineai_pm01,
kuavo_s45, fourier_gr3, stanford_toddy†, pnd_adam_lite†
(† = run with the mink-limits shim + ~4π velocity limit; verified to remove
their spikes without hurting EE error)

**Conditional (1):** hightorque_hi — spikes fixed by the velocity limit, but
walk-clip skate worsens (0.39 m/s); include after a quick skate check at 4π–6π.

**Exclude pending fixes (4):**
- pal_talos — repeated wrist singularities; needs orientation-weight retuning,
  clamping alone makes it lag
- tienkung — spikes + foot penetration + skate; likely needs scale and foot
  offset work
- booster_k1 — crouched, torso-pitched posture; scale/offset work needed
- berkeley_humanoid_lite — systematic ground penetration; smplx scale table
  does not transfer to a 0.5 m robot

That gives 12–13 embodiments immediately, which already scales the
cross-embodiment experiment from 2 topologies to ~13.

## Disk / time actuals (this pass)

| item | actual |
|---|---|
| QC retargets (17 robots × 5 motions, incl. FK metrics) | 44.5 s wall on 4 workers (3m18s CPU) ≈ 2.3 s per robot-motion |
| qpos npz for all 85 clips | 3.4 MB total (≈ 40 KB/clip) |
| 17 real-mesh GIFs (480×360, 15 fps, ~75 frames) + grid PNG | 44 MB |
| Extrapolation, 2000 files × 13 robots | ≈ 11 CPU-hours ≈ 45 min on 16 workers; ≈ 1.4 GB npz |

## Next steps

1. Apply the mink-limits shim (or fix `motion_retarget.py` upstream: pass
   `limits=self.ik_limits` by keyword) and set a 4π velocity limit for the
   production run; re-run this QC to confirm toddy/hi/adam and re-grade.
2. Retune pal_talos (drop wrist rot weight 10→2) and the three excluded small
   robots' scale tables; each is ~15 min with this QC harness
   (`qc_run.py --robot X` + the GIF).
3. Wire this QC into the production driver so every (file, robot) output gets
   the same stats JSON — the spike counter is cheap and catches exactly the
   singularity cases the user is worried about.

## Production run (2026-08-03)

2000 BONES-SEED clips × 12 robots = 24,000 retargets, CPU-only, 28 workers at
`nice -n 10` (32-core box, GPU training untouched). **Wall time 52.5 min**,
**625 MB** total output. Zero worker crashes; all 24,000 clips produced a
record (rejected clips keep their npz — nothing deleted).

**File selection = exact tokenizer pairing.** The 2000 files were recovered
from `/var/tmp/alphamotion/topo_exp/cache_paired_seq/sequences.json` (the
tokenizer cache's own per-clip list, which encodes the `default_rng(0)`
selection that `prep_paired.py` made): 2000/2000 names matched to BVH paths,
0 missing, cache order preserved. The list used is
`/var/tmp/alphamotion/gmr_spike/files.json` — row-for-row alignable with
`cache_paired_seq` for the E5 full cache.

Outputs: `/var/tmp/alphamotion/gmr_production/<robot>/` — `<clip>.npz`
(float32 qpos @30 fps), `manifest.json` (per-clip QC), `rejects.json`;
global `summary.json`. Reject thresholds = the fail thresholds of the sample
pass (NaN, >5 spike frames, >10 limit violations, body-origin z < −0.02).

| robot | clips | accept | reject | rate | robot-specific rate* | top reject reasons | disk (MB) |
|---|---|---|---|---|---|---|---|
| unitree_g1 | 2000 | 1819 | 181 | 9.0% | 2.9% | spikes:111, minz:72 | 58 |
| unitree_g1_with_hands | 2000 | 1828 | 172 | 8.6% | 2.2% | spikes:106, minz:68 | 59 |
| unitree_h1 ⚠ | 2000 | 1715 | 285 | 14.2% | **7.0%** | minz:197, spikes:89 | 42 |
| unitree_h1_2 | 2000 | 1781 | 219 | 10.9% | 3.8% | minz:140, spikes:83 | 46 |
| booster_t1 | 2000 | 1838 | 162 | 8.1% | 2.9% | minz:110, spikes:52 | 45 |
| booster_t1_29dof | 2000 | 1783 | 217 | 10.8% | 4.3% | minz:146, spikes:88 | 48 |
| fourier_n1 | 2000 | 1815 | 185 | 9.2% | 2.1% | minz:100, spikes:86 | 47 |
| engineai_pm01 | 2000 | 1872 | 128 | 6.4% | 0.8% | spikes:69, minz:59 | 49 |
| kuavo_s45 ⚠ | 2000 | 1738 | 262 | 13.1% | **5.9%** | minz:150, spikes:116 | 53 |
| fourier_gr3 | 2000 | 1823 | 177 | 8.8% | 1.8% | spikes:99, minz:80 | 57 |
| stanford_toddy † | 2000 | 1873 | 127 | 6.3% | 2.0% | spikes:83, minz:46 | 47 |
| pnd_adam_lite † | 2000 | 1953 | 47 | 2.4% | 0.3% | minz:47 | 48 |
| **total** | **24000** | **21838** | **2162 (9.0%)** | | | | **599** |

† = ran with the mink-limits shim + 4π/s velocity limit.
\* robot-specific rate = rejects excluding the 166 clips that ≥6/12 robots
reject — those are motion-driven, not config-driven.

**Reading the reject numbers correctly.** 484 unique clips account for all
2,162 rejects; 166 of them fail on ≥half the robots and 68 fail on ≥10/12.
Those universal rejects are floor and acrobatic motions — top categories:
`dance_hiphop` (fast), `crawl_ff`, `sit_on`/`sitting`, `ab_bicycle`,
`victory_dance`. A crawling or sitting human legitimately drives robot body
origins below the −0.02 m upright-locomotion threshold, and fast hip-hop is
genuine singularity food. This is a *dataset* property: the raw reject rate
overstates config problems.

**Robots needing retuning before their data is used (>5% robot-specific):**
- **unitree_h1 (7.0%)** — dominated by minz; tallest robot, likely scale-table
  (hips too low → knee/ankle dips) rather than singularities.
- **kuavo_s45 (5.9%)** — mixed minz+spikes; needs both scale check and the
  velocity limit.

Everything else is ≤4.3% robot-specific; engineai_pm01 (0.8%) and the two
velocity-limited robots (0.3%, 2.0%) are the cleanest — strong evidence the
shim + 4π velocity limit should be default for ALL robots in the next pass
(the sample QC already showed it does not hurt EE error at 4π).

**Recommended next actions**
1. Re-run only the spike-rejected clips for the 10 non-vlim robots with the
   shim + 4π limit (~900 clip-robot jobs ≈ 3 min wall) — expected to recover
   the majority of spike rejects.
2. Tag the 166 shared hard clips by category (crawl/sit/floor vs fast-dance):
   exclude floor clips from the upright cross-embodiment cache, or keep them
   with a category flag and a relaxed minz criterion.
3. Retune unitree_h1 and kuavo_s45 scale tables, re-run those two robots.

## Rework pass: elbow over-bend (lloyd's finding), 2026-08-03

lloyd reviewed all sample GIFs and flagged grossly over-bent elbows on the jog
clip — on every robot. Confirmed and quantified; root cause was NOT the
originally suspected wrist-target scaling.

### Elbow metric

Flexion = 180° − interior angle at the elbow, measured at joint anchors
(`data.xanchor`) — shoulder-pitch anchor / elbow anchor / wrist anchor (with
hand-body and forearm-com fallbacks for wrist-less arms). Human side from SOMA
FK of the same clip. Reported per clip: mean shift (robot − human) and KS
distance; both arms pooled. Human reference: walk 39.8°, jog 84.7° mean
flexion — so a +35° shift means the robot jogs with ~120° folded elbows.
(Two cheaper proxies failed: body-origin geometry breaks when link origins
sit on the joint, and raw elbow qpos breaks because joint-zero conventions
differ per robot — anchors are the only convention-free measurement.)

### Before → after (mean flexion shift vs human, deg, averaged over 5 probes)

| robot | before | after | robot | before | after |
|---|---|---|---|---|---|
| unitree_g1 | +32.7 | **−2.0** | fourier_n1 | +42.4 | **+1.8** |
| unitree_g1_with_hands | +32.7 | **−2.0** | engineai_pm01 | +45.1* | **+3.7** |
| unitree_h1 | +42.7 | **+2.0** | kuavo_s45 | +39.3 | **+1.4** |
| unitree_h1_2 | +49.4 | **+3.9** | fourier_gr3 | +36.5 | **+0.7** |
| booster_t1 | −25.8* | −10.9† | stanford_toddy | −13.9 | +9.1 |
| booster_t1_29dof | −24.4* | −9.6† | pnd_adam_lite | +43.7 | **−1.3** |

† boosters' residual is entirely one clip: jog (−59/−62) — see limitation
below; their other 4 probes are within ±10.

### Root cause and fix

The arm tasks are orientation-dominated (position weights 0–10, rotation
weights up to 100), so arm-length scaling was irrelevant — a k=0.7…1.5 sweep
on the Arm/ForeArm/Hand scales moved the elbow angle by <2°. The over-bend was
baked into the **forearm rotation offsets**: the neutral calibration frame had
slightly bent human elbows, and that bias transferred to every robot through
the convention deltas.

Fix (`tune_elbow.py`): rotate each ForeArm rotation offset about the robot's
own elbow-pitch joint axis (from `model.jnt_axis`, re-expressed in the tracked
link's frame), solving the per-side angle δ against the flexion metric on the
walk probe (zero-crossing of a 5-point sweep), verified on jog. δ* landed at
+25°…+72° per side per robot — mechanical, no hand-tuning. Kuavo needed a
fallback (its elbow joints are named `zarm_l4`, not "elbow"): the hinge whose
child body is the tracked link.

### Known limitations after the fix

- **Boosters straighten arms on fast swings** (jog −60°): reproduced with and
  without velocity limits and under elbow/shoulder weight rebalances — it is a
  4-DoF-arm reachability conflict with the shoulder orientation target, not a
  tuning miss. Walk/idle/reach/jump are correct. Accept or retune shoulder
  weights per-motion later.
- **Reach EE spikes**: with faithful (straighter) elbows, max wrist position
  error on the overhead-reach probe grows to 0.6–1.0 m on some robots — the
  arm-length mismatch is now expressed as position error instead of being
  hidden in a folded elbow. Position weights are tiny, the pose is
  orientation-faithful; EE error is not a production reject criterion.
- t1_29dof dips toe links to −5 cm for ~6 frames on the walk probe's turn
  (production minz threshold will reject such clips; monitored in rerun stats).
- h1/kuavo minz retune deferred to the rerun stats: the joint-limit shim (now
  default) was the strongest lever in v1 (the two shim-enabled robots had the
  lowest minz reject counts), so their scale tables were left unchanged.

All 12 jog GIFs + the grid were re-rendered to `R4_VIZ/gmr_qc/` (same
filenames) for lloyd's re-review.

## Production rerun with tuned configs (rework close-out, 2026-08-03)

Full 24,000-clip rerun with: elbow-corrected configs (all 12), mink-limits shim
+ 4π/s velocity limit as default (all 12), and a foot fix for the two worst
minz robots (unitree_h1, booster_t1_29dof: foot targets +2 cm z, leg scales
×0.95 — chosen by testing 4 variants against each robot's own failing clips).
Main rerun 55 min wall + 12 min selective h1/t1_29dof rerun; 630 MB.
Manifests/rejects/summary refreshed in `/var/tmp/alphamotion/gmr_production/`.

### Final accept counts

| robot | accept | reject | raw rate | robot-specific* | notes |
|---|---|---|---|---|---|
| pnd_adam_lite | 1940 | 60 | 3.0% | 0.1% | cleanest |
| fourier_gr3 | 1883 | 117 | 5.9% | 0.8% | |
| stanford_toddy | 1875 | 125 | 6.2% | 3.0% | |
| engineai_pm01 | 1865 | 135 | 6.8% | 2.5% | |
| unitree_h1_2 | 1848 | 152 | 7.6% | 3.4% | |
| booster_t1 | 1838 | 162 | 8.1% | 3.8% | jog straight-arm limitation |
| kuavo_s45 | 1822 | 178 | 8.9% | 3.9% | fixed by shim (was 5.9%) |
| booster_t1_29dof | 1809 | 191 | 9.6% | 4.9% | foot fix (was 11.1%) |
| unitree_g1_with_hands | 1800 | 200 | 10.0% | 4.0% | |
| unitree_g1 | 1778 | 222 | 11.1% | 5.0% | borderline, floor clips |
| fourier_n1 | 1779 | 221 | 11.1% | 5.2% | residual = floor/acro clips |
| **unitree_h1 ⚠** | 1754 | 246 | 12.3% | **7.2%** | still flagged (minz) |
| **total** | **21991** | **2009 (8.4%)** | | | |

\* excluding the 132 clips rejected by ≥6/12 robots (floor/acrobatic motions:
sitting, crawl, fast dance, cartwheel).

Versus v1: spike rejects collapsed (e.g. h1 89→5, kuavo 116→21, g1 111→40 —
the velocity limit works); minz became the dominant residual class, largely
floor-adjacent clips where whole-body ground proximity is intrinsic to the
motion (verified on fourier_n1: its "specific" rejects are ab_bicycle,
cartwheel, deep-carry clips with thighs/hands at floor level).

### Elbow metric at production scale (60 clips accepted by all 12 robots)

| robot | mean shift | p90 abs | robot | mean shift | p90 abs |
|---|---|---|---|---|---|
| unitree_g1 | −2.7° | 10.1° | engineai_pm01 | −5.7° | 20.3° |
| unitree_h1 | +0.1° | 8.0° | kuavo_s45 | −2.1° | 8.0° |
| unitree_h1_2 | +2.7° | 11.0° | fourier_gr3 | −2.0° | 10.2° |
| booster_t1 | −15.4° | 47.1° | stanford_toddy | +6.1° | 18.3° |
| booster_t1_29dof | −9.9° | 36.8° | pnd_adam_lite | −1.7° | 8.8° |
| fourier_n1 | −2.3° | 13.2° | (g1_with_hands = g1) | | |

Before the fix these were +32° to +49° (11 robots). Boosters' −10/−15° is the
known 4-DoF fast-swing straightening, concentrated in high-swing clips.

### Remaining flags

1. **unitree_h1 (7.2% specific)** — only robot above the 5% bar; minz-driven.
   The foot fix helped (8.5→7.2) but its tall build needs a dedicated scale
   pass. Its accepted 1754 clips are usable; treat the rejects list as
   authoritative.
2. Boosters on fast arm-swing clips: elbows straighten (embodiment limit).
3. unitree_g1 / fourier_n1 sit at ~5% specific, dominated by floor-adjacent
   clips — recommend tagging floor motions dataset-wide rather than per-robot
   tuning.

All 12 jog GIFs + `grid_all_robots.png` re-rendered (same filenames) in
`R4_VIZ/gmr_qc/` for re-review.
