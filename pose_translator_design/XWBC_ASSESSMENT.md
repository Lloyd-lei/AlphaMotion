# XWBC Retarget Dataset Assessment

Dataset: `TonyZhang1999/xwbc-retarget-data` (HuggingFace, MIT license, ~138 GB)
Assessed: 2026-08-03. Sample workspace: `/var/tmp/alphamotion/xwbc_sample/`
Sampled robots: unitree_g1_29dof, unitree_h1, fourier_n1 (+ human_motion archive).

## TL;DR verdict

**Strong candidate — recommend downloading.** Same AMASS clip set per robot
(paired by filename), data is exact qpos for GMR's own MuJoCo XMLs (FK matches
to <1 mm), 30 fps, joint names included, feet planted (no penetration/float in
sampled clips), and it ships the paired **human motion** (22 SMPL bodies) so we
may not even need raw AMASS for pairing. Covers 11 of our 12 production robots.

## 1. Repo layout

- `amass/<robot>.tar.gz` x 18 robots + `amass/human_motion.tar.gz` (~125 GB total,
  5.1-8.1 GB per robot, README claims ~13k clips per robot)
- `lafan/<robot>.tar.gz` x 18 robots + `lafan/human_motion.tar.gz` (~13 GB total,
  0.5-0.85 GB per robot)
- Monolithic tar.gz per robot; no per-clip remote files. Inside each tar:
  `<robot_name>/<clip>.npz`.
- Robots (18 archives, README lists 16): unitree_g1_29dof, unitree_g1_23dof,
  unitree_h1, unitree_h1_2, unitree_h2, unitree_r1, berkeley_humanoid_lite,
  booster_k1, booster_t1, booster_t1_29dof, engineai_pm01, fourier_gr3,
  fourier_n1, hightorque_hi, kuavo_s45, pal_talos, pnd_adam_lite, tienkung.

## 2. Pairing check (make-or-break)

- AMASS head-samples (first ~30-40 clips streamed from each tar):
  - unitree_h1 vs fourier_n1: 33/33 overlap; both vs human_motion: 29/29.
    These three tars share an identical (shuffled) entry order.
  - unitree_g1_29dof tar is ordered alphabetically (all ACCAD first), so its
    head-sample has 0 overlap with the shuffled tars — an artifact of tar entry
    order, not of clip-set mismatch.
- LAFAN full-archive check (all three robots downloaded completely):
  - Each robot tar contains exactly 77 clips = the complete LAFAN1 set
    (standard names: walk/run/dance/fight/obstacles/fallAndGetUp/..._subjectN).
  - unitree_h1 vs fourier_n1: 77/77 identical filenames (100% overlap).
  - unitree_g1_29dof vs unitree_h1: 77/77; vs fourier_n1: 77/77.
  - **Verdict: 100% pairing — identical clip filename sets across robots.**
  - LAFAN clips are long continuous takes (e.g. h1 walk1_subject1: 7839
    frames = 4.4 min at 30 fps) — same npz schema as AMASS clips. Good fit
    for long-variable-length-sequence training.
- Clip filenames are identical strings across robots and across the
  human_motion archive, so pairing is by exact filename match.

## 3. Format spec

Robot NPZ keys (e.g. g1_29dof, T frames):
- `fps`: [30] (int64) — all sampled clips 30 fps
- `joint_pos`: (T, n_dof) float32 — actuated joint angles, radians
- `joint_vel`: (T, n_dof) float32 — exact central difference of joint_pos at 30 fps (verified, err 0.0)
- `body_pos_w` (T, n_body, 3), `body_quat_w` (T, n_body, 4) wxyz,
  `body_lin_vel_w`, `body_ang_vel_w` — world-frame per-body kinematics
- `joint_names`: (n_dof,) strings — Isaac-Lab BFS joint order (interleaved
  left/right), NOT MuJoCo depth-first order; remap by name.

n_dof per sampled robot: g1_29dof=29, h1=19, fourier_n1=23.
n_body: 32 / 20 / 24 (body index 0 = floating-base root; no body_names key in
robot files).

`human_motion` NPZ: `schema_version` = human_motion_v1, `fps` = 30,
`body_names` (22 SMPL joints: Pelvis...L/R_Wrist), `body_pos_w`,
`body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`. This is the paired human
side — usable directly as translator source without raw AMASS.

### qpos compatibility with GMR XMLs (verified)

Reconstruction: `qpos = [body_pos_w[t,0], body_quat_w[t,0](wxyz), joint_pos[t] remapped by joint_names]`.
FK cross-check (MuJoCo forward vs stored body_pos_w, nearest-neighbor over bodies):

| robot | GMR XML | npz dof | XML nq | joint-name match | FK err |
|---|---|---|---|---|---|
| unitree_g1_29dof | unitree_g1/g1_mocap_29dof.xml | 29 | 36 (=7+29) | exact set | 0.0 mm |
| unitree_h1 | unitree_h1/h1.xml | 19 | 26 (=7+19) | exact set | 0.0 mm |
| fourier_n1 | fourier_n1/n1_mocap.xml | 23 | 30 (=7+23) | exact set | 0.0 mm |

The data was almost certainly produced with the same (GMR-lineage) MuJoCo
models we already use — drop-in compatible.

## 4. Quality first impressions (renders)

GIFs (full-body + hands/feet closeup side by side), 10 clips:
`/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/R4_VIZ/xwbc_sample/`
- `unitree_g1_29dof__{run,walk,reach,sidestep}.gif`
- `unitree_h1__{run,walk,reach}.gif` (run = SOMA run_001, walk = KIT circle walk, reach = GRAB teapot pour)
- `fourier_n1__{run,walk,reach}.gif`

First impressions (final call is Lloyd's, from the GIFs):
- Feet: planted and flat during stance in walk/run strips; numeric
  min-body-z over sampled clips is 0.000-0.022 m (no penetration, no floating).
- Hands/arms: GRAB teapot-pour and ACCAD pick-up-box show articulated,
  non-degenerate wrist/elbow motion; no arm-flailing seen in samples.
- Gait looks natural (proper knee flexion, no skating evident at GIF frame rate).
- Sampled clip lengths: 50-1521 frames (1.7-50 s), mean ~7.7 s at 30 fps.

## 5. AMASS mapping

- Filenames are AMASS SMPL-X (stageii) ids with `__` as path separator:
  `CMU__09__09_11_stageii.npz` <-> AMASS `CMU/09/09_11_stageii.npz`,
  `ACCAD__Female1Running_c3d__C3_-_Run_stageii.npz` <->
  `ACCAD/Female1Running_c3d/C3 - Run_stageii.npz`. Mapping is trivial.
- Local raw AMASS spot-check FAILED for a different reason: the stated local
  copy at `data/amass/` does not exist (searched the project tree, /mnt,
  /var/tmp; only found third-party retargets using the same stageii naming,
  e.g. `/var/tmp/alphamotion/public_retargets/g1_mink_openhe/ACCAD_retargeted/`,
  whose names match 1:1). The raw-AMASS location note appears stale.
- Mitigation: the dataset's own `amass/human_motion.tar.gz` (5.1 GB) carries
  the paired human motion at 30 fps with identical filenames — sufficient for
  human<->robot pairing without raw AMASS. Raw AMASS only needed if we want
  SMPL-X parameters (betas/poses) rather than 22-body world kinematics.

## 6. Associated paper

- Likely **XHugWBC: "Scalable and General Whole-Body Control for
  Cross-Humanoid Locomotion"** (arXiv 2602.05791, xhugwbc.github.io; Shanghai
  AI Lab / SJTU; 12 simulated + 7 real humanoids).
- Caveats: neither the HF card nor the project page links the other
  explicitly; code is "coming soon" on the project page. HF uploader
  "TonyZhang1999" has fullname "SII-juntongz" (SII = Shanghai Innovation
  Institute), consistent with the Shanghai group. Call it "very likely, not
  formally confirmed".

## 7. Coverage vs our 12 production robots

| our robot | in dataset | note |
|---|---|---|
| unitree_g1 | YES (unitree_g1_29dof, also 23dof) | body-only |
| unitree_g1_with_hands | no | no hand-dof variant in dataset |
| unitree_h1 | YES | |
| unitree_h1_2 | YES | |
| booster_t1 | YES | |
| booster_t1_29dof | YES | |
| fourier_n1 | YES | |
| engineai_pm01 | YES | |
| kuavo_s45 | YES | |
| fourier_gr3 | YES | verified: dataset gr3 has 31 dof with joint names identical to our gr3v2_1_1 XML (GMR maps key "fourier_gr3" to that model) |
| stanford_toddy | NO | only real gap |
| pnd_adam_lite | YES | |

11/12 covered. Bonus robots we don't have in prod: unitree_h2, unitree_r1,
berkeley_humanoid_lite, booster_k1, hightorque_hi, pal_talos, tienkung,
unitree_g1_23dof.

## 8. Download recommendation

- Recommended minimal pull (our 11 covered robots + human pairing):
  amass: g1_29dof 8.1 + h1 5.2 + h1_2 6.7 + booster_t1 6.1 + t1_29dof 7.6 +
  n1 6.2 + pm01 6.4 + kuavo 7.0 + gr3 8.0 + adam_lite 6.5 + human_motion 5.1
  = **~73 GB**; matching lafan tars ≈ +7.3 GB → **~80 GB total**.
- Full mirror is ~138 GB if we want the bonus robots for cross-embodiment
  scaling experiments (cheap increment, MIT license — no restriction).
- Observed throughput here ~0.4-1.5 MB/s per connection; use
  `huggingface-cli download --include` with several workers, expect hours.
- Storage note: tars expand; NPZ is uncompressed float32 — budget ~1.5-2x
  archive size after extraction, or read directly from tar. The / volume
  currently has ~298 GB free — the ~80 GB plan fits comfortably, the full
  138 GB mirror also fits if we skip extraction of unused robots.

## Open items

- Definitive per-archive clip counts / total AMASS coverage (need full tar
  listings; head samples span ACCAD, BMLmovi, CMU, CNRS, EyesJapan, GRAB, KIT,
  PosePrior, SOMA, TotalCapture, Transitions, WEIZMANN — broad AMASS coverage,
  incl. GRAB tabletop reaches which our GMR pipeline lacked).
- Confirm XHugWBC linkage when their code releases.
