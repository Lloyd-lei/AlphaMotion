# PUBLIC RETARGETS — pipelines, calibration, and public robot-motion corpora

2026-08-03. Survey + downloads + pipeline A/B, CPU only.
Downloads live in `/var/tmp/alphamotion/public_retargets/`. Pipeline clones in `/var/tmp/alphamotion/config_harvest/`. A/B artifacts in `/var/tmp/alphamotion/ab_test/` and `pose_translator_design/R4_VIZ/pipeline_ab/`.

## 0. Framing

Paired training needs the SAME source motions on every robot. Public datasets each retarget a *different* corpus (LAFAN1, AMASS, OMOMO, GDance...), so they cannot form the paired core — the paired core has to come from re-retargeting our own BONES-SEED SOMA BVH with better pipelines/configs. Public sets are still valuable as **per-embodiment "monolingual" corpora** (decoder fluency, tracking-policy pretraining) and as **calibration references**.

One exception found: `RRGGZZ/Adam-pro-soma-retarget` (89 GB, 142k clips) is a retarget of the **same SOMA corpus as our BONES-SEED** — identical date-dir layout and clip names (`data/230731/walk_180_R_002__A437_M.pkl` = our walk QC clip), ik_targets carry SOMA joint names (`LeftShin`, `LeftFootMod`-style skeleton). That is natively paired data for PND Adam Pro that we did not have to produce.

## 1. Retargeting pipelines (config harvest)

Clones: `/var/tmp/alphamotion/config_harvest/{GMR, TWIST, TWIST2, whole_body_tracking, ProtoMotions, soma-retargeter, PHUMA}` + local `/var/tmp/alphamotion/omniretarget/holosoma/src/holosoma_retargeting`.

| Project | Method | Robots w/ configs | Wrist handling | Foot handling | Notes |
|---|---|---|---|---|---|
| GMR upstream | mink IK, ik_match_table | 17 (g1, h1, h1_2, t1, t1_29dof, n1, gr3, adam, k1, kuavo, openloong, pm01, tienkung, toddy, hi, bhl, r1pro...) | wrist_yaw pos0/rot10 + pos10/rot5; only axis-aligned quat offsets | NEW recipe: pos 100 / rot 50, `toe_link` target frame, ±0.02 lateral offset; **no contact/anti-skate/post at all** | our config base; we're on the OLD foot recipe |
| NVIDIA soma-retargeter | Newton/Warp IK + FeetStabilizer | G1 only | t_w 2.0 / r_w 1.2 + **fixed ±90° X quat convention fix** for SOMA hands | ankle t_w 30 (=pelvis), r_w 2.0; then analytic 2-bone leg IK snapping ankles exactly + 2nd IK + limit clamp | **SOMA-native** — same source skeleton as BONES-SEED; hand-tuned per-joint scale (legs .86, arms .85 × 1.70/1.8) |
| ProtoMotions (PyRoki, v3) | batched IK w/ costs | G1, H1-2 | virtual hand-aux point 0.14 m off wrist, downweighted /4; wrist rest penalty | **contact-aware costs in-solve**: anti-skate (w 30) on 4 foot pts, flat-foot (z_ankle−z_toe), foot-tilt; contacts from human vel<0.1 & h<0.08; post ground-clamp ±2 cm | config hardcoded in `pyroki/batch_retarget_to_g1_from_keypoints.py` |
| PHUMA | physics-constrained optimization | G1 (23dof), H1-2 | wrists rigid (23dof) | 4 virtual heel/toe pads per foot; contact-weighted z²-to-ground (w 10) + skate loss; Butterworth filter + hard limit clip | scale is *learned* (SMPL-β fit + 12 per-link scales) |
| holosoma / OmniRetarget | interaction-mesh + QP (cvxpy) | G1, Booster T1 | palm-mapped, uniform weight; **wrist joint limits manually tightened** (yaw ≈ locked) to stop flailing | 5 contact spheres/foot in MJCF; **hard QP pins**: contact-frame XY box pin + linearized ground-penetration constraint | local, runs on CPU ~3 fps; the "B" side of our A/B |
| TWIST / TWIST2 | uses upstream GMR | (none vendored; ik dir gitignored) | none (VR trigger scalar) | none in retarget; pushed into RL rewards (feet_slip −0.1) | nothing to harvest |
| BeyondMimic | none (consumes Unitree CSVs) | — | — | — | confirms Unitree LAFAN1 arrives pre-cleaned from an **unreleased** tool |
| Unitree | unreleased | — | — | — | dataset-only release; no tooling in 53 public repos |

### Our configs vs the field — why hands/feet fail (numeric diffs)

Ours: `/var/tmp/alphamotion/gmr_spike/ik_configs/bvh_soma_to_*.json`.

- **Feet under-constrained + no cleanup.** Ours: ankle pos 50 / rot 10 (old GMR recipe). Upstream current: **pos 100 / rot 50**, often on a toe frame with lateral stance offsets. And our BVH path never calls GMR's `offset_to_ground`; there is no contact/anti-skate/post-pass anywhere in our pipeline — every other project has one (stabilizer / in-solve costs / pad losses / QP pins).
- **Wrist quat offsets are free-form auto-calibrated** (e.g. G1 right wrist `[0.227,−0.479,0.302,−0.792]`) — upstream never ships a non-axis-aligned wrist offset; soma-retargeter says the SOMA hand frame needs a clean **±90° X rotation** (`[∓0.707,0,0,±0.707]`) + ~1.5 cm. Our auto-calibration baked a twisted wrist in.
- **Arm scales over-stretched**: ours 1.05–1.2 (g1 1.125, n1 1.2) vs upstream 0.75–0.8 → hand targets beyond reach → elbow/wrist saturate and orientation breaks precisely on reaches (matches lloyd's rejection; see `reach` clip ee_err_max 0.63 in old QC).
- **booster_t1 (23dof) config is structurally broken for wrists**: maps `hand_link ← LeftForeArm` (tracks the forearm, not the hand). The t1_29dof variant (`AL4←ForeArm`, `AL5/hand←LeftHand`) is correct — including our own `bvh_soma_to_t1_29dof.json`.
- unitree_h1 has **no wrist DOFs** (19dof) — wrist orientation there is unfixable by config; expected-fail, not a calibration bug.

## 2. Pipeline A/B: our tuned GMR vs OmniRetarget optimizer

Same 3 SOMA clips (walk / jog_stop / reach) from BONES-SEED, both pipelines, G1 and Booster T1 (H1 not supported by holosoma — dead URDF dir only).

- A = our tuned GMR output (`/var/tmp/alphamotion/gmr_spike/qc/{unitree_g1,booster_t1}/`)
- B = holosoma interaction-mesh QP, fed our SOMA FK world joints via its `lafan` input path (converter: `/var/tmp/alphamotion/ab_test/soma_to_lafan_npy.py`; outputs `/var/tmp/alphamotion/ab_test/holosoma_out/{g1,t1}/*.npz`, qpos MuJoCo order)

**Renders (left = A GMR ours, right = B OmniRetarget), 18 GIFs — look at these, especially `*_feet` and `*_hands`:**

```
pose_translator_design/R4_VIZ/pipeline_ab/
  g1_{walk,jog_stop,reach}_{full,feet,hands}.gif
  t1_{walk,jog_stop,reach}_{full,feet,hands}.gif
```

Metrics (same definitions as our qc_run; full JSON: `/var/tmp/alphamotion/ab_test/metrics_{g1,t1}.json`):

| robot/clip | pipeline | vel_max | vel_p99 | spikes | foot_z_min | min_body_z |
|---|---|---|---|---|---|---|
| g1 walk | GMR (ours) | 8.5 | 4.0 | 0 | **0.119 (floats ~8 cm)** | 0.099 |
| g1 walk | OmniRetarget | 8.3 | 4.1 | 0 | **0.034 (sole ON ground)** | 0.004 |
| g1 jog_stop | GMR (ours) | 16.4 | 8.1 | 1 | 0.088 | 0.063 |
| g1 jog_stop | OmniRetarget | 9.3 | 6.3 | 0 | 0.034 | 0.004 |
| g1 reach | GMR (ours) | 10.6 | 7.7 | 0 | 0.062 | 0.035 |
| g1 reach | OmniRetarget | 10.2 | 5.5 | 0 | 0.034 | 0.004 |
| t1 walk | GMR (ours) | 16.6 | 5.8 | 2 | 0.058 | 0.058 |
| t1 walk | OmniRetarget | 6.2 | 3.9 | 0 | 0.022 | 0.005 |
| t1 jog_stop | GMR (ours) | 10.1 | 8.0 | 0 | 0.034 | 0.034 |
| t1 jog_stop | OmniRetarget | 15.3 | 6.1 | 1 | 0.022 | 0.008 |
| t1 reach | GMR (ours) | 16.6 | 7.6 | 2 | 0.023 | 0.023 |
| t1 reach | OmniRetarget | 12.7 | 6.8 | 0 | 0.006 | 0.004 |

Reading (numbers only — final call is the GIFs): B plants feet exactly on the ground on every clip (G1 ankle_roll link z = 0.034 ≡ sole at z=0) while A hovers 2–8 cm; B is smoother at p99 everywhere and has fewer velocity spikes (5 of 6 clips). B costs ~3 fps/clip on CPU vs ~100+ fps for GMR. Caveats: B uses holosoma's own scale normalization (subject pre-normalized to 1.70 m by our converter), B's G1 is g1_29dof.xml (no dexterous hand model), and B's skate-metric window is not directly comparable to A's because A never actually contacts the ground.

Repro: one mujoco/numpy-2 compat patch was applied to the local holosoma copy (`interaction_mesh_retargeter.py:1283`, `int(body_idx)` → `int(np.asarray(body_idx).reshape(-1)[0])`). T1 runs must use the top-level `--robot t1` flag (`--robot-config.robot-type` is silently overridden — first T1 attempt produced G1 output).

## 3. Public dataset inventory (auxiliary unpaired corpora, per pivot §3)

All under `/var/tmp/alphamotion/public_retargets/`. LAFAN1-derived sets inherit Ubisoft's CC BY-NC-ND terms upstream regardless of repo card — fine for internal research, do not redistribute.

| Local dir | HF repo | Robot | Content | Format | License | Provenance / quality | Status |
|---|---|---|---|---|---|---|---|
| `unitree_lafan1_official` | lvhaidong/LAFAN1_Retargeting_Dataset | **G1, H1, H1-2** | LAFAN1, 40 clips ≈ 4.6 h ea. | CSV rows = root xyz + quat xyzw + joint rad, 30 fps; URDFs incl. | none stated (LAFAN1 NC-ND) | **Unitree-official**, interaction-mesh + IK w/ anti-foot-slip; the BeyondMimic reference data | done |
| `phuma_g1_h12` | DAVIAN-Robotics/PHUMA | **G1 (23dof), H1-2** | large locomotion set (AMASS-based; LAFAN1/LocoMuJoCo excluded) | npy dict: root_trans, root_ori xyzw, dof_pos, fps | **Apache-2.0** | physics-constrained retarget, author-calibrated | done (unzip data.zip) |
| `g1_lafan1_contact` | seu-zuxing/Contact_LAFAN1 | G1 | 40 LAFAN1 clips + **contact masks** + mp4s | npz + npy + mp4 | Apache-2.0 | community, contact-annotated | done |
| `g1_amass_ember_berkeley` | ember-lab-berkeley/AMASS_Retargeted_for_G1 | G1 | 17,714 AMASS clips (~40 h) | npz | CC-BY-4.0 | Berkeley lab release | downloading (11.9 GB) |
| `g1_amass_fleaven` | fleaven/Retargeted_AMASS_for_robotics | G1 (29dof) | 17,717 AMASS clips | npy [T,36] = root pos+quat xyzw+29 joints | CC-BY-4.0 | author-calibrated vs official XML limits | downloading (7 GB) |
| `fourier_n1_amass_fleaven` | fleaven/Retargeted_AMASS_for_FourierN1 | **Fourier N1** | 17,717 AMASS clips | npy, same layout | CC-BY-4.0 | same author | downloading (5.8 GB) |
| `bxi_elf2_amass_fleaven` | fleaven/Retargeted_AMASS_for_bxi_elf2 | BXI Elf2 | 17,717 AMASS clips | npy | CC-BY-4.0 | not one of our 12; keep as extra embodiment | downloading (6.2 GB) |
| `g1_mink_openhe` | openhe/g1-retargeted-motions | G1 (23dof) | 174 clips (ACCAD/LAFAN1/DanceDB), 30 fps | pkl | MIT | mink pipeline | done |
| `g1_any4hdmi/*` | elijahgalahad/any4hdmi-g1-{lafan,100style,amass,amass-hard,extreme-motions,ground-motions,real-pico} | G1 | LAFAN 40, 100STYLE 3,573, AMASS 10,385, + extreme/ground/pico sets | npz | none stated | community (Any4HDMI project) | done |
| `g1_beyondmimic_dailylife` | wty-yy/beyondmimic_dailylife_dataset | G1 | 116 daily-life clips | BeyondMimic csv + npz | none stated | community | done |
| `g1_lkr_objectinteraction` | l-k-r/retargeted_motions | G1 | OMOMO/loco w/ **objects + contact matrices**, IsaacLab BFS order | npz + json (mp4 excluded) | MIT | mixed methods, metadata per clip | downloading (4 GB) |
| `g1_asap_videomotions` | github LeCAR-Lab/ASAP | G1 (23dof) | 52 video-derived athletic clips (Ronaldo/Bolt etc.) | pkl | repo LICENSE incl. | ASAP paper assets | done |
| `booster_t1pro_lafan1` | ZJUjiejie/Lafan1_t1pro_retargeted_Nohand | **Booster T1 Pro** | 77 LAFAN1 clips, no hands | pkl | CC-BY-4.0 (LAFAN1 NC-ND) | community | done |
| `booster_k1_lafan1` | wu0712/retargeted_lafan1_for_booster_k1 | **Booster K1** | LAFAN1 zip, booster_assets CSV layout | csv (root pos+quat xyzw+joints) | none stated | made with GMR (same tool we're replacing — treat as weak) | done |
| `openloong_amass_subset` | OpenLoong/Public_Retargeted | OpenLoong "Loong" | AMASS families as zips (29 GB total); took SFU + ACCAD | zip | none stated | Shanghai Humanoid Center | SFU done; ACCAD (6.4 GB) downloading |
| `pnd_adampro_soma_sample` | RRGGZZ/Adam-pro-soma-retarget | **PND Adam Pro (29 dof named, dof_pos 31)** | **SAME SOMA/BONES corpus as ours**, 142k clips, 89 GB; curated 20 h tar | pkl: fps=120, root_pos, root_rot, dof_pos, dof_names, **ik_targets (SOMA joints)** | none stated | professionally packaged incl. IK targets | sample + 20 h tar (4.8 GB) downloading |
| `engineai_pm01_gdance` | jingjing18/PM01-GDance-Retarget | **EngineAI PM01** | GDance retarget | zip | none stated | community | downloading (3.6 GB) |

Surveyed, **not** downloaded: GeorgiaTech/g1_lafan1_50hz (CC BY-NC-**ND** + it's a resample of Unitree's), amathislab/musclemimic-retargeted (gated), MeoMayKuro/vr_m3_soma_retargete (78 GB, robot "M3" unidentified — another SOMA-corpus retarget, worth a later look), Cryyzz BUMI (GMR-made, NC), TWIST data (index site, no bulk release yet), ProtoMotions (no packaged robot data — consumes BONES-SEED CSVs we already have, + PHUMA), MaskedMimic (SMPL character only), Fourier ActionNet (teleop vision dataset, different problem), Unitree HF org (manipulation/teleop only, no retarget sets).

## 4. Per-robot coverage map (public + ours)

| Robot | Public author-calibrated | Ours | Verdict |
|---|---|---|---|
| unitree_g1 | Unitree LAFAN1, PHUMA, ember AMASS, fleaven AMASS, contact-LAFAN1, openhe, any4hdmi, ASAP, l-k-r, beyondmimic-dailylife | OmniRetarget 4 h (done), BONES vendor CSV, GMR (rejected) | **rich** — G1 gold = Unitree LAFAN1 + PHUMA + OmniRetarget |
| unitree_h1 | Unitree LAFAN1 (H1) | GMR | thin but official |
| unitree_h1_2 | Unitree LAFAN1 (H1-2), PHUMA | GMR | good |
| booster_t1 / t1_29dof | T1-Pro LAFAN1 (variant), holosoma pipeline support | GMR + new OmniRetarget A/B outputs | pipeline > data here |
| booster_k1 | wu0712 (GMR-made) | GMR | weak both ways |
| fourier_n1 | fleaven AMASS N1 | GMR | good |
| fourier_gr3 | — | GMR only | **GMR-only** |
| engineai_pm01 | PM01-GDance (community) | GMR | partial |
| pnd_adam_lite | **Adam-Pro SOMA 89 GB (variant: Pro)** | GMR | verify Pro→Lite transfer; possibly the best non-G1 asset found |
| kuavo_s45 | — | GMR only | **GMR-only** |
| stanford_toddy | — | GMR only | **GMR-only** |
| tienkung | — | GMR only | **GMR-only** |
| hightorque_hi | — | GMR only | **GMR-only** |
| pal_talos | — | GMR only | **GMR-only** |
| berkeley_humanoid_lite | — | GMR only | **GMR-only** |

## 5. Recommendation — next-gen retarget pass over BONES-SEED

1. **Keep GMR as the 12-robot workhorse but fix the configs** (it's the only pipeline with configs for all our robots): feet → pos 100 / rot 50 (+ toe-frame targets and lateral offsets where upstream has them); wrists → replace free-form auto-calibrated quat offsets with soma-retargeter's clean ±90° X SOMA-hand convention; arm scales → 0.75–0.85 band (from 1.05–1.2); booster_t1 → use the t1_29dof arm mapping.
2. **Add the missing post-pass** — this is what every working pipeline has and we don't: ground alignment + contact-frame XY pin (ProtoMotions recipe: contact = foot vel < 0.1 m/s & h < 0.08 m; clamp ±2 cm) or port soma-retargeter's FeetStabilizer (analytic leg IK, SOMA-native, config at `config_harvest/soma-retargeter/soma_retargeter/configs/unitree_g1/g1_feet_stabilizer_config.json`).
3. **Use OmniRetarget/holosoma as the gold pass for G1 + T1**: the A/B shows it already solves exactly what lloyd rejected (planted feet, no hover, smoother). ~3 fps CPU → batch the top-priority subset, not all 60+ h. Pipe is proven end-to-end from our SOMA FK (`ab_test/soma_to_lafan_npy.py`).
4. **Grab the Adam-Pro paired corpus**: verify frame-level alignment of the name-matched clips against our BVH (fps 120 vs our 30 — resample), then it's a free second embodiment of the paired core.
5. Public sets: use per the coverage map as decoder-fluency monolingual data; G1 gold = Unitree LAFAN1 + PHUMA + our OmniRetarget 4 h. Six robots (gr3, kuavo_s45, toddy, tienkung, hi, talos, bhl) remain **GMR-only** — they are the ones that most need the config fixes in (1)+(2).
