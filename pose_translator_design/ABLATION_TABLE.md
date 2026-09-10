# Non-curriculum ablations, 0826–0829 (X2 / mjlab-MuJoCo-Warp / PPO)

Read-only reconstruction from `logs/rsl_rl/*/params/{env,agent}.yaml`, the saved
checkpoints, the training logs in `/home/arenalabs/tmp/` and the exam JSONs.
No GPU was used; parameter counts come from `torch.load(..., map_location='cpu')`
on each run's last checkpoint. Raw table: `/home/arenalabs/tmp/ablation_table.csv`.

Hardware: one NVIDIA RTX PRO 6000 Blackwell (95 GiB), so **GPU-hours = wall-clock hours**.
GPU-hours are computed as `iterations x median(Iteration time)` from the log,
excluding exam/render time. For resumed arms both the incremental and the
cumulative (parent included) cost are given.

---

## 0. What the exam actually measures

`scripts/horizon_exam.py`, 16 repeats per window (32 for the hidyn experts,
11 for the r9 generalist), no observation noise, no pushes, no spawn randomisation.

* **OLD criterion** (files written before 0828): `mastered = success >= 0.70 AND travel_ratio >= 0.80`.
* **NEW criterion** (files with `steps_ps` / `skate` / `ref_steps_ps`): the same,
  **plus**, for any window whose reference walks (`ref_steps_ps >= 0.5`):
  `steps_ps >= 0.5 * ref_steps_ps` **AND** `skate <= 0.15 m/s`.

The two criteria are not interchangeable. Every 2000-iteration file has been
**rescored under the old criterion** so one comparable column exists
(`mastered_old` below); the reverse is impossible, because the 600-iteration
files never measured the feet.

Pool used by every multi-window arm here: **224 windows**, of which **155 are gait
windows** (`ref_steps_ps >= 0.5`, mean reference cadence 1.45 steps/s).
`horizon_x2_hz3/pool_r12x.npz` and `horizon_x2_hz4/pool_r0.npz` carry the same
content on every field checked (identical `window_ids`, `clip_bounds`, `joint_pos`,
`ref_steps_ps`; same key set), and `pool_r1.npz` carries the same 224 ids — so the ab_xattn,
fs4_duel, wp_arm and hz4-round-1 arms are all scored on the *same* windows.

Exam repeatability: `fs4_mastery_TFS.json` and `horizon_x2_hz4/mastery_r0.json`
are two independent exams of the **same** checkpoint (md5-identical: TFS4
`model_999.pt` == hz4 `seed/model_seed.pt`). They return 49 vs 50 mastered,
0.345 vs 0.348 mean success. **Treat +/-1 window and +/-0.005 success as noise.**

---

## 1. The table

`params` = actor only (critic is a separate 2.08M / 3.98M / 4.51M MLP).
`obs` = what the **actor** sees; the critic always additionally sees the full
reference (`command`, `body_pos`, `body_ori`) — "ref hidden" means hidden from the actor only.

### 1a. Architecture / configuration

| arm | actor | actor params | token window (H x stride) | contact preview | ref visible to actor | reward set | pool | iters | envs |
|---|---|---|---|---|---|---|---|---|---|
| **X600**   | XAttn 3L/8H d=256 ff=1024, 31 joint queries, KV = 32 Equator temporal tokens | 3.36M (10.8k frozen `proj_out`) | no spatial window; 32x6 A3 codes + sin/cos phase | no | no | dense BM 6-term | pool_r12x (224w) | 600 | 4096 |
| **T600**   | Transformer 4L/4H d=256 ff=1024, `[ACT]` readout | 3.40M | 4 x 512 pooled Greenwich feats, stride 1 (0.08 s) | no | no | dense BM 6-term | pool_r12x | 600 | 4096 |
| **XW600**  | = X600, core warm-started from `xattn_pretrain.pt` (64 tensors) | 3.36M | as X600 | no | no | dense BM 6-term | pool_r12x | 600 | 4096 |
| **XF600**  | XAttn 2L/8H d=256 ff=512, KV = 2 x 256 raw code latents | 1.80M (10.8k frozen) | raw codes, frames (0, 2) x 256 x 20 | no | no | dense BM 6-term | pool_r12x | 600 | 4096 |
| **XF2000** | = XF600, resumed from XF600@599 | 1.80M | as XF600 | no | no | dense BM 6-term | pool_r12x | 1400 (2000 total) | 4096 |
| **T2000**  | = T600, resumed from T600@599 | 3.40M | as T600 | no | no | dense BM 6-term | pool_r12x | 1400 (2000 total) | 4096 |
| **TFS4**   | Transformer 4L/4H d=256 ff=1024 (`AM_TT_CFG` token_dim 2560 / frames 5) | 3.41M | 5 x 512, **stride 5** (0.5 s) | **yes** (2x15) | no | dense 6-term + foot 3-term (1.5 / -3.0 / -1.5), friction 0.7-1.2 | pool_r12x | 1000 | 4096 |
| **XFS4b**  | XAttn 2L/8H ff=512, KV = 5 x 256 raw code latents (1280 tokens) | 1.86M | raw codes, frames (0,5,10,15,25) | **yes** | no | dense 6-term + foot 3-term | pool_r12x | 200 scored / 361 burned | 4096 |
| **WP2000** | `TokenTransformerModel5x512`, resumed from TFS4@999 | 3.41M | 5 x 512, stride 5 | **yes** | no | **SPARSE**: waypoint pos 12.0 / ori 6.0 every 25 f; root pos/ori dense; foot 3-term; body lin/ang-vel terms **removed** | pool_r0 (= pool_r12x) | 2000 | 4096 |
| **hz4_r1** | `TokenTransformerModel5x512`, resumed from the **same** TFS4@999 | 3.41M | 5 x 512, stride 5 | **yes** | no | **DENSE** 6-term + foot 3-term | pool_r1 (same 224 ids) | 2000 | 4096 |
| **BM**     | MLP 512-256-128 (stock BeyondMimic) | 256k | none | no | **yes** (`command`) | dense 6-term only, friction 0.3-1.2 | amass_00115 (1 window) | 794 / 2000 | 2048 |
| **BMFS**   | MLP 512-256-128 | 256k | none | no | **yes** | + `foot_contact_match` **0.5** / `foot_slide` **-0.3**, friction 0.3-1.2 | amass_00115 | 682 / 2000 | 2048 |
| **BMFS2**  | MLP 512-256-128 | 256k | none | no | **yes** | + `foot_contact_match` **1.5** / `foot_slide` **-1.5**, friction **0.7-1.2** | amass_00115 | 2000 | 2048 |
| **BMFS3**  | MLP 512-256-128 (+30 preview inputs) | 272k | none | **yes** (2x15) | **yes** | + `foot_contact_match` **1.5** / `foot_slide` **-3.0** / `swing_refusal` **-1.5**, friction 0.7-1.2 | amass_00115 | 2500 | 2048 |
| **DANCE**  | MLP + preview | 272k | none | yes | yes | BMFS3 recipe | dance_lafan (150 f) | 494 / 3000 | 2048 |
| **DANCE2** | MLP + preview | 272k | none | yes | yes | BMFS3 recipe | dance_lafan2 | 1086 / 3000 | 2048 |
| **DANCE3** | MLP + preview | 272k | none | yes | yes | BMFS3 recipe + `spawn_planted_only` | dance_lafan2 | 1026 / 6000 | 2048 |
| **HIDYN-00492** | `TokenTransformerModel5x512` from scratch | 3.41M | 5 x 512, stride 5 | yes | no | dense + foot 3-term | amass_00492 (ref 1.61 steps/s) | 2000 | 4096 |
| **HIDYN-00526** | idem | 3.41M | 5 x 512, stride 5 | yes | no | dense + foot 3-term | amass_00526 (ref 2.01 steps/s) | 2000 | 4096 |
| **HIDYN-00499** | idem | 3.41M | 5 x 512, stride 5 | yes | no | dense + foot 3-term | amass_00499 (ref 3.01 steps/s) | 2000 | 4096 |

Reward weights shared by every arm: `motion_global_root_pos` 1.0, `motion_global_root_ori` 0.5,
`motion_body_pos` 1.0, `motion_body_ori` 1.0, `motion_body_lin_vel` 1.0, `motion_body_ang_vel` 1.0,
`action_rate_l2` -0.1, `joint_limit` -10.0, `self_collisions` -10.0.
Terminations shared by every arm: `anchor_pos` 0.30 m, `anchor_ori` 0.80 rad, `ee_body_pos` 0.25 m.

### 1b. Cost and result

| arm | s/iter | GPU-h (this run) | GPU-h (cumulative) | exam criterion | mastered / 224 | **rescored OLD** / 224 | gait mastered / 155 | gait mastered OLD | mean success | mean travel | mean steps/s | mean skate | gait steps/s (ref 1.45) | gait skate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| X600   | 11.77 | 1.96 | 1.96 | old | 60 | **60** | – | 30 | 0.321 | 0.794 | – | – | – | – |
| T600   | **2.81** | **0.47** | **0.47** | old | 68 | **68** | – | 34 | 0.398 | 0.822 | – | – | – | – |
| XW600  | 11.79 | 1.96 | 1.96 | old | 55 | **55** | – | 28 | 0.287 | 0.821 | – | – | – | – |
| XF600  | 17.82 | 2.97 | 2.97 | old | 63 | **63** | – | 31 | 0.334 | 0.778 | – | – | – | – |
| XF2000 | 17.79 | 6.92 | 9.89 | new | 34 | **94** | **0** | 60 | 0.545 | 0.863 | 0.631 | 0.186 | 0.725 | 0.223 |
| T2000  | 2.76 | 1.07 | 1.54 | new | 39 | **101** | **0** | 62 | 0.548 | 0.887 | 0.638 | 0.208 | 0.754 | 0.257 |
| TFS4   | 4.22 | 1.17 | 1.17 | new | 49 | **62** | **19** | 32 | 0.345 | 0.880 | 0.974 | 0.107 | 1.225 | 0.133 |
| XFS4b  | 44.22 | 2.46 (4.43 burned) | 2.46 | new | 5 | **17** | **0** | 12 | 0.097 | 0.974 | 0.841 | 0.102 | 0.869 | 0.102 |
| WP2000 | 6.51 | 3.62 | 4.79 | new | 61 | **81** | **20** | 40 | 0.472 | 0.845 | 0.903 | 0.091 | 1.180 | 0.116 |
| hz4_r1 | 3.09 | 1.72 | 2.89 | new | **79** | **100** | **35** | 56 | 0.583 | 0.877 | 0.940 | 0.093 | 1.239 | 0.117 |
| *(hz4_r0 = the shared seed)* | – | – | 1.17 | new | 50 | 63 | 19 | 32 | 0.348 | 0.874 | 0.977 | 0.106 | 1.221 | 0.132 |

Single-window arms (1 window, own exam file):

| arm | s/iter | GPU-h | mastered | success | travel | steps/s | ref steps/s | skate |
|---|---|---|---|---|---|---|---|---|
| BM     | 0.43 | 0.09 | *not exam'd* | 0.500 @400 it (`bm_400.json`) | **1.128** | – | 0.60 | – |
| BMFS   | 0.43 | 0.08 | *not exam'd* | 0.375 @400 it | 1.016 | – | 0.60 | – |
| BMFS2  | 0.43 | 0.24 | *not exam'd* | 1.000 @1800 it | 0.933 | – | 0.60 | – |
| BMFS3  | 0.44 | 0.31 | **yes** | 1.000 | 0.979 | **0.82** | 0.60 | **0.049** |
| DANCE  | 3.57 | 0.49 | no | 0.000 | 1.000 | 0.00 | 2.61 | 0.000 |
| DANCE2 | 3.46 | 1.04 | no | 0.000 | 1.000 | 6.22 | 2.61 | 0.507 |
| DANCE3 | 5.69 | 1.62 | no | 0.000 | 1.000 | 5.57 | 2.61 | 0.353 |
| HIDYN-00492 | 3.12 | 1.73 | **yes** | 1.000 | 0.970 | 1.71 | 1.61 | 0.072 |
| HIDYN-00526 | 3.20 | 1.78 | no | 0.156 | 1.000 | 1.63 | 2.01 | 0.134 |
| HIDYN-00499 | 3.11 | 1.73 | no (**skater**) | 1.000 | 1.000 | 0.82 | 3.01 | 0.125 |

---

## 2. Answer 1 — cost-normalised architecture ranking

The project metric is **mastered windows per GPU-hour**. Because the criterion
changed mid-series, the ranking is given twice.

### Under the OLD criterion (the only column every arm shares)

| rank | arm | mastered_old | GPU-h (cumulative) | **windows / GPU-h** |
|---|---|---|---|---|
| 1 | **T600** (transformer, 0.08 s window) | 68 | 0.47 | **144.7** |
| 2 | T2000 | 101 | 1.54 | 65.6 |
| 3 | TFS4 (transformer, 0.5 s window + foot terms) | 62 | 1.17 | 53.0 |
| 4 | hz4_r1 | 100 | 2.89 | 34.6 |
| 5 | X600 (xattn, Equator KV) | 60 | 1.96 | 30.6 |
| 6 | XW600 (xattn + warm start) | 55 | 1.96 | 28.1 |
| 7 | XF600 (xattn, raw-code KV) | 63 | 2.97 | 21.2 |
| 8 | WP2000 (sparse) | 81 | 4.79 | 16.9 |
| 9 | XF2000 | 94 | 9.89 | 9.5 |
| 10 | XFS4b (xattn, 5 code frames) | 17 | 2.46 | 6.9 |

### Under the NEW (foot-gated) criterion — arms that measured feet

| rank | arm | mastered | GPU-h (cumulative) | **windows / GPU-h** |
|---|---|---|---|---|
| 1 | **TFS4** | 49 | 1.17 | **41.9** |
| 2 | hz4_r1 | 79 | 2.89 | 27.3 |
| 3 | T2000 | 39 | 1.54 | 25.3 |
| 4 | WP2000 | 61 | 4.79 | 12.7 |
| 5 | XF2000 | 34 | 9.89 | 3.4 |
| 6 | XFS4b | 5 | 2.46 | 2.0 |

**The verdict is the same under both criteria: the pooled-feature transformer
actor dominates the raw-code cross-attention actor, and the gap is almost
entirely throughput, not sample efficiency.**

Head-to-head at matched iterations, matched pool, matched reward:

* **T600 vs X600, 600 iters**: 68 vs 60 mastered (transformer +13%) at **2.81 vs 11.77 s/iter**
  (4.2x). Cost per mastered window: 0.007 vs 0.033 GPU-h.
* **T2000 vs XF2000, 2000 iters**: 101 vs 94 old-criterion, 39 vs 34 new-criterion,
  at **2.76 vs 17.79 s/iter** (6.4x). XF2000 burned **9.89 GPU-h to reach a strictly
  worse result than T2000's 1.54** — a 6.4x cost for -7 windows.
* **TFS4 vs XFS4b**, same reward, same preview, same pool: **4.22 vs 44.22 s/iter (10.5x)**.
  XFS4b was OOM-killed at iteration 361 and only reached 5 mastered.

The cost driver is the KV set size: the xattn actor's `Learning time` is
16.3 s of XF600's 17.8 s and 40.4 s of XFS4b's 44.2 s. KV tokens per env: 32
(X600) -> 512 (XF600, 2 frames x 256 latents) -> 1280 (XFS4b, 5 frames x 256).
Its 31 joint queries each cross-attend that whole set every forward pass, in
every one of 5 PPO epochs x 4 minibatches.

**Warm-starting the xattn core hurt**: XW600 55 vs X600 60 mastered, 0.287 vs
0.321 mean success, identical cost. The pretrained core is not a free head start.

*Caveat*: the two rankings are not one ranking. The old-criterion column credits
skating (see Answer 5); the new-criterion column is the honest one but excludes
the four 600-iteration arms entirely.

---

## 3. Answer 2 — the foot-term dose series (BM -> BMFS -> BMFS2 -> BMFS3)

One window (`amass_00115`, 249 f, reference 0.60 steps/s, reference feet planted
82.5% of frames), stock BeyondMimic MLP, reference fully visible to the actor.
Only the foot terms and the friction floor change.

| dose | `foot_contact_match` | `foot_slide` | `swing_refusal` | contact preview | friction | iters run |
|---|---|---|---|---|---|---|
| BM    | – | – | – | no | 0.3–1.2 | 794 (killed) |
| BMFS  | 0.5 | -0.3 | – | no | 0.3–1.2 | 682 (killed) |
| BMFS2 | 1.5 | -1.5 | – | no | **0.7**–1.2 | 2000 |
| BMFS3 | 1.5 | **-3.0** | **-1.5** | **yes** | 0.7–1.2 | 2500 |

### 3a. How the per-checkpoint foot numbers were recovered

`scripts/foot_audit.py` prints steps/s, planted fraction and skate, but its stdout
was never persisted anywhere on this machine (`grep -r "steps/s" /home/arenalabs/tmp`
returns nothing), and re-running it needs the GPU. The per-checkpoint
`bm*_NNN.json` files are `evaluate_local.py` output and carry **no** foot fields.

So the series below is **derived from the reward terms in the training logs**,
which are the same physical quantities the audit measures:

```
Episode_Reward/<term> = weight * dt * sum_over_episode(value) / max_episode_length_s
=> mean per-step raw value = Episode_Reward * max_episode_length_s / (|weight| * dt * mean_ep_len)
   with dt = 0.02 s (decimation 4 x timestep 0.005), max_episode_length_s = 10.0
```
Both `Episode_Reward` and `Mean episode length` are rolling medians over +/-15 iterations.
The conversion was validated on bounded terms: it returns
`motion_global_root_pos` 0.76–0.79 and `motion_body_pos` 0.85–0.88 — inside [0,1], as required.

Derived quantities, with `p_ref = 0.8253`, `swing_weight = 4`, `D = p_ref + 4(1-p_ref) = 1.5241`:

* `swing_refusal_feet` **S** = mean feet planted while the reference swings (direct, BMFS3 only).
* `b` = P(planted | reference swinging) = `S / (2(1-p_ref))`.
* `a` = P(planted | reference planted) = `(M*D - 4(1-p_ref)(1-b)) / p_ref`, `M` = raw contact-match.
* `planted fraction` = `p_ref*a + (1-p_ref)*b`.
* `swing compliance` = `(M*D - p_ref) / (4(1-p_ref))` — 0 for an always-planted skater, 1 for a perfect match.
  (BMFS/BMFS2 have no `swing_refusal` term, so their planted fraction is reported under the
  assumption `a = 1`, i.e. the robot always plants when the reference does. BMFS3 shows
  `a ≈ 0.72`, so BMFS/BMFS2's planted fraction is an **upper** bound.)
* `skate per planted foot` = `slide_2ft / (2 * planted fraction)`.

**These are training-time values** — with exploration noise, pushes every 1–3 s and
velocity randomisation — so they are 3–4x the deterministic exam skate and are
comparable **across the four arms**, not against the exam. The one exam
ground truth (BMFS3 final) is quoted separately.

### 3b. The collapse-vs-survival curve

| arm | iter | survival (s) | contact match | swing compliance | planted frac | slide (m/s, 2 ft) | skate / planted foot | swing refusal (feet) |
|---|---|---|---|---|---|---|---|---|
| **BMFS** | 50 | 0.63 | 0.846 | 0.665 | <=0.884 | 0.421 | 0.238 | – |
| | 100 | 0.77 | 0.823 | 0.613 | <=0.893 | 0.375 | 0.210 | – |
| | 200 | 1.30 | 0.796 | 0.556 | <=0.903 | 0.484 | 0.268 | – |
| | 300 | 2.89 | 0.774 | **0.508** | <=0.911 | 0.583 | 0.320 | – |
| | 400 | 4.06 | 0.785 | 0.530 | <=0.907 | 0.597 | **0.329** | – |
| | 670 | 5.51 | 0.792 | 0.547 | <=0.904 | 0.591 | 0.327 | – |
| **BMFS2** | 50 | 0.65 | 0.823 | 0.614 | <=0.893 | 0.329 | 0.184 | – |
| | 200 | 1.17 | 0.775 | 0.508 | <=0.911 | 0.306 | 0.168 | – |
| | 600 | 3.99 | 0.808 | 0.581 | <=0.899 | 0.336 | 0.187 | – |
| | 1000 | 5.43 | 0.857 | 0.688 | <=0.880 | 0.340 | 0.193 | – |
| | 1400 | 6.17 | 0.881 | **0.741** | <=0.871 | 0.328 | 0.188 | – |
| | 1985 | 6.71 | 0.856 | 0.686 | <=0.880 | **0.294** | 0.167 | – |
| **BMFS3** | 50 | 0.62 | 0.853 | 0.680 | 0.769 | 0.304 | 0.197 | 0.067 |
| | 100 | 0.70 | 0.855 | 0.685 | 0.705 | 0.247 | 0.175 | 0.040 |
| | 250 | 1.09 | 0.799 | 0.562 | 0.574 | 0.216 | 0.188 | 0.022 |
| | 500 | 2.60 | 0.808 | 0.582 | 0.577 | 0.239 | 0.207 | 0.018 |
| | 1000 | 4.61 | 0.846 | 0.665 | 0.629 | 0.239 | 0.190 | 0.015 |
| | 1500 | 5.80 | 0.835 | 0.641 | 0.606 | 0.218 | 0.180 | 0.013 |
| | 2000 | 6.27 | 0.843 | 0.657 | 0.615 | 0.215 | 0.175 | 0.012 |
| | 2485 | 6.62 | 0.840 | 0.651 | 0.608 | 0.209 | **0.172** | **0.011** |
| **BM** | 400 | 4.31 | *no foot terms logged* | – | – | – | – | – |
| | 780 | 6.01 | – | – | – | – | – | – |

### 3c. The separation, at matched budget (iteration 680, the point where BM/BMFS were killed)

| arm | survival (s) | mean reward | contact match | swing compliance | planted frac | slide (2 ft) | skate / foot | swing refusal |
|---|---|---|---|---|---|---|---|---|
| BM    | **5.80** | 13.32 | – | – | – | – | – | – |
| BMFS  | 5.72 | 13.87 | 0.764 | **0.485** | <=0.915 | **0.567** | **0.310** | – |
| BMFS2 | 4.27 | 11.58 | 0.847 | 0.667 | <=0.883 | 0.344 | 0.195 | – |
| BMFS3 | **3.60** | 9.74 | 0.802 | 0.569 | **0.563** | **0.239** | 0.212 | 0.016 |

**Reading the curve.**

1. **Dose 0 and dose 1 collapse *while* survival rises.** BMFS's swing compliance
   *falls* from 0.665 (iter 50) to 0.508 (iter 300) and its slide *rises* from
   0.42 to 0.60 m/s over exactly the window where survival climbs from 0.6 s to
   2.9 s. The policy is buying survival with sliding, and a 0.5 / -0.3 foot dose
   is too weak to stop it. Its planted fraction never drops below 0.90 against a
   reference that lifts 17.5% of the time. BM (no foot terms) cannot be measured
   from its log, but its `travel_ratio` in `bm_200/400.json` is **1.148 / 1.128**
   — the root out-travels the reference by 13–15% — against BMFS's 1.03 / 1.02
   and BMFS2/BMFS3's 0.93–0.99. Over-travel with a planted foot is the sliding signature.
2. **Dose 2 stops the erosion but does not fix the feet.** BMFS2 dips to 0.508
   compliance at iter 200, then recovers monotonically to 0.741 at 1400 and holds
   ~0.69–0.71. Slide is roughly halved versus BMFS (0.29–0.35 vs 0.57–0.60).
   Planted fraction stays ~0.88 — the robot is stepping in place, not swinging properly.
3. **Dose 3 breaks the planted fraction.** `swing_refusal` drops **0.202 -> 0.067
   (iter 50) -> 0.022 (iter 250) -> 0.011 (iter 2485)** — a 95% reduction, and the
   only arm where the planted fraction actually falls (0.926 -> 0.61), i.e. `a ≈ 0.72`:
   the robot lifts its feet **more** than the reference. This is corroborated
   independently by the exam: `bmfs3_final_exam.json` reports **steps/s 0.82 vs
   reference 0.60**, exactly the over-stepping the derived `a < 1` predicts, with
   **skate 0.049** (gate 0.15) and `mastered: true`.
4. **The price is a training-time delay, not a ceiling.** At iteration 680 the dose
   ladder costs survival monotonically: 5.80 / 5.72 / 4.27 / 3.60 s. By the end of
   their full budgets BMFS2 reaches 6.71 s and BMFS3 6.62 s — the same plateau BM
   was heading for, plus honest feet. **BMFS3 is the only arm in the whole
   corpus that passes the honest gate on this window, and it cost 0.31 GPU-h.**
5. `evaluate_local` success ladder: BM 0.125@200 / 0.50@400; BMFS 0.125@200 /
   0.375@400; BMFS2 0.00@200 / 0.875@400 / 1.00@1800; BMFS3 0.00@250 / 0.50@500 /
   1.00@1750 / 1.00@2499. The heavier doses start slower and finish higher.

**Not comparable because…** BM (794 iters) and BMFS (682) were killed early to
free the GPU, so their *end states* cannot be compared with BMFS2 (2000) and
BMFS3 (2500); only the matched-budget row at iteration 680 is a fair comparison.
BMFS also ran on friction 0.3–1.2 (ice floors allowed) while BMFS2/BMFS3 ran on
0.7–1.2, so part of BMFS's slide advantage-loss is the floor, not the reward.
BMFS3 additionally adds 30 contact-preview inputs to the actor (272k vs 256k
params), so dose 3 is *reward + observation*, not reward alone.

---

## 4. Answer 3 — dense vs sparse reward

This is the cleanest pair in the corpus.

* Same actor (`TokenTransformerModel5x512`, 3.41M).
* Same start: both `--agent.resume` from a checkpoint verified **md5-identical**
  (`e774686327aaadccea67e2f84fc47483`) — `wp_arm/seed/model_seed.pt` ==
  `x2_horizon_hz4/seed/model_seed.pt` == `fs4_duel/.../TFS4/model_999.pt`.
* Same windows: `pool_r0` and `pool_r1` carry the same 224 ids (verified).
* Same budget: 2000 iterations, 4096 envs each.
* Same foot terms, same terminations, same contact preview.

| | seed (hz4_r0) | **DENSE** (hz4_r1) | **SPARSE** (WP2000) |
|---|---|---|---|
| body-tracking reward | per-frame pos 1.0 / ori 1.0 / lin-vel 1.0 / ang-vel 1.0 | same | waypoint pos **12.0** / ori **6.0** every 25 f (0.5 s); lin/ang-vel terms **deleted** |
| mastered / 224 (new) | 50 | **79** | 61 |
| rescored old / 224 | 63 | **100** | 81 |
| gait mastered / 155 | 19 | **35** | 20 |
| mean success | 0.348 | **0.583** | 0.472 |
| mean travel ratio | 0.874 | 0.877 | 0.845 |
| mean steps/s | 0.977 | 0.940 | 0.903 |
| mean skate | 0.106 | 0.093 | **0.091** |
| gait steps/s (ref 1.45) | 1.221 | **1.239** | 1.180 |
| gait skate | 0.132 | 0.117 | **0.116** |
| s / iteration | – | **3.09** | 6.51 |
| GPU-h for the 2000 iters | – | **1.72** | 3.62 |
| **windows gained over the seed** | – | **+29** | +11 |
| **gained per GPU-h** | – | **16.9** | 3.0 |

**Dense wins decisively: 2.6x the mastery gain at 47% of the cost — 5.6x on the
cost-normalised metric.** The hypothesis that sparse scoring buys physical
solidity is **not supported**: sparse's skate advantage is 0.091 vs 0.093
(0.116 vs 0.117 on gait windows), i.e. inside exam noise, while it loses 15
mastered windows and 15 gait windows. Sparse also *lost* cadence (0.903 vs 0.940
steps/s) rather than gaining it.

Sparse is also **2.1x slower per iteration** (6.51 vs 3.09 s) — the waypoint terms
compute `quat_error_magnitude` over all tracked bodies every step and then
multiply by a 0/1 mask, so 24/25 of that work is thrown away.

**Not comparable because…** two confounds favour the dense arm and must be stated:
1. **Sampler.** hz4_r1 uses `HorizonTokenMotionCommand`: unmastered windows are
   drawn **5x** more often and unmastered round-0 seeds **25x** (the log prints
   `[Horizon] 224 windows, mastered 50, frontier weight 5.0, unmastered seeds
   boosted x25.0: 8`). WP2000 uses plain uniform `TokenMotionCommand`. Part of
   dense's +29 is frontier weighting, not reward density.
2. **Not a pure dense->sparse swap.** WP also *deletes* `motion_body_lin_vel` and
   `motion_body_ang_vel` entirely and re-weights the surviving pos/ori terms 12x/6x.
   The correct reading is "dense 6-term vs sparse-2-term-plus-deletions", not
   "same reward, sparser schedule".

A clean rerun would be: WP with frontier weighting on, or hz4_r1 rerun with
uniform sampling.

---

## 5. Answer 4 — single-window expert vs generalist

Same three windows, same exam script, same criterion.

| window | ref steps/s | | single-window expert (2000 it, 4096 env, 32 reps) | hz4 r0 (seed) | hz4 r1 | hz4 r9 generalist (11 reps) |
|---|---|---|---|---|---|---|
| **amass_00492** | 1.61 | success | **1.000** | 0.000 | 0.000 | 0.182 |
| | | travel | 0.970 | 0.241 | 0.497 | 1.010 |
| | | steps/s | **1.71** | 0.70 | 0.86 | 1.14 |
| | | skate | **0.072** | 0.065 | 0.059 | 0.086 |
| | | **mastered** | **YES** | no | no | **no** |
| **amass_00526** | 2.01 | success | 0.156 | 0.000 | 0.000 | 0.000 |
| | | travel | 1.000 | 1.000 | 1.000 | 1.000 |
| | | steps/s | 1.63 | 2.94 | 2.93 | 2.58 |
| | | skate | 0.134 | 0.158 | 0.199 | 0.180 |
| | | **mastered** | no | no | no | no |
| **amass_00499** | 3.01 | success | **1.000** | 0.000 | 0.000 | 0.000 |
| | | travel | 1.000 | 1.000 | 1.000 | 1.000 |
| | | steps/s | **0.82** (needs >= 1.505) | 7.59 | 4.13 | 1.19 |
| | | skate | 0.125 | 0.208 | 0.194 | 0.150 |
| | | **mastered** | **no — skater** | no | no | no |

**Score: expert 1/3, generalist 0/3.** Cost: 1.73–1.78 GPU-h per expert
(5.24 GPU-h for all three) versus **18.47 GPU-h** cumulative for the r9
generalist (seed 1.17 + rounds r1–r9, 2000 iters each at 3.07–6.49 s/iter).

Three findings:

1. **On the one window either side can do (00492, 1.61 steps/s), the expert
   masters it outright while the generalist is at 0.182 success after 10x the
   compute.** The generalist is closer than r0/r1 (steps/s 0.70 -> 0.86 -> 1.14,
   travel 0.24 -> 0.50 -> 1.01), so the horizon curriculum is moving toward it, but
   after nine rounds it has not arrived.
2. **The foot clause caught a false master.** `amass_00499`: the expert has
   **success 1.00 and travel 1.00** — a perfect pass under the old criterion — but
   **0.82 steps/s against a 3.01 steps/s reference**. It survives the entire
   window by shuffling. Without the foot clause this would have been logged as a
   mastered high-dynamics window. It is a skater.
3. **The generalist is the only arm whose cadence *converges downward* on 00499**
   (7.59 -> 4.13 -> 1.19 steps/s across r0/r1/r9) while its skate falls
   0.208 -> 0.194 -> 0.150 — it is learning to stop flailing before it learns to step.

**Not comparable because…**
* **Repeat count differs**: 32 reps for the experts, 16 for r0/r1, **11** for r9
  (`4096 // 363 windows`). The r9 success estimates are the noisiest of the three.
* **The mastery threshold moved for 00526**: `pool_r0/r1` record
  `ref_steps_ps = 1.81`, `pool_r9` and `motions/hidyn/amass_00526.npz` record
  **2.01**. The clip is the same length (249 f) and the same window, but the
  cadence was recomputed between rounds, so the `0.5 * ref` bar for that window
  moved from 0.905 to 1.005 between r1 and r9.
* **The generalist is asked to hold 363 windows** while each expert holds one; the
  18.47 vs 1.73 GPU-h comparison is not a per-window budget comparison.

The DANCE arms bound the same question from the other side: three attempts at a
2.61 steps/s LAFAN dance window as a single-window expert produced **0/1 mastered
every time** (success 0.000 at 494, 1086 and 1026 iterations; DANCE2/DANCE3 reach
5.6–6.2 steps/s with skate 0.35–0.51, i.e. flailing, not dancing), for 3.15 GPU-h.
Single-window brute force works at 1.61 steps/s and fails at 2.01, 2.61 and 3.01.

---

## 6. Answer 5 — iteration scaling within one arm

### The incomparability, stated first

`ab_mastery_{XF,T}.json` were written **before** the foot clauses existed: they
contain only `success`, `travel_ratio`, `mastered`, `n`. `ab_mastery_{XF2000,T2000}.json`
were written **after**, and their `mastered` flag additionally requires
`steps_ps >= 0.5*ref_steps_ps` and `skate <= 0.15` on the 155 gait windows.
**The raw `mastered` counts (63 -> 34, 68 -> 39) therefore cannot be read as a
regression.** The 2000-iteration files have been rescored under the old
criterion below; the 600-iteration files cannot be rescored under the new one,
because `steps_ps` and `skate` were never measured for them.

### XF600 -> XF2000 (cross-attention, raw codes, 3.33x compute)

| | XF600 | XF2000 | delta |
|---|---|---|---|
| iterations | 600 | 2000 | 3.33x |
| GPU-h (cumulative) | 2.97 | 9.89 | 3.33x |
| mastered, **old criterion** | 63 | **94** | **+31 (+49%)** |
| gait mastered, **old criterion** | 31 | **60** | +29 (+94%) |
| mean success | 0.334 | 0.545 | +63% |
| mean travel ratio | 0.778 | 0.863 | +11% |
| mastered, **new criterion** | *unmeasurable* | **34** | – |
| gait mastered, **new criterion** | *unmeasurable* | **0** | – |
| gait steps/s (ref 1.45) | – | 0.725 | – |
| gait skate (gate 0.15) | – | **0.223** | – |

### T600 -> T2000 (transformer, 3.33x compute)

| | T600 | T2000 | delta |
|---|---|---|---|
| iterations | 600 | 2000 | 3.33x |
| GPU-h (cumulative) | 0.47 | 1.54 | 3.28x |
| mastered, **old criterion** | 68 | **101** | **+33 (+49%)** |
| gait mastered, **old criterion** | 34 | **62** | +28 (+82%) |
| mean success | 0.398 | 0.548 | +38% |
| mean travel ratio | 0.822 | 0.887 | +8% |
| mastered, **new criterion** | *unmeasurable* | **39** | – |
| gait mastered, **new criterion** | *unmeasurable* | **0** | – |
| gait steps/s (ref 1.45) | – | 0.754 | – |
| gait skate (gate 0.15) | – | **0.257** | – |

### What 3.3x the compute bought

**Under the old criterion it bought +49% mastered windows, in both arms, almost
identically (63->94 and 68->101). Under the honest criterion it bought nothing on
gait: both 2000-iteration policies master ZERO of the 155 gait windows.**

Every one of XF2000's 34 and T2000's 39 mastered windows is a **non-gait** window
(`ref_steps_ps < 0.5`; 69 such windows exist). On the 155 gait windows the two
2000-iteration policies average **0.725 / 0.754 steps/s against a 1.45 steps/s
reference — exactly on the 0.5x bar — and skate 0.223 / 0.257, well over the 0.15
gate.** The extra 1400 iterations of dense body tracking with no foot supervision
bought more confident sliding: 60 and 62 gait windows now clear
`success >= 0.7 AND travel >= 0.8` while none of them is actually walking.

The control that proves the point is **TFS4**: same pool, same actor family,
**1000 iterations and 1.17 GPU-h**, but with the foot 3-term and the 0.7 friction
floor. It masters **19 gait windows honestly** with gait steps/s **1.225** (85% of
the reference) and gait skate **0.133**. Compare XF2000's 9.89 GPU-h for 0 gait
windows. **On gait windows the foot terms are worth more than 8x the compute.**

### Further "not comparable because…"

* **XF2000 and T2000 are resumptions, not independent runs.** `resume: true`,
  `load_run: XF600 / T600b`, `load_checkpoint: model_599.pt`, `max_iterations: 1400`.
  The optimizer state, the observation normaliser and the action std carry over.
  Their cumulative cost includes the parent (2.97 and 0.47 GPU-h respectively).
* **X600 has a discarded twin.** `ab_xattn/2026-08-26_22-08-08_X600` exists with
  only `model_0.pt`; the scored run is `..._22-40-01_X600b`.
* **XFS4b is the weakest row in the table and should not be read as an
  architecture verdict.** It was OOM-killed at iteration 361 of 1000 (the shell
  reports `Killed`), and the exam ran on `model_200.pt`, so its 5 mastered
  windows reflect **200** iterations of learning against TFS4's 1000. Its
  `--agent.algorithm.num-mini-batches 16` also differs from every other arm's 4.
  What it does establish is the **cost**: 44.22 s/iteration, 10.5x TFS4.
* **fs_duel `XFS1000b`** (foot terms at the weaker 0.5 / -0.3 dose, friction
  0.3–1.2) died at iteration 197 with no exam at all and is excluded from every
  ranking; its config is in `logs/rsl_rl/fs_duel/2026-08-27_15-17-14_XFS1000b/params/`.
* **BM/BMFS/BMFS2 have no honest exam.** Only BMFS3 was run through
  `horizon_exam.py`. The dose ladder in Answer 2 rests on log-derived quantities
  plus `evaluate_local.py` success rates, not on the exam criterion.
* **The 2048-env single-window arms (BM/BMFS/BMFS2/BMFS3/DANCE*) and the 4096-env
  pool arms differ in batch size** (2048x24 vs 4096x24 steps per iteration), so
  their s/iteration and GPU-h are not on the same scale as the pool arms'.
* **hz4_r9 is a curriculum arm** and appears here only as the generalist
  comparator for Answer 4; its 363-window pool is a superset of the 224 used by
  every other arm, so its `mastered/363` must never be compared to a `/224` count.

---

## 7. Files

| what | where |
|---|---|
| raw table (one row per arm, 31 columns) | `/home/arenalabs/tmp/ablation_table.csv` |
| exam JSONs | `/home/arenalabs/tmp/{ab_mastery_*,fs4_mastery_*,wp_mastery,bmfs3_final_exam,dance*_exam,hidyn_amass_*_scratch}.json` |
| curriculum comparators | `/home/arenalabs/alphamotion_artifacts/horizon_x2_hz4/mastery_r{0,1,9}.json` |
| training logs | `/home/arenalabs/tmp/*.log`, `/home/arenalabs/alphamotion_ctrl/logs/train_horizon_hz4_r*.log` |
| per-run frozen config | `/home/arenalabs/alphamotion_ctrl/logs/rsl_rl/<exp>/<run>/params/{env,agent}.yaml` |
| task definitions | `/home/arenalabs/alphamotion_ctrl/amctrl/agibot_x2/__init__.py` |
| actor architectures | `/home/arenalabs/alphamotion_ctrl/amctrl/rl/{token_transformer,token_xattn}.py` |
| reward terms | `/home/arenalabs/alphamotion_ctrl/amctrl/token/{foot_rewards,waypoint_rewards}.py` |
| sampler | `/home/arenalabs/alphamotion_ctrl/amctrl/token/curriculum.py` |
| exam criterion | `/home/arenalabs/alphamotion_ctrl/scripts/horizon_exam.py` |
