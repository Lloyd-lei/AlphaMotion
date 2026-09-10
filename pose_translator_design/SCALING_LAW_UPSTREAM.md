# 上游两层的 scaling law — Greenwich（空间 codec）与 Equator（时序栈）
### Scaling laws of the two upstream models, mined from the 119 training runs on this box

**2026-08-29 · read-only mining, zero GPU touched.**
Sources: 119 `config.json` under `/var/tmp/alphamotion/topo_exp/runs/`, their `eval_log.json`,
the verdict dumps in `pose_translator_design/verdicts/`, and the written analyses
(`MODEL_VERSIONS.md`, `R7_LATENT/ANALYSIS.md`, `R8_DIM/ANALYSIS_DIM.md`,
`APPENDIX_ARCH_ABLATION.md`, `ENCODER_BAKEOFF.md`, `LIT_TEMPORAL_FSQ.md`, `R10_RATE_PROBES.md`).

Raw machine-readable tables (every column, including the ones trimmed out of the markdown below):
* `/home/arenalabs/tmp/scaling_greenwich.csv` — 47 rows × 61 columns
* `/home/arenalabs/tmp/scaling_equator.csv` — 72 rows × 106 columns

---

## 0. 一句话结论 / One-paragraph answer

两层都**已经饱和在比特轴和参数轴上**，唯一还在给回报的是**具身数量**——而且只在零样本那一栏。
Greenwich 的误差在 ~2500 bits/frame 以上就撞上 **8.03° 的地板**（拟合 R²=0.9997），
再多比特买不动；参数从 3.67M→18M 有效（−4.17°），18M→57.8M 完全无效（+0.36°，反而变差）。
Equator 的 S 扫描把比特开到 +60% 只买回 0.19 cm，而**零样本误差反而随比特单调上升**
（r=+0.999，每 100 bits/window 付 0.40 cm），机制是身体泄漏随比特上升（kNN leakage 0.085→0.131）。
CTRLa/CTRLb 的 2×2 判决量化后是：**架构主效应 −2.04 cm（语料内），数据主效应 −1.05 cm（零样本）**，
两者几乎不相交。这句话就是给更大联合模型的处方：**加具身、别加比特、别加参数。**

Both stacks are already saturated on the rate axis and the parameter axis. The only axis still
paying is **number of embodiments**, and it pays only in the zero-shot column.

---

## 1. 口径与公式 / Definitions, formulas, and which ruler each number came from

### 1.1 Greenwich bits per frame — the formula

From `exp_topo/model.py` (`FSQ.bits`, `ResidualFSQ.bits`) and `exp_topo/model_dual.py`:

```
bits_per_token  = fsq_stages × Σ_i log2(levels_i)
slots           = n_latent            if rot_only   (position stream aliased away)
                = 2 × n_latent        otherwise     (separate pos + rot query banks)
bits_per_frame  = slots × bits_per_token
```

`single_codebook=True` also gives `2 × n_latent` slots through one shared bank — same total.
Cross-check: `PROD_c1_91k_dim20_tail_0810` = 128 latents × 2 streams × 1 stage × 20 dims × log2(8)
= 256 × 60 = **15 360 bits/frame**; `E19_K128_0805` = 256 × 39 = **9 984 bits/frame**, which is
exactly the `bits_per_frame_nominal = 9984` recorded in `LIT_TEMPORAL_FSQ.md §2`. Formula confirmed.

**Caveat on `rot_only`:** `model_dual.py` sets `self.bits = (fsqp.bits + fsqr.bits) * n_latent`
even when `rot_only=True`, i.e. the model's own attribute double-counts. `fk_duel.py` uses the
correct `slots = n_latent * (1 if rot_only else 2)`. Table A follows `fk_duel.py`. FK1/FK2 are the
only `rot_only` runs.

### 1.2 Equator bits per second

Every V7-era config carries a pre-registered `bit_budget` block, asserted before step 0:

```
bits_per_register    = Σ log2(content levels)                 (13.932 for 5^6 = 15 625)
path_bits_per_window = n_path × bits_per_register
time_bits_per_window = n_time × Σ log2(time levels)           (8.42 for 7^3 = 343)
total_bits_per_s     = (path + time) / window_seconds          (window = 2.0 s = 60 frames @30 fps)
```

This reproduces `MODEL_VERSIONS.md` exactly: V8 line 8×13.932 + 1×13.932 = 125.38 / 2 s = **62.69 bit/s**;
S1 (speed token 6×5 → 3×7) = 111.45 + 8.42 = 119.87 / 2 = **59.94**; S2 (content 6→8 dims @5) =
148.60 + 8.42 = 157.02 / 2 = **78.51**; S3 (content 8 dims @8 levels) = 192.00 + 8.42 = 200.42 / 2 = **100.21**.
T32 (32 path tokens) = 459.74 / 2 = **229.87 bit/s**.

### 1.3 Parameter counts

Temporal configs record `n_params` directly. For spatial runs (no such field) the count is read
from the `.pt` zip central directory: Σ(uncompressed bytes of `model/data/*`) / 4. Validated against
`S0_arclen_4b_0807`: zip gives 8 333 766 vs config `n_params` 8 333 634 — a +132 difference, i.e.
the zip figure includes non-trainable buffers (`_levels`, hop tables). Treat spatial parameter
counts as **±0.01 % (buffers included)**, which is far below any effect measured here.

### 1.4 The rulers, and what each one is NOT

| ruler | where | what it is | what it is not |
|---|---|---|---|
| `eval_log.json` (spatial) | `dual_common.evaluate()` | geodesic ° and FK cm on a **random FRAME-level** val split (`split_indices` permutes rows, not clips) | **not** a held-out-motion number — adjacent frames of the same clip sit on both sides of the split |
| `eval_log.json` (temporal) | `train_v7.py` | `key_pos_cm` / `key_rot_deg` on 6 key joints, plus `zeroshot_*` rows | same frame-split caveat |
| `v7_eval.py` (`*_verdict_*`) | V7/V8/CTRL/ALPHA/SIZE verdicts | the canonical duel ruler: per-family energy, warp invariance, content share, zero-shot G1, transitions | window-sampled, so its absolute cm differ slightly from `eval_log` |
| `E21_data_axis_heldout.json` | `eval_data_axis.py` | **clip-level held-out** eval, 600 disjoint clips, arm/leg/wrist split, paired bootstrap CI | the only leakage-free spatial data-axis number we have |
| FK ruler | `FK_DUEL_0813` | FK on predicted rotations vs FK on GT rotations, per body part | the only meaningful ruler for `rot_only` models |
| `BRIDGE_STACK_v2` | `bridge_stack_eval2.py` | decode → held-out G1, scored against the real paired retarget FK | v1 (`BRIDGE_STACK_verdict_0810`) is the **dirty** self-decode ruler; v1 and v2 numbers are not comparable |

**Self-reconstruction is degenerate wherever `vis_residual=True`.** Every run from `E28_resid_0807`
onward (all PROD/PT/FK/SMALL rows) decodes visible joints as a residual on the observed pose, so
`X->X` reads a trivial 0.1° / 0.00 cm. Use the `ST_*` swing/twist column instead. Table A carries
both and flags this per row in the CSV `notes` field.

---

## 2. Table A — Greenwich (cross-embodiment spatial codec)

47 runs that have both a config and at least one eval number. Sorted by name.
Columns trimmed for reading; the CSV has 61 columns including round-trip fidelity, FSQ usage,
FK arm/leg/wrist split, amplitude ratio and per-row caveat notes.

**Column meanings** — `self rot°/cm` = mean over trained bodies of `X->X`; `xlate` = mean over
trained robots of `human_smpl->X` (or `soma77->X` in the 0803–04 caches); `0-shot` =
`human_smpl->unitree_g1_29dof`, a body never sampled in training (the eval only ever translates
into `holdout[0]`); `ST train/hold` = mean `full_rot` of the swing/twist self-translation report
over trained / held-out bodies; `scope=decoder` marks the frozen-lattice decoder-only retrains.

| run | date | d_model | enc | dec | heads | K=n_latent | fsq_levels | stg | bits/frame | params | steps | corpus (cache) | #train bodies | holdout | self rot° | self cm | xlate rot° | xlate cm | 0-shot rot° | 0-shot cm | ST train° | ST hold° | scope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0b_final_0803 | 2026-08-03 | — | — | — | — | 32 | 8,8,8,5,5,5 | 2 | 2043.6 | 17677541 | 8000 | cache_paired_fixed | 2 | — | 7.5 | 3.88 | 9.4 | 3.15 | — | — | 10.45 | — | — |
| E14_grad_0804 | 2026-08-04 | — | — | — | — | 32 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 2496.0 | 17951759 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 7.28 | 3.479 | 8.921 | 4.377 | 19.6 | 9.3 | 7.28 | 32.033 | — |
| E14_obs_0804 | 2026-08-04 | — | — | — | — | 32 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 2496.0 | 17951759 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 7.207 | 3.513 | 9.071 | 4.384 | 20.1 | 9.62 | 7.207 | 31.667 | — |
| E15_dof_0804 | 2026-08-04 | — | — | — | — | 32 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 2496.0 | 17956623 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.833 | 3.061 | 8.45 | 3.918 | 18.0 | 8.11 | 6.833 | 28.9 | — |
| E16_dof_k64_0804 | 2026-08-04 | — | — | — | — | 64 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 4992.0 | 17973007 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.853 | 2.97 | 8.414 | 3.843 | 16.9 | 9.1 | 6.853 | 28.6 | — |
| E19_K128_0805 | 2026-08-05 | — | — | — | — | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18005775 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.453 | 2.926 | 8.171 | 3.89 | 15.5 | 7.72 | 6.453 | 27.533 | — |
| E19_K16_0805 | 2026-08-05 | — | — | — | — | 16 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 1248.0 | 17948431 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 7.94 | 3.369 | 9.257 | 4.124 | 17.1 | 9.17 | 7.94 | 30.133 | — |
| E19_K96_0805 | 2026-08-05 | — | — | — | — | 96 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 7488.0 | 17989391 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.493 | 2.908 | 8.164 | 3.839 | 17.0 | 9.08 | 6.493 | 26.667 | — |
| E20_K64_L15_0805 | 2026-08-05 | — | — | — | — | 64 | 15,15,15,15,15,15,15,15,15,15,15,15,15 | 1 | 6501.1 | 17973007 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.04 | 2.836 | 7.807 | 3.794 | 16.3 | 8.99 | 6.04 | 27.667 | — |
| E20_K96_L15_0805 | 2026-08-05 | — | — | — | — | 96 | 15,15,15,15,15,15,15,15,15,15,15,15,15 | 1 | 9751.6 | 17989391 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.187 | 2.748 | 7.936 | 3.704 | 17.3 | 8.91 | 6.187 | 27.167 | — |
| E21_data1000_0805 | 2026-08-05 | — | — | — | — | 64 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 4992.0 | 17973007 | 8000 | cache_amass_1000 | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 7.28 | 3.451 | 13.021 | 5.091 | 25.1 | 11.46 | 7.28 | 28.667 | — |
| E21_data2000_0805 | 2026-08-05 | — | — | — | — | 64 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 4992.0 | 17973007 | 8000 | cache_amass_2000 | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 7.56 | 3.277 | 12.971 | 4.926 | 22.6 | 11.78 | 7.56 | 27.0 | — |
| E21_data4000_0805 | 2026-08-05 | — | — | — | — | 64 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 4992.0 | 17973007 | 8000 | cache_amass_4000 | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 7.9 | 3.414 | 13.779 | 5.192 | 23.3 | 11.81 | 7.9 | 26.933 | — |
| E21_data500_0805 | 2026-08-05 | — | — | — | — | 64 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 4992.0 | 17973007 | 8000 | cache_amass_500 | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 7.647 | 3.531 | 13.093 | 5.087 | 25.6 | 11.31 | 7.647 | 29.0 | — |
| E22_synth_0805 | 2026-08-05 | — | — | — | — | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18007317 | 8000 | cache_xwbc_lafan_synth | 21 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 7.0 | 2.954 | 8.915 | 4.0 | 16.5 | 8.45 | 7.0 | 27.467 | — |
| E23_K128_s1_0805 | 2026-08-05 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18005775 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.38 | 2.801 | 8.057 | 3.752 | 16.4 | 9.9 | 6.38 | 28.9 | — |
| E23_K128_s2_0805 | 2026-08-05 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18005775 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.513 | 2.867 | 8.171 | 3.782 | 16.0 | 9.37 | 6.513 | 29.733 | — |
| E23_K32_s1_0805 | 2026-08-05 | 256 | 4 | 4 | 8 | 32 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 2496.0 | 17956623 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.987 | 3.091 | 8.621 | 3.926 | 16.5 | 8.64 | 6.987 | 27.833 | — |
| E23_K32_s2_0805 | 2026-08-05 | 256 | 4 | 4 | 8 | 32 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 2496.0 | 17956623 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.927 | 3.057 | 8.436 | 3.856 | 18.2 | 7.98 | 6.927 | 30.433 | — |
| E24_large_0805 | 2026-08-05 | 384 | 6 | 6 | 12 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 57780479 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 7.06 | 2.792 | 8.386 | 3.597 | 17.2 | 8.54 | 7.06 | 29.733 | — |
| E24_mid_0805 | 2026-08-05 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18005775 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.3 | 3.006 | 8.029 | 3.908 | 17.2 | 8.97 | 6.3 | 28.167 | — |
| E24_small_0805 | 2026-08-05 | 128 | 3 | 3 | 4 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 3671511 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 9.6 | 4.097 | 12.2 | 5.583 | 20.1 | 11.65 | 9.6 | 31.733 | — |
| E25_dofhead_0805 | 2026-08-05 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18006546 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 9.027 | 2.927 | 10.4 | 3.824 | 32.8 | 8.88 | 9.027 | 37.933 | — |
| E25_dofloss_0805 | 2026-08-05 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18005775 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 31.14 | 2.878 | 33.2 | 3.76 | 38.9 | 8.92 | 31.14 | 44.1 | — |
| E26_fixed_0805 | 2026-08-05 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18005775 | 8000 | cache_xwbc_lafan_fixed | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.32 | 2.897 | 8.05 | 3.799 | 16.6 | 8.0 | 6.32 | 27.467 | — |
| E27_qwen_0806 | 2026-08-06 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18071311 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.513 | 2.853 | 8.214 | 3.806 | 17.5 | 8.82 | 6.513 | 29.633 | — |
| E27_sig_0806 | 2026-08-06 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18005775 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.8 | 2.987 | 8.429 | 3.871 | 17.7 | 9.07 | 6.8 | 27.833 | — |
| E28_resid_0807 | 2026-08-07 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18009112 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 0.1 | 0.0 | 8.143 | 3.819 | 16.0 | 8.31 | 6.647 | 29.133 | — |
| E29_wristaug_0807 | 2026-08-07 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18006803 | 8000 | cache_xwbc_lafan_wrist | 19 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.268 | 2.753 | 7.894 | 3.671 | 17.3 | 8.08 | 6.268 | 27.267 | — |
| E5_n2_0803 | 2026-08-03 | — | — | — | — | 32 | 8,8,8,5,5,5 | 2 | 2043.6 | 17677541 | 8000 | cache_paired_multi4 | 2 | soma_nohand;soma_scaled | 7.15 | 3.58 | 9.0 | 2.87 | 101.4 | 17.65 | 10.2 | 55.05 | — |
| E5_n3_0803 | 2026-08-03 | — | — | — | — | 32 | 8,8,8,5,5,5 | 2 | 2043.6 | 17677541 | 8000 | cache_paired_multi4 | 3 | soma_nohand | 7.3 | 4.517 | 8.05 | 4.44 | 77.4 | 20.67 | 11.533 | 81.6 | — |
| E5f_n13_0803 | 2026-08-03 | — | — | — | — | 32 | 8,8,8,5,5,5 | 2 | 2043.6 | 17677541 | 8000 | cache_paired_13 | 13 | kuavo_s45 | 7.323 | 3.015 | 8.117 | 3.148 | 43.6 | 22.2 | 7.354 | 41.1 | — |
| E5f_n2_0803 | 2026-08-03 | — | — | — | — | 32 | 8,8,8,5,5,5 | 2 | 2043.6 | 17677541 | 8000 | cache_paired_13 | 2 | kuavo_s45;booster_t1;booster_t1_29dof;engineai_pm01;fourier_gr3;fourier_n1;pnd_adam_lite;stanford_toddy;unitree_g1;unitree_g1_with_hands;unitree_h1;unitree_h1_2 | 6.1 | 3.58 | 7.5 | 2.9 | 129.6 | 45.4 | 8.75 | 121.292 | — |
| E5v2_n13_0804 | 2026-08-04 | — | — | — | — | 32 | 8,8,8,5,5,5 | 2 | 2043.6 | 17677541 | 8000 | cache_paired_13v2 | 13 | kuavo_s45 | 6.969 | 2.973 | 7.717 | 3.041 | 44.8 | 22.95 | 7.008 | 50.3 | — |
| E5v2_n2_0804 | 2026-08-04 | — | — | — | — | 32 | 8,8,8,5,5,5 | 2 | 2043.6 | 17677541 | 8000 | cache_paired_13v2 | 2 | kuavo_s45;booster_t1;booster_t1_29dof;engineai_pm01;fourier_gr3;fourier_n1;pnd_adam_lite;stanford_toddy;unitree_g1;unitree_g1_with_hands;unitree_h1;unitree_h1_2 | 7.15 | 3.23 | 8.7 | 2.68 | 56.7 | 35.24 | 9.9 | 57.575 | — |
| FIX_translate | 2026-08-02 | — | — | — | — | 32 | — | — | — | 17677285 | 8000 | cache_paired_fixed | 2 | — | 7.3 | 3.375 | 9.1 | 2.59 | — | — | 10.25 | — | — |
| FK1_rot128_0812 | 2026-08-12 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 6144.0 | 18072868 | 24000 | cache_xwbc_lafan,cache_xwbc_amass_full | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 0.1 | 0.0 | 11.3 | 35.556 | 21.1 | 33.96 | 5.98 | 26.7 | all |
| FK2_rot256_0812 | 2026-08-12 | 256 | 4 | 4 | 8 | 256 | 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 12288.0 | 18138404 | 24000 | cache_xwbc_lafan,cache_xwbc_amass_full | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 0.1 | 0.0 | 11.257 | 3.802 | 21.5 | 6.73 | 5.76 | 26.8 | all |
| N3f_varfloor_0807 | 2026-08-07 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18071311 | 8000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 6.447 | 2.937 | 8.164 | 3.876 | 17.6 | 8.78 | 6.447 | 28.767 | — |
| PROD_c1_91k_dim20_tail_0810 | 2026-08-10 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 15360.0 | 18081844 | 91000 | cache_xwbc_lafan,cache_xwbc_amass_full | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 0.1 | 0.0 | 9.829 | 3.489 | 20.0 | 8.8 | 4.82 | 25.933 | — |
| PROD_data13k_0809 | 2026-08-09 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18074648 | 24000 | cache_xwbc_lafan,cache_xwbc_amass_full | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 0.1 | 0.0 | 11.129 | 4.136 | 20.9 | 9.58 | 5.527 | 26.133 | — |
| PROD_dim16_13k_0809 | 2026-08-09 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 12288.0 | 18077732 | 24000 | cache_xwbc_lafan,cache_xwbc_amass_full | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 0.1 | 0.0 | 11.029 | 4.133 | 20.7 | 9.29 | 5.22 | 25.967 | — |
| PROD_spatial_0808 | 2026-08-08 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 9984.0 | 18074648 | 24000 | cache_xwbc_lafan | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 0.1 | 0.0 | 5.314 | 2.503 | 13.5 | 7.53 | 4.34 | 25.133 | — |
| PT_c1dec_arm_0811 | 2026-08-11 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 15360.0 | 18081844 | 20000 | cache_xwbc_lafan,cache_xwbc_amass_full | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 0.1 | 0.0 | 9.836 | 3.494 | 20.1 | 8.82 | 4.86 | 26.133 | decoder |
| PT_c1dec_reenc_0812 | 2026-08-12 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 15360.0 | 18081844 | 20000 | cache_xwbc_lafan,cache_xwbc_amass_full | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 0.1 | 0.0 | 9.857 | 3.471 | 19.9 | 8.77 | 4.793 | 25.8 | decoder |
| PT_c1dec_wrist_0811 | 2026-08-11 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 15360.0 | 18081844 | 30000 | cache_xwbc_lafan,cache_xwbc_amass_full | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 0.1 | 0.0 | 9.893 | 3.556 | 20.1 | 8.89 | 4.92 | 26.733 | decoder |
| SMALL_dim16_reenc_0812 | 2026-08-12 | 256 | 4 | 4 | 8 | 128 | 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8 | 1 | 12288.0 | 18072868 | 24000 | cache_xwbc_lafan,cache_xwbc_amass_full | 15 | unitree_g1_29dof;unitree_g1_23dof;pal_talos | 0.1 | 0.0 | 11.379 | 4.189 | 20.6 | 10.44 | 5.567 | 26.467 | all |

### 2.1 Reading Table A — the traps that are live in it

1. **`E25_dofloss_0805` self_rot = 31.1°** is a *training failure*, not a scaling point
   (the DOF-projection loss is irreversible — `RUNS_INDEX.md`: "E25 失败（投影不可撤）"). Exclude it
   from every fit.
2. **`FK1_rot128_0812` xlate_pos = 35.56 cm is invalid.** It is `rot_only`; the position head is
   gone, so `eval_log`'s `pos_cm` column measures |GT| and nothing else. `FK_DUEL_0813` says this
   explicitly. FK1's real numbers are in the FK-ruler columns (`arm mean 6.62 cm`).
3. **`PROD_spatial_0808` looks best in-corpus (5.31° / 2.50 cm) but it is not comparable** to the
   PROD_c1 / PROD_dim16 / PROD_data13k line. It trained *and evaluated* on LAFAN alone (77 clips,
   496 593 rows); the others trained and evaluated on LAFAN + AMASS-full (13 358 clips, 5.23 M rows).
   Its val split is a random frame split of 77 clips — the easiest possible test. `MODEL_VERSIONS.md`
   quotes Greenwich = `PROD_spatial_0808` at 6.39°/2.82 cm in-corpus and 23.02°/10.52 cm held out,
   from a different eval script again; neither set of numbers should be put on the same axis as
   the AMASS-scale rows.
4. **Round-trip / re-encode fidelity has a bugfix boundary at 2026-08-12 08:00.**
   `blame_split.py` before the fix computed `fsqp.ints(fsqp.features(zp))` on an already-encoded
   `zp` — two stacked projections, one of which lives on the *decoder* side, so a decoder-only
   finetune silently changed the measured quantity while the real codes stayed bit-identical.
   `REENC_ABLATION_0812.txt` (written 05:55) uses the **pre-fix** numbers; the `BLAME_*.json`
   files (re-run 08:06–08:10) are post-fix. Both are in the CSV; only the post-fix set is
   defensible:

   | run | A enc-invariance | B decoder fidelity | C chain | note |
   |---|---|---|---|---|
   | `PROD_c1_91k_dim20_tail_0810` (Greenwich/C1) | 0.592 | 0.278 | 0.275 | reenc off |
   | `PT_c1dec_reenc_0812` (crowned decoder) | **0.592** | **0.679** | **0.644** | decoder-only; A identical to C1 ⇒ encoder provably untouched |
   | `SMALL_dim16_reenc_0812` (full retrain, reenc=1) | 0.622 | 0.680 | 0.710 | |
   | `PROD_dim16_13k_0809` (its reenc=0 twin) | 0.597 | 0.294 | 0.299 | single-variable control |

   The single-variable ablation is `PROD_dim16_13k_0809` → `SMALL_dim16_reenc_0812`
   (identical recipe, `--reenc_consistency 0 → 1`): **decoder round-trip 0.294 → 0.680**.
   `PT_c1dec_reenc_0812` reproduces that (0.278 → 0.679) at zero cascade cost, which is why it is
   the crowned production decoder. The "11 % → 61 %" figure quoted elsewhere is the same move
   expressed above each model's own chance floor; the floors differ between models
   (`REENC_ABLATION_0812` measures 0.568 for C1, 0.625 for the ablation), so the normalised
   percentages should never be compared across rows without their floors.
5. **The re-encode switch costs zero-shot.** At equal steps: in-distribution bodies pay +1.3–2.3 %
   position error, the held-out G1 pays **+12.4 %** (`REENC_ABLATION_0812`). The switch buys
   round-trip and spends generalisation.
6. **FSQ usage is healthy everywhere it was measured** — `PROD_c1`: per-dim entropy 2.601–2.801 of
   a possible 3.000 (median 2.75), **0 dead dims, 0 saturated dims**, 185 118 / 166 361 distinct
   codes in 200 k frames. `SMALL_dim16_reenc`: 2.564–2.798, 0 dead, 182 467 / 132 197 distinct.
   The re-encode constraint's rising chance floor is therefore *not* codebook collapse; it is the
   code becoming more predictable from geometry. Independently, `R7_LATENT`: on unseen AMASS
   **94.5 % of slots use ≥ 7 of the 8 nominal levels** (the 8th and 0th level are `round(tanh·3.5)`
   saturation artefacts at 0.1–0.5 % each). **There is no capacity being wasted; the bottleneck is
   not the codebook.**
7. **Arm vs leg split, on the FK ruler** (`FK_DUEL_0813`, 24 contiguous 90-frame val windows × 4
   cross-body directions):

   | model | slots | bits/frame | arm mean | arm p95 | leg mean | wrist p95 |
   |---|---|---|---|---|---|---|
   | `PROD_dim16_13k_0809` | 256 | 12 288 | 6.78 | 29.28 | 4.21 | 37.63 |
   | `SMALL_dim16_reenc_0812` | 256 | 12 288 | 7.17 | 29.44 | 4.37 | 37.62 |
   | `FK1_rot128_0812` (rot_only) | 128 | **6 144** | **6.62** | **28.51** | 4.37 | 37.60 |
   | `FK2_rot256_0812` (rot_only) | 256 | 12 288 | 6.81 | 28.54 | 4.21 | **37.18** |

   Halving the bit budget (FK2 → FK1) changes arm p95 by **+0.03 cm (+0.1 %)** and wrist p95 by
   **−0.42 cm (−1.1 %)**. Half of Greenwich's storage is free.
8. **The arm tail is a data wall, not a model-size or rate problem.** `DECODER_PT_verdict_0811`:
   a readout probe from the codes reaches the human wrist at p95 ≈ 12 cm but the G1-paired wrist at
   p95 ≈ 30 cm — the same 30 cm the decoder itself achieves. Two dedicated decoder post-trains
   (20 k gentle, 30 k hot with a wrist-position term) moved wrist p95 by **−0.4 %** and **< 1 %**.
   The human→G1 paired GT (GMR arm retarget) contains ~30 cm-p95 variance that is not predictable
   from the human motion. **No scaling axis in this report can cross that wall.**
9. **A retracted claim that used to live here:** the "all models compress motion to 30–44 % of GT
   amplitude" finding was measured on ONE punch window and does not survive 24 (or 300) windows.
   Measured properly, amplitude retention is 0.885–0.912 overall, 0.897–0.935 on arms, 0.911–0.954
   on legs, and `corr(GT amplitude, retention) = −0.073`. The amplitude columns in the CSV are the
   corrected ones.

---

## 3. Table B — Equator (temporal bridge stack)

72 rows: every temporal run with a config, including the P1 generation, the V5/V6/V7/V8/V9
generations, the S / A / RATE / T32 / CTRL scans, and the BRIDGE A/B stack.

`bits/win` and `bit/s` are the pre-registered `bit_budget` block. `key cm` / `key °` are
`eval_log.json`'s `val.human_smpl.key_pos_cm` / `key_rot_deg` at the final step. `0-shot cm/°` are
the mean over the `zeroshot_*` rows (both G1s). `v7ev cm` / `warp inv` / `body leak` come from the
canonical `v7_eval.py` verdict dumps and exist only for the runs that were dueled.
`bridgeCE` / `ppl` / `tok acc` are the BRIDGE A/B token-prediction metrics.

| run | date | arch | arm | n_path | n_time | n_grp | fsq levels | time lv | bits/win | bit/s | d | enc | ctx | ro | dec | params | steps | corpus | key cm | key ° | endpt rel | energy | 0-shot cm | 0-shot ° | v7ev cm | warp inv | body leak | bridgeCE | ppl | tok acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0.0_arclen_4b_0807 | 2026-08-07 | v7 | arclen | 8 | 1 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 8333634 | 60000 | 4 emb / 4000 clips | 5.942 | 9.53 | 0.0822 | 0.915 | 6.917 | 18.345 | 5.71 | 0.835 | 0.093 | — | — | — |
| A0.25_arclen_4b_0807 | 2026-08-07 | v7 | arclen | 8 | 1 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 8333634 | 60000 | 4 emb / 4000 clips | 5.778 | 9.61 | 0.0919 | 0.908 | 6.9255 | 18.35 | 5.65 | 0.885 | 0.085 | — | — | — |
| A0.5_arclen_4b_0807 | 2026-08-07 | v7 | arclen | 8 | 1 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 8333634 | 60000 | 4 emb / 4000 clips | 5.632 | 9.33 | 0.0881 | 0.914 | 7.277 | 18.28 | 5.65 | 0.894 | 0.101 | — | — | — |
| A0.75_arclen_4b_0807 | 2026-08-07 | v7 | arclen | 8 | 1 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 8333634 | 60000 | 4 emb / 4000 clips | 5.742 | 9.41 | 0.0881 | 0.912 | 6.798 | 18.145 | 5.71 | 0.911 | 0.078 | — | — | — |
| BRIDGE_A2_full_0810 | 2026-08-10 | bridge_a | — | — | — | — | — | — | — | — | — | — | — | — | — | 20353798 | 40000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | — | — | 0.7412 |
| BRIDGE_A3_prod_0810 | 2026-08-10 | bridge_a | — | — | — | — | — | — | — | — | 384 | 6 | — | — | 4 | 40006278 | 100000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | — | — | 0.7855 |
| BRIDGE_A_pilot_0810 | 2026-08-10 | bridge_a | — | — | — | — | — | — | — | — | — | — | — | — | — | 13777158 | 40000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | — | — | 0.6248 |
| BRIDGE_B2_maskgit_0810 | 2026-08-10 | bridge_b | — | — | — | — | — | — | — | — | — | — | — | — | — | 12833545 | 30000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | 2.9709 | 19.5 | — |
| BRIDGE_B2_v1ctl_0810 | 2026-08-10 | bridge_b | — | — | — | — | — | — | — | — | — | — | — | — | — | 12833545 | 30000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | 2.8441 | 17.2 | — |
| BRIDGE_B3_prod_0810 | 2026-08-10 | bridge_b | — | — | — | — | — | — | — | — | — | — | — | — | — | 12866313 | 80000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | 3.1175 | 22.6 | — |
| BRIDGE_B4_text_0810 | 2026-08-10 | bridge_b4_text | — | — | — | — | — | — | — | — | — | — | — | — | — | 16868617 | 30000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | — | — | — |
| BRIDGE_B4v2_gdrop_0810 | 2026-08-10 | bridge_b4_text | — | — | — | — | — | — | — | — | — | — | — | — | — | 16868617 | 30000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | — | — | — |
| BRIDGE_B4v3_textlr_0810 | 2026-08-10 | bridge_b4_text | — | — | — | — | — | — | — | — | — | — | — | — | — | 16868617 | 50000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | — | — | — |
| BRIDGE_B4v4_scratch_0811 | 2026-08-11 | bridge_b4_text | — | — | — | — | — | — | — | — | — | — | — | — | — | 16868617 | 80000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | — | — | — |
| BRIDGE_B4v5_fixed_0812 | 2026-08-12 | bridge_b4_text | — | — | — | — | — | — | — | — | — | — | — | — | — | 16868617 | 30000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | — | — | — |
| BRIDGE_B5_xattn_0814 | 2026-08-14 | bridge_b5_xattn | — | — | — | — | — | — | — | — | — | — | — | — | — | 14719753 | 24000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | — | — | — |
| BRIDGE_B6_fwd_0811 | 2026-08-11 | bridge_b | — | — | — | — | — | — | — | — | — | — | — | — | — | 12866313 | 40000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | 2.9439 | 19.0 | — |
| BRIDGE_B7_rollout_0811 | 2026-08-11 | bridge_b_rollout | — | — | — | — | — | — | — | — | — | — | — | — | — | 12866313 | 20000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | 3.0994 | 22.2 | — |
| BRIDGE_B8_cycle_0811 | 2026-08-11 | bridge_b_rollout | — | — | — | — | — | — | — | — | — | — | — | — | — | 12866313 | 16000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | 3.612 | 37.0 | — |
| BRIDGE_B_pilot_0810 | 2026-08-10 | bridge_b | — | — | — | — | — | — | — | — | — | — | — | — | — | 12833545 | 30000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | 4.5693 | 96.5 | — |
| CTRLa_maskfsq_4body_0807 | 2026-08-07 | v7 | maskfsq | 9 | 0 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 8055868 | 60000 | 4 emb / 4000 clips | 5.663 | 8.8 | 0.0858 | 0.933 | 7.01 | 18.08 | 5.6 | 0.58 | 0.074 | — | — | — |
| CTRLb_v5c_16body_0807 | 2026-08-07 | — | pathint | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8309826 | 60000 | 16 emb / 4000 clips | 8.055 | 14.06 | 0.1186 | 0.737 | — | — | 8.0 | 0.997 | 0.041 | — | — | — |
| EQ_B5_semigroup_0811 | 2026-08-11 | bridge_b_semigroup | — | — | — | — | — | — | — | — | — | — | — | — | — | 12866313 | 8000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | 3.6883 | 40.0 | — |
| N3a_regAR_0807 | 2026-08-07 | — | pathint | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8313666 | 60000 | 4 emb / 4000 clips | 7.428 | 12.85 | 0.1096 | 0.786 | — | — | — | — | — | — | — | — |
| N3b_xbody_0807 | 2026-08-07 | — | pathint | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8311362 | 60000 | 4 emb / 4000 clips | 8.235 | 14.1 | 0.1186 | 0.719 | — | — | — | — | — | — | — | — |
| N3c_root_0807 | 2026-08-07 | — | pathint | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8316747 | 60000 | 4 emb / 4000 clips | 7.19 | 12.7 | 0.0863 | 0.797 | — | — | — | — | — | — | — | — |
| P1_oat_z_0805 | 2026-08-05 | oat | — | — | — | — | 5,5,5,5,5,5 | — | — | — | 384 | 5 | — | 2 | 3 | 30739340 | 30000 | 5 emb / — | — | — | — | — | — | — | — | — | — | — | — | — |
| P1_unord_z_0805 | 2026-08-05 | unordered | — | — | — | — | 5,5,5,5,5,5 | — | — | — | 384 | 5 | — | 2 | 3 | 30739340 | 30000 | 5 emb / — | — | — | — | — | — | — | — | — | — | — | — | — |
| P1b_full_0806 | 2026-08-06 | oat | — | — | — | — | 5,5,5,5,5,5 | — | — | — | 384 | 5 | — | 2 | 3 | 30736274 | 30000 | 5 emb / — | — | — | — | — | — | — | — | — | — | — | — | — |
| P1b_motion_vel_0806 | 2026-08-06 | oat | — | — | — | — | 5,5,5,5,5,5 | — | — | — | 384 | 5 | — | 2 | 3 | 30739340 | 30000 | 5 emb / — | — | — | — | — | — | — | — | — | — | — | — | — |
| P1b_motion_w2_0806 | 2026-08-06 | oat | — | — | — | — | 5,5,5,5,5,5 | — | — | — | 384 | 5 | — | 2 | 3 | 30739340 | 30000 | 5 emb / — | — | — | — | — | — | — | — | — | — | — | — | — |
| P1b_motion_w5_0806 | 2026-08-06 | oat | — | — | — | — | 5,5,5,5,5,5 | — | — | — | 384 | 5 | — | 2 | 3 | 30739340 | 30000 | 5 emb / — | — | — | — | — | — | — | — | — | — | — | — | — |
| P1b_n4_0806 | 2026-08-06 | oat | — | — | — | — | 5,5,5,5,5,5 | — | — | — | 384 | 5 | — | 2 | 3 | 30736268 | 30000 | 5 emb / — | — | — | — | — | — | — | — | — | — | — | — | — |
| P1c_C1_conv_fps10_0806 | 2026-08-06 | conv | — | — | — | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 384 | 5 | — | 2 | 3 | 33223302 | 30000 | 5 emb / — | — | — | — | — | — | — | — | — | — | — | — | — |
| P1c_C1_conv_hi_0806 | 2026-08-06 | conv | — | — | — | — | 5,5,5,5,5,5 | — | 236.84 | 118.42 | 384 | 5 | — | 2 | 3 | 33225606 | 30000 | 5 emb / — | — | — | — | — | — | — | — | — | — | — | — | — |
| P1c_C1_conv_lo_0806 | 2026-08-06 | conv | — | — | — | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 384 | 5 | — | 2 | 3 | 36769158 | 30000 | 5 emb / — | — | — | — | — | — | — | — | — | — | — | — | — |
| P1c_C1_conv_lo_mres_0806 | 2026-08-06 | conv | — | — | — | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 384 | 5 | — | 2 | 3 | 36769158 | 30000 | 5 emb / — | — | — | — | — | — | — | — | — | — | — | — | — |
| P1c_C2_freq_hi_0806 | 2026-08-06 | freq | — | — | — | — | 5,5,5,5,5,5 | — | 236.84 | 118.42 | 384 | 5 | — | 2 | 3 | 21561238 | 30000 | 5 emb / — | — | — | — | — | — | — | — | — | — | — | — | — |
| P1c_C2_freq_lo_0806 | 2026-08-06 | freq | — | — | — | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 384 | 5 | — | 2 | 3 | 21555086 | 30000 | 5 emb / — | — | — | — | — | — | — | — | — | — | — | — | — |
| PRODT_arclen_0808 | 2026-08-08 | v7 | arclen | 8 | 1 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 8333634 | 120000 | 16 emb / 4000 clips | 7.906 | 12.02 | 0.1322 | 0.858 | 6.293 | 12.14 | — | — | — | — | — | — |
| PRODT_jepa_0808 | 2026-08-08 | v7 | jepa | 8 | 1 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 9997634 | 120000 | 16 emb / 4000 clips | — | — | — | — | — | — | — | — | — | — | — | — |
| RATE_jepa_l3_0807 | 2026-08-07 | v7 | jepa | 8 | 1 | 9 | 3,3,3,3,3,3 | — | 85.59 | 42.79 | 256 | 5 | 2 | 2 | 3 | 9997634 | 60000 | 4 emb / 4000 clips | 7.066 | 12.06 | 0.1108 | 0.835 | 7.1525 | 18.225 | — | — | — | — | — | — |
| RATE_jepa_l4_0807 | 2026-08-07 | v7 | jepa | 8 | 1 | 9 | 4,4,4,4,4,4 | — | 108.0 | 54.0 | 256 | 5 | 2 | 2 | 3 | 9997634 | 60000 | 4 emb / 4000 clips | 6.945 | 11.62 | 0.1459 | 0.86 | 6.9405 | 18.595 | — | — | — | — | — | — |
| RATE_jepa_l5_0807 | 2026-08-07 | v7 | jepa | 8 | 1 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 9997634 | 60000 | 4 emb / 4000 clips | 6.665 | 11.24 | 0.1036 | 0.882 | 6.653 | 19.525 | — | — | — | — | — | — |
| RATE_mf_l3_0807 | 2026-08-07 | v7 | maskfsq | 9 | 0 | 9 | 3,3,3,3,3,3 | — | 85.59 | 42.79 | 256 | 5 | 2 | 2 | 3 | 8055868 | 60000 | 4 emb / 4000 clips | 6.104 | 9.84 | 0.1034 | 0.91 | 6.792 | 16.395 | — | — | — | — | — | — |
| RATE_mf_l4_0807 | 2026-08-07 | v7 | maskfsq | 9 | 0 | 9 | 4,4,4,4,4,4 | — | 108.0 | 54.0 | 256 | 5 | 2 | 2 | 3 | 8055868 | 60000 | 4 emb / 4000 clips | 6.136 | 10.05 | 0.1492 | 0.843 | 7.122 | 19.08 | — | — | — | — | — | — |
| S0_arclen_4b_0807 | 2026-08-07 | v7 | arclen | 8 | 1 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 8333634 | 60000 | 4 emb / 4000 clips | 5.884 | 9.4 | 0.0929 | 0.91 | 6.6015 | 18.565 | 5.75 | 0.925 | 0.089 | — | — | — |
| S1_speed37_0807 | 2026-08-07 | v7 | arclen | 8 | 1 | 9 | 5,5,5,5,5,5 | 7,7,7 | 119.87 | 59.94 | 256 | 5 | 2 | 2 | 3 | 8332095 | 60000 | 4 emb / 4000 clips | 6.143 | 10.09 | 0.1081 | 0.899 | 6.6105 | 18.755 | 6.01 | 0.931 | 0.085 | — | — | — |
| S2_c85_0807 | 2026-08-07 | v7 | arclen | 8 | 1 | 9 | 5,5,5,5,5,5,5,5 | 7,7,7 | 157.02 | 78.51 | 256 | 5 | 2 | 2 | 3 | 8333121 | 60000 | 4 emb / 4000 clips | 5.823 | 9.09 | 0.102 | 0.923 | 6.6255 | 17.035 | 5.64 | 0.92 | 0.12 | — | — | — |
| S3_open88_0807 | 2026-08-07 | v7 | arclen | 8 | 1 | 9 | 8,8,8,8,8,8,8,8 | 7,7,7 | 200.42 | 100.21 | 256 | 5 | 2 | 2 | 3 | 8333121 | 60000 | 4 emb / 4000 clips | 5.678 | 8.75 | 0.0847 | 0.937 | 6.8035 | 17.81 | 5.56 | 0.926 | 0.131 | — | — | — |
| T32_arclen_0809 | 2026-08-09 | v7 | arclen | 32 | 1 | 9 | 5,5,5,5,5,5 | — | 459.74 | 229.87 | 256 | 5 | 2 | 2 | 3 | 8352066 | 60000 | 16 emb / 13281 clips | 6.036 | 8.72 | 0.0807 | 0.955 | 5.5805 | 10.445 | — | — | — | — | — | — |
| T32_jepa_0809 | 2026-08-09 | v7 | jepa | 32 | 1 | 9 | 5,5,5,5,5,5 | — | 459.74 | 229.87 | 256 | 5 | 2 | 2 | 3 | 10009922 | 60000 | 16 emb / 13281 clips | 6.214 | 8.94 | 0.0815 | 0.944 | 5.8885 | 11.485 | — | — | — | — | — | — |
| TEXT_ALIGN_v1_0810 | 2026-08-10 | text_align_v1 | — | — | — | — | — | — | — | — | — | — | — | — | — | 7825932 | 300 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | — | — | — |
| TEXT_ALIGN_v2_0810 | 2026-08-10 | text_align_v1 | — | — | — | — | — | — | — | — | — | — | — | — | — | 7825932 | 12000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | — | — | — |
| TEXT_ALIGN_v3_fixed_0812 | 2026-08-12 | text_align_v1 | — | — | — | — | — | — | — | — | — | — | — | — | — | 7825932 | 12000 | A3/B3 token cache | — | — | — | — | — | — | — | — | — | — | — | — |
| V5_baseline_frameCE_0806 | 2026-08-06 | — | baseline_frameCE | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8531362 | 60000 | 4 emb / 4000 clips | 4.969 | 8.77 | 0.2447 | 0.89 | — | — | — | — | — | — | — | — |
| V5_pathint_0806 | 2026-08-06 | — | pathint | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8309314 | 60000 | 4 emb / 4000 clips | 7.558 | 13.26 | 0.1134 | 0.773 | — | — | — | — | — | — | — | — |
| V5_pathint_noaug_0806 | 2026-08-06 | — | pathint_noaug | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8309314 | 60000 | 4 emb / 4000 clips | 7.008 | 11.87 | 0.1097 | 0.821 | — | — | — | — | — | — | — | — |
| V5b_Bonly_0806 | 2026-08-06 | — | pathint | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8309314 | 60000 | 4 emb / 4000 clips | 7.748 | 13.35 | 0.1131 | 0.767 | — | — | — | — | — | — | — | — |
| V5b_cdrop_0806 | 2026-08-06 | — | pathint | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8309826 | 60000 | 4 emb / 4000 clips | 8.2 | 14.16 | 0.1095 | 0.745 | — | — | — | — | — | — | — | — |
| V5b_full_0806 | 2026-08-06 | — | pathint | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8309314 | 60000 | 4 emb / 4000 clips | 7.706 | 13.3 | 0.104 | 0.783 | — | — | — | — | — | — | — | — |
| V5b_noC_0806 | 2026-08-06 | — | pathint | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8309314 | 60000 | 4 emb / 4000 clips | 7.763 | 12.75 | 0.1118 | 0.786 | — | — | — | — | — | — | — | — |
| V5c_0806 | 2026-08-06 | — | pathint | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8309826 | 60000 | 4 emb / 4000 clips | 7.695 | 13.39 | 0.1097 | 0.762 | — | — | 7.66 | 0.995 | 0.119 | — | — | — |
| V6_mae_0807 | 2026-08-07 | v6 | — | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8304706 | 60000 | 4 emb / 4000 clips | 7.683 | 12.84 | 0.0912 | 0.78 | — | — | — | — | — | — | — | — |
| V6_mae_resid_0807 | 2026-08-07 | v6 | — | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8333122 | 60000 | 4 emb / 4000 clips | 7.657 | 12.74 | 0.1073 | 0.763 | — | — | 7.61 | 0.954 | 0.067 | — | — | — |
| V6_regar_0807 | 2026-08-07 | v6 | — | 8 | 1 | — | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | — | 2 | 3 | 8308546 | 60000 | 4 emb / 4000 clips | 7.759 | 13.05 | 0.1077 | 0.775 | — | — | — | — | — | — | — | — |
| V7_jepa_0807 | 2026-08-07 | v7 | jepa | 8 | 1 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 9997634 | 60000 | 16 emb / 4000 clips | 6.511 | 10.39 | 0.1028 | 0.904 | 5.9395 | 12.785 | 6.39 | 0.867 | 0.056 | — | — | — |
| V7_maskfsq_0807 | 2026-08-07 | v7 | maskfsq | 9 | 0 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 8055868 | 60000 | 16 emb / 4000 clips | 6.044 | 9.3 | 0.0948 | 0.926 | 5.759 | 12.305 | 5.99 | 0.542 | 0.071 | — | — | — |
| V8_arclen_0807 | 2026-08-07 | v7 | arclen | 8 | 1 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 8333634 | 60000 | 16 emb / 4000 clips | 6.301 | 9.99 | 0.1003 | 0.906 | 5.546 | 12.25 | 6.24 | 0.922 | 0.06 | — | — | — |
| V8u_arclen_unif_0807 | 2026-08-07 | v7 | arclen | 8 | 1 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 8333634 | 60000 | 16 emb / 4000 clips | 6.229 | 9.62 | 0.093 | 0.909 | 5.6015 | 11.94 | 6.26 | 0.895 | 0.059 | — | — | — |
| V9_arclen_pt_0808 | 2026-08-08 | v7 | arclen | 8 | 1 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 8333634 | 60000 | 16 emb / 4000 clips | 6.233 | 9.51 | 0.1011 | 0.916 | 5.562 | 11.445 | — | — | — | — | — | — |
| V9_jepa_pt_0808 | 2026-08-08 | v7 | jepa | 8 | 1 | 9 | 5,5,5,5,5,5 | — | 125.38 | 62.69 | 256 | 5 | 2 | 2 | 3 | 9997634 | 60000 | 16 emb / 4000 clips | 6.467 | 10.24 | 0.1048 | 0.91 | 5.6115 | 12.52 | — | — | — | — | — | — |

### 3.1 Reading Table B — what is missing and what is invalid

* **There is no Equator model-size ladder.** Every V5/V6/V7/V8/V9/S/A/RATE/T32/CTRL run is
  `d_model=256, enc=5, ctx=2, readout=2, dec=3`. Parameter counts differ only by FSQ projection
  width and arm (8.06 M maskfsq / 8.33 M arclen / 10.00 M jepa). The P1 generation is
  `d_model=384, enc=5, dec=3` at 21–37 M, but it is a different architecture (global pooling / conv /
  freq arms) evaluated with a different metric. **Question 2 has no answer on the Equator side of a
  matched comparison.** The only size ladder anywhere in the temporal stack is BRIDGE-A (§5.2), and
  it is confounded with training steps.
* **P1 rows carry no cm metric at all** — only `budget_level_acc` (per-frame argmax hit rate), which
  `LIT_TEMPORAL_FSQ.md` flags as decoupled from motion-energy retention (the LARP rFVD/gFVD case).
  They are in the table for completeness; do not fit anything to them.
* **V5-era rows** (`V5*`, `N3a/b/c`, `CTRLb`) log `val` keyed by prefix budget `n1/n2/n4/n8` and have
  **no zero-shot row** in `eval_log.json`. Their zero-shot numbers exist only in the `v7_eval.py`
  verdicts (the `ve_*` CSV columns).
* **`content_share` is invalid on several rows and the CSV flags them.** The metric's denominator is
  "what the registers buy in cm"; `MODEL_VERSIONS.md §1` retires it below ~4 cm. Live cases:
  `CTRLb_v5c_16body` (registers buy 1.689 cm → content_share 0.658 meaningless),
  `V5c_0806` (1.410 cm → 0.583 meaningless), `S1_speed37` (3.566 cm → the reported 1.497, and
  3.238/2.744 on the zero-shot G1s, which is how it flew out of [0,1] in the first place).
  **Do not use content_share as a scaling variable.** The `registers buy [cm]` absolute is the
  honest quantity and it is in the CSV.
* **`.norm(-1)` vs `.norm(dim=-1)`**: the p=−1 norm reads ≈0 and masquerades as a perfect model.
  No metric reported in this document is a norm-based similarity, so this trap is not live here —
  but it is why every "similarity ≈ 1.0" claim in older files needs its call site checked before
  it is quoted.
* **`VERSION_AUDIT_0812` contract break.** A3 trains on the *masked* encode path, B3 on the
  *unmasked* `tokens_of` path; measured stream overlap **0.15 %**. Consequences carried into the CSV
  `notes`:
  * class 1 (A-internal) and class 2 (B-internal) comparisons **stand** — that includes the whole
    BRIDGE_STACK_v2 verdict, the knob sweep, the semigroup duel and the B3/B6/B7/B8 series *as
    B-vs-B comparisons*;
  * class 3 (B tokens into A3's decoder) are real measurements of the **shipped** path but not of
    the intended architecture — phase-space, homecoming, rainbow, response spectrum, gauge;
  * class 4 is **invalid**: `align_text.py` aligned text to masked tokens while B4 conditions the
    unmasked prior. **Every `text_gain` number in Table B (B4, B4v2, B4v3, B4v4, B4v5) and every
    TEXT_ALIGN retrieval number is therefore not interpretable as text conditioning.**
    `BRIDGE_B5_xattn_0814`'s `text_gain 0.0947` is on the newer cross-attention path and is the
    only text number not caught by this.
* **BRIDGE_STACK v1 vs v2.** `BRIDGE_A_pilot`/`BRIDGE_B_pilot` were scored on the v1 "dirty" ruler
  (self-decode on the human side). v2 decodes into the **held-out G1** and scores against the real
  paired retarget FK. On v2 the same generation reads endpoint hit 14.05 cm instead of 11.90, and
  sample diversity 2.33 cm instead of 0.00. The v1 diversity of exactly 0.00 was mode collapse that
  the dirty ruler could not see. **v1 and v2 columns must never be put in the same fit.**

### 3.2 Equator quality metrics that are not in the per-run table

| metric | value | source |
|---|---|---|
| endpoint hit (A3+B3, clean ruler) | mean 14.05 cm, p95 33.26 | `BRIDGE_STACK_v2_BRIDGE_B3_prod_0810` |
| anchor FK vs GT (A3+B3) | mean 15.64 cm, p95 31.74 | same |
| generation energy ratio (argmax / soft) | 0.65 / 0.50 | same |
| sample diversity | 2.33 cm (v1 was 0.00 = collapse) | same |
| sparse-pin token accuracy k = 0/8/16/24 | 0.209 / 0.344 / 0.430 / 0.477 | same |
| cross-clip bridge junction snap in / out | 21.21 / 31.85 cm (T32 recon reference: 3.79 / 3.75) | same |
| A3 interior token top-1 accuracy | 0.6028 (pilot); A3 final `val_acc` **0.7855**, `val_ce` 0.5023 | `BRIDGE_A_verdict_0810`, `eval_log` |
| A copy-rate (frame-AR disease) | 36.96 % (frame-level AR was 93.9 %) | `BRIDGE_A_verdict_0810` |
| B3 token NLL / ppl | CE 3.1175, ppl 22.6 (uniform vocabulary = 15 625) | `eval_log` |
| **retiming semigroup gap** | gap 2.69 cm vs one-hop sampling noise 3.89 cm → **ratio 0.69** | `HOMECOMING_0811` |
| same, re-measured on matched batches | gap 1.78, noise 2.28 → ratio 0.78 (B3); 1.70 / 2.04 → 0.84 (EQ_B5) | `SEMIGROUP_DUEL_0811` |
| semigroup probe variance | **±0.1 on the ratio**, and only comparable within one batch set | `SEMIGROUP_DUEL_0811` note |
| Equator-as-judge NLL ordering | real 3.423 < reversed 3.906 < spliced 5.314 < shuffled 5.382 | `EQUATOR_NLL_JUDGE.txt` |
| **speed / warp invariance** | **best ever = V8_arclen 0.922**; V8u 0.895; V7_jepa 0.867; V7_maskfsq 0.542. Oracle 0.9606, control 0.0176 | `V8_verdict_0807` |

**On speed invariance specifically:** the acceptance line quoted in `MODEL_VERSIONS.md` is 0.95 and
**nothing has ever passed it**; the `v7_eval` gate is the looser 0.90, which V8_arclen passes and
V8u fails. The V5c/CTRLb arms score 0.995 — but they were *trained against warps* and they buy that
score by discarding window content (registers buy only 1.41–1.69 cm). Warp invariance is only
comparable between arms that share the `inv_mode` training contract.

---

## 4. Fit 1 — Rate–distortion

### 4.1 Greenwich: the K-ladder (matched corpus, matched steps, matched everything else)

`E19_K16 / E23_K32×2 / E19_K96 / {E19_K128, E23_K128×2, E24_mid}` — all `cache_xwbc_lafan`,
8 000 steps, 13 dims × 8 levels, 15 trained bodies, holdout {g1_29, g1_23, talos}.
Only `n_latent` moves. Replicates averaged; sd shown where n > 1.

| bits/frame | n runs | xlate rot ° | xlate cm | 0-shot rot ° | 0-shot cm |
|---|---|---|---|---|---|
| 1 248 (K=16) | 1 | 9.257 | 4.124 | 17.10 | 9.17 |
| 2 496 (K=32) | 2 | 8.529 ± 0.093 | 3.891 ± 0.035 | 17.35 ± 0.85 | 8.31 ± 0.33 |
| 7 488 (K=96) | 1 | 8.164 | 3.839 | 17.00 | 9.08 |
| 9 984 (K=128) | 4 | **8.107 ± 0.065** | 3.833 ± 0.067 | 16.27 ± 0.62 | 8.99 ± 0.80 |

**Fits (rotation, in-corpus):**

```
power law      err = 14.01 · bits^(−0.0604)                 R² = 0.917
log-linear     err = −0.3616 · log2(bits) + 12.829          R² = 0.909
floor + power  err = 8.026 + 1.086e4 · bits^(−1.274)        R² = 0.9997   ← the one that fits
```

*(Honesty note on that R²: the floor+power form has 3 free parameters fitted to 4 points, so it is
nearly interpolating. What is load-bearing is not the R² but the fact that the two 2-parameter forms
both leave a systematic 0.21–0.22° residual with the same sign pattern — the data has a floor, and
any form without one misfits it.)*

The floor+power form is decisive: **Greenwich's cross-embodiment error asymptotes at 8.03° and no
number of bits reaches below it.** Marginal returns per doubling of the bit budget:

| segment | Δ octaves | Δ rot ° | °/octave | Δ cm | cm/octave |
|---|---|---|---|---|---|
| 1 248 → 2 496 | 1.00 | −0.728 | −0.728 | −0.233 | −0.233 |
| 2 496 → 7 488 | 1.58 | −0.365 | −0.230 | −0.052 | −0.033 |
| 7 488 → 9 984 | 0.42 | −0.057 | −0.137 | −0.003 | −0.007 |

**Knee: ~2 500 bits/frame.** Below it, one octave of bits buys 0.73°. Above it, an octave buys
0.14–0.23° and essentially nothing in centimetres (the position channel is flat from 2 496 bits up:
3.891 → 3.833 cm across a 4× budget increase, i.e. inside the seed band).

**And the shape of the code beats the size of the code.** `E20_K64_L15` uses 13 dims × **15** levels
at K=64 = 6 501 bits/frame and scores **7.807° / 3.794 cm** — *better than every 8-level run
including the 9 984-bit K=128 set (8.107 ± 0.065°)*, at **35 % fewer bits**. The 0.30° gap is 4.6×
the sd of the four-run K=128 set and 1.6× the largest observed seed-pair difference (0.185°). `E20_K96_L15` (9 752
bits) scores 7.936° / 3.704 cm. Finer quantisation per dimension is worth more than more tokens at equal bits. This is the same direction as the video
literature (`LIT_TEMPORAL_FSQ.md`: VidTok — "allocating information to channel dimensions yields
better performance than temporal dimensions at equivalent total capacity"; MAGVIT-v2 — larger
vocabulary monotonically improves reconstruction).

**Matched-corpus confirmation at production scale** (`cache_xwbc_lafan + cache_xwbc_amass_full`,
24 k steps, `vis_residual`, everything else identical — only the FSQ dim count moves):

| run | dims | bits/frame | xlate rot ° | xlate cm | 0-shot ° | 0-shot cm | ST train ° |
|---|---|---|---|---|---|---|---|
| `PROD_data13k_0809` | 13 | 9 984 | 11.129 | 4.136 | 20.9 | 9.58 | 5.527 |
| `PROD_dim16_13k_0809` | 16 | 12 288 | 11.029 | 4.133 | 20.7 | 9.29 | 5.220 |

**+23 % bits buys −0.100° and −0.003 cm in-corpus.** (`PROD_c1` at 20 dims / 15 360 bits reads
9.829° / 3.489 cm, but it also trained for 91 k steps instead of 24 k — that comparison is
confounded by a 3.8× step increase and is not a rate point.)

**And the FK duel closes it**: `FK1` at 6 144 bits/frame matches `FK2` at 12 288 to within
+0.1 % on arm p95 and −1.1 % on wrist p95. Greenwich can be halved with no measurable loss.

### 4.2 Equator: the S0/S1/S2/S3 scan

Canonical numbers from `SIZE_verdict_0807` (`v7_eval.py` ruler, 4 bodies, identical losses,
identical steps, identical everything except the code shape). These reproduce `MODEL_VERSIONS.md`'s
quoted deltas exactly (S1 +0.26 cm, S2 −0.11 cm, S3 −0.19 cm).

| run | bit/s | bits/window | key pos cm | key rot ° | energy | 0-shot g1_29 cm | 0-shot g1_23 cm | kNN body leakage |
|---|---|---|---|---|---|---|---|---|
| `S1_speed37` | 59.94 | 119.88 | 6.01 | 9.7 | 0.922 | 7.12 | 6.98 | 0.085 |
| `S0_arclen_4b` | 62.69 | 125.38 | 5.75 | 9.3 | 0.925 | 7.13 | 6.99 | 0.089 |
| `S2_c85` | 78.51 | 157.02 | 5.64 | 8.8 | 0.935 | 7.26 | 7.09 | 0.120 |
| `S3_open88` | 100.21 | 200.42 | 5.56 | 8.6 | 0.942 | 7.44 | 7.22 | 0.131 |

**Marginal cm per bit, segment by segment along the monotone ladder:**

| segment | Δ bits/window | Δ key pos | **cm per bit** |
|---|---|---|---|
| S1 → S0 | +5.50 | −0.260 cm | **−0.0473** |
| S0 → S2 | +31.64 | −0.110 cm | **−0.00348** |
| S2 → S3 | +43.40 | −0.080 cm | **−0.00184** |

**The knee is at S0 = 62.69 bit/s = 125.4 bits per 2-second window.** The first 5.5 bits below the
knee are worth **13.6×** as much per bit as the next 31.6, and **25.7×** as much as the 43.4 after
that. Fits over the four points:

```
power law      key = 10.64 · bits^(−0.1236)                 R² = 0.738
log-linear     key = −0.4932 · log2(bits) + 9.293           R² = 0.734
floor + power  key = 5.546 + 3.48e12 · bits^(−6.233)        R² = 0.875
```

*(3 parameters on 4 points again, and here it barely helps — the fitted floor lands on S3 itself,
which means the scan never got far enough past its knee to locate one.)*

The exponent −0.124 is close to Greenwich's −0.060 in the pre-knee regime, but **all three fits are
weak (R² 0.73–0.88) because the ladder has only four points and it is not a pure rate ladder** —
S1 reshapes the *speed* register (6 dims × 5 levels → 3 × 7), S2 and S3 reshape the *content*
registers. S1 is the only point that removes bits, and it removes them from a different register
than the one S2/S3 add to. **The clean statement is the segment table, not the fit.**

**The independent vocabulary ladder corroborates the knee.** `RATE_*` holds the register count fixed
and drops the FSQ levels (5⁶ → 4⁶ → 3⁶), on the same 4-body corpus:

| bit/s | jepa arm key cm | maskfsq arm key cm |
|---|---|---|
| 42.79 (3⁶) | 7.066 | 6.104 |
| 54.00 (4⁶) | 6.945 | 6.136 |
| 62.69 (5⁶) | 6.665 | **5.663** (= CTRLa) |

jepa: `key = 12.29 · bits^(−0.146)`, R² = 0.877, monotone. maskfsq is non-monotone at 4⁶ (6.104 →
6.136 → 5.663), R² = 0.566 — i.e. **the maskfsq rate ladder is not resolvable above the run-to-run
scatter**. Both arms improve as they approach 62.69 and the S-scan shows them flattening above it.

**T32 is not a matched rate point.** `T32_arclen_0809` runs 32 path tokens = 229.87 bit/s and reads
6.036 cm / 8.72° in-corpus and 5.58 cm / 10.45° zero-shot, against V8_arclen's 6.301 / 9.99 and
5.55 / 12.25. But T32 trained on `cache_xwbc_amass_full` (13 281 clips) while the whole V7/V8/S/A/
RATE line trained on `cache_xwbc_amass` (4 000 clips) — the straight-line baseline itself moves
(9.508 vs 9.242 cm), which is the tell. **Rate and corpus both changed 3.3×; the −0.265 cm cannot
be attributed.** Reported here as nominal only: −0.00079 cm/bit.

---

## 5. Fit 2 — Model size

### 5.1 Greenwich: E24, the only clean size ladder in the whole run set

`E24_small / E24_mid / E24_large` — matched rate (9 984 bits/frame, K=128, 13×8), matched corpus
(`cache_xwbc_lafan`, 15 trained bodies), matched steps (8 000), matched losses. Only
`(d_model, enc_layers, dec_layers)` moves.

| run | d_model | enc/dec | params | xlate ° | xlate cm | 0-shot ° | 0-shot cm | ST hold ° |
|---|---|---|---|---|---|---|---|---|
| `E24_small` | 128 | 3/3 | 3 671 511 | 12.200 | 5.583 | 20.1 | 11.65 | 31.733 |
| `E24_mid` | 256 | 4/4 | 18 005 775 | **8.029** | 3.908 | **17.2** | 8.97 | 28.167 |
| `E24_large` | 384 | 6/6 | 57 780 479 | 8.386 | **3.597** | 17.2 | **8.54** | 29.733 |

| segment | Δ octaves of params | Δ xlate ° | Δ xlate cm | Δ 0-shot ° | Δ 0-shot cm |
|---|---|---|---|---|---|
| 3.67 M → 18.0 M | 2.29 | **−4.171** | −1.675 | **−2.90** | −2.680 |
| 18.0 M → 57.8 M | 1.68 | **+0.357** | −0.311 | **0.00** | −0.430 |

A single power law over all three points reads `err = 100.9 · params^(−0.1435)` with **R² = 0.798
and a max residual of 1.15°** — i.e. it does not describe the data. What the data actually says is a
**hard elbow at ≈18 M parameters**: 2.3 octaves below the elbow buy 4.17°, 1.7 octaves above buy
−0.36° (the rotation channel gets *worse*; the position channel improves by 0.31 cm, which is at the
edge of the seed band). This is exactly `RUNS_INDEX.md`'s "3.67M→18M 有效, 18M→57.8M 无效", now with
the numbers attached.

Every production Greenwich sits at that elbow: `PROD_spatial_0808` 18.07 M, `PROD_c1_91k` 18.08 M,
`PT_c1dec_reenc_0812` 18.08 M. **Nobody has ever trained a spatial codec above 57.8 M, and the one
run at 57.8 M was not better.**

### 5.2 Equator: no matched size ladder exists

All 40-odd V5→V9 / S / A / RATE / CTRL / T32 runs share `d_model=256, enc=5, ctx=2, readout=2,
dec=3`. The parameter spread (8.06 M → 10.01 M) tracks the *arm*, not a size choice.

The closest thing to a size ladder is BRIDGE-A, and it is confounded:

| run | params | steps | val_ce | val_acc |
|---|---|---|---|---|
| `BRIDGE_A_pilot_0810` | 13 777 158 | 40 000 | 0.9156 | 0.6248 |
| `BRIDGE_A2_full_0810` | 20 353 798 | 40 000 | 0.6108 | 0.7412 |
| `BRIDGE_A3_prod_0810` (d=384, 6/4) | 40 006 278 | 100 000 | **0.5023** | **0.7855** |

pilot → A2 is +0.56 octaves of parameters at matched steps: **CE −0.305, accuracy +11.6 points**.
A2 → A3 is +0.97 octaves **and** 2.5× the steps: CE −0.109, accuracy +4.4 points. The A2→A3 step
cannot be split between size and steps with the runs on disk. **Report the pilot→A2 number; do not
report an A3 scaling exponent.**

**Conclusion for question 2: at matched data and matched rate, Greenwich's parameter axis is closed
above 18 M, and Equator's parameter axis has never been measured.** If a bigger joint model is to be
justified on parameters, the experiment does not exist yet.

---

## 6. Fit 3 — Data axis

Two different things get called "data" here and they behave in opposite ways. Keep them apart.

### 6.1 Data VOLUME (more clips of the same kind): dead

`E21_data500 / 1000 / 2000 / 4000` — matched architecture (K=64, 4 992 bits/frame), matched steps
(8 000), matched 15 trained bodies, only the AMASS clip pool changes. Scored on the **only
leakage-free spatial ruler we have**, `E21_data_axis_heldout.json`: 600 clips disjoint from every
training pool, 4 096 sampled frames, paired bootstrap over 575 clips.

| train clips | human→g1_29 all ° | all cm | legs ° | arms ° | wrists ° | 95 % CI on ° | paired Δ vs 500 |
|---|---|---|---|---|---|---|---|
| 500 | 24.40 | 11.03 | 16.42 | 30.10 | 36.09 | 22.37–26.45 | — |
| 1 000 | 24.43 | 11.23 | 18.68 | 29.14 | 32.59 | 22.41–26.53 | +0.03° **n.s.** |
| 2 000 | 23.25 | 11.93 | 16.07 | 28.07 | 33.78 | 21.21–25.37 | **−1.15°** (CI −1.35…−0.96) |
| 4 000 | 23.53 | 11.83 | 16.74 | 27.92 | 34.43 | 21.51–25.61 | **−0.87°** (CI −1.09…−0.66) |

**8× the data buys 0.87–1.15° out of 24.4°, i.e. 3.6–4.7 %, and 4 000 clips is worse than 2 000.**
The `unitree_h1→g1_29` direction reads the same shape (20.95 → 19.97).

This is not a training artefact — it is what the corpus contains. `R8_DIM/ANALYSIS_DIM.md §5`
measures the intrinsic dimension of the *raw* motion, no model involved:

| clips | 100 | 500 | 1 000 | 2 000 | 4 000 |
|---|---|---|---|---|---|
| 1-frame PCA 95 % var | 13 | 13 | 13 | 13 | 13 |
| 1-frame PCA 99 % var | 21 | 23 | 23 | 23 | 23 |
| 32-frame PCA 95 % var | 30 | 33 | 34 | 34 | 34 |
| 32-frame PCA 99 % var | 76 | 81 | 84 | 85 | 85 |

**500 → 4 000 clips adds exactly zero 95 %-variance dimensions.** (`R8` also calibrates the ruler on
synthetic data of known dimension and retracts `R7`'s "participation ratio ≈ 10" as an intrinsic-
dimension estimate; only the PCA thresholds are used above, and they are the estimator that is
immune to density dilution.) The training-side null and the data-side null agree.

### 6.2 Number of EMBODIMENTS: alive, and only in the zero-shot column

**Equator — the designed 2×2.** `CTRLa_maskfsq_4body` vs `CTRLb_v5c_16body` were run precisely to
separate architecture from data, and with `V7_maskfsq_0807` and `V5c_0806` they close a full
factorial. All four at 62.69 bit/s, same losses, same steps, `v7_eval.py` ruler
(`CTRL_verdict_0807`):

**in-corpus `key pos (all latents)` [cm]**

| | 4 bodies | 16 bodies | Δ(data) |
|---|---|---|---|
| maskfsq (no OAT) | **5.60** | 5.99 | **+0.39** |
| v5c (with OAT) | 7.66 | 8.00 | **+0.34** |
| **Δ(arch)** | **−2.06** | **−2.01** | |

→ architecture main effect **−2.035 cm**, data main effect **+0.365 cm** (wrong sign), interaction
**−0.025 cm**. Motion energy tells the same story: maskfsq 0.939/0.939 vs v5c 0.805/0.780 →
architecture +0.147, data −0.013. **The amplitude fix is dropping OAT, full stop** — which is
exactly the verdict `MODEL_VERSIONS.md` recorded, now with an effect size.

**zero-shot, held-out `unitree_g1_29dof` key pos [cm]**

| | 4 bodies | 16 bodies | Δ(data) |
|---|---|---|---|
| maskfsq | 7.69 | **6.34** | **−1.35** |
| v5c | 8.08 | 7.34 | **−0.74** |
| **Δ(arch)** | −0.39 | −1.00 | |

→ data main effect **−1.045 cm**, architecture main effect **−0.695 cm**, interaction −0.305 cm.
`g1_23dof` replicates: data −1.045, architecture −0.705. **Zero-shot is the data axis**, again
exactly as recorded, now with the split.

**And it replicates across 21 further runs.** Every 4-body run in the S / A / RATE / CTRLa family
versus every 16-body run in the V7 / V8 / V9 / PRODT family, on the `eval_log.json` ruler:

| corpus | n runs | 0-shot key rot ° | 0-shot key pos cm |
|---|---|---|---|
| 4 embodiments | 14 | 18.23 ± 0.75 (16.39–19.52) | 6.873 ± 0.207 |
| 16 embodiments | 7 | 12.20 ± 0.40 (11.45–12.79) | 5.759 ± 0.253 |

**A 6.03° / 1.12 cm gap at 8.0 sd of separation**, with zero overlap between the two groups across
14 vs 7 runs spanning three different architectures and five different bit budgets. This is the
single most robust effect in the entire run set.

**Greenwich — the E5 series.** The 18-body xwbc caches always train on 15 bodies, so the spatial
embodiment axis exists only in the 0803–04 `soma77/g1_30` era. Two matched pairs (same cache, same
held-out target `kuavo_s45`, only the number of trained bodies moves):

| cache | trained bodies | in-corpus `soma77→g1_30` | zero-shot `soma77→kuavo_s45` |
|---|---|---|---|
| `cache_paired_13` | 2 (`E5f_n2`) | 7.5° / 2.90 cm | **129.6° / 45.40 cm** |
| `cache_paired_13` | 13 (`E5f_n13`) | 8.6° / 3.01 cm | **43.6° / 22.20 cm** |
| `cache_paired_13v2` | 2 (`E5v2_n2`) | 8.7° / 2.68 cm | **56.7° / 35.24 cm** |
| `cache_paired_13v2` | 13 (`E5v2_n13`) | 7.7° / 2.67 cm | **44.8° / 22.95 cm** |

Same shape as Equator: **in-corpus moves by ±1° with no consistent sign; zero-shot improves by
11.9° / 12.3 cm (v2, the fixed cache) to 86.0° / 23.2 cm (v1).** The 73° disagreement between the
two 2-body runs is the honest measure of how unstable a 2-body codec is — treat the v2 pair as the
estimate and the v1 pair as an existence proof that a 2-body codec can fail outright.
(`E5_n2` → `E5_n3` on `cache_paired_multi4` reads 101.4° → 77.4° into `soma_nohand`, but the third
body is a rescaled human, so it is a weak point.)

**The mechanism is invariance, not capacity.** `R8_DIM §6` measures human / g1 / talos at intrinsic
dimension 7.9 / 7.5 / 7.0 — *retargeting the same motions onto 18 robots creates zero new intrinsic
dimensions*. `R7_LATENT` measures what it does create: cross-body same-frame cosine AUC **0.960**,
but kNN still identifies the body **71.6 %** of the time (chance 16.7 %), and `pal_talos` — the
out-of-family holdout — is identified **99.9 %** of the time while `g1_29dof` is only 72 % because it
resembles the trained unitree siblings. **More embodiments do not add information; they remove the
model's ability to lean on body identity.** That is why the benefit lands entirely on held-out
bodies.

**Two things that look like the data axis but are not:**
* `E22_synth_0805` adds 6 *synthetic* embodiments (21 total). Against `E19_K128` on the otherwise
  identical recipe: in-corpus 8.171 → 8.915° (worse), zero-shot 15.5 → 16.5° (worse). **Synthetic
  body augmentation did not buy the embodiment effect.**
* `E29_wristaug_0807` (19 bodies, wrist-augmented cache) is the second-best in-corpus run of the 8 k
  family (7.894° / 3.671 cm — only `E20_K64_L15`'s 7.807° beats it) but slightly worse zero-shot
  (17.3°). That is a different augmentation, not more bodies.

---

## 7. Fit 4 — Zero-shot generalisation vs the three axes

The whole point. Held-out-body error against each axis, at matched everything else:

| axis | Greenwich | Equator | verdict |
|---|---|---|---|
| **rate** | 1 248 → 9 984 bits/frame: 17.10 → 16.27 ± 0.62° — **inside the seed band** (E23 seed pairs differ by 0.4° and 1.7° on this exact metric). Matched-corpus PROD ladder +23 % bits: 20.9 → 20.7°, i.e. nothing. | S1 → S3, +67 % bits: **7.12 → 7.44 cm, monotonically WORSE**, r = +0.9993 | rate does **not** buy zero-shot; on Equator it actively costs |
| **size** | 3.67 M → 18 M: **−2.90° / −2.68 cm**. 18 M → 57.8 M: **0.00° / −0.43 cm** | never measured | size buys zero-shot only up to the same 18 M elbow, then stops |
| **data (embodiments)** | 2 → 13 bodies: **−11.9° / −12.3 cm** (v2 cache) | 4 → 16 bodies: **−1.045 cm** (2×2 main effect), **−6.03° / −1.12 cm** across 21 runs at 8 sd | **dominates** |
| **data (volume)** | 500 → 4 000 clips: **−0.87° of 24.4°**, and 4 000 < 2 000 | (not run) | dead |

**The rate–generalisation anti-correlation on Equator, and its mechanism.** Regressing the S-scan
against bits per window (n = 4, matched corpus, matched steps, matched arm):

```
0-shot g1_29 key_pos = +0.004035 · bits/window + 6.630     r = +0.9993
0-shot g1_23 key_pos = +0.003018 · bits/window + 6.615     r = +0.9997
in-corpus  key_pos   = −0.004425 · bits/window + 6.407     r = −0.8344
kNN body leakage     = +0.000587 · bits/window + 0.0178    r = +0.9551
```

Per +100 bits per 2-second window: **in-corpus −0.56 cm, zero-shot +0.40 cm, body leakage
+5.7 points.** Going from S1 (119.9 bits) to S3 (200.4 bits) buys 0.45 cm in-corpus and pays 0.32 cm
zero-shot, while kNN body identifiability rises 0.085 → 0.131. **Extra bits are spent memorising the
body.** `APPENDIX_ARCH_ABLATION.md` independently reports the same adversarial relationship on the
spatial side (leakage-probe accuracy 71 % @ 1 012 bits → 99 % @ 1 548 bits), and `R7_LATENT` gives
the static picture (71.6 % kNN body identifiability at AUC 0.960, 99.9 % for the out-of-family
talos). **Three independent probes, two layers, same sign.**

Caveats on that regression, stated loudly: **n = 4**; the ladder changes code *shape* as well as
code size; and the total zero-shot swing (0.32 cm) is only 1.3× the arc_alpha family's run-to-run
scatter on the same metric (0.71 cm spread across five runs at identical rate — see §8). The
**direction** is corroborated by the leakage probes and by two independent layers; the **slope** is
not something to quote to three digits.

**Which axis dominates: data, by an order of magnitude.**
On Equator's matched 2×2, going 4 → 16 embodiments buys 1.045 cm of zero-shot; the entire 67 % rate
increase costs 0.32 cm; there is no size measurement. On Greenwich, going 2 → 13 embodiments buys
11.9°; the entire 8× rate range buys 0.83° (inside noise); doubling parameters past 18 M buys 0.00°.

**And there is an interaction worth naming.** In the 2×2, architecture buys −0.39 cm of zero-shot at
4 bodies but −1.00 cm at 16 bodies, and data buys −1.35 cm for maskfsq but only −0.74 cm for v5c.
The generalisation gap (zero-shot minus in-corpus) collapses with data: maskfsq at 4 bodies is
+2.09 cm, at 16 bodies +0.35 cm. **A better architecture only cashes in its zero-shot advantage once
there are enough bodies to train it on.**

---

## 8. The noise floor — what any claim above has to clear

Nothing in this document means anything without this section.

| source of scatter | measurement | magnitude |
|---|---|---|
| Greenwich seed (identical config, seed 1 vs 2) | `E23_K128_s1/s2` | xlate **0.114°**; zero-shot **0.4°** |
| Greenwich seed | `E23_K32_s1/s2` | xlate **0.185°**; zero-shot **1.7°** |
| Greenwich, 4 runs at K=128 | `E19_K128`, `E23_s1`, `E23_s2`, `E24_mid` | sd **0.065°** in-corpus, **0.62°** zero-shot |
| Greenwich, reported band | `RUNS_INDEX.md` | ±0.9°, "后修正为 2.2°" |
| Greenwich, data-axis arm | `R8_DIM §5` | ±0.9° |
| Equator, five runs at identical rate/data (arc_alpha family) | `ALPHA_verdict_0807` | in-corpus key pos spread **0.10 cm** (sd 0.039); zero-shot g1_29 spread **0.71 cm** (sd 0.243) |
| Equator eval-script jitter (same checkpoint scored twice) | `V7_maskfsq` in V7 vs CTRL verdicts | key pos 5.99 vs 5.99; content share 0.620 vs 0.623 → **eval is essentially deterministic; all scatter above is training scatter** |
| Equator semigroup probe | `SEMIGROUP_DUEL_0811` | ratio ±0.1, only comparable within one batch set |

**Consequences you must carry:**

* The entire S-scan spans 0.45 cm in-corpus. The arc_alpha family's scatter at *fixed* rate is
  0.10 cm. So the scan's dynamic range is **4.5 seed-sigmas** — real, but the individual S0↔S2
  step (0.11 cm) is **≈1σ and should not be quoted as a result on its own**.
* Every Equator zero-shot rate effect (0.32 cm total) is **smaller than the arc_alpha zero-shot
  scatter (0.71 cm spread)**. The anti-correlation survives only because it is monotone across four
  points *and* because the leakage mechanism is independently measured.
* On Greenwich the whole 8× rate range moves zero-shot by 0.83°, against a seed band of 0.4–1.7°.
  **The honest statement is that Greenwich's zero-shot error is flat in rate**, not that it improves.
* The arc_alpha sweep itself: key pos 5.71 / 5.65 / 5.65 / 5.71 / 5.75 across α = 0 → 1 is **flat
  inside noise**. `MODEL_VERSIONS.md`'s "单调：α=1 最优" is true of **warp invariance**
  (0.835 → 0.885 → 0.894 → 0.911 → 0.925 — monotone across all five runs, total range 0.090; we have
  no seed replicate for warp invariance, so the range is reported without an error bar), not of
  reconstruction error. Do not let that claim migrate to the wrong column.

---

## 9. Fit 5 — Where the upstream is saturating, and what that means for a bigger joint model

### 9.1 The saturation map

| axis | Greenwich | Equator | headroom left |
|---|---|---|---|
| bits per frame / per second | asymptote **8.03°** fitted at R² = 0.9997; knee ≈ 2 500 bits/frame; production sits at 9 984–15 360, i.e. **4–6× past the knee**; FK1 shows half the budget is free | knee at **62.69 bit/s**; +60 % bits buys 0.19 cm at 0.0018–0.0025 cm/bit and costs zero-shot | **none** |
| parameters | elbow at **18 M**; 57.8 M is not better | **never measured** | unknown on Equator, none on Greenwich |
| training steps | 24 k → 91 k is the one visibly unsaturated knob (`PROD_dim16` → `PROD_c1`: −1.2°, −0.64 cm, ST −0.40°) but it is confounded with a rate change | 60 k is the standard everywhere; **no completed run has ever gone past it** (`PRODT_arclen_0808` was configured for 120 k but its `eval_log` stops at step 12 500, and `PRODT_jepa_0808` has no `eval_log` at all — both are dead runs, not long-training points) | **untested on Equator** |
| data volume | **0 new PCA-95 dimensions** from 500 → 4 000 clips; 8× buys 0.87° | not run | none |
| **number of embodiments** | 2 → 13 buys 11.9° zero-shot | 4 → 16 buys 1.05 cm / 6.03° zero-shot at 8 sd | **the only open axis** |
| code *shape* at fixed size | `E20` 13×15 levels at 6 501 bits beats 13×8 at 9 984 by 0.30° (~3σ) | `S1` shows a 3×7 speed register is a bad trade; `S2`'s 8-dim content is the good one | **open, cheap, untested at scale** |
| the ceiling underneath everything | paired-GT arm variance: readout probe saturates at **~30 cm p95** on the G1 wrist, decoder sits on it, two post-trains moved it < 1 % | — | **hard wall; no axis crosses it** |

### 9.2 What the field's numbers say about our operating point

From `LIT_TEMPORAL_FSQ.md`, the only external anchors that are like-for-like:

* Equator runs at **4 tokens/s, ~56–62 bit/s**. T2M-GPT (motion) is 5 tok/s ≈ 45 bit/s;
  MoMask is 30 tok/s ≈ 270 bit/s. **The one published motion rate-distortion point** is
  MoMask vs T2M-GPT: 5 → 30 tok/s improves reconstruction FID by 3.7–4.2× (HumanML3D 0.070 → 0.019,
  KIT-ML 0.472 → 0.112).
* Stable-codec/TAAE uses the *identical* FSQ (6 dims × 5 levels = 15 625) at **25 tok/s** and its own
  ablation shows quantisation is nearly free (quantised SI-SDR 4.73 vs continuous 5.08) — **the
  constraint is token rate, not the quantiser.**
* ProMAG's temporal-compression ablation: 4× → rFVD 57.9, 8× → 219.4, 16× → 255.5. Equator's
  temporal compression is **7.5×**, i.e. already in the painful zone by that curve.
* Our own rough rate-distortion lower bound (intrinsic dim ≈ 8, ~2 Hz per-dim bandwidth,
  4–6 bit/real) is **128–192 bit/s**. We run at 62.69, i.e. **2–3× short**.

**This is the one place where our measurements and the literature disagree, and the disagreement is
the interesting part.** Our S-scan says +60 % bits buys 0.19 cm and the marginal is collapsing;
the literature says 6× the token rate buys 4× the reconstruction FID. The reconciliation is that
**our scan raised bits per token, not tokens per second** — S2 and S3 widen the content register from
6 to 8 dims and open the levels, while the *register count stays at 8 per 2-second window* and the
time axis stays pooled away. T32 is the only run that raised the token count (8 → 32), and it is
confounded with corpus. **The rate knee we measured is a knee in code width, and the axis the
literature says matters — temporal resolution — has one confounded data point on it.**

### 9.3 What this implies for a bigger joint model

1. **Do not buy parameters.** Greenwich's parameter axis closed at 18 M and 57.8 M was worse on
   rotation. A joint model that is bigger *for the sake of being bigger* has no evidence behind it.
   If a size claim is wanted, the missing experiment is an Equator size ladder at matched steps —
   the cheapest high-value run in this whole report.
2. **Do not buy bits per token.** Both layers are past their knee, and on Equator extra bits are
   measurably spent on body identity (leakage +5.7 points per 100 bits/window) at a cost of 0.40 cm
   of zero-shot per 100 bits. `FK1` proves Greenwich can be **halved** (6 144 bits/frame) at no cost,
   which also halves Equator's input width and halves what Stage B must predict.
3. **Buy embodiments.** It is the only axis with a clean, replicated, 8-sd effect, it lands exactly
   where a generalist needs it (held-out bodies), and `R8` explains why it is free of the
   dimension-saturation problem: retargeting adds no intrinsic dimension, so it costs the model
   nothing to absorb, while removing its ability to lean on body identity.
4. **Buy motion difficulty, not motion volume.** `R8 §2`: dance 15.4 intrinsic dims vs walk 7.1, and
   the largest bucket in the corpus (3 023 "other" clips ≈ 75 % of the AMASS sample) is the
   *thinnest* at 6.0. 8× more of the same clips bought 0.87°. The corpus currently holds 60 dance
   and 74 jump clips.
5. **Spend the freed budget on temporal resolution, not on code width.** This is the only
   rate direction the evidence does not close: our scan varied width and found a knee; every external
   anchor that matters (MoMask, TAAE, ProMAG, VidTok) varies token *rate* and finds large effects.
   The clean experiment is T32 re-run on `cache_xwbc_amass` so it is a matched rate point against V8.
6. **Fix the contract before spending anything on the text/generation path.** `VERSION_AUDIT_0812`:
   A3 and B3 train on streams with 0.15 % overlap. Every B4 text-conditioning result and every
   TEXT_ALIGN retrieval number is class-4 invalid, and the whole B3→B6→B7→B8 autonomy series is
   class-3 (shipped path, not intended architecture). Scaling anything on top of that is scaling on
   sand.
7. **Accept the arm wall or attack it upstream.** No axis in this report — rate, size, data, steps,
   loss wiring — moved the G1 arm tail by more than 2.5 %. The readout probe says the information is
   not in the paired labels. Either fix the GMR arm retarget and re-pair, or change the yardstick to
   human-side / multi-embodiment-consensus arm acceptance. Everything else is fitting noise.

---

## 10. What we can and cannot claim

### 10.1 CAN claim — matched comparisons with an effect larger than the noise floor

| claim | evidence | effect / noise |
|---|---|---|
| Greenwich's cross-embodiment rotation error asymptotes at **8.03°** and the knee is ≈ 2 500 bits/frame | E19/E20/E23/E24 K-ladder, matched corpus/steps/losses, 8 runs, floor+power fit R² = 0.9997 | 1.15° range vs 0.065° sd at the top rung |
| Half of Greenwich's bit budget is free | `FK_DUEL_0813`, FK1 vs FK2, matched recipe, FK ruler | arm p95 +0.1 %, wrist p95 −1.1 % |
| Greenwich's parameter axis closes at 18 M | E24 small/mid/large, matched rate/corpus/steps | −4.17° then +0.36° |
| Equator's rate knee is 62.69 bit/s; marginal falls from 0.047 to 0.0018 cm/bit across the scan | `SIZE_verdict_0807`, 4 matched runs, canonical ruler | 25.7× change in marginal |
| Dropping OAT is the amplitude fix, worth **−2.04 cm** in-corpus, independent of corpus size | CTRLa/CTRLb/V7_maskfsq/V5c full 2×2, one ruler | interaction −0.03 cm |
| Number of embodiments is the zero-shot axis, worth **−1.05 cm / −6.03°** on Equator and **−11.9°** on Greenwich | 2×2 main effect + 21-run replication at 8 sd + matched E5 pairs | zero group overlap |
| More clips of the same motion buys ~4 % and saturates by 2 000 | `E21_data_axis_heldout` paired bootstrap on 600 disjoint clips + `R8` PCA-95 flat at 13 | CI-significant but tiny; two independent methods agree |
| Extra bits are spent on body identity | S-scan kNN leakage r = +0.955 · `APPENDIX_ARCH_ABLATION` 71 %→99 % · `R7` 71.6 % / talos 99.9 % | three probes, two layers, same sign |
| The re-encode switch takes decoder round-trip from 0.294 to 0.680 and costs 12 % zero-shot | `PROD_dim16_13k` → `SMALL_dim16_reenc` single-variable ablation; post-bugfix `BLAME_*.json` | 2.3× on the target metric |
| `PT_c1dec_reenc_0812`'s codes are bit-identical to C1's | `DUEL_wrist_0811` code-invariance PASS (3 bodies × 4096 frames × 2 streams) and identical enc-invariance column (0.592) | exact |
| The G1 arm tail is a paired-label wall, not a model wall | readout probe 12 cm human / 30 cm G1-paired = decoder's own 30 cm; two post-trains < 1 %; FK loss −2.5 % | four independent attacks, all null |
| Equator's warp/speed invariance has never passed 0.95; best is V8_arclen 0.922 | `V8_verdict_0807` | oracle 0.9606 |

### 10.2 CANNOT claim

* **Any Equator model-size scaling.** No matched size ladder exists. BRIDGE-A's pilot→A2 step
  (+0.56 octaves, CE −0.305, acc +11.6 pts) is the only clean size measurement anywhere in the
  temporal stack, and A2→A3 is confounded with 2.5× the steps.
* **A rate law with a trustworthy exponent on Equator.** The S-scan has four points, changes code
  *shape* not just size, and its in-corpus dynamic range is 4.5σ while its zero-shot range is *below*
  the run-to-run scatter. Quote the segment marginals, not the fitted exponent.
* **That temporal token rate is saturated.** Our scan varied bits per token at a fixed 8 registers
  per 2 s. T32 is the only run that varied register count and it changed corpus at the same time.
  The literature's strongest rate effects are all on the axis we did not clean.
* **That `PROD_spatial_0808` (5.31° / 2.50 cm) is better than `PROD_c1_91k` (9.83° / 3.49 cm).**
  Different corpora, different val sets, different difficulty. `MODEL_VERSIONS.md`'s Greenwich
  headline (6.39°/2.82 cm, 23.02°/10.52 cm) comes from a third eval script again.
* **Any spatial embodiment-count result at production scale.** The E5 pairs are 0803–04, on the
  `soma77/g1_30` caches, at 2 043 bits/frame with `fsq_stages=2` — a different regime from every
  E19+ run. Their *direction* agrees with Equator's 2×2; their *magnitude* should not be transferred.
* **Anything about text conditioning.** `VERSION_AUDIT_0812` class 4: 0.15 % stream overlap between
  what the text layer aligned to and what B4 conditions. Every `text_gain` in Table B is void.
  `BRIDGE_B5_xattn_0814` (`text_gain 0.0947`, `gen_R5 0.688`) is on the newer path and is the only
  survivor, unreplicated.
* **`content_share` as a scaling variable, ever.** It flew to 1.497 / 3.238 / 2.744 on S1 because its
  denominator collapsed to 3.57 cm. `MODEL_VERSIONS.md` retires it below ~4 cm; four rows in Table B
  are below that and are flagged in the CSV.
* **Held-out-*motion* generalisation for anything except the E21 arm.** `dual_common.split_indices`
  permutes **rows, not clips** — the spatial val split puts adjacent frames of the same clip on both
  sides. Every `eval_log.json` number in Table A is a within-clip interpolation number.
  `E21_data_axis_heldout.json` (600 disjoint clips) is the only leakage-free spatial eval on disk.
  The temporal `eval_log` inherits the same split discipline.
* **Any number from `REENC_ABLATION_0812.txt`'s round-trip table as-is.** It was written at 05:55 on
  0812; the `blame_split.py` bugfix landed and the tables were re-run at 08:06–08:10. Use the
  `BLAME_*.json` values (reproduced in §2.1) and note that the two sets disagree by roughly 2×.
* **Class-3 Equator studies as architecture results.** Phase-space, homecoming, rainbow, response
  spectrum and gauge were all measured with B's unmasked tokens fed into A3's masked-trained decoder.
  They describe the shipped system. They do not describe the intended one, and `REENC_CROSSOVER_0812`
  showed that repairing the decoder changes B3's failure *mode* (return-map slope −0.09 → +0.25)
  without changing the verdict.
* **That any of this crosses the GMR ceiling.** `APPENDIX_ARCH_ABLATION`: GMR is the invisible
  ceiling of the paired data; translation metrics must not be pushed above GMR quality. Several
  numbers in Table A are already at it.

### 10.3 The three cheapest experiments this analysis says are missing

1. **An Equator size ladder** — `d_model ∈ {192, 256, 384, 512}` at fixed 62.69 bit/s, fixed 16-body
   corpus, fixed 60 k steps. It is the one axis with no measurement at all, and every other axis is
   closed.
2. **T32 on `cache_xwbc_amass`** (4 000 clips) so that 229.87 bit/s becomes a matched rate point
   against V8's 62.69, turning the one confounded temporal-resolution data point into a real one.
3. **A 2 → 4 → 8 → 16 embodiment sweep on a modern (E19+) recipe.** The only live axis is currently
   supported by one 2×2 at 4-vs-16 and a pair of 0803-era runs. A four-point sweep would give it a
   curve instead of two points, and it is the number a bigger joint model would be justified on.

---

### Provenance

Every number above is traceable to one of: a `config.json` field, the final entry of an
`eval_log.json`, a named `*_verdict_*` / `BLAME_*` / `FSQ_USAGE_*` / `*_axis_heldout.json` dump, or a
named section of a written analysis. No run was re-executed and no GPU was touched. Runs whose only
numbers live in a `.log` rather than a json (`E0a_*`, `E0b_*` except `final`, `E0c_single`,
`P1c_*` eval dumps, `BRIDGE_*.log`) were **not** used in any fit and are absent from both tables —
they have no `config.json` in the 119, or no parseable eval, and folding log-scraped numbers into a
scaling fit would have been the exact kind of unmatched comparison this document is about.
