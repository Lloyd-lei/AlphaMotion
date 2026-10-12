# AlphaMotion early scaling experiment

This directory contains six seed-0 A/B × S/M/L runs, their actual data lineage,
and matched kinematic evaluations. A uses the shared skeleton model; B adds
surface patches, semantic features and geometric losses. It is an early,
single-seed comparison, not an established scaling law or robot validation.

All six use the same expanded data mixture. There is no XWBC-only control,
so these runs do not isolate the causal gain from adding UniMate. The older
warmstart pilot is not a strict data-ablation control.

## Current state

All six runs completed 6,000 optimizer steps (3,000 initial + 3,000 common
second phase). Training and all 14 fixed phase-2 evaluation jobs exited cleanly.
The 24 final evaluation summaries match the registered case sets, with no case
execution failures. These software/coverage checks do not imply model success.

Final strict-reference P1 actual-q refinement at 100 updates gives Adroit
40.62–76.56% and Franka 0–55.60% episode-macro all-target passing-frame rates
(1 cm AND 5 degrees). These are seen-body completion on three heldout source
episodes, not trajectory/grasp success or zero-shot hand-to-gripper mapping.
Franka B-L's three selected valid subclips pass 15/35, 14/31 and 26/33 frames;
the equal-episode mean is 55.6021%, distinct from pooled 55/99 = 55.5556%.
Its all-frame mean is 10.85 mm / 10.88 degrees. A-L has a similar mean position
error (10.73 mm) but a lower passing-frame rate (15.15%); B-L is not universally
more accurate. At 20 P1 updates, all Franka models still have 0% passing frames.

Native Franka opening has not recovered: at 6,000 steps all M/L networks have
100% exact slide saturation in the strict-control diagnostic; S outputs are
near saturation. All six unrefined opening errors are about 12.142 mm.
Real-q refinement improvement must not be presented as native-output recovery.

XWBC whole-body-heldout paired mapping and UniMate whole-body-heldout same-body
reconstruction are reported separately. XWBC's 24 model/body combinations have
8 lower and 16 higher mean errors at 6,000 versus 3,000. These are descriptive
error directions, not 24 independent trials or a significance test. There is no
cross-task consistent monotonic size benefit in this one-seed experiment.

The second phase restores each own model and Adam state, uses new shared NumPy
streams and a common cosine learning-rate restart, and applies one declared
numerical sigmoid-backward correction to all six models. Original NumPy states
were not saved, so this is not an exact uninterrupted continuation. Changes from
3,000 to 6,000 cannot be attributed solely to additional steps.

Final human-readable results are in the existing Google Docs experiment tab:
https://docs.google.com/document/d/1gH9WaPSgqUcTdrPYeBCNaLkWrVoG0qzNsiJLPVsniBU/edit?tab=t.2w3c93da5d23
The 77-page export's experiment pages 22–27 passed full-page visual review;
all 10 other native tabs are unchanged. Source cases, failed-threshold frames,
P0, P1 and raw-versus-actual-joint results are retained.

## Evidence entry points

| Area | Entry |
|---|---|
| Data and split lineage | [data_manifest.json](data_manifest.json), [data/data_manifest_v2.json](data/data_manifest_v2.json) |
| Initial training | [logs/queue_state.json](logs/queue_state.json), `runs/*/result.json`, `runs/*/train.jsonl` |
| Original P0 | [logs/evaluation_queue/state.json](logs/evaluation_queue/state.json), [evaluation/execution_plan_io_v2.json](evaluation/execution_plan_io_v2.json) |
| Supplemental P1 | [evaluation_p1/preregistration.json](evaluation_p1/preregistration.json), [evaluation_p1/logs/state.json](evaluation_p1/logs/state.json) |
| Common second phase | [continuation_v3/execution_plan.json](continuation_v3/execution_plan.json), [continuation_v3/logs/queue_state.json](continuation_v3/logs/queue_state.json) |
| Restoration and numeric gate | [continuation_v3/validation_receipt.json](continuation_v3/validation_receipt.json), [continuation_v3/numeric_repair_receipt.json](continuation_v3/numeric_repair_receipt.json) |
| Second-phase evaluation | [evaluation_phase2_v2/execution_plan.json](evaluation_phase2_v2/execution_plan.json), [evaluation_phase2_v2/logs/state.json](evaluation_phase2_v2/logs/state.json) |
| Final compact numerical report | [compact_final_review.md](evaluation_phase2_v2/reports/compact_final_review.md), [p1_step_0_20_100.json](evaluation_phase2_v2/reports/p1_step_0_20_100.json), [Franka episode counts](evaluation_phase2_v2/reports/franka_BL_episode_counts.json) |
| Training costs and common-budget comparisons | [two_phase_results.json](reports/two_phase_results.json), [completion receipt](reports/phase2_training_completion_receipt.json) |
| Final curves / paired visuals | `evaluation_phase2_v2/reports/{locomotion,manipulation,slide,document_figures,locomotion_visuals,manipulation_visuals_P0,manipulation_visuals_P1}/` |
| Actual visual review | [final_visual_review.json](evaluation_phase2_v2/reports/final_visual_review.json); 9 decoded frames per actual MP4, not full playback |
| Final native Doc / PDF QA | `reports/final_document_20261004/`; other tabs preserved, affected pages inspected |
| Native-joint oracle diagnostic | [reports/native_q_oracle_diagnostic_v1.json](reports/native_q_oracle_diagnostic_v1.json) |
| Saturation failure diagnosis | [reports/continuation_prismatic_gradient_diagnostic.json](reports/continuation_prismatic_gradient_diagnostic.json) |
| Initial curves and paired renderings | `reports/locomotion_comparison/`, `reports/manipulation_comparison/`, `reports/manipulation_visuals_labeled_P0/`, `reports/manipulation_visuals_labeled_P1/`, `reports/locomotion_visuals/` |
| Resource downloads, independently monitored | [latest resource evidence](../../pretraining_foundation/logs/resource_monitor/heartbeat_latest.json) |

## Reading the comparison

- The 28-body fixed data manifest has 20 training bodies and 8 whole-body
  heldouts. XWBC provides paired human-to-robot examples; UniMate, Adroit and
  Franka provide same-body reconstruction or completion. These hand/arm results
  do not establish cross-end-effector transfer.
- Metric bodies use verified metric lengths. UniMate canonical units must not
  be labeled millimetres. Confidence intervals resample source episodes, not
  independent training seeds. Three heldout UniMate bodies each have only one
  source episode, so their episode confidence intervals are unavailable.
- P0 preserves the original objective. P1 fixes its acceptance-scaled weighting
  and normalizes actual joint coordinates. Both keep the same learned initial
  output, conditions, cases, 0–100 update budget, and default-posture control.
  P1 is a supplemental solver protocol, not replacement evidence for P0.
- Free global joint rotations can fit endpoints without forming a realizable
  joint posture. Read native-joint projection, constrained refinement and
  gripper opening measurements before making manipulation claims.
- The initial B-M/A-L/B-L Franka slide sigmoid outputs saturated exactly at 1
  in float32. The correction preserves every forward value and computes the
  algebraically identical derivative without `1 - sigmoid(x)` cancellation.
  A nonzero derivative does not demonstrate recovery. The second-phase queue
  records matching pure-SE(3) Franka logits, exact saturation rates and opening
  errors at 3,000 / 4,500 / 6,000, plus final P0/P1 real-joint evaluations.
- Logged optimizer times are observed synchronized training times, not an
  exclusive hardware benchmark. Initial B-L includes the separately documented
  CPU diagnostic overlap. FLOPs are model-based estimates.

## Operational boundaries

Training and evaluation use one shared GPU lock and identity receipts. Never
restart from a PID alone. `continuation_v2` retains the failed uncorrected
restoration gate; `evaluation_phase2` retains an unlaunched superseded plan.
The completed versions are `continuation_v3` and `evaluation_phase2_v2`; do not restart them.

The combined fixed download plan, 15 GiB safety margin and 1 GiB metadata overhead
remain protected. Training and evaluation have separate bounded allocations;
archive extraction is not included. Existing downloads and services are not
stopped or modified. Full cache files, partial transfers and content-verified
resources are different states. SMPL-X reuse is verified; commercial Truebones
raw motion has not been acquired.

No result here establishes contact, grasping, dynamics, simulation or physical
robot success. Independent surface-contact training labels remain unavailable.
## Matching source-mesh kinematic rollouts

Open `reports/rollout_mesh_v1/index.html` locally. The package contains 24 matched
Adroit/Franka full/detail videos and one G1 paired video, versioned offline
bundles, and the reusable `rollout_tools/` CLI. Read
`reports/rollout_mesh_v1/README.md` for timing, camera policy, mesh gaps and
reproduction commands; `visual_qa.json`, `browser_qa.json` and `docs_qa/` hold
verification receipts. These are kinematic replays, not simulation execution.
