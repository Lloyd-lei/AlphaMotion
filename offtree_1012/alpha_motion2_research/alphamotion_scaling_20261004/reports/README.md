# Six controlled AlphaMotion experiments

This directory contains the evidence for the separately authorized A/B × S/M/L experiments. The fixed resource-download monitor is a different task; its cache progress does not establish dataset content verification or model quality.

The first common milestone is 3,000 optimizer steps, seed 0, effective batch 128. Every model starts from random motion-model weights. Frozen Qwen features are shared. The earlier warmstart pilot is preserved in its own research directory and is excluded from these six results.

| Size | A main parameters | B main parameters | Auxiliary judge, each |
| --- | ---: | ---: | ---: |
| S | 6,567,365 | 6,817,237 | 134,412 |
| M | 25,813,317 | 26,542,293 | 265,484 |
| L | 75,489,061 | 76,926,517 | 396,556 |

Frozen Qwen has 595,776,512 parameters and is counted separately. Both A and B have the same typed hinge/slide geometry and prismatic prediction path. B adds stable mesh patch observations, text features, attention and geometric losses. A/B therefore compares a package of information, architecture and losses; it does not isolate the causal effect of surface information.

## Read the results

- `../logs/queue_state.json`: live training queue state. Pair the process ID with start ticks, boot ID and command line before any process action.
- `../runs/<run>/train.jsonl`: one record per real optimizer update, including actual selected rows, sampling digest, loss denominators, effective tokens, estimated dense FLOPs and synchronized training time.
- `../runs/<run>/result.json` and `exit_status.json`: successful completion and actual exit state. A checkpoint alone is not a completed run.
- `live_results.json`: derived sampling/exposure checks and unrefined checkpoint metrics. `live_results.png` plots shared loss terms, because total A/B losses have different extra terms.
- `../runs/<run>/eval/`: exact common checkpoint evaluations. Individual-row results do not establish temporal or world-space slip quality.
- `../evaluation/results/`: formal continuous-clip and manipulation refinement results, when executed. Integration/diagnostic folders are separate and must not be reported as formal experiments.
- `../validation_receipt.json`: all six real CUDA gates, bound to exact data, configuration and source hashes, with zero optimizer updates during acceptance.
- `hidden_observation_execution_receipt.json`: actual B-S checkpoint audit. All four strict pose/SE3 protocols preserve outputs and effective encoder/patch/query inputs when hidden answers are changed. The separate reference recipe with explicitly observed contact is labeled separately.
- `../evaluation/execution_plan.json` and `../logs/evaluation_queue/`: the sealed evaluation follower waits for all six successful training exits and the same GPU lease. Its 18 jobs cover final-checkpoint manipulation, final-checkpoint continuous locomotion, and the seven earlier locomotion checkpoints for every run.
- `summarize_manipulation.py`: reads saved curves for every integer refinement step, keeps original and strictly limit-valid reference cohorts separate, and forms paired A/B episode-cluster intervals. Missing or failed cases remain in the comparison.
- `render_locomotion_pairs.py` and `render_manipulation_pairs.py`: render already saved, matched FK results with shared spatial limits. Examples follow a fixed case/frame rule independent of quality. Palm/gripper detail views use the actual named model subtree. These are kinematic visualizations.

## Data and tasks

The sealed loader includes 16 XWBC bodies, 10 UniMate bodies, Adroit pen and the full Franka arm with two independently recorded slides. Four complete XWBC bodies are held out. XWBC supplies paired cross-body labels; UniMate, Adroit and Franka are same-body reconstruction/completion tasks. Additional dexterous hand or end-effector cross-body mapping is not established by these data.

The sampling domain called `unimate` in the inherited training code also contains the separately identified Adroit and Franka tasks. Each direction records its actual dataset. No cross-body pairing is invented for these sources.

Pure SE(3) completion provides only the named sparse target poses, static body information and fully hidden source pose. Full-source mapping provides the source pose and its observed surface geometry; no dynamic target surface/contact is provided. The old `reference_mask_mix` training task additionally allows declared source contact observations. All formal no-contact evaluation protocols are labeled separately from that training task.

Metric XWBC/Adroit/Franka results can use millimetres after explicit conversion. UniMate remains in canonical units or static-reach normalization. Full-body error in a sparse IK problem is a posture diagnostic, since sparse conditions do not specify a unique full-body answer.

Franka's original recordings include small soft-limit excursions. Original reference clips remain unchanged and are reported separately from control intervals whose every covered native frame satisfies the known joint bounds. Neither category proves collision freedom, grasp success, dynamic feasibility or physical-robot execution.

## Evaluation and interpretation

Training checkpoints are saved at steps 100, 250, 500, 1,000, 1,500, 2,000, 2,500 and 3,000. Fixed-exposure comparisons require matching actual sampling decisions, not just a shared seed. Fixed-compute comparisons use common covered budgets and identify an earlier saved checkpoint as under budget when it is not equal; they do not invent interpolated results. Dense FLOPs are analytic estimates with their exclusions recorded. Measured training time excludes loading, checkpoint I/O and evaluation.

Manipulation evaluation records every integer refinement step from 0 through 100. Free posture, projection to verified native DOFs, constrained optimization and an identical optimizer starting at an actual default valid q are kept separate. Only the visible SE(3) constraints and declared valid constraints enter optimization; full-body reference motion enters detached diagnostics only. A known-joint-limit check is not a physics or grasp check.

The six seed-0 experiments are finite-budget comparisons. Clip-cluster confidence intervals describe variation across the recorded clips and do not substitute for multiple training seeds or establish a scaling law. Any continuation must be a separately versioned common schedule for all six models, after examining the initial curves, remaining time and disk capacity.

## Storage and preservation

The fixed download plan is 200,505,865,539 bytes after repository-content deduplication. Remaining downloads plus 15 GiB reserve and 1 GiB overhead have priority. Archive extraction needs a separate estimate. Training and evaluation own a single GPU queue lock. Original downloaders, services, oldpilot artifacts and old checkpoints are preserved.

Detailed paths, hashes, commands and per-file evidence belong in the JSON receipts and logs, while the existing Google Docs experiment page contains the readable conclusions, tables and plots.
