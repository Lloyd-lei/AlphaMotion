# Hidden-observation audit

The strict pure-SE3, generic partially observed pose, and full-source protocols
passed the new joint mutation check. All eight output tensors and all captured
effective input/query tensors were unchanged (maximum difference0). This receipt
uses actual Franka MJCF geometry/states, a small random-weight B model, unencoded
text and synthetic contact stress labels. It is a wiring audit, not a claim about
the live checkpoint. The same script can read the actual sealed data and B
checkpoint on CPU through `--actual-root` and `--checkpoint`.

`audit_hidden_observations.json` records four cases: sparse-only with2 selected
SE3 values, generic masked with2 values, full-source with0 target conditions, and
full-source with2. Each keeps the visible source and supplied SE3 fixed, replaces
hidden source pose/q with NaNs, changes hidden source contacts, poisons all
unselected target pose and target q/contact values, and injects unusable dynamic
fields into target geometry. It captures pose/contact/slide embeddings, first
encoder input, patch geometry/motion/semantics, target patch attention, and
condition projection. Interleaving an unrelated forward and poisoning the old
`last_pos` attribute also leaves the repeated query unchanged.

The raw API still carries full source pose/q plus a hidden mask; the invariant
is the effective computational input after masking. It would be inaccurate to
claim the raw source tensor itself is invariant in partial/pure modes.

Precise source evidence, relative to the frozen scaling root:

| Path and lines | Boundary |
|---|---|
| `base/pilot_data.py:208–236` | Pure source pose mask is all-hidden; generic hidden contacts become NaN; pure contacts are None; only selected target joints enter condition values. |
| `surface/surface_model.py:85–106` | Pose is zeroed before source projection/binding. Posed patch points and normals are recomputed from that masked source. A patch is observed only when all bound joints are visible; masking also occurs after its MLP. |
| `prismatic/kinematics.py:118–130` | Hidden source slides are masked before projection; no target slide is read. |
| `surface/surface_model.py:69–83` | Target patch attention uses static rest geometry, binding, static semantics and topology only. |
| `base/pilot_model.py:113–151` | Decoder accepts latent codes, static target geometry and explicit sparse condition. It predicts slide q; FK reads that predicted q. |
| `orchestration/scaling_data.py:19–30,118–130` | Body geometry is built from static descriptors/typed mechanics and expanded static patch assets; no motion row is inserted into geometry. |
| `orchestration/train_scaling.py:24–25,119–132` | Actual forward wrapper passes the declared inputs. The existing real-data CUDA gate holds conditions fixed while replacing source pose/q with NaNs, and checks joint-position equality. The new audit broadens this to all outputs and intermediate inputs. |

Existing tests already covered hidden pose and zero hidden gradients
(`surface/test_surface_model.py:80–92`), hidden prismatic q and zero gradients
(`prismatic/test_prismatic_model.py:72–82`), and real-asset pose occlusion
(`surface/test_real_surface_assets.py`). The new report adds the joint
target/contact/cache mutation and all-output/query checks.

The reference XWBC mask mixture is accurately recorded as
`observed_contact_extra_task`, shared by A and B. It intentionally retains source
contact observations at pose-hidden joints (`base/recipe_training.py:29–35`).
The actual training dispatch invokes this helper for all8 XWBC main directions
(`base/recipe_training.py:78–81`, called by
`orchestration/train_scaling.py:270–275`). Configured XWBC step probability is.55
and per-row masking gate probability is.30; `cask:.15,cgive:.10` also use contact
to select masks. These are configured rates, not measured live frequencies.
The all-valid synthetic contact probe sampled136 hidden rows out of512 and
retained3894 hidden contact joint-frames; it is not a dataset prevalence estimate.

Changing only hidden-joint contact distance by.05 in this extra task changes
the `con_emb` input by.05. The fresh model's contact projection is zero-initialized,
so this particular untrained model's encoder/output values do not yet change.
The receipt explicitly records that fact instead of interpreting an output
plateau as contact independence. An actual checkpoint probe records contact
embedding, first encoder token and output sensitivity separately. This additional
task is permitted and is not labeled a pure-SE3 result.

Only `reports/` files were added. No sealed model/data/orchestration source,
configuration, training process, GPU or remote service was changed.
