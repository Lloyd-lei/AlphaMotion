# Surface B integration contract

The current mixed-joint contract is detailed in `../prismatic/README.md`. Both A
and B now support verified slide joints and fixed bases. The old arm-only Franka
asset is archived; active training uses `relay_franka_full_v2`, J11 including both
independent moving fingers.

`SurfaceMegatron` extends the sibling `base.pilot_model.ConditionalMegatron`.
Forward/decode arguments and the `(global_rot6, raw_hpos, contact_logits)` return
are unchanged. Both FSQ branches and the shared target decoder remain. Surface
position is obtained only from `predict_bound_surface(rot6, g)`: native-unit FK
followed by a verified binding. There is no independently predicted surface head.

```python
from surface import (SurfaceMegatron, load_surface_asset, scale_surface_units,
                     expand_surface, surface_losses, surface_metrics)
static = load_surface_asset(path, frozen_vectors_by_text_key, J,
                            qwen_cache_sha256=actual_cache_sha256)
# External hand/arm assets are meters; XWBC assets already are centimeters.
if external_meters_to_cm:
    static = scale_surface_units(static, 100, destination_unit='cm')
g['surface'] = expand_surface(static, batch)
# g['parents']: one parent-first topology; g['kinematic_rest_off']: native FK offsets
# g['pose_scale']: [B] static maximum root-to-joint path, in the SAME native units
# g['rest_off']: separate legacy descriptor units, never reused implicitly for FK
prediction = model.forward_kinematic(x_source_pose9, g_source, g_target, **same_A_args)
aux = surface_losses(prediction['rot6'], target_pose9[..., :6], g_target,
                     pred_slide_native=prediction['slide_native'],
                     target_slide_native=actual_target_slide_native)
loss = base_loss + aux['total']
common_metrics = surface_metrics(prediction['rot6'], target_pose9[..., :6], g_target,
                                 pred_slide_native=prediction['slide_native'],
                                 target_slide_native=actual_target_slide_native)
```

Actual frozen Qwen vectors are required by the asset loader. Diagnostic
`allow_unencoded=True` is explicit and must not be used for B training. Text
requirements are frozen in `required_surface_texts.json`: 217 entries keyed by
SHA256 of the exact prompt; each entry carries its body/link/patch association.
Prismatic v2 adds10 exact prompts in `required_surface_texts_incremental_prismatic.json`;
`required_surface_texts_v2.json` is the227-entry union. The original217-entry
file is unchanged. Original whole-file SHA256: `fda606c98c5bd7a6188827d309af7543ebd17297bbd06d6fe9f4d54ab15af0ed`.
The root task owns actual GPU encoding; no Qwen encoder or optimizer was run here.

Static patches use real compiled triangle geometry, original-file and compiled-mesh
hashes, fixed face IDs and barycentric coordinates. Every candidate binding was
tested in neutral plus seven deterministic random joint states. This handles the
XWBC merged rotation frames and the external hand/arm `R(q) Rrest.T` rebasing.
`patch_build_receipt.json` has no build failures. Each asset records omissions and
the forward kinematic evidence; names alone are never accepted as binding proof.

| Coverage | Bodies | Patches / body | Native length |
|---|---|---:|---|
| XWBC | unitree_g1_29dof, unitree_g1_23dof, booster_t1, booster_k1, fourier_n1, pnd_adam_lite, hightorque_hi | 96 | cm |
| Articulated hand | adroit_pen, J25 and 24 hinges | 96 | m |
| Arm and sliding gripper | relay_franka_full_v2, J11 complete tree | 96 | m |

G1 variants share a physical model. Some merged-cache geometries cannot bind
rigidly to a cached rotation frame and are explicitly omitted. Franka's full v2 includes both slide
finger bodies and meshes, with verified mixed FK and independent true q values.
The fixed end-effector marker is retained. No UniMate/SMPL skin binding is invented.
These are geometric/kinematic checks, not grasp success or robot execution.

The source encoder consumes J+P tokens. Visible source pose binds dynamic patch
position/normal; any hidden binding joint suppresses that dynamic observation.
Static geometry and frozen local functional text remain available. Attachment
hop distances bias joint/patch attention. Shared target queries attend to their
static patches. `surface_cost` reports allocated/valid tokens and attention cells.

Fixed loss weights for all sizes: position 1.0, normal 0.1, penetration 1.0,
contact 0.1. Position is area-weighted point distance divided by static reach;
normal is area-weighted `1-dot`. Ground penetration is squared depth divided by
metric static reach. Ground requires an explicit trusted Y=0 datum and metric
root height; otherwise its loss is zero. Contact also requires source SHA256,
valid source labels and valid ground partner. Missing contacts/pressure are never
negative labels. Until such labels exist, pass contact arguments as None. Object
contact/tactile supervision is not implemented. Patch GT is the same real mesh
bound to target skeleton GT: it is geometry supervision, not an independent
surface scan or contact observation.

`surface_metrics` accepts **either A or B** rotation predictions and returns the
same fixed-point native/normalized/metric errors, normals, validity counts and
normalization metadata. Unknown physical scale yields NaN metric results. Metrics
are root-relative; any global-root translation metric must be reported separately.
All motion normalizers are geometry-only. No holdout motion is used for binding,
scale fitting, point selection or semantics.

| Size | B main | Judge | Added surface parameters |
|---|---:|---:|---:|
| S: d128, 6/6 | 6,817,237 | 134,412 | 249,872 |
| M: d256, 6/6 | 26,542,293 | 265,484 | 728,976 |
| L: d384, 8/8 | 76,926,517 | 396,556 | 1,437,456 |

Counts exclude the frozen 595,776,512-parameter Qwen tower. A/B differences are
the complete surface information + token/attention structure + loss package;
the comparison alone cannot attribute changes to one ingredient. Separate modes
`drop_surface`, `static_only`, `no_patch_semantics`, `no_attachment_bias` are
available for diagnostic ablations. The last removes distance structure for mixed
encoder pairs; their learned hop-zero constant remains.

Validation: `test_surface_model.py` has 17 CPU numerical/gradient/masking tests;
`test_real_surface_assets.py` tests all nine actual assets with random-weight small
models, neutral geometry, no encoded text, and no optimizer. Reports contain
surface-branch gradients, ablation output changes, neutral reconstruction errors,
and zero hidden-GT leakage. They establish wiring/numerics, not learned quality.
Global-rotation-to-actual-q projection is not implemented here; supplied verified
q-to-FK metadata is insufficient by itself to claim mechanical executability.
