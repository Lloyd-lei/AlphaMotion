# Shared A/B slide kinematics

The new scaling A and B models both have the same shared joint-type embedding,
static axis/limit projection, visible-source slide projection, and target slide
head. No parameters are body-specific. Original pilot files are untouched.

```python
prediction = model.forward_kinematic(
    source_pose9, source_geom, target_geom,
    source_hidden=hidden, source_contact=source_contact, condition=sparse_SE3,
    source_slide_normalized=source_q01)
# Explicit per-call result, safe to retain across subsequent calls:
# rot6 [B,J,6], raw_hpos [B,J,3], contact_logits [B,J]
# slide_normalized [B,J], slide_native [B,J], slide_valid [B,J] bool
# joint_position [B,J,3], joint_rotation [B,J,3,3]
```

`decode_kinematic(zp,zr,g,condition=None)` returns the same fields. The legacy
`forward` three-tuple and `encode/decode` remain for the rotational XWBC auxiliary
recipe. That auxiliary cycle does not cover prismatic data; all prismatic training
and evaluation must use the explicit dictionary. There is no `last_prismatic`
cache. The historical `raw_hpos` head remains separate from actual FK positions.

Static `g` fields:

| Key | Shape | Meaning |
|---|---|---|
| `joint_type` | B,J | unknown0 / fixed1 / revolute2 / prismatic3 / root4 |
| `prismatic_axis` | B,J,3 | true unit axis in canonical rest basis before parent G |
| `slide_lower/upper/reference` | B,J | native-length limits and q used to form rest offsets |
| `kinematic_fixed_rotation` | B,J,3,3 | verified C for `G_child=G_parent@C`; alias `prismatic_fixed_rotation` accepted |
| `kinematic_rest_off` | B,J,3 | native-length rest offsets, independent of descriptor units |
| `parents` | J or B,J | shared parent-first topology |
| `pose_scale` | B | native-length, geometry-only static reach |
| `root_fixed` | bool or B bool | explicit verified fixed-base flag; default false |

Static arrays may also omit the batch dimension. Metadata without slide types
retains the rotational path and returns all-zero slide values with false masks.
Missing active slide axes/limits or source-visible q fail loudly. Unknown slots
may carry NaNs in metadata; inactive values are sanitized before arithmetic.

Normalized source/target q uses `(q-lower)/(upper-lower)`. Visible source values
must be finite, but recorded soft-limit excursions remain outside [0,1]; they are
not clipped. Hidden source q is zeroed before projection, then its token
contribution is masked after the MLP. Target GT q is never passed into the model.
The shared query head produces sigmoid q01 and maps it to the actual limits.
Use normalized slide L1 with weight1.0 on valid target slide slots, identically
for all six runs. This strict prediction head cannot reproduce source soft-limit
violations; report their number and the resulting representational error floor.

`mixed_forward_kinematics(rot6,g,slide_native=None)` returns a dictionary with
derived rot6, joint_rotation, joint_position, slide_native and slide_valid.
Joint positions are root-relative **native length units**:

```
G_fixed_or_slide[j] = G[parent[j]] @ C[j]
P[j] = P[parent[j]] + G[parent[j]] @ (
    kinematic_rest_off[j] + axis[j] * (q[j] - q_reference[j]))
```

The independent rot6 prediction at fixed/prismatic slots has zero effect and
zero gradient. For verified Adroit and Franka fixed bases, root_fixed derives
canonical rebased root G=I without reading GT. XWBC/UniMate remain free-root.
Rotational hinge outputs otherwise remain free global rotations; this module
does not implement their inverse q projection or prove mechanical executability.

All Franka dimensions use the verified v2 J11 order: link0…7, leftfinger8,
rightfinger9, exact source end-effector marker10. Two actual independent slide
qpos entries7/8 have limits[0,.04]m. No mimic equality exists in the source model
and none is invented. Fixed/root inheritance and mixed FK were independently
verified against actual MJCF states. Body-origin aperture is signed separation
along the parent-rotated left slide axis; it equals q_left+q_right. This is not a
measurement of contact-pad clearance or grasp success.

For m→cm, multiply offsets, pose_scale, q, q limits and q_reference by100; leave
axes, rotations and q01 unchanged. Convert surface geometry with
`scale_surface_units(...,100,destination_unit='cm')`. Both A and B call the same
surface metrics, using the same fixed mesh points:

```python
surface_losses(pred_rot, target_rot, g,
               pred_slide_native=prediction['slide_native'],
               target_slide_native=actual_target_q)
surface_metrics(pred_rot, target_rot, g,
                pred_slide_native=prediction['slide_native'],
                target_slide_native=actual_target_q)
```

The new active asset `surface/patches/relay_franka_full_v2.json` includes both
moving finger meshes. The arm-only v1 file remains archived, and must not be
sampled as a second dataset. Frozen original217 text requests are unchanged;
`required_surface_texts_incremental_prismatic.json` contains10 additional exact
prompts. Their union is `required_surface_texts_v2.json`. Actual Qwen encoding is
owned by the root task; tests use explicit unencoded text diagnostics.

`test_prismatic_model.py`: seven targeted CPU tests verify actual MJCF-derived
states, fixed/root inheritance, limits, hidden-source non-leakage, A/B gradients,
independent fingers/aperture/moving meshes, explicit GT q requirements and unit
invariance. `numerical_review.json` records the observed errors and gradients.
These checks use random small models and diagnostic backward, never an optimizer.

AMP boundary: static mechanics validation, derived rotations, slide decoding,
mixed FK, surface binding, surface losses and surface metrics disable device
autocast internally and explicitly use float32 geometry operands. Neural
projections/attention remain in the caller's bf16 context. The regression script
`test_autocast_geometry.py --device cuda --batch-size 128` requires real CUDA
and has no CPU fallback or optimizer. Its CPU bf16 run verifies exact AMP-on/off
geometry and gradient parity plus bf16 neural projections for actual Franka
geometry. The root task owns CUDA acceptance before training.

| Size | A main | B main | Judge | Common added slide parameters |
|---|---:|---:|---:|---:|
| S d128 6/6 | 6,567,365 | 6,817,237 | 134,412 | 18,561 |
| M d256 6/6 | 25,813,317 | 26,542,293 | 265,484 | 69,889 |
| L d384 8/8 | 75,489,061 | 76,926,517 | 396,556 | 153,985 |
