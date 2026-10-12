# Real geometry assets for the G1 to H2 case

All PNGs use real robot visual meshes and CPU FK, rendered through MuJoCo EGL. No dynamics stepping, training, service changes, or robot execution occurred. The AlphaMotion forward pass belongs to ../alpha1/; this renderer consumes that result.

- G1: `g1_reference_frame_{1800,2000,2600}_mesh_skeleton.png`. Raw XWBC retargeted reference from `fallAndGetUp3_subject1`, 30 fps. These are the actual three source model-input poses, independently verified against its cached FK.
- H2: `h2_predicted_frame_{1800,2000,2600}_mesh_skeleton.png`. Actual fresh-MJCF query predictions. Rotations and FK positions are used directly, without hinge fitting or IK. Root is at zero as a visualization gauge. No root XY, floor contact, control, or zero-shot result is implied.
- Go2: `go2_reference_mesh_skeleton.png`; provided model keyframe 0, clip=null. Highlight `FL_calf_joint` / `FL_calf`, q=-1.8 rad, local axis [0,1,0].
- Panda: `panda_reference_mesh_skeleton.png`; exact Franka Emika Panda, not FR3, provided keyframe 0, clip=null. Highlight `joint4` / `link4`, q=-1.57079 rad, local axis [0,0,1]. Its finger slides are omitted by the legacy mechanical descriptor. This is not a completed Panda adapter run.

Go2 and Panda are independent input-type reference cards, not simultaneous motion inputs or evidence of three-body fusion. Each `*_projection.json` carries precise image pixels, camera, actual joint axis projection, and crop mapping. `*_joint_closeup.png` is a 600px crop; coordinates refer to the full 1000px image unless explicitly named closeup. Transparent PNGs and skeleton-only PNGs are also supplied. Skeleton lines deliberately remain visible through the mesh.

The H2 MJCF is an existing local conversion. Its 34 meshes match pinned official Unitree git blobs exactly; all 31 movable joint origins and axes match the pinned URDF up to recorded compiler rounding. The fresh 36-slot descriptor preserves actual link names. It merges co-located waist_roll/torso rotations, so direct prediction rendering omits only the unresolved waist_roll_link visual mesh. Torso and all other visual links remain. Unprojected links can disconnect because this is the network result. Full qpos reference renders have no omitted link.

## Reuse on ArenaLabs / lloyd

Python: `/home/arenalabs/miniconda3/envs/ardy/bin/python`. Set `MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1` before import. Run `build_case_assets.py` in this directory to rebuild the displayed PNGs, reference arrays, and manifest. Original mesh trees, dataset caches, and AlphaMotion checkpoint remain in their original locations. Local copies contain only derived PNG/NPZ/JSON/scripts and license notices, not source meshes or weights.

```python
import render_case as R
# Fresh PG.build_merge order ONLY; H2 is 36 named slots.
meta = R.render_globals(
    R.H2_XML, global_rotations_y, fk_positions_root_relative_cm_y,
    'output_prefix', expected_joint_names=joint_names)
# Exact qpos reference or separately labeled projected qpos:
meta = R.render_qpos(xml_path, qpos, 'output_prefix',
                     label='explicit source or projected-qpos label')
# A direct position head is a separate skeleton, never silently substituted for FK.
R.render_skeleton(position_head_cm_y, parents, joint_names,
                  'direct_position_head', camera=meta['camera'])
```

`render_globals` accepts [J,3,3] global rotations or [J,6] column-6D. Inputs are Y-up and cm. It applies R_z=AX.T@R_y@AX and p_z=(p_y/100)@AX, where AX maps raw (x,y,z) to cache (y,z,x). Optional root_world_yup_cm adds an explicitly supplied translation. Raw NPZ positions are Z-up meters and quaternions wxyz. Camera framing translates only the camera; meshes are not rescaled or floor-snapped.

Do not pass `predicted_position_head_raw` as centimeters: that raw array's output units must be decoded by the parent AlphaMotion adapter first. The figure uses actual FK positions. Global rotation does not directly establish a valid scalar hinge angle; predicted H2 target angle is explicitly null.

`geometry_manifest.json` records checksums, exact source and target identities, licenses, axes, reference qpos, predicted target vectors, topology, and limitations. `renderer_validation.json` records source-input agreement and independent camera-marker projection error. `h2_urdf_mjcf_binding_check.json` records the URDF/MJCF joint comparison. `source_derived_frames.npz`, `h2_fresh_geometry_descriptor.npz`, and the two reference-card NPZ files are derived numerical companions.
