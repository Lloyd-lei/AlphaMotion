# UniMate real sample trace

Actual clip: Jumping_Up-000. 8 released poses,22 Mixamo joints,30fps. Caption: An object jumps up from an idle pose.
Selected joint: mixamorig:LeftFoot, zero-based index9; parent LeftLeg,index6.

build_case.py executes official numpy preprocessing/recovery bodies and official Motion quaternion/FK reference code on CPU. No installation, neural forward pass, ODE sampling, robot retargeting, simulation or hardware execution.

Frames0/3/6 map to feature rows0/3/6 and use frames1/4/7 for the delta channels. Eight poses yield7 feature rows; frame7 has no feature row.
12D = RIFKE position3 + T-pose-relative,HML-reordered rotation6 + local displacement3. LeftFoot rotation channels store rebased parent LeftLeg rotation. Executable encoding uses matrix columns0 and1.
Y-up,Z-forward, canonical body units with skeleton diameter2.0; not asserted meters. Delta is body units/frame, not m/s; it is not neural flow velocity.
Cached official statistics normalize the actual values. Shape(7,22,12) becomes(60,22,12) then(1,22,12,60), with only time0-6 valid. This is preprocessing output, not a neural forward-pass result.

QC: {"finite_features": true, "ric_max_abs_error_body_units": 1.7100135196734811e-10, "fk_max_abs_error_body_units": 1.5991813903659668e-07, "normalization_roundtrip_max_abs_error": 1.1102230246251565e-16, "raw_feature_shape": [7, 22, 12], "model_motion_array_shape": [1, 22, 12, 60], "padded_frames_exact_zero": true}

Figures: skeleton_tpose_motion.png/.svg, left_foot_12d_values.png/.svg and named foot crops. joint_pixels.json gives top-left PNG pixel positions and projected coordinate axes. Local trace.json and model_inputs_derived.npz retain the delivered numeric stages. Remote-only derived_preprocessing.npz also retains original selected source poses and is intentionally excluded from this local delivery; asset_manifest.json records its remote location separately.

model_inputs_derived.npz is the shareable computed-tensor export: features_12d/normalized_12d [7,22,12], model_motion_array_bjdt [1,22,12,60], time_mask [60], joint_mask [22], T-pose arrays [22,12]/[22,3], statistics and topology. Original source pose/quaternion arrays stay remote. build_case.py is reproduction code targeting the pinned ArenaLabs cache and external source checkout; it is not standalone without the source dependencies listed in the manifest.

No textured mesh is loaded. This does not establish G1/Go2/Franka-to-H2 mapping. Raw references/dependency sources remain remote; only derived figures, selected trace, computed tensor export and reproduction script may be copied to local presentation assets.
