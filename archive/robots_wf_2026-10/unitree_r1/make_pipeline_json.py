"""Step 2 -- freeze the unitree_r1 dedicated pipeline config -> pipeline/pipeline.json.
Inputs: staged assets, assets_check.json (step 1), patches.npz, DATA_SPEC_v2 thresholds, QA clip set."""
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import r1_common as C

ck = json.load(open(C.QADIR / 'assets' / 'assets_check.json'))
st = json.load(open(C.RDIR / 'staging.json'))
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
CODE = Path('/home/arenalabs/umr_factory_code')
patches = np.load(C.PDIR / 'patches.npz')

cfg = dict(
    robot=C.ROBOT,
    frozen_at=time.strftime('%Y-%m-%d %H:%M:%S'),
    spec='/home/arenalabs/umr_factory_data/DATA_SPEC_v2.md',
    assets=dict(
        mjcf=str(C.XML), mjcf_sha256=sha(C.XML),
        mjcf_source=st['src_mjcf'], nq=st['nq'], nv=st['nv'],
        surface=str(C.SURF), surface_sha256=sha(C.SURF), n_dense_points=ck['n_points'],
        surface_policy='4096 exterior points (UMR first-hit sampler) on VISUAL meshes, bound to geoms; '
                       '+ 33863 convex-hull vertices of visual geoms (exact ground gauge)',
        umr_robot_json=str(C.RDIR / 'umr_robot.json'), umr_robot_json_sha256=sha(C.RDIR / 'umr_robot.json'),
        staging_json=str(C.RDIR / 'staging.json'), staging_json_sha256=sha(C.RDIR / 'staging.json'),
        patches=str(C.PDIR / 'patches.npz'), patches_sha256=sha(C.PDIR / 'patches.npz'),
    ),
    tpose=dict(qpos=st['tpose'], metrics=st['tpose_metrics_final'], note='least-squares arm-chain T-pose (draft failed: forearm 16 deg)'),
    scale_and_correspondence=dict(
        point_cloud_center='body:waist_yaw_link', smpl_center='bbox_ratio', bbox_center_ratio=st['bbox_center_ratio'],
        human_scale='UMR auto: robot_height / human_height per clip (smpl_scale), recorded per output as human_scale',
        smplx_plain=dict(dir='/home/arenalabs/umr_factory_data/corr/unitree_r1/smplx', template=ck['correspondence']['smplx'].get('names', [None])[0],
                         slots_sha256=ck['correspondence']['smplx'].get('slots_sha256'), used_for='plain SMPL-X (LAFAN)'),
        shared_beta0=dict(dir='/home/arenalabs/umr_factory_data/corr/unitree_r1/beta0', template=ck['correspondence']['beta0'].get('names', [None])[0],
                          slots_sha256=ck['correspondence']['beta0'].get('slots_sha256'),
                          used_for='HOI / HiPHI with per-actor betas (reuse_across_smplx_betas, fixed face+bary transfer)'),
    ),
    regions=C.REGIONS,
    region_to_bodies=ck['region_bodies'],
    region_point_counts=ck['region_counts'],
    collision_geoms_per_region={r: [g['body'] for g in v] for r, v in ck['collision_geoms_per_region'].items()},
    contact_capable=dict(
        ground=['L_foot', 'R_foot', 'L_shin', 'R_shin', 'L_thigh', 'R_thigh', 'L_hand', 'R_hand', 'L_forearm', 'R_forearm',
                'L_upper_arm', 'R_upper_arm', 'trunk'],
        object=['L_hand', 'R_hand', 'L_forearm', 'R_forearm', 'L_upper_arm', 'R_upper_arm', 'L_foot', 'R_foot', 'L_shin', 'R_shin',
                'L_thigh', 'R_thigh', 'trunk'],
        note='region level = the 13 label regions shared with the human (SMPL-X part segmentation). Knee = L/R_knee patch '
             '(shin+thigh points within 8 cm of the knee joint), elbow = L/R_elbow patch (upper_arm+forearm within 6 cm of '
             'the elbow joint), pelvis/hips = pelvis patch (pelvis_link + imu), torso/back = torso_back patch.'),
    contact_patches=dict(
        definition={
            'L_foot_all/R_foot_all': 'every dense point of the foot region (ankle_pitch, ankle_roll, constraint_A/B links). '
                                     'ANY point may carry ground contact: edges/heel/toe tip allowed (spec v2: 1 point suffices)',
            'L_sole/R_sole': 'informational: foot points within 15 mm of the rest-pose foot minimum with downward normal (40 / 40)',
            'L_knee/R_knee': 'shin+thigh points within 8 cm of the knee joint anchor (rest pose)',
            'L_elbow/R_elbow': 'upper_arm+forearm points within 6 cm of the elbow joint anchor (T-pose)',
            'L_hand/R_hand': 'all points of wrist_roll_link (fixed fist, no finger DoF)',
            'pelvis': 'pelvis_link + imu_in_pelvis_link points', 'torso_back': 'waist_roll_link + waist_yaw_link points'},
        counts=ck['patch_counts'],
        dense_spacing_mm=ck['dense_nn_spacing'],
    ),
    self_collision=dict(
        gauge='MuJoCo collision geoms (34 meshes, convex hulls), mj_geomDistance; every pair of different, non parent/child bodies',
        exclusions=st['exclusions'],
        exclusions_reason='none needed: no pair overlaps at qpos0 or T-pose; no pair overlaps in >=95% of 300 random in-limit poses',
        n_pairs_tested=ck['n_self_pairs_tested'],
        rigid_pairs_tested_never_overlap=ck['rigid_pairs_tested'],
        frequent_overlap_pairs_random_poses={k: v for k, v in ck['random_pose_overlap_frac'].items() if v >= 0.05},
    ),
    floor_convention=dict(
        output='calibrated floor = z 0 (per clip, from the human source: median lowest foot point over quasi-static stance '
               'frames); floor_offset_removed stored in every output npz; the same floor is used for robot and human labels',
        robot_qpos0='root z = 0 puts the soles at -0.741 m; the factory floor alignment handles it (no robot offset table)'),
    thresholds_spec_v2=dict(
        contact_clearance_m=[0.0, 0.001], min_contact_points=1, penetration_tol_m=0.0005,
        slip_speed_max_m_s=0.01, min_contact_segment_frames=3, hysteresis_enter_m=0.001, hysteresis_leave_m=0.002,
        human_contact='human region in contact (same ruler: enter < 1 cm / leave > 2 cm on SMPL-X slots, floor calibrated) '
                      '-> robot region must be solid', human_off_m=0.02, robot_off_required_m=0.005,
        hover_band_m=[0.001, 0.02], fps=30),
    engine=dict(
        name='umrf CPU reference (UMR d1cdaed + strict post-pass)', code=str(CODE),
        module_sha256={p.name: sha(p)[:16] for p in sorted((CODE / 'umrf').glob('*.py'))},
        solver_overrides='pipeline.SOLVER_OVERRIDES (ground hard, self hard+slack, object hard for HOI)',
        latch_pin=0.0, latch_note='LATCH_PIN=0 reproduces UMR slip -> does NOT meet spec v2 no-slip',
        umr_python='/var/tmp/alphamotion/umr_factory/umr/env/bin/python', cuda='disabled (CUDA_VISIBLE_DEVICES="") for this robot run',
        runner=str(C.PDIR / 'run_qa_clips.py')),
    quirks=[
        'hands are fixed fists (wrist_roll_link, no finger DoF); hand contact = any fist point',
        'ankle parallel linkage (ankle_A/B_link + rods, constraint_A/B) has NO joints and NO equality constraints: the '
        'rods are welded to the knee link / ankle roll link and do not follow ankle pitch/roll (visual artefact only; '
        'rods have no collision geom)',
        'ankle_pitch_link and the 4 ankle rods have no collision geom (visual only); ground gauge still covers them via '
        'visual hull vertices + dense points',
        'head_yaw_link vs waist_yaw_link collision meshes are 0-5 mm apart at rest and overlap 1-6 mm for most head '
        'pitch/yaw values (76% of random in-limit poses): not adjacent, not always overlapping -> NOT excluded; '
        'head motion is pushed back by the self constraint',
        'dense point spacing ~13 mm (4096 points / 1.23 m robot): a tilted foot touching with an edge may have its '
        'lowest DENSE point 1-5 mm up while the hull vertex is at 0; the engine gauge uses hull vertices',
        'small foot: 163/158 dense points per foot, 40 per sole',
    ],
    qa_clips=[
        dict(id='lafan_dance', kind='smplx', source='/var/tmp/alphamotion/umr_factory/umr/UMR/sample_data/lafan1_smplx/dance1_subject2.npz',
             window_frames=[1900, 2500], stride=1, fps=30, note='windowed copy under pipeline/src (run_umr has no smplx window)'),
        dict(id='soccer_kick', kind='hoi', data='/var/tmp/alphamotion/umr_factory/umr/UMR/sample_data/omnicontact/soccer/case3_kick_right',
             seq='20260330000586_1_1775269234', window_src_frames=[450, 1950], stride=3, fps=30, shared_corr=True),
        dict(id='box_carry', kind='hoi', data='/home/arenalabs/umr_factory_data/smplx/omnicontact/box/case1_carry',
             seq='20260330000166_1_1775007076', window_src_frames=[450, 2250], stride=3, fps=30, shared_corr=True),
        dict(id='hiphi_kneel', kind='hoi', data='/home/arenalabs/umr_factory_data/smplx/hiphi/data/Change_posture/lean',
             seq='Change_posture-lean_0025', window_src_frames=[450, 2250], stride=3, fps=30, shared_corr=True,
             note='kneel facing a chair, both forearms on the seat (knees on the ground from src frame ~723); copied to '
                  'pipeline/src/hiphi so the CoACD chair collision asset is written under this robot'),
    ],
)
C.dump(C.PDIR / 'pipeline.json', cfg)
print('wrote', C.PDIR / 'pipeline.json')
