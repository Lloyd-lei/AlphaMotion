"""Freeze the unitree_g1_29dof per-robot pipeline config -> pipeline/pipeline.json (all assets pinned by sha256)."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np

ROBOT = 'unitree_g1_29dof'
DATA = Path('/home/arenalabs/umr_factory_data')
RD = DATA / 'robots' / ROBOT
PD = RD / 'pipeline'
sys.path.insert(0, '/home/arenalabs/umr_factory_code')
from umrf import pipeline as P, labels as L, postpass as PP        # read-only: record the engine constants in force


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


ac = json.load(open(PD / 'assets' / 'assets_check.json'))
st = json.load(open(RD / 'staging.json'))
S = np.load(RD / 'surface.npz')
regions = [str(r) for r in S['regions']]
corr = DATA / 'corr' / ROBOT
cfg = dict(
    robot=ROBOT, version='g1-pipeline-v1', frozen=time.strftime('%Y-%m-%d %H:%M:%S'),
    spec='/home/arenalabs/umr_factory_data/DATA_SPEC_v2.md', spec_sha256=sha(DATA / 'DATA_SPEC_v2.md'),
    engine=dict(
        code='/home/arenalabs/umr_factory_code (shared, read-only for this pipeline)',
        entry='umrf.pipeline.run_umr + umrf.pipeline.postprocess (= scripts/run_clip.py body; per-robot driver '
              'pipeline/run_qa.py adds the frame window and redirects RUNS/OUT into this pipeline dir)',
        code_sha256={f: sha(f'/home/arenalabs/umr_factory_code/umrf/{f}') for f in
                     ('pipeline.py', 'postpass.py', 'geometry.py', 'labels.py', 'objects.py', 'surface.py')},
        umr_commit='d1cdaed', umr_python='/var/tmp/alphamotion/umr_factory/umr/env/bin/python',
        solver_overrides=P.SOLVER_OVERRIDES, hoi_overrides=P.HOI_OVERRIDES, latch_pin=P.LATCH_PIN,
        engine_label_cfg=L.CFG, engine_pen_tol_m=PP.PEN_TOL, engine_contact_tol_m=PP.CONTACT_TOL,
        umr_device='CPU (CUDA_VISIBLE_DEVICES="" -- GPU reserved for EGL rendering)'),
    assets=dict(
        mjcf=str(RD / f'{ROBOT}.xml'), mjcf_sha256=sha(RD / f'{ROBOT}.xml'),
        mjcf_source=st['src_mjcf'], nq=st['nq'], nv=st['nv'], facing='+x', up='+z',
        surface=str(RD / 'surface.npz'), surface_sha256=sha(RD / 'surface.npz'), n_dense_points=int(len(S['region'])),
        staging=str(RD / 'staging.json'), staging_sha256=sha(RD / 'staging.json'),
        umr_robot=str(RD / 'umr_robot.json'), umr_robot_sha256=sha(RD / 'umr_robot.json'),
        patches=str(PD / 'assets' / 'patches.npz'), patches_sha256=sha(PD / 'assets' / 'patches.npz'),
        renders=[str(PD / 'assets' / f) for f in ('tpose_regions.png', 'rest_regions.png', 'patches.png',
                                                    'soles_from_below.png')]),
    scale_and_correspondence=dict(
        tpose_qpos=st['tpose'], tpose_metrics=st['tpose_metrics_final'], point_cloud_center='body:waist_yaw_link',
        smpl_center='bbox_ratio', bbox_center_ratio=st['bbox_center_ratio'],
        human_scale='per clip, computed by UMR (robot_height 1.32246 m from slots y-extent / SMPL-X height); stored as human_scale in every output npz',
        smplx_plain=dict(dataset=str(corr / 'smplx/correspondence_dataset.npz'),
                         slots=str(corr / 'smplx/train/correspondence_slots_final.npz'),
                         slots_sha256=sha(corr / 'smplx/train/correspondence_slots_final.npz'),
                         note='per-robot; used for kind=smplx clips (LAFAN)'),
        hoi_shared_beta0=dict(dataset=str(corr / 'beta0/correspondence_dataset.npz'),
                              slots=str(corr / 'beta0/train/correspondence_slots_final.npz'),
                              slots_sha256=sha(corr / 'beta0/train/correspondence_slots_final.npz'),
                              binding=str(corr / 'beta0/smplx_surface_binding.npz'),
                              note='beta-zero shared correspondence (UMR HiPHI mechanism) for every HOI clip: no per-clip training')),
    regions=regions,
    region_bodies={r: v['bodies'] for r, v in ac['regions_rest'].items()},
    region_points={r: v['n'] for r, v in ac['regions_rest'].items()},
    contact_capable=dict(
        ground=['L_foot', 'R_foot', 'L_shin', 'R_shin', 'L_hand', 'R_hand', 'L_forearm', 'R_forearm', 'L_upper_arm',
                'R_upper_arm', 'trunk', 'L_thigh', 'R_thigh'],
        object=['L_hand', 'R_hand', 'L_forearm', 'R_forearm', 'L_upper_arm', 'R_upper_arm', 'trunk', 'L_thigh',
                'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot'],
        note='knees = shin/thigh points near the knee anchor (patch L_knee/R_knee), elbows = upper_arm/forearm points near the '
             'elbow anchor, pelvis/torso = trunk; thighs are contact-capable for sitting on the floor'),
    contact_patches=dict(
        file=str(PD / 'assets' / 'patches.npz'), counts=ac['patches'],
        rules={
            'sole': 'foot points within 3 mm of the foot minimum at rest (flat bottom face)',
            'foot_edge': 'foot points 3-20 mm above the sole plane at rest (rim, toe tip, heel edge) -- point/edge contact allowed',
            'foot_upper': 'foot points > 20 mm above the sole (instep, ankle_pitch link) -- ground contact here = foot flipped, flagged',
            'knee': 'thigh/shin points within 10 cm of the knee joint anchor with normal_x > -0.2 (front/sides of the knee)',
            'shin_front': 'shin points with normal_x > 0.3 further than 10 cm from the knee anchor (kneeling with shin on floor)',
            'elbow': 'upper_arm/forearm points within 7 cm of the elbow joint anchor',
            'hand': 'all hand-region points (rubber mitten + wrist links): palm, back, fingertips and edges may all carry contact',
            'pelvis': 'trunk points on the pelvis body (sitting on the floor)', 'torso': 'trunk points on torso_link (incl. head mesh)'},
        ground_allowed={'L_foot': ['L_sole', 'L_foot_edge'], 'R_foot': ['R_sole', 'R_foot_edge']},
        min_points_for_solid_contact=1),
    self_collision=dict(
        n_collision_geoms=st['n_collision_geoms'], explicit_exclusions=st['exclusions'],
        implicit_parent_child_filter=dict(pairs=ac['exclusions']['mujoco_parent_child_filtered_pairs'],
                                          reason='MuJoCo filters parent-child body pairs automatically (adjacent links share a joint; their meshes touch by construction)'),
        rest_overlaps=ac['exclusions']['rest_overlaps'], tpose_overlaps=ac['exclusions']['tpose_overlaps'],
        random_pose_overlap_top=ac['exclusions']['random_pose_overlap_freq_top'][:8],
        verdict=ac['exclusions']['verdict']),
    floor=dict(convention='calibrated floor = z 0 in every output (qpos root z shifted by -floor); raw floor stored as '
                          'floor_offset_removed', calibration='per clip (session): median lowest human foot slot over '
                          'quasi-static stance frames (engine labels.floor_from_points)', model_floor_plane='geom 0 at z=0 of the staged MJCF (used only for the MuJoCo cross-check after the shift)'),
    thresholds=dict(contact_solid_m=[0.0, 0.001], penetration_max_m=0.0005, slip_max_mps=0.01, min_segment_frames=3,
                    hysteresis_enter_m=0.001, hysteresis_leave_m=0.002, human_contact_enter_m=0.01,
                    human_contact_leave_m=0.02, human_release_m=0.02, robot_release_required_m=0.005,
                    hover_band_m=[0.001, 0.02], fps=30),
    quirks=[
        'hands are fixed rubber mittens (no hand DoF); left/right_rubber_hand have NO collision geom -> hand-object / hand-self '
        'penetration is measured on dense points + visual hull only (wrist_yaw collision mesh ends at the wrist)',
        'feet collision = 4 spheres r=5 mm per foot at the sole corners (bottoms 0.4 mm above the visual sole at rest); '
        'visual sole is the real contact surface; ankle_pitch_link mesh is part of the foot region',
        'sole has only ~40 dense points per foot (~2 cm spacing): exact ground clearance comes from hull vertices / '
        'collision support points; the dense-point minimum can sit a few mm above the true minimum on edge contacts',
        'waist_yaw/waist_roll links have visual mesh but no collision; head mesh lives on torso_link (head_link has no geom)',
        'GMR g1_mocap_29dof: wrist pitch/yaw kept (29 DoF); elbow range [-1.047, 1.7] rad; T-pose from UMR shipped config (never changed)',
        'robot height 1.32 m vs human ~1.7 m: human_scale ~0.75-0.8 applied by UMR to the human source'],
    qa_clips=[
        dict(id='qa_dance', kind='smplx', data='/var/tmp/alphamotion/umr_factory/umr/UMR/sample_data/lafan1_smplx',
             seq='dance1_subject2', window=[1900, 2500, 1], note='LAFAN 30 fps, window sliced into pipeline/src'),
        dict(id='qa_kick', kind='hoi', data='/var/tmp/alphamotion/umr_factory/umr/UMR/sample_data/omnicontact/soccer/case3_kick_right',
             seq='20260330000586_1_1775269234', window=[0, 1800, 3], shared_corr=True,
             note='90 fps -> 30 fps; foot/shin on ball f0-480 + kick touch ~f1511 (source frames)'),
        dict(id='qa_carry', kind='hoi', data='/home/arenalabs/umr_factory_data/smplx/omnicontact/box/case1_carry',
             seq='20260330000166_1_1775007076', window=[0, 1800, 3], shared_corr=True, note='box carry, hands+trunk on box'),
        dict(id='qa_ground', kind='hoi', data=str(PD / 'src/hiphi/data/Cause_motion/push'), seq='Cause_motion-push_0211',
             window=[900, 2700, 3], shared_corr=True,
             note='HiPHI push ball across the floor while kneeling: knees + hands near/on floor (private copy of the '
                  'sequence + Ball_A_2.obj so UMR CoACD writes stay in this pipeline dir)'),
    ],
    outputs=dict(engine_runs=str(PD / 'runs'), engine_out=str(PD / 'out'), qa=str(DATA / 'qa' / ROBOT)),
)
json.dump(cfg, open(PD / 'pipeline.json', 'w'), indent=1, default=str)
print('wrote', PD / 'pipeline.json')
