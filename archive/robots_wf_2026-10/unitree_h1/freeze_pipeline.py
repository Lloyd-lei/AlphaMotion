"""unitree_h1 dedicated pipeline -- step 2: freeze pipeline.json + contact_patches.npz (dense-point index sets).
Usage: <umr py> freeze_pipeline.py"""
import hashlib, json, time
from pathlib import Path
import numpy as np, mujoco

ROBOT = 'unitree_h1'
D = Path('/home/arenalabs/umr_factory_data')
RD = D / 'robots' / ROBOT; PD = RD / 'pipeline'
st = json.load(open(RD / 'staging.json')); ac = json.load(open(PD / 'assets_check.json'))
m = mujoco.MjModel.from_xml_path(st['staged_mjcf']); d = mujoco.MjData(m)
S = np.load(RD / 'surface.npz'); REG = [str(x) for x in S['regions']]
reg = S['region'].astype(int); geom = S['geom'].astype(int); loc = S['points_local'].astype(float)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))

# rest pose (qpos0) for patch geometry
mujoco.mj_resetData(m, d); mujoco.mj_forward(m, d)
X = np.einsum('nij,nj->ni', d.geom_xmat[geom].reshape(-1, 3, 3), loc) + d.geom_xpos[geom]
jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
patches = {}
for s, side in (('L', 'left'), ('R', 'right')):
    kn = d.xanchor[jid(f'{side}_knee_joint')]; el = d.xanchor[jid(f'{side}_elbow_joint')]
    legs = np.isin(reg, [REG.index(f'{s}_thigh'), REG.index(f'{s}_shin')])
    arms = np.isin(reg, [REG.index(f'{s}_upper_arm'), REG.index(f'{s}_forearm'), REG.index(f'{s}_hand')])
    patches[f'{s}_foot'] = np.flatnonzero(reg == REG.index(f'{s}_foot'))
    patches[f'{s}_knee'] = np.flatnonzero(legs & (np.linalg.norm(X - kn, axis=1) < 0.08))
    patches[f'{s}_hand'] = np.flatnonzero(reg == REG.index(f'{s}_hand'))
    patches[f'{s}_elbow'] = np.flatnonzero(arms & (np.linalg.norm(X - el, axis=1) < 0.07))
pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
patches['pelvis'] = np.flatnonzero((reg == 0) & (m.geom_bodyid[geom] == pel))
patches['torso'] = np.flatnonzero((reg == 0) & (m.geom_bodyid[geom] != pel))
np.savez_compressed(PD / 'contact_patches.npz', **{k: v.astype(np.int32) for k, v in patches.items()})

br = S['body_region']
cfg = dict(
    robot=ROBOT, frozen=time.strftime('%F %T'), engine='CPU reference factory /home/arenalabs/umr_factory_code (run via pipeline/run_qa_clips.py)',
    mjcf=dict(path=st['staged_mjcf'], sha256=sha(st['staged_mjcf']), source=st['src_mjcf'], nq=int(m.nq), nv=int(m.nv),
              n_hinge=int((m.jnt_type == 3).sum()), facing='+x', up='+z'),
    surface=dict(path=str(RD / 'surface.npz'), sha256=sha(RD / 'surface.npz'), n_points=int(len(reg)),
                 points_per_region=ac['points_per_region'], hull_vertices=int(len(S['hull_local'])),
                 hand_split=json.loads(str(S['split_note']))),
    umr_robot_config=dict(path=str(RD / 'umr_robot.json'), sha256=sha(RD / 'umr_robot.json'), tpose=st['tpose'],
                          point_cloud_center='body:torso_link', bbox_center_ratio=st['bbox_center_ratio']),
    correspondence=dict(smplx=dict(dir=str(D / 'corr' / ROBOT / 'smplx'), **{k: ac['correspondence']['smplx'].get(k) for k in ('slot_names', 'dataset_sha256_16', 'slots_sha256_16')}),
                        hoi_shared_beta0=dict(dir=str(D / 'corr' / ROBOT / 'beta0'), **{k: ac['correspondence']['beta0'].get(k) for k in ('slot_names', 'dataset_sha256_16', 'slots_sha256_16')}),
                        scale='UMR per_sample_height normalisation (robot height 1.806 m); human scale per clip = UMR smpl_scale (stored as human_scale)',
                        hoi_policy='HOI clips use the shared beta-zero correspondence (UMR HiPHI mechanism) -> no per-clip GPU training'),
    region_to_body={rn: sorted({bn(b) for b in range(1, m.nbody) if REG[int(br[b])] == rn}) for rn in REG},
    region_note='L/R_forearm and L/R_hand share body left/right_elbow_link (no wrist, no hand DoF); dense points split '
                'geometrically at 17.8 cm lateral from the elbow anchor; one convex collision hull covers both',
    contact_capable_regions=['L_foot', 'R_foot', 'L_shin', 'R_shin', 'L_thigh', 'R_thigh', 'L_hand', 'R_hand',
                             'L_forearm', 'R_forearm', 'L_upper_arm', 'R_upper_arm', 'trunk'],
    contact_patches=dict(file=str(PD / 'contact_patches.npz'),
                         counts={k: int(len(v)) for k, v in patches.items()},
                         definition={
                             'foot': 'every L/R_foot dense point (ankle_link shell, 30 x 3 cm footprint). Solid contact may be a single '
                                     'point / heel or toe edge / sole line (only 11-12 points lie within 1 mm of the foot minimum when flat)',
                             'knee': 'thigh+shin points within 8 cm of the knee joint anchor (rest pose)',
                             'hand': 'L/R_hand points = distal end of elbow_link (fixed, no fingers); any single point may carry contact',
                             'elbow': 'upper_arm/forearm/hand points within 7 cm of the elbow joint anchor',
                             'pelvis': 'trunk points on the pelvis body (sitting / lying)', 'torso': 'trunk points on torso_link (lying, leaning)'}),
    self_collision=dict(
        staged_exclusions=st['exclusions'], mjcf_contact_exclude=0,
        implicit_parent_child_filter=dict(pairs=ac['implicit_parent_child_filter']['pairs'],
                                          reason='MuJoCo filterparent (joint-adjacent bodies, always overlapping at the joint); not configurable per pair'),
        not_excluded_non_adjacent_pairs_seen_overlapping={k: v for k, v in ac['non_adjacent_overlap_random_poses'].items() if v['frac'] >= 0.05},
        rest_overlaps=ac['non_adjacent_overlap_rest'], tpose_overlaps=ac['non_adjacent_overlap_tpose'],
        rule='only adjacent (parent/child) pairs excluded; no non-adjacent pair overlaps at rest/T-pose or in >=95% of 300 random '
             'poses, so none excluded; hip_yaw|hip_pitch (2 hops, 36-43% of random poses) stays a real collision pair'),
    floor_convention='per-clip floor from the human source (median lowest foot slot over quasi-static stance), outputs shifted '
                     'so floor z = 0; raw offset stored as floor_offset_removed',
    thresholds_DATA_SPEC_v2=dict(contact_clearance_m=[0.0, 0.001], min_points_in_contact=1, penetration_tol_m=0.0005,
                                 slip_speed_max_m_s=0.01, min_segment_frames=3, hysteresis_enter_m=0.001, hysteresis_leave_m=0.002,
                                 human_contact_requires_robot_solid=True, human_clear_m=0.02, robot_clear_when_human_clear_m=0.005,
                                 near_surface_store_m=0.02, fps=30),
    engine_thresholds_in_use=dict(label_hysteresis_enter_m=0.01, label_hysteresis_leave_m=0.02, slip_flag_speed_m_s=0.03,
                                  pen_tol_m=0.0005, contact_tol_m=0.001, LATCH_PIN=0.0,
                                  note='engine label hysteresis (1/2 cm) and slip flag (3 cm/s) are looser than v2 (1/2 mm, 1 cm/s); '
                                       'LATCH_PIN=0 reproduces UMR slip'),
    quirks=[
        'no hands: elbow_link ends in a fixed rounded fist; no wrist joint, no finger DoF (hand_type none/fixed)',
        'ankle is 1-DoF (pitch only): foot cannot roll to conform to the ground; flat contact is a 3 cm wide line',
        'mesh collision geoms are used by MuJoCo as CONVEX HULLS (pelvis, torso, elbow_link, knee_link...): self/object '
        'collision gauge is conservative (hull fills concavities), e.g. elbow_link hull vs torso hull',
        'hip_yaw_link collision is a cylinder (r 6 cm, half-length 3.5 cm), not the visual mesh',
        'logo_link visual mesh on torso has no collision geom (decorative, inside torso hull)',
        'torso_link visual mesh includes the head; the head belongs to region trunk',
        'no dedicated knee / elbow regions in the 13-region label set: knee contact is labelled on L/R_shin or L/R_thigh, '
        'elbow contact on upper_arm/forearm; patches above give the dense-point subsets',
    ],
    qa_clips=[
        dict(id='dance', kind='smplx', data='/var/tmp/alphamotion/umr_factory/umr/UMR/sample_data/lafan1_smplx',
             seq='dance1_subject2', start=1900, end=2500, stride=1, src_fps=30, cid='lafan1_smplx__dance1_subject2__f1900-2500'),
        dict(id='kick', kind='hoi', data='/var/tmp/alphamotion/umr_factory/umr/UMR/sample_data/omnicontact/soccer/case3_kick_right',
             seq='20260330000586_1_1775269234', start=300, end=1800, stride=3, src_fps=90,
             cid='omni_soccer_case3_kick_right__20260330000586_1_1775269234__f300-1800s3',
             why='ball resting against the left foot/shin (src 300-482) then kicks (ball speed up to 12.9 m/s at src 1500-1800)'),
        dict(id='box_carry', kind='hoi', data='/home/arenalabs/umr_factory_data/smplx/omnicontact/box/case1_carry',
             seq='20260330000166_1_1775007076', start=0, end=1800, stride=3, src_fps=90,
             cid='omni_box_case1_carry__20260330000166_1_1775007076__f0-1800s3',
             why='same window as the G1 pilot; hands on the box from out-frame ~190, carried to the end'),
        dict(id='kneel', kind='hoi', data='/home/arenalabs/umr_factory_data/smplx/hiphi/data/Change_posture/lean',
             seq='Change_posture-lean_0025', start=600, end=2400, stride=3, src_fps=90,
             cid='hiphi_Change_posture-lean_0025__f600-2400s3',
             why='HiPHI: kneel facing a chair (knees on the ground), forearms on the seat, lean in; kneels by src ~900'),
    ],
    outputs=dict(runs=str(PD / 'runs'), out=str(PD / 'out'), qa=str(D / 'qa' / ROBOT)),
)
json.dump(cfg, open(PD / 'pipeline.json', 'w'), indent=1)
print(json.dumps(cfg['contact_patches']['counts']))
