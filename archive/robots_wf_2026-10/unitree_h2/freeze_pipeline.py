"""unitree_h2 dedicated pipeline -- step 2: freeze pipeline.json + contact_patches.npz (per-robot, versioned by sha).
Run: <umr python> freeze_pipeline.py"""
import hashlib, json, time
from pathlib import Path
import mujoco, numpy as np

ROBOT = 'unitree_h2'
DATA = Path('/home/arenalabs/umr_factory_data')
RD = DATA / 'robots' / ROBOT; PD = RD / 'pipeline'
XML = RD / f'{ROBOT}.xml'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
st = json.load(open(RD / 'staging.json'))
ac = json.load(open(PD / 'assets_check.json'))
surf = dict(np.load(RD / 'surface.npz', allow_pickle=True))
REG = [str(r) for r in surf['regions']]
m = mujoco.MjModel.from_xml_path(str(XML)); d = mujoco.MjData(m)
bn = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
d.qpos[:] = m.qpos0; d.qpos[3:7] = [1, 0, 0, 0]; mujoco.mj_forward(m, d)
g = surf['geom'].astype(int); body = surf['body'].astype(int); reg = surf['region'].astype(int)
X = np.einsum('nij,nj->ni', d.geom_xmat[g].reshape(-1, 3, 3), surf['points_local'].astype(float)) + d.geom_xpos[g]


def local(bname, idx):
    b = bid(bname)
    return (X[idx] - d.xpos[b]) @ d.xmat[b].reshape(3, 3)


# ------------------------------------------------------------------ contact patches (which dense points may carry contact)
P = len(reg)
patch = {}
for s, side in (('L', 'left'), ('R', 'right')):
    # foot: sole + rim/edges = lower 2.0 cm of the ankle_pitch_link mesh (in its own frame); ankle housing / top excluded
    fi = np.flatnonzero(body == bid(f'{side}_ankle_pitch_link'))
    lz = local(f'{side}_ankle_pitch_link', fi)[:, 2]
    patch[f'{s}_foot'] = fi[lz <= lz.min() + 0.020]
    # hand: wrist_yaw_link + hand_link only (H2 'hand' region = wrist_roll subtree also contains the distal forearm tube)
    patch[f'{s}_hand'] = np.flatnonzero(np.isin(body, [bid(f'{side}_wrist_yaw_link'), bid(f'{side}_hand_link')]))
    # forearm (incl. elbow): elbow_link + wrist_roll_link + wrist_pitch_link
    patch[f'{s}_forearm'] = np.flatnonzero(np.isin(body, [bid(f'{side}_elbow_link'), bid(f'{side}_wrist_roll_link'),
                                                          bid(f'{side}_wrist_pitch_link')]))
    # upper arm: all points (elbow tip of the shoulder_yaw link included)
    patch[f'{s}_upper_arm'] = np.flatnonzero(reg == REG.index(f'{s}_upper_arm'))
    # knee/shin: whole shin link (kneeling = shin front on the ground) ; thigh: whole thigh (sitting on the floor)
    patch[f'{s}_shin'] = np.flatnonzero(reg == REG.index(f'{s}_shin'))
    patch[f'{s}_thigh'] = np.flatnonzero(reg == REG.index(f'{s}_thigh'))
patch['trunk'] = np.flatnonzero(reg == REG.index('trunk'))          # pelvis (sitting), back/head (lying)
patch_region = np.full(P, -1, np.int16)                              # region id carried by each point for CONTACT purposes
for k, idx in patch.items():
    patch_region[idx] = REG.index(k)
excluded = np.flatnonzero(patch_region < 0)
np.savez_compressed(PD / 'contact_patches.npz', patch_region=patch_region, regions=np.array(REG),
                    **{f'idx_{k}': v.astype(np.int32) for k, v in patch.items()})
# knee point (for reporting): shin points within 10 cm of the knee joint
cfg = dict(
    robot=ROBOT, frozen=time.strftime('%Y-%m-%d %H:%M:%S'), spec='/home/arenalabs/umr_factory_data/DATA_SPEC_v2.md',
    engine=dict(cpu_reference='/home/arenalabs/umr_factory_code (umrf.pipeline.run_umr + umrf.pipeline.postprocess, '
                              'called by pipeline/run_qa.py with outputs redirected to this robot)',
                gpu_engine='/home/arenalabs/umr_gpu (separate agent; not used here)', umr_commit='d1cdaed',
                python='/var/tmp/alphamotion/umr_factory/umr/env/bin/python', umr_device='cpu (CUDA_VISIBLE_DEVICES="")'),
    assets=dict(mjcf=str(XML), mjcf_sha256=sha(XML), src_mjcf=st['src_mjcf'],
                surface=str(RD / 'surface.npz'), surface_sha256=sha(RD / 'surface.npz'), n_dense_points=int(P),
                umr_robot_json=str(RD / 'umr_robot.json'), umr_robot_json_sha256=sha(RD / 'umr_robot.json'),
                staging_json=str(RD / 'staging.json'), contact_patches=str(PD / 'contact_patches.npz'),
                contact_patches_sha256=sha(PD / 'contact_patches.npz'), nq=int(m.nq), nv=int(m.nv)),
    scale_correspondence=dict(
        scaling='UMR smpl_scale = robot_height / human_height per clip (robot_height 1.83303 m inferred from slots)',
        point_cloud_center='body:waist_yaw_link', bbox_center_ratio=st['bbox_center_ratio'],
        tpose=st['tpose'], tpose_metrics=st['tpose_metrics_final'],
        smplx_corr=ac['correspondence']['smplx'], shared_beta0_corr=ac['correspondence']['beta0'],
        hoi_policy='shared beta0 correspondence (reuse_across_smplx_betas); no per-clip training'),
    region_body_map={k: sorted({bn(int(b)) for b in np.flatnonzero(surf['body_region'] == i) if b > 0}) for i, k in enumerate(REG)},
    points_per_region=ac['points_per_region'],
    contact_capable_regions=dict(
        ground=['L_foot', 'R_foot', 'L_shin', 'R_shin', 'L_thigh', 'R_thigh', 'L_hand', 'R_hand', 'L_forearm', 'R_forearm',
                'L_upper_arm', 'R_upper_arm', 'trunk'],
        object=REG,
        named={'feet': ['L_foot', 'R_foot'], 'knees': ['L_shin', 'R_shin'], 'hands': ['L_hand', 'R_hand'],
               'elbows': ['L_forearm', 'R_forearm', 'L_upper_arm', 'R_upper_arm'], 'pelvis_torso': ['trunk'],
               'thighs_sitting': ['L_thigh', 'R_thigh']}),
    contact_patches=dict(
        rule='a region is in solid contact if min clearance over its PATCH points is in [0, 1.0 mm] (>= 1 point; a '
             'point, an edge or a small patch is enough). Points outside every patch never carry contact.',
        foot='ankle_pitch_link points in the lowest 2.0 cm of the foot mesh (own frame): sole + toe/heel/side rims; '
             'ankle housing excluded; %d/%d (L) points' % (len(patch['L_foot']), int((reg == REG.index('L_foot')).sum())),
        hand='wrist_yaw_link + hand_link (%d L / %d R points); the surface region L/R_hand also holds wrist_roll/'
             'wrist_pitch links (= distal forearm tube, 84/88 pts) which are moved to the forearm patch' % (len(patch['L_hand']), len(patch['R_hand'])),
        forearm='elbow_link + wrist_roll_link + wrist_pitch_link (elbow contact = forearm patch near the elbow)',
        knee='whole shin (left/right_knee_link): kneeling puts the knee cap and the shin front on the floor',
        thigh='whole thigh (sitting on the floor)', trunk='pelvis + torso + head (sitting / lying)',
        excluded_points=int(len(excluded)), sizes={k: int(len(v)) for k, v in patch.items()}),
    self_collision=dict(
        n_collision_geoms=ac['n_collision_geoms'], staged_exclusions=st['exclusions'],
        mjcf_exclude_elements=ac['staged_exclude_elements_in_mjcf'],
        filtered_by_mujoco='parent-child body pairs only (MuJoCo default filterparent); list in assets_check.json',
        reason='no collision-geom pair overlaps at qpos0 or T-pose (closest non-adjacent rest gap: pelvis|left_hip_roll '
               '6.2 mm, torso|shoulder_roll 10.5 mm); no pair overlaps in >= 95% of 300 random in-limit poses (max '
               'torso|shoulder_roll 11%) -> nothing may be excluded; every non-adjacent pair counts as penetration',
        random_pose_overlap_top=ac['random_pose_overlap_frac_top'][:5]),
    floor=dict(convention='output z = 0 is the calibrated floor; per clip (session) floor from the HUMAN source: '
                          'median lowest foot point over quasi-static stance frames (umrf.labels.floor_from_points); '
                          'floor_offset_removed stored in every output npz; ground gauge uses z of the floor-shifted pose',
               up='+z', robot_forward='+x', left='+y'),
    thresholds=dict(contact_max_m=0.001, contact_min_m=0.0, penetration_tol_m=0.0005, slip_speed_max_m_s=0.01,
                    min_contact_segment_frames=3, hysteresis_enter_m=0.001, hysteresis_leave_m=0.002,
                    human_contact_enter_m=0.01, human_contact_leave_m=0.02, human_separated_m=0.02,
                    robot_must_separate_m=0.005, hover_band_m=[0.001, 0.02], fps=30,
                    note='human contact label uses the engine human rule (enter 1 cm / leave 2 cm on SMPL-X slots, '
                         'same floor); robot rules are DATA_SPEC_v2 section 1 verbatim'),
    quirks=[
        'H2 has no hand DoF: hands are rigid mitts (hand_link mesh); grasp = palm/finger-side contact of a rigid shell',
        'surface region L/R_hand = wrist_roll subtree: includes the distal forearm tube (wrist_roll + wrist_pitch links, '
        '~84/88 points); contact patch reassigns them to the forearm patch so a forearm touching a box does not count as hand contact',
        'knee_link collision geom is a thin CYLINDER (r 2.5 cm, half-length 17.5 cm) while the visual shin reaches 4.4 cm '
        '(p50) / 7.9 cm (max) from the link axis -> MuJoCo knee/shin collision sits 2-5 cm inside the visual surface: a '
        'kneeling clip that is exact on the visual points leaves the physics shin 2-5 cm above the floor (sim would sink)',
        'hip_roll collision geom is a 5 cm sphere (visual mesh larger)',
        'no collision geom on hip_pitch_link (thigh, 76/82 pts), wrist_roll/wrist_pitch (forearm tube, 84/88 pts), '
        'waist_yaw/waist_roll/head_pitch (trunk 48 pts), ankle_roll (foot 3/2 pts): MuJoCo self/object cross-check is '
        'blind there; the dense-point gauge must cover them',
        'qpos0 has elbows at 0 = forearms pointing forward horizontally (not arms-down)',
        'UMR T-pose shipped with UMR (shoulder roll +-90 deg, elbow 90 deg) kept unchanged',
        'foot sole is flat: 22/136 points within 2 mm of the floor at rest, sole patch 22.0 x 7.5 cm'],
    qa_clips=[
        dict(id='dance', kind='smplx', data='/var/tmp/alphamotion/umr_factory/umr/UMR/sample_data/lafan1_smplx',
             seq='dance1_subject2', start=1900, end=2500, stride=1, src_fps=30),
        dict(id='soccer_kick', kind='hoi', data='/var/tmp/alphamotion/umr_factory/umr/UMR/sample_data/omnicontact/soccer/case3_kick_right',
             seq='20260330000586_1_1775269234', start=0, end=1800, stride=3, src_fps=90, shared_corr=True,
             note='ball resting against the left foot/shin (src 0-481) then dribble/kick; first window 3150-4650 had only 4 human-ball contact frames and was replaced'),
        dict(id='box_carry', kind='hoi', data='/home/arenalabs/umr_factory_data/smplx/omnicontact/box/case1_carry',
             seq='20260330000166_1_1775007076', start=0, end=1800, stride=3, src_fps=90, shared_corr=True),
        dict(id='hiphi_kneel', kind='hoi', data='/home/arenalabs/umr_factory_data/robots/unitree_h2/pipeline/src/hiphi/data/Change_posture/lean',
             seq='Change_posture-lean_0025', start=600, end=2100, stride=3, src_fps=90, shared_corr=True,
             note='kneel facing a chair, forearms on the seat: knee/shin + foot ground contact, forearm/hand object contact; private copy of smplx/hiphi/data/Change_posture/lean/Change_posture-lean_0025 + object_meshes/Chair_S_1.obj so the CoACD collision asset UMR builds is written under this pipeline')],
    outputs=dict(pipeline_dir=str(PD), qa_dir=str(DATA / 'qa' / ROBOT), disk_budget_gb=2),
)
json.dump(cfg, open(PD / 'pipeline.json', 'w'), indent=1, default=str)
print(json.dumps(cfg['contact_patches'], indent=1)); print('region_body_map', cfg['region_body_map'])
