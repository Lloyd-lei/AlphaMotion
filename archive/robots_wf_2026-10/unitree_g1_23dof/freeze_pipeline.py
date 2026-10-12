"""Step 2 -- freeze the dedicated per-robot pipeline config: pipeline/pipeline.json (unitree_g1_23dof)."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mujoco
import numpy as np
import g1common as C
import g1measure as M
from run_qa import CLIPS, cid_of

m = C.load_model(); surf = C.load_surface()
st = json.load(open(C.RDIR / 'staging.json'))
ac = json.load(open(C.QADIR / 'assets' / 'assets_check.json'))
bn = [C.name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
breg = surf['body_region']
P = np.load(C.PDIR / 'contact_patches.npz')
corr = Path('/home/arenalabs/umr_factory_data/corr') / C.ROBOT
cfg = dict(
    schema='umr_factory per-robot pipeline v2', robot=C.ROBOT, frozen=time.strftime('%Y-%m-%d %H:%M:%S'),
    engine=dict(reference='CPU umrf (umr_factory_code: pipeline.run_umr -> postprocess), UMR d1cdaed',
                runner=str(C.PDIR / 'run_qa.py'), gpu_engine='/home/arenalabs/umr_gpu (separate agent; must reproduce this config)',
                umr_python='/var/tmp/alphamotion/umr_factory/umr/env/bin/python', smplx_device='cpu'),
    assets=dict(
        mjcf=str(C.XML), mjcf_sha256=C.sha256(C.XML, None), source_mjcf=st['src_mjcf'], nq=m.nq, nv=m.nv,
        surface=str(C.SURF), surface_sha256=C.sha256(C.SURF, None), n_dense_points=int(len(surf['region'])),
        n_hull_vertices=int(len(surf['hull_local'])),
        staging=str(C.RDIR / 'staging.json'), staging_sha256=C.sha256(C.RDIR / 'staging.json', None),
        umr_robot_json=str(C.RDIR / 'umr_robot.json'), contact_patches=str(C.PDIR / 'contact_patches.npz'),
        contact_patches_sha256=C.sha256(C.PDIR / 'contact_patches.npz', None),
        asset_check=str(C.QADIR / 'assets' / 'assets_check.json')),
    scale_and_correspondence=dict(
        smpl_scale='per clip by UMR = robot_height / human_height (robot_height 1.32246 m T-pose); dance 0.7155, '
                   'box 0.7699 (stored per output as human_scale)',
        point_cloud_center='body:waist_yaw_link', bbox_center_ratio=0.598916,
        smplx_plain=dict(dataset=str(corr / 'smplx/correspondence_dataset.npz'), slots=str(corr / 'smplx/train/correspondence_slots_final.npz'),
                         slots_sha256=ac['correspondence']['smplx'].get('slots_sha256'), names=ac['correspondence']['smplx'].get('slots_names')),
        shared_beta0=dict(dataset=str(corr / 'beta0/correspondence_dataset.npz'), slots=str(corr / 'beta0/train/correspondence_slots_final.npz'),
                          slots_sha256=ac['correspondence']['beta0'].get('slots_sha256'), names=ac['correspondence']['beta0'].get('slots_names'),
                          use='HOI / HiPHI / any SMPL-X body shape (no per-clip GPU training)')),
    regions=dict(names=C.REGIONS,
                 region_to_bodies={r: sorted({bn[b] for b in range(1, m.nbody) if breg[b] == i}) for i, r in enumerate(C.REGIONS)},
                 n_points={r: int((surf['region'] == i).sum()) for i, r in enumerate(C.REGIONS)},
                 human_side='SMPL-X part segmentation -> same 13 regions (umrf.surface.SMPLX_PART_REGION)'),
    contact=dict(
        capable_regions={'ground': ['L_foot', 'R_foot', 'L_shin', 'R_shin', 'L_thigh', 'R_thigh', 'L_hand', 'R_hand',
                                    'L_forearm', 'R_forearm', 'L_upper_arm', 'R_upper_arm', 'trunk'],
                         'object': ['L_hand', 'R_hand', 'L_forearm', 'R_forearm', 'L_upper_arm', 'R_upper_arm', 'trunk',
                                    'L_thigh', 'R_thigh', 'L_foot', 'R_foot', 'L_shin', 'R_shin']},
        patches={k: dict(n_points=int(len(P[k])), rule=v) for k, v in {
            'L_sole': 'L_foot points within 20 mm of the sole plane at rest (sole + rim/edges: point/edge contact allowed)',
            'R_sole': 'same, right', 'L_knee': 'L_shin/L_thigh points within 8 cm of the knee joint anchor',
            'R_knee': 'same, right', 'L_shin_front': 'L_shin points in front of the knee-ankle mid line (kneeling)',
            'R_shin_front': 'same, right', 'L_elbow': 'upper-arm/forearm points within 6 cm of the elbow anchor',
            'R_elbow': 'same, right', 'L_hand': 'all L_hand points (wrist roll/pitch/yaw links + fixed rubber hand)',
            'R_hand': 'same, right', 'pelvis': 'trunk points on pelvis + pelvis_contour_link (sitting/lying)',
            'torso': 'trunk points on torso_link except head mesh (lying prone/supine, box against chest)',
            'head': 'head mesh points (contact allowed only if the human head touches)'}.items()},
        patch_point_indices='contact_patches.npz (int32 indices into surface.npz points)',
        expected_patch_per_region={'L_foot': 'L_sole', 'R_foot': 'R_sole', 'L_shin': 'L_knee|L_shin_front', 'R_shin': 'R_knee|R_shin_front',
                                   'L_hand': 'L_hand', 'R_hand': 'R_hand', 'L_forearm': 'L_elbow', 'R_forearm': 'R_elbow',
                                   'trunk': 'pelvis|torso|head'},
        sole_point_spacing_mm_p50=14.1,
        known_resolution_limit='87/90 dense points per sole (14 mm spacing): tilted heel/toe/edge contacts leave the '
                               'nearest DENSE point 0.5-4 mm above the floor while the mesh touches; see qa report'),
    self_collision=dict(
        staged_exclude_list=st['exclusions'], mjcf_nexclude=int(m.nexclude), filterparent=True,
        mujoco_implicit_filters=ac['exclusions']['implicitly_filtered_body_pairs'],
        nonadjacent_filtered=ac['exclusions']['nonadjacent_filtered'],
        pairs_checked=ac['exclusions']['n_geom_pairs_checked_by_mujoco'], pairs_total=ac['exclusions']['n_geom_pairs_total'],
        rest_or_tpose_overlaps=ac['exclusions']['nonfiltered_pairs_penetrating_at_rest'] + ac['exclusions']['nonfiltered_pairs_penetrating_at_tpose'],
        always_overlapping=ac['exclusions']['always_overlapping_candidates'],
        verdict='only adjacent (parent-child through one joint) or rigidly welded pairs are skipped; no explicit '
                'exclusions needed; shoulder/elbow/wrist vs torso overlap in 4-22 % of random in-limit poses = real '
                'collisions, kept'),
    floor=dict(convention='calibrated floor = z 0 in every output (qpos root z shifted by -floor); raw value stored as '
                          'floor_offset_removed; floor_height = 0',
               calibration='per session (here per clip) from the human source: median lowest foot point over quasi-static stance frames',
               robot_rest_sole_z_m=0.0007),
    thresholds=dict(DATA_SPEC_v2=dict(contact_min_clearance_mm=[0.0, 1.0], penetration_tol_mm=0.5, slip_max_cms=1.0,
                                      min_contact_segment_frames=3, hysteresis_enter_mm=1.0, hysteresis_leave_mm=2.0,
                                      human_contact='hysteresis enter < 1 cm, leave > 2 cm, >= 3 frames (same ruler as robot labels)',
                                      human_off_cm=2.0, robot_off_when_human_off_mm=5.0, fps=30),
                    audit_hover_band_mm=[1.0, 20.0]),
    quirks=[
        '23 DoF = GMR 29-DoF G1 with waist_roll/waist_pitch and wrist_pitch/wrist_yaw joints REMOVED (nq 30); the '
        'removed links are welded (waist_roll_link + torso_link to waist_yaw_link; wrist pitch/yaw links to wrist_roll_link).',
        'Hands are fixed rubber hands (no finger DoF). left/right_rubber_hand have NO collision geom: hand-object and '
        'hand-ground contact can only be verified with dense points + the wrist link collision meshes; in MuJoCo '
        'physics a palm/finger contact does not exist.',
        'Feet: collision = 4 spheres r 5 mm per foot (corners at x -0.05/+0.12 m), bottoms 0.41 mm above the flat '
        'visual sole; visual sole overhangs the spheres by 1.1 cm (heel) and 1.7 cm (toe) -> heel/toe-edge contacts '
        'and any top/side-of-foot contact (soccer kick) have no physics counterpart.',
        'waist_yaw_link and waist_roll_link have no collision geom (covered by pelvis_contour and torso meshes).',
        'Head mesh is merged into torso_link (head_link body has no geom).',
        'G1 elbow = 0 rad means forearm horizontal (90 deg flexion) at qpos0.',
        'Sole has only ~88 dense points (14 mm spacing) -> spec-literal dense-point contact gauge under-reports edge contacts.'],
    qa_clip_set={k: dict(cid=cid_of(k, c), **c) for k, c in CLIPS.items()},
)
json.dump(cfg, open(C.PDIR / 'pipeline.json', 'w'), indent=1)
print('wrote', C.PDIR / 'pipeline.json')
