import json, os
from surface_retargeting.config import CFG as SRCFG
OUT = f'{SRCFG.factory_root}/robots'   # registry work dir (outputs); code lives in this package
HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f'{OUT}/robots_raw.json'))
EXCL = {'galaxea_r1pro': 'wheeled base, no freejoint, no visual-only meshes',
        'rby1': 'wheeled base (no legs/feet)', 'agility_cassie': 'no arms, closed-chain legs (4 eq), not humanoid',
        'berkeley_humanoid': 'no arms (legs only)'}
TIER = {}
for n in ['booster_k1', 'booster_t1', 'booster_t1_29dof', 'fourier_n1', 'hightorque_hi', 'pnd_adam_lite', 'tienkung',
          'unitree_g1_23dof', 'unitree_g1_29dof', 'unitree_h1', 'unitree_h1_2', 'fourier_gr3', 'kuavo_s45', 'unitree_h2',
          'unitree_r1', 'engineai_pm01', 'pal_talos', 'agibot_x2_ultra', 'agibot_a3_t3d0', 'agibot_a2', 'openloong']:
    TIER[n] = 'A_core20'
for n in ['unitree_g1_29dof_dex3', 'agibot_a3_ultra_t3d0']:
    TIER[n] = 'B_dexterous_variant(object contact)'
for n in ['unitree_g1_29dof_vendor', 'unitree_g1_23dof_vendor']:
    TIER[n] = 'B_alt_mjcf'
for n in ['apptronik_apollo', 'berkeley_humanoid_lite', 'stanford_toddy', 'toddlerbot_2xm', 'robotis_op3']:
    TIER[n] = 'C_optional_extra'
EXTRA_FLAGS = {
    'unitree_g1_23dof': ['GMR XML is the 29-DoF tree with 6 joints locked; UMR should use a true 23-DoF MJCF (unitree_g1_23dof_vendor) or lock joints via joint_limits'],
    'unitree_g1_29dof': ['GMR g1_mocap_29dof.xml has hand-narrowed joint limits (15/29, per prep_xwbc_cache LIMIT_URDF note) and head_link has mass but no geom -> head missing from point cloud; prefer unitree_g1_29dof_vendor for UMR'],
    'pal_talos': ['GMR talos.xml has no gripper joints (XWBC data has gripper_left/right_joint, dropped); 20.6 mm geometric mismatch vs XWBC data (prep_xwbc_cache DATA_TREE)'],
    'unitree_h2': ['~40 mm geometric mismatch between unitree_ros MJCF and XWBC data on several bodies (prep_xwbc_cache DATA_TREE); UMR ships its own H2_stl_mjcf.xml'],
    'fourier_gr3': ['asymmetric foot collision: right foot collision on right_foot_pitch_link (does not follow ankle roll), left on left_foot_roll_link; proposed T-pose arm goes 17 cm backward / +7 cm up -> set shoulder_pitch/yaw in UMR Studio'],
    'pnd_adam_lite': ['degenerate mesh torso_0 (9 verts / 3 faces); wristYaw_Left/Right unlimited; hands have no collision geoms; T-pose hand +5 cm above shoulder'],
    'engineai_pm01': ['foot collision lives on child body LINK_FOOT_L/R (fixed), not on ankle-roll body: fine for MuJoCo, note for foot-body label code'],
    'agibot_x2_ultra': ['agibot_x2_ultra_mjcf.xml is a 2-line <include> wrapper of vendor X2-Ultra.xml (identical model); proposed T-pose hand +9.5 cm above shoulder -> check in UMR Studio; qpos0 root z=0.68'],
    'agibot_a3_t3d0': ['qpos0 root z=1.3 leaves soles 22.4 cm above floor (convention only, UMR solves root)'],
    'agibot_a3_ultra_t3d0': ['12 equality constraints (finger couplings); 16 hand DoF/side'],
    'unitree_g1_29dof_dex3': ['Dex3 hands 7 DoF/side; same body/leg geometry as G1 29dof'],
    'apptronik_apollo': ['ALL 18 collision geoms have contype=conaffinity=0 (menagerie class collision group 3, no <pair>): no physics contact until contype/conaffinity fixed; UMR point clouds unaffected (visual meshes)'],
    'toddlerbot_2xm': ['only 2/26 collision geoms physics-enabled (contacts via 65 explicit <pair>); 13 eq constraints (gear/linkage)'],
    'robotis_op3': ['12+ joints without limits (UMR joint-limit box missing)'],
    'berkeley_humanoid_lite': ['root body "base" frame is at foot level (body z=0, meshes at 0.7 m): body:base gives bbox ratio -0.05 -> use point_cloud_center "bbox_ratio_0.6" or body:imu_2; XWBC data exists (77 clips) but excluded from AM2 cache (prep_xwbc_cache EXCLUDE)'],
    'stanford_toddy': ['arm bodies have no side tokens (names via joints only); 0.56 m tall'],
    'unitree_r1': ['qpos0 root z=0 (soles at -0.74 m)'],
    'booster_k1': ['qpos0 root z=0 (soles at -0.53 m); UMR ships its own K1_serial_tpose_mjcf.xml'],
    'booster_t1': ['qpos0 root z=1.68 (soles 1.0 m above floor)'],
}
rows = []
for r in R:
    n = r['name']
    flags = list(EXTRA_FLAGS.get(n, []))
    if not r.get('load_ok'):
        flags.append('LOAD FAIL: ' + r.get('load_error', ''))
    else:
        if not r.get('feet_have_collision'):
            flags.append('feet: missing collision geoms or feet not found')
        if r['feet'] and not all(v['physics_enabled'] for v in r['feet'].values()):
            flags.append('feet collision geoms not physics-enabled (contype/conaffinity=0)')
        if not r.get('hands_have_collision') and r.get('hands'):
            flags.append('hands: no collision geoms near hand body')
        rat = r.get('pcc_bbox_ratio_tpose')
        if rat is not None and not (0.45 <= rat <= 0.75):
            flags.append(f'point_cloud_center bbox ratio {rat} outside [0.45,0.75]')
        if r['umr_surface_policy'] != 'visual_mesh':
            flags.append('UMR surface falls back to ' + r['umr_surface_policy'])
        if r.get('degenerate_meshes'):
            flags.append('degenerate meshes: ' + ','.join(r['degenerate_meshes']))
    fl = {k: v for k, v in r.items()}
    fl['tier'] = 'X_excluded: ' + EXCL[n] if n in EXCL else TIER.get(n, '?')
    fl['flags'] = flags
    fl['needs_fix'] = any(('LOAD FAIL' in f or 'not physics-enabled' in f or 'missing collision' in f or 'degenerate' in f
                           or 'asymmetric' in f or 'outside' in f) for f in flags)
    rows.append(fl)
    if r.get('load_ok') and n not in EXCL:
        cfg = {'robot': {'name': n, 'xml': r['mjcf'],
                         'point_cloud_center': 'bbox_ratio_0.6' if n == 'berkeley_humanoid_lite' else r['umr_point_cloud_center'],
                         'tpose_qpos': r.get('umr_tpose_qpos_proposed') or {}},
               'correspondence': {'dataset': {'smpl_center': 'bbox_ratio',
                                              'bbox_center_ratio': 0.6 if n == 'berkeley_humanoid_lite' else r.get('pcc_bbox_ratio_tpose')}},
               '_provenance': 'DRAFT auto-generated by umr_factory/robots/inventory.py; tpose_qpos = heuristic (verify in UMR Studio); bbox_center_ratio = point_cloud_center z-ratio in proposed T-pose (method reproduces UMR booster_k1 0.5662 and unitree_h2 0.6157 examples)'}
        json.dump(cfg, open(f'{OUT}/umr_configs/humanoid_retarget_{n}.json', 'w'), indent=2)
doc = {'schema': 'umr_factory_robot_inventory_v1', 'generated': '2026-10-04', 'mujoco': '3.6.0 (ardy env, read-only)',
       'code': [f'{HERE}/inventory.py', f'{HERE}/finalize.py', f'{HERE}/render_tpose.py'],
       'tpose_contact_sheet': f'{OUT}/png/contact_sheet_tpose.png',
       'umr_reference_configs': f'{OUT}/umr_ref',
       'not_found_locally': {'booster_t3': 'no MJCF/URDF anywhere on / (bounded find); needs vendor download',
                             'engineai_t800 / hightorque_pipluspro': f'only in UMR repo assets (not cloned locally; {SRCFG.umr_root} is empty)',
                             'roboterax_star1': 'URDF only (models_roboterax/star1), no MJCF'},
       'robots': rows}
json.dump(doc, open(f'{OUT}/robots.json', 'w'), indent=1)
# table
hdr = '| robot | tier | AM2 | nq | hinge | H (m) | vis mesh | col (phys) | feet col | hands | pcc / ratio | caches | flags |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n'
lines = []
for r in rows:
    if not r.get('load_ok'):
        continue
    f = r['feet']
    fc = '/'.join(str(f[s]['n_collision']) for s in ('L', 'R') if s in f) or '-'
    hd = r['hand_type'] + ('' if not r['hands'] else f"({max(v['n_hand_dof'] for v in r['hands'].values())}dof)")
    lines.append(f"| {r['name']} | {r['tier'][:28]} | {r['am2_role'].split('(')[0]} | {r['nq']} | {r['n_hinge']} | {r.get('height_tpose_m')} | {r['n_visual_mesh_geoms']} | {r['n_collision_geoms']} ({r['n_collision_geoms_physics_enabled']}) | {fc} | {hd} | {r['umr_point_cloud_center']} / {r.get('pcc_bbox_ratio_tpose')} | {len(r['topo_exp_caches'])} | {'FIX ' if r['needs_fix'] else ''}{len(r['flags'])} |")
open(f'{OUT}/robots_table.md', 'w').write(hdr + '\n'.join(lines) + '\n')
print(hdr + '\n'.join(lines))
