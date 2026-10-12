"""Robot MJCF inventory for the UMR data factory (read-only on all inputs).

Run: /home/arenalabs/miniconda3/envs/ardy/bin/python inventory.py
Writes robots.json + robots_table.md next to this file.  CPU only.
"""
import json, os, re, sys, glob, hashlib, traceback
from pathlib import Path
import numpy as np
import mujoco

OUT = Path(__file__).resolve().parent
GMR = '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/external/GMR/assets'
RM = '/var/tmp/alphamotion/robot_models'
SOLE = '/home/arenalabs/Desktop/alphamotion video/resource/report/32_ground_gauge_audit/sole_audit.json'
TOPO = '/var/tmp/alphamotion/topo_exp'
XWBC_EXT = '/var/tmp/alphamotion/xwbc_lafan/extracted'
SURF = '/var/tmp/alphamotion/am2exp/assets/surface_v2'

TRAIN = ['booster_k1', 'booster_t1', 'booster_t1_29dof', 'fourier_n1', 'hightorque_hi', 'pnd_adam_lite',
         'tienkung', 'unitree_g1_23dof', 'unitree_g1_29dof', 'unitree_h1', 'unitree_h1_2']
HELDOUT = ['fourier_gr3', 'kuavo_s45', 'unitree_h2', 'unitree_r1']

sole = json.load(open(SOLE))
CAND = {}
for b, v in sole.items():
    if b == 'human_smpl':
        continue
    m = v.get('mjcf')
    CAND[b] = dict(mjcf=m, source='sole_audit.json (report 32)')
CAND['pal_talos'] = dict(mjcf=f'{GMR}/pal_talos/talos.xml', source='GMR params.py ROOT_XML_DICT[pal_talos] (sole_audit had None)')
CAND['unitree_g1_29dof_dex3'] = dict(mjcf=f'{RM}/unitree_ros_official/robots/g1_description/g1_29dof_with_hand.xml',
                                     source='unitree_ros_official (vendor G1 + Dex3 hands)')
CAND['agibot_x2_ultra'] = dict(mjcf=f'{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml', source='robot_models (used by prior X2 zero-shot code, 23 refs)')
CAND['unitree_g1_29dof_vendor'] = dict(mjcf=f'{RM}/unitree_ros_official/robots/g1_description/g1_29dof.xml', source='unitree_ros_official vendor G1 29dof (alt. to GMR mocap XML)')
CAND['unitree_g1_23dof_vendor'] = dict(mjcf=f'{RM}/unitree_ros_official/robots/g1_description/g1_23dof.xml', source='unitree_ros_official vendor G1 23dof (true 23-DoF tree)')
CAND['agibot_a3_t3d0'] = dict(mjcf=f'{RM}/agibot_a3/a3_t3d0/mjcf/agibot_a3_t3d0_mjcf.xml', source='robot_models')
CAND['agibot_a3_ultra_t3d0'] = dict(mjcf=f'{RM}/agibot_a3/a3_ultra_t3d0/mjcf/agibot_a3_ultra_t3d0_mjcf.xml', source='robot_models')
CAND['agibot_a2'] = dict(mjcf=f'{RM}/agibot_a2_gmr/agibot_a2_gmr_mjcf.xml', source='robot_models (GMR agibot_a2 converted)')
CAND['openloong'] = dict(mjcf=f'{GMR}/openloong/AzureLoong.xml', source='GMR assets (not in params.py dict)')
CAND['berkeley_humanoid_lite'] = dict(mjcf=f'{GMR}/berkeley_humanoid_lite/berkeley_humanoid_lite.xml', source='GMR assets')
CAND['stanford_toddy'] = dict(mjcf=f'{GMR}/stanford_toddy/toddy_mocap.xml', source='GMR assets')
CAND['apptronik_apollo'] = dict(mjcf=f'{RM}/menagerie/apptronik_apollo/apptronik_apollo.xml', source='mujoco_menagerie')
CAND['toddlerbot_2xm'] = dict(mjcf=f'{RM}/menagerie/toddlerbot_2xm/toddlerbot_2xm.xml', source='mujoco_menagerie')
CAND['robotis_op3'] = dict(mjcf=f'{RM}/menagerie/robotis_op3/op3.xml', source='mujoco_menagerie')
CAND['berkeley_humanoid'] = dict(mjcf=f'{RM}/menagerie/berkeley_humanoid/berkeley_humanoid.xml', source='mujoco_menagerie')
CAND['galaxea_r1pro'] = dict(mjcf=f'{GMR}/galaxea_r1pro/r1_pro.xml', source='GMR assets (wheeled base)')
CAND['agility_cassie'] = dict(mjcf=f'{RM}/menagerie/agility_cassie/cassie.xml', source='mujoco_menagerie (no arms)')
CAND['rby1'] = dict(mjcf=f'{RM}/menagerie/rainbow_robotics_rby1/rby1m_1.3.xml', source='mujoco_menagerie (wheeled)')

J = mujoco.mjtJoint
G = mujoco.mjtGeom


def bname(m, i):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f'body{i}'


def jname(m, i):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or f'joint{i}'


def subtree(m, root):
    out = [root]
    for b in range(m.nbody):
        p = b
        while p > 0:
            p = m.body_parentid[p]
            if p == root:
                out.append(b)
                break
    return out


def ancestors(m, b):
    a = []
    while b > 0:
        a.append(b)
        b = m.body_parentid[b]
    return a


def geom_world_points(m, d, gids):
    pts = []
    for g in gids:
        R = d.geom_xmat[g].reshape(3, 3)
        p = d.geom_xpos[g]
        t = m.geom_type[g]
        if t == G.mjGEOM_MESH and m.geom_dataid[g] >= 0:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            pts.append(v @ R.T + p)
        elif t == G.mjGEOM_PLANE:
            continue
        else:
            s = m.geom_size[g]
            if t == G.mjGEOM_SPHERE:
                e = np.array([s[0]] * 3)
            elif t in (G.mjGEOM_CAPSULE,):
                e = np.array([s[0], s[0], s[1] + s[0]])
            elif t == G.mjGEOM_CYLINDER:
                e = np.array([s[0], s[0], s[1]])
            else:
                e = s[:3]
            c = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * e
            pts.append(c @ R.T + p)
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def is_visual(m, g):
    # UMR definition (mujoco_geom_surface.mesh_geom_ids): contype==0 & conaffinity==0 & group!=3
    return m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and m.geom_group[g] != 3


def collides(m, g):
    return m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0


def floor_like(m, g):
    lab = (bname(m, m.geom_bodyid[g]) + ' ' + (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or '')).lower()
    return m.geom_bodyid[g] == 0 or 'floor' in lab or 'ground' in lab or m.geom_type[g] == G.mjGEOM_PLANE


def umr_surface_gids(m):
    """Mirror of UMR mujoco_geom_surface.surface_geom_ids(policy='auto')."""
    ids = [g for g in range(m.ngeom) if m.geom_type[g] == G.mjGEOM_MESH]
    vis = [g for g in ids if is_visual(m, g) and m.geom_group[g] != 3]
    fv = [g for g in vis if not floor_like(m, g)]
    if fv:
        return fv, 'visual_mesh'
    fm = [g for g in ids if not floor_like(m, g)]
    if fm:
        return fm, 'all_mesh(no visual-only meshes)'
    prim = [g for g in range(m.ngeom) if m.geom_type[g] in (G.mjGEOM_SPHERE, G.mjGEOM_CAPSULE, G.mjGEOM_ELLIPSOID,
                                                           G.mjGEOM_CYLINDER, G.mjGEOM_BOX) and not floor_like(m, g)]
    return prim, 'primitives'


SIDE_L = re.compile(r'(^|[_\W])(l|left)([_\W]|$)|left|_l$|^l_|L_|_L\b|Left', re.I)


def side_of(name):
    n = name
    nl = n.lower()
    if re.search(r'left|(^|_)l(_|$)|_l\d|^l\d|(^|_)l[a-z]*$', nl) and not re.search(r'right', nl):
        if re.search(r'left', nl) or re.search(r'(^|_)l(_|$)', nl) or nl.endswith('_l') or nl.startswith('l_'):
            return 'L'
    if re.search(r'right', nl) or re.search(r'(^|_)r(_|$)', nl) or nl.endswith('_r') or nl.startswith('r_'):
        return 'R'
    # vendor styles: leg_l5_link, zarm_l7, LINK_ANKLE_ROLL_L, anklePitchLeft
    if re.search(r'(leg|arm)_?l\d', nl) or re.search(r'left', nl):
        return 'L'
    if re.search(r'(leg|arm)_?r\d', nl) or re.search(r'right', nl):
        return 'R'
    return None


HAND_KW = re.compile(r'finger|thumb|index|middle|ring|pinky|little|gripper|hand_|_hand|claw|palm|dex|grip', re.I)


def label(m, b):
    js = [jname(m, j) for j in range(m.njnt) if m.jnt_bodyid[j] == b]
    for t in [bname(m, b)] + js:
        s = side_of(t)
        if s:
            return t
    return bname(m, b)


def analyse(name, path):
    r = dict(name=name, mjcf=path)
    if path is None or not os.path.exists(path):
        r['load_ok'] = False
        r['load_error'] = 'file missing'
        return r
    r['mjcf_sha256'] = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]
    try:
        m = mujoco.MjModel.from_xml_path(path)
    except Exception as e:
        r['load_ok'] = False
        r['load_error'] = str(e).splitlines()[0][:300]
        return r
    r['load_ok'] = True
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r['nq'], r['nv'], r['nu'], r['nbody'], r['ngeom'], r['nmesh'] = int(m.nq), int(m.nv), int(m.nu), int(m.nbody), int(m.ngeom), int(m.nmesh)
    jt = [int(m.jnt_type[j]) for j in range(m.njnt)]
    r['n_free'] = jt.count(J.mjJNT_FREE)
    r['n_hinge'] = jt.count(J.mjJNT_HINGE)
    r['n_slide'] = jt.count(J.mjJNT_SLIDE)
    r['n_ball'] = jt.count(J.mjJNT_BALL)
    r['n_eq'] = int(m.neq)
    r['n_contact_pairs'] = int(m.npair)
    r['n_keyframes'] = int(m.nkey)
    if any(t == J.mjJNT_FREE for t in m.jnt_type):
        r['qpos0_root_xyz'] = [round(float(x), 4) for x in m.qpos0[:3]]
    r['has_freejoint'] = r['n_free'] > 0
    root_children = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0]
    r['root_body'] = bname(m, root_children[0]) if root_children else None
    r['n_root_children'] = len(root_children)
    hinge = [j for j in range(m.njnt) if m.jnt_type[j] in (J.mjJNT_HINGE, J.mjJNT_SLIDE)]
    lim = [j for j in hinge if m.jnt_limited[j]]
    r['n_limited'] = len(lim)
    r['unlimited_joints'] = [jname(m, j) for j in hinge if not m.jnt_limited[j]][:12]
    zero_out = [jname(m, j) for j in lim if not (m.jnt_range[j, 0] - 1e-6 <= m.qpos0[m.jnt_qposadr[j]] <= m.jnt_range[j, 1] + 1e-6)]
    r['qpos0_outside_limits'] = zero_out
    # geoms
    vis = [g for g in range(m.ngeom) if is_visual(m, g) and not floor_like(m, g)]
    col = [g for g in range(m.ngeom) if not is_visual(m, g) and not floor_like(m, g)]
    vis_mesh = [g for g in vis if m.geom_type[g] == G.mjGEOM_MESH]
    col_mesh = [g for g in col if m.geom_type[g] == G.mjGEOM_MESH]
    r['n_visual_geoms'], r['n_visual_mesh_geoms'] = len(vis), len(vis_mesh)
    r['n_collision_geoms'], r['n_collision_mesh_geoms'] = len(col), len(col_mesh)
    r['n_collision_geoms_physics_enabled'] = len([g for g in col if collides(m, g)])
    r['collision_geom_types'] = sorted({mujoco.mjtGeom(int(m.geom_type[g])).name.replace('mjGEOM_', '') for g in col})
    r['floor_geom_in_model'] = any(floor_like(m, g) for g in range(m.ngeom))
    sg, pol = umr_surface_gids(m)
    r['umr_surface_policy'] = pol
    r['umr_surface_n_geoms'] = len(sg)
    # degenerate meshes
    bad = []
    for k in range(m.nmesh):
        if m.mesh_facenum[k] < 4 or m.mesh_vertnum[k] < 4:
            bad.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, k))
    r['degenerate_meshes'] = bad
    # bodies with mass but no surface geom
    sg_bodies = {int(m.geom_bodyid[g]) for g in sg}
    r['bodies_without_surface_geom'] = [bname(m, b) for b in range(1, m.nbody) if b not in sg_bodies and m.body_mass[b] > 0.05][:15]
    # height at qpos0 from surface geoms
    P = geom_world_points(m, d, sg)
    if len(P):
        lo, hi = P.min(0), P.max(0)
        r['height_qpos0_m'] = round(float(hi[2] - lo[2]), 4)
        r['min_z_qpos0_m'] = round(float(lo[2]), 4)
        r['extent_xy_qpos0_m'] = [round(float(hi[0] - lo[0]), 3), round(float(hi[1] - lo[1]), 3)]
    # feet: lowest bodies at qpos0 per side
    leaf_low = []
    for b in range(1, m.nbody):
        gb = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and not floor_like(m, g)]
        if not gb:
            continue
        pz = geom_world_points(m, d, gb)[:, 2].min()
        leaf_low.append((pz, b))
    leaf_low.sort()
    feet = {}
    for pz, b in leaf_low:
        # side by ancestry name
        s = None
        for a in ancestors(m, b):
            s = side_of(label(m, a))
            if s:
                break
        if s and s not in feet and pz < (leaf_low[0][0] + 0.05):
            feet[s] = (b, pz)
    r['feet'] = {}
    for s, (b, pz) in feet.items():
        near = set(subtree(m, b)) | {int(m.body_parentid[b])}
        cg = [g for g in range(m.ngeom) if m.geom_bodyid[g] in near and not is_visual(m, g)]
        r['feet'][s] = dict(body=bname(m, b), min_z=round(float(pz), 4), n_collision=len(cg),
                            n_collision_on_foot_body=len([g for g in cg if m.geom_bodyid[g] == b]),
                            physics_enabled=bool(cg) and all(collides(m, g) for g in cg),
                            collision_types=sorted({mujoco.mjtGeom(int(m.geom_type[g])).name.replace('mjGEOM_', '') for g in cg}))
    r['feet_have_collision'] = bool(feet) and all(v['n_collision'] > 0 for v in r['feet'].values()) and len(feet) == 2
    # hands: arm chains = side bodies whose names contain shoulder/arm, find most distal descendant
    hand_joints = [jname(m, j) for j in hinge if HAND_KW.search(jname(m, j)) and not re.search(r'wrist|elbow|shoulder', jname(m, j), re.I)]
    hands = {}
    for s in ('L', 'R'):
        sh = [b for b in range(1, m.nbody) if side_of(label(m, b)) == s and re.search(r'shoulder|sho_|arm|zarm|upper_arm', label(m, b), re.I)]
        if not sh:
            continue
        top = min(sh, key=lambda b: len(ancestors(m, b)))
        st = subtree(m, top)
        deepest = max(st, key=lambda b: len(ancestors(m, b)))
        # wrist/hand body: deepest non-finger body on chain
        chain = [b for b in reversed(ancestors(m, deepest)) if b in st]
        hb = [b for b in chain if not re.search(r'finger|thumb|index|middle|ring|pinky|little|tip', bname(m, b), re.I)]
        hand_body = hb[-1] if hb else deepest
        sub_j = [j for j in hinge if m.jnt_bodyid[j] in subtree(m, hand_body) and m.jnt_bodyid[j] != hand_body]
        near = set(subtree(m, hand_body)) | {int(m.body_parentid[hand_body])}
        cg = [g for g in range(m.ngeom) if m.geom_bodyid[g] in near and not is_visual(m, g)]
        arm_j = [jname(m, j) for j in hinge if m.jnt_bodyid[j] in chain]
        hands[s] = dict(hand_body=bname(m, hand_body), n_hand_dof=len(sub_j), n_collision=len(cg), arm_chain_joints=arm_j[:12],
                        top_body=bname(m, top))
    r['hands'] = hands
    nh = max([v['n_hand_dof'] for v in hands.values()] + [0])
    r['hand_type'] = 'none/fixed' if nh == 0 else ('gripper' if nh <= 2 else 'dexterous')
    r['hand_like_joints'] = hand_joints[:20]
    r['hands_have_collision'] = bool(hands) and all(v['n_collision'] > 0 for v in hands.values())
    # point cloud center suggestion
    cands = [b for b in range(1, m.nbody) if re.search(r'waist_yaw|torso|trunk|chest|waist', bname(m, b), re.I)
             and m.body_mass[b] > 0.2 and not re.search(r'frame|camera|imu|gear', bname(m, b), re.I)]
    pcc = None
    for pat in (r'waist_yaw', r'torso|trunk|chest', r'waist'):
        c = [b for b in cands if re.search(pat, bname(m, b), re.I)]
        if c:
            pcc = min(c, key=lambda b: len(ancestors(m, b)))
            break
    if pcc is None and root_children:
        pcc = root_children[0]
    r['umr_point_cloud_center'] = f'body:{bname(m, pcc)}' if pcc else None
    r['_m'] = m
    r['_pcc'] = pcc
    r['_sg'] = sg
    return r


def tpose_search(r):
    """Propose tpose_qpos: per arm, pick (joint, angle) among first 3 arm joints that brings the hand most lateral,
    then pick elbow angle maximising shoulder-hand distance. Heuristic; must be eyeballed in UMR Studio."""
    m = r['_m']
    d = mujoco.MjData(m)
    q = m.qpos0.copy()
    out = {}
    notes = []
    for s, sign in (('L', 1.0), ('R', -1.0)):
        h = r['hands'].get(s)
        if not h:
            continue
        hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, h['hand_body'])
        js = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in h['arm_chain_joints']]
        js = [j for j in js if m.jnt_type[j] == J.mjJNT_HINGE]
        if len(js) < 2:
            continue
        tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, h['top_body'])

        def hand_pos(qq):
            d.qpos[:] = qq
            mujoco.mj_kinematics(m, d)
            return d.xpos[hb].copy(), d.xpos[tb].copy()

        hp0, sp0 = hand_pos(q)
        notes.append(f'{s} qpos0: hand-shoulder dy={hp0[1]-sp0[1]:+.3f} dz={hp0[2]-sp0[2]:+.3f} m')
        reach = np.linalg.norm(hp0 - sp0)
        if sign * (hp0[1] - sp0[1]) > 0.85 * reach and abs(hp0[2] - sp0[2]) < 0.05:
            notes.append(f'{s}: qpos0 already T-pose (no change)')
            continue
        best = None
        for j in js[:3]:
            for a in (np.pi / 2, -np.pi / 2):
                qq = q.copy()
                lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.inf, np.inf)
                qq[m.jnt_qposadr[j]] = np.clip(a, lo, hi)
                hp, sp = hand_pos(qq)
                # T-pose score: lateral outward, same height as shoulder
                score = sign * (hp[1] - sp[1]) - 1.0 * abs(hp[2] - sp[2]) - 0.5 * abs(hp[0] - sp[0])
                if best is None or score > best[0]:
                    best = (score, j, float(qq[m.jnt_qposadr[j]]))
        qq = q.copy()
        qq[m.jnt_qposadr[best[1]]] = best[2]
        picks = {jname(m, best[1]): round(best[2], 6)}
        # elbow: try remaining chain joints to straighten
        elb = [j for j in js if re.search(r'elbow|(^|_)el(_|$)|_el\b', jname(m, j), re.I)] or js[3:4]
        for j in elb[:1]:
            lo, hi = m.jnt_range[j] if m.jnt_limited[j] else (-np.pi, np.pi)
            grid = np.unique(np.concatenate([np.linspace(max(lo, -np.pi), min(hi, np.pi), 721), [a for a in (np.pi/2, -np.pi/2, 0.0) if lo <= a <= hi]]))
            sc = []
            for a in grid:
                q2 = qq.copy()
                q2[m.jnt_qposadr[j]] = a
                hp, sp = hand_pos(q2)
                sc.append(sign * (hp[1] - sp[1]) - 1.0 * abs(hp[2] - sp[2]))
            sc = np.asarray(sc); a = float(grid[int(np.argmax(sc))])
            for snap in (np.pi / 2, -np.pi / 2):  # prefer exact +-pi/2 if within 1 mm of the optimum
                if lo <= snap <= hi:
                    q2 = qq.copy(); q2[m.jnt_qposadr[j]] = snap; hp, sp = hand_pos(q2)
                    if sign * (hp[1] - sp[1]) - abs(hp[2] - sp[2]) > sc.max() - 0.001:
                        a = snap
            base = sign * (hand_pos(qq)[0][1] - hand_pos(qq)[1][1]) - abs(hand_pos(qq)[0][2] - hand_pos(qq)[1][2])
            if max(sc) > base + 0.01:
                qq[m.jnt_qposadr[j]] = a
                picks[jname(m, j)] = round(a, 6)
        hp, sp = hand_pos(qq)
        out.update(picks)
        notes.append(f'{s}: hand-shoulder dy={hp[1]-sp[1]:+.3f} dz={hp[2]-sp[2]:+.3f} dx={hp[0]-sp[0]:+.3f} m')
        q = qq
    # bbox ratio of point cloud center in proposed T-pose (UMR robot side)
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    P = geom_world_points(m, d, r['_sg'])
    ratio = None
    if len(P) and r['_pcc'] is not None:
        lo, hi = P.min(0), P.max(0)
        ratio = float((d.xpos[r['_pcc']][2] - lo[2]) / (hi[2] - lo[2]))
        r['height_tpose_m'] = round(float(hi[2] - lo[2]), 4)
        r['armspan_tpose_m'] = round(float(hi[1] - lo[1]), 4)
    r['umr_tpose_qpos_proposed'] = out
    r['umr_tpose_check'] = notes
    r['pcc_bbox_ratio_tpose'] = None if ratio is None else round(ratio, 4)


def caches(name):
    found = []
    for c in sorted(glob.glob(f'{TOPO}/cache*')):
        if glob.glob(f'{c}/{name}_spec*.npz') or glob.glob(f'{c}/{name}_paired_local6d.pt'):
            found.append(os.path.basename(c))
    return found


def main():
    rows = []
    for name, info in CAND.items():
        try:
            r = analyse(name, info['mjcf'])
            if r.get('load_ok'):
                tpose_search(r)
        except Exception as e:
            r = dict(name=name, mjcf=info['mjcf'], load_ok=False, load_error='analysis crash: ' + repr(e))
            traceback.print_exc()
        r.pop('_m', None); r.pop('_pcc', None); r.pop('_sg', None)
        r['mjcf_source'] = info['source']
        base = {'unitree_g1_29dof_dex3': 'unitree_g1_29dof', 'agibot_x2_ultra_vendor': 'agibot_x2_ultra'}.get(name, name)
        r['am2_role'] = 'xwbc_train' if base in TRAIN else ('xwbc_heldout' if base in HELDOUT else
                        ('xwbc_data_only(not in AM2 lists)' if os.path.isdir(f'{XWBC_EXT}/{base}') else 'not_in_am2'))
        r['xwbc_lafan_extracted'] = os.path.isdir(f'{XWBC_EXT}/{base}') and len(glob.glob(f'{XWBC_EXT}/{base}/*.npz'))
        r['topo_exp_caches'] = caches(base)
        r['surface_v2_asset'] = os.path.exists(f'{SURF}/{base}.npz')
        sa = sole.get(base, {})
        r['sole_audit_label_feet'] = sa.get('label_feet')
        r['sole_audit_predicted_bias_cm'] = sa.get('predicted_bias_cm')
        rows.append(r)
        print(name, r.get('load_ok'), r.get('load_error', ''), flush=True)
    json.dump(rows, open(OUT / 'robots_raw.json', 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))


if __name__ == '__main__':
    main()
