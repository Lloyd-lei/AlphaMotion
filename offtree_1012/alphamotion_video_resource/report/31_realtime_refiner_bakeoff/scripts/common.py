"""Shared setup for the real-time refiner bake-off: Megatron, target packs (G1 bundled spec; H2 = 30-node H2_loop build so Eric's
sole model and our decode share one skeleton), sources (ARDY-G1 prompts, ARDY-Core prompts, BoneSeed official G1 clips), decode cache."""
import os, sys, json, glob, time, numpy as np, torch, mujoco
os.environ.setdefault('MUJOCO_GL', 'egl')
S = '/tmp/claude-1000/-home-arenalabs-Desktop--be-water--robot-/eff459e5-3969-4f6d-9ec8-eae021b0a07a/scratchpad'
R29 = '/home/arenalabs/Desktop/alphamotion video/resource/report/29_ardy_greenwich_5robots'
R30 = '/home/arenalabs/Desktop/alphamotion video/resource/report/30_ardy_core_greenwich_5robots'
for p in (f'{S}/alphamotion-locomotion-toolkit/src', f'{S}/alphamotion-locomotion-toolkit/toolkits/greenwich-soma-multibody/scripts', f'{S}/alphamotion-locomotion-toolkit/runtime/scripts', R29, R30, f'{S}/refiner_duel', f'{S}/bake'):
    if p not in sys.path: sys.path.insert(0, p)
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as Rot
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.engine.constraints import project, fk_from_angles, lm_fit, _fit_angles_greedy, _limits
from alphamotion.engine.spatial import key_joints, fk_pos
from alphamotion.engine.nets.rotations import SkeletonSpec, matrix_to_rot6d, rot6d_to_matrix, geodesic_deg
from alphamotion.embodiment import registry
from arena.eval.visual import resolve_xml
from arena.refiner.robots import profiles
from greenwich_umi_proof.sole_geometry import build_foot_sole_model
from greenwich_motion_sdk.support_geometry import build_endpoint_surface_model
import ruler
AX = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
H2_LOOP_XML = '/home/arenalabs/tmp/greenwich_workbench_0911/h2_assets/H2_loop.xml'
DEV = 'cuda'
_g = None
def megatron():
    global _g
    if _g is None: _g = Greenwich.load(model='megatron', device=DEV)
    return _g

class Pack:
    def __init__(self, name, ruler_robot, spec, dof, rest, qn, xml, leg_cm):
        self.name, self.ruler_robot, self.spec, self.xml, self.leg_cm = name, ruler_robot, spec, xml, leg_cm
        self.dof, self.rest = torch.as_tensor(dof).float(), torch.as_tensor(rest).float()
        self.qn = qn; self.slot = {n: (k, s) for k, names in enumerate(qn) for s, n in enumerate(names)}
        self.keys = key_joints(spec)[0]; self.root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
        self.feet = [self.keys[4], self.keys[5]]; self.hands = [self.keys[2], self.keys[3]]
        self.sole = build_foot_sole_model(xml, spec, self.rest.numpy(), ['left_foot', 'right_foot'], self.feet)
        self.support = build_endpoint_surface_model(xml, spec, self.rest.numpy(), ['left_foot', 'right_foot', 'left_hand', 'right_hand'], self.feet + self.hands)
        self.hull = []                                            # convex-hull vertices of each foot's visual mesh, foot-joint frame, cm
        for role in ('left_foot', 'right_foot'):
            f = self.sole.feet[role]; v = np.asarray(f.vertices_local_cm, np.float64)
            try: v = v[ConvexHull(v).vertices]
            except Exception: pass
            self.hull.append(torch.as_tensor(v, dtype=torch.float32))
        self.contact_nodes = [[self.feet[0]], [self.feet[1]]]        # decoder contact logits read on these nodes per foot
        for side, i in (('left', 0), ('right', 1)):
            for j, n in enumerate(spec.joint_names):
                if n.startswith(side) and 'toe' in n and j not in self.contact_nodes[i]: self.contact_nodes[i].append(j)
        self.hinges = ruler.hinge_names(ruler.model(ruler_robot)[0])
        self.lo, self.hi, self.limited = _limits(self.dof, 1.0)
        prof = profiles().get(name, None); self.standing_root_cm = prof['pelvis_height_rest_cm'] if prof else None

def foot_low(pack, gR, gp_world):
    """lowest visual-mesh vertex per foot (cm, Y up) and foot joint world position: gR [T,J,3,3], gp_world [T,J,3] -> low [T,2], pos [T,2,3]"""
    lows, poss = [], []
    for i, j in enumerate(pack.feet):
        v = pack.hull[i].to(gR.dtype); cloud = torch.einsum('tab,vb->tva', gR[:, j], v) + gp_world[:, j, None]
        lows.append(cloud[:, :, 1].min(1).values); poss.append(gp_world[:, j])
    return torch.stack(lows, 1), torch.stack(poss, 1)

_packs = {}
def pack(name):
    if name in _packs: return _packs[name]
    if name == 'unitree_g1_29dof':
        e = registry.load(name); xml = e.xml or resolve_xml(name); sx, dx, rx, qn, _ = build_from_mjcf(xml, name)
        assert list(sx.joint_names) == list(e.spec.joint_names) and float(np.abs(dx - e.dof).max()) == 0.0
        p = Pack(name, 'g1', e.spec, e.dof, e.rest, qn, xml, 0.5 * (profiles()[name]['left']['leg_len_cm'] + profiles()[name]['right']['leg_len_cm']))
    elif name == 'unitree_h2':
        sx, dx, rx, qn, _ = build_from_mjcf(H2_LOOP_XML, name)
        p = Pack(name, 'h2', sx, dx, rx, qn, H2_LOOP_XML, 0.5 * (profiles()[name]['left']['leg_len_cm'] + profiles()[name]['right']['leg_len_cm']))
    else: raise KeyError(name)
    _packs[name] = p; return p

# ---------------------------------------------------------------- sources
def stance_from_ardy(contacts):
    st = np.zeros((len(contacts), 2), bool)
    for c, side in enumerate([0, 0, 1, 1]): st[:, side] |= contacts[:, c] > 0.5   # L heel, L toe, R heel, R toe
    return st
def src_ardy_g1(key):
    from ardy_in import load_ardy, SPEC, DOF, REST, NAMES
    a = load_ardy(f'{S}/ardy5/gen/{key}.npz'); st = stance_from_ardy(a['contacts'])
    con = torch.zeros(len(st), SPEC.J)
    for side, col in (('left', 0), ('right', 1)):
        for n in (f'{side}_ankle_roll_link', f'{side}_toe_link'): con[:, NAMES.index(n)] = torch.as_tensor(st[:, col]).float()
    idx = {n: i for i, n in enumerate(a['ardy_names'])}; P = a['ardy_P']
    foot_h = np.stack([np.minimum(P[:, idx['left_ankle_roll_skel'], 1], P[:, idx['left_toe_base'], 1]), np.minimum(P[:, idx['right_ankle_roll_skel'], 1], P[:, idx['right_toe_base'], 1])], 1)
    return dict(name=f'ardyG1/{key}', kind='ardy_g1', spec=SPEC, dof=DOF, rest=REST, gR=torch.as_tensor(a['gR']).float(), gp=a['gp'].float(), root_R=a['root_R'], root_cm=a['root_cm'], fps=a['fps'], stance=st, con=con,
                q_native=a['q'], leg_cm=80.0, foot_h_cm=foot_h, text=a['text'],
                foot_xz_cm=np.stack([P[:, idx['left_ankle_roll_skel']][:, [0, 2]], P[:, idx['right_ankle_roll_skel']][:, [0, 2]]], 1))
def src_ardy_core(key):
    import ardy_core_in as C
    a = C.load_ardy(f'{R30}/{key}_ardy_core.npz'); st = stance_from_ardy(a['contacts'])
    con = torch.zeros(len(st), C.SPEC.J)
    for side, col in (('left', 0), ('right', 1)):
        for j in C.FEET_IDX[side]: con[:, j] = torch.as_tensor(st[:, col]).float()
    P = a['ardy_P']; foot_h = np.stack([P[:, C.FEET_IDX['left'], 1].min(1), P[:, C.FEET_IDX['right'], 1].min(1)], 1)
    neutral_h = np.array([C.neutral[C.FEET_IDX['left'], 1].min(), C.neutral[C.FEET_IDX['right'], 1].min()]) * 100
    return dict(name=f'ardyCore/{key}', kind='ardy_core', spec=C.SPEC, dof=C.DOF, rest=C.REST, gR=a['gR'].float(), gp=a['gp'], root_R=a['root_R'], root_cm=a['root_cm'], fps=a['fps'], stance=st, con=con,
                q_native=None, leg_cm=C.LEG_CM, foot_h_cm=foot_h - neutral_h, text=a['text'],
                foot_xz_cm=np.stack([P[:, C.FEET_IDX['left'][0]][:, [0, 2]], P[:, C.FEET_IDX['right'][0]][:, [0, 2]]], 1))
def src_official_g1(stem):
    from ardy_in import SPEC, DOF, REST, GMAP
    p1 = pack('unitree_g1_29dof'); path = glob.glob(f'{S}/refiner_duel/g1_official/**/{stem}.csv', recursive=True)[0]
    pos_m, quat, qj, fps = ruler.load_boneseed_csv(path, 'g1'); T = len(qj)
    q = np.zeros((T, SPEC.J, 3), np.float32)
    for i, h in enumerate(p1.hinges): k, s = GMAP[h]; q[:, k, s] = qj[:, i]
    Rn = Rot.from_quat(quat).as_matrix(); root_R = (AX @ Rn @ AX.T).astype(np.float32); root_cm = (pos_m * 100 @ AX.T).astype(np.float32)
    gR, gp = fk_from_angles(torch.as_tensor(q), SPEC, DOF, rest=REST, root_R=torch.as_tensor(root_R))
    low, _ = foot_low(p1, gR, gp + torch.as_tensor(root_cm)[:, None]); low = low.numpy()
    st = low < 1.0; con = torch.zeros(T, SPEC.J)
    for side, col in (('left', 0), ('right', 1)):
        for n in (f'{side}_ankle_roll_link', f'{side}_toe_link'): con[:, list(SPEC.joint_names).index(n)] = torch.as_tensor(st[:, col]).float()
    return dict(name=f'official/{stem}', kind='official_g1', spec=SPEC, dof=DOF, rest=REST, gR=gR.float(), gp=gp.float(), root_R=root_R, root_cm=root_cm, fps=fps, stance=st, con=con,
                q_native=q, leg_cm=80.0, foot_h_cm=np.maximum(low, 0.0), text=stem,
                foot_xz_cm=(gp[:, [list(SPEC.joint_names).index('left_ankle_roll_link'), list(SPEC.joint_names).index('right_ankle_roll_link')]] + torch.as_tensor(root_cm)[:, None]).numpy()[:, :, [0, 2]])

PROMPTS = ['p1_box', 'p2_hurdle', 'p3_wave', 'p4_kick', 'p5_sit']
SOMA10 = sorted(os.path.basename(f)[:-4] for f in glob.glob(f'{S}/soma-retargeter/assets/motions/bvh/*.bvh'))

# ---------------------------------------------------------------- decode cache
def decode(src, pk, cache_dir=f'{S}/bake/cache'):
    """Megatron encode (with the source's contact channel) + decode_full on the pack. Cached. Returns R [T,J,3,3], root_h_cm [T], con_logit [T,J], codes, timing."""
    os.makedirs(cache_dir, exist_ok=True); f = f"{cache_dir}/{src['name'].replace('/', '__')}__{pk.name}.npz"
    if os.path.exists(f):
        z = np.load(f); return dict(R=torch.as_tensor(z['R']), root_h_cm=z['root_h_cm'], con=z['con'], gp_dec=torch.as_tensor(z['gp_dec']), enc_ms=float(z['enc_ms']), dec_ms=float(z['dec_ms']))
    g = megatron(); torch.cuda.synchronize(); t = time.perf_counter()
    pose, _ = g.pose9(matrix_to_rot6d(src['gR']).float(), src['spec'], is_global=True); codes = g.encode(pose, src['spec'], src['dof'], contact=src['con'].to(DEV)); torch.cuda.synchronize(); enc_ms = 1000 * (time.perf_counter() - t)
    t = time.perf_counter(); r6, pos, conl = g.decode_full(codes, pk.spec, pk.dof, contact=True); torch.cuda.synchronize(); dec_ms = 1000 * (time.perf_counter() - t)
    R = rot6d_to_matrix(r6.cpu()); gp_dec = torch.as_tensor(fk_pos(r6.cpu(), pk.spec)).float()
    out = dict(R=R.float(), root_h_cm=(pos[:, pk.root, 0].cpu().numpy() * 100).astype(np.float64), con=conl.cpu().numpy(), gp_dec=gp_dec, enc_ms=enc_ms, dec_ms=dec_ms)
    np.savez(f, R=out['R'].numpy(), root_h_cm=out['root_h_cm'], con=out['con'], gp_dec=gp_dec.numpy(), enc_ms=enc_ms, dec_ms=dec_ms); return out

def src_root_xz(src, pk):
    """source horizontal trajectory passed through, scaled by leg length, first frame at the origin"""
    r = src['root_cm'].astype(np.float64).copy(); r[:, [0, 2]] -= r[:1, [0, 2]]; r[:, [0, 2]] *= pk.leg_cm / src['leg_cm']; return r

# ---------------------------------------------------------------- ruler adapter
def to_ruler(pk, q, root_R, root_cm):
    qd = {n: q[:, k, s] for n, (k, s) in pk.slot.items()}
    qj = np.stack([qd[h] if h in qd else np.zeros(len(q)) for h in pk.hinges], 1).astype(np.float64)
    pos_m = (root_cm @ AX) / 100.0; quat = np.stack([Rot.from_matrix(AX.T @ root_R[t] @ AX).as_quat() for t in range(len(q))])
    return pos_m, quat, qj

def finish_src(src):
    """v_thr: the source's own foot speed p95 while it labels the foot planted (cm/s)"""
    sp = np.linalg.norm(np.diff(src['foot_xz_cm'], axis=0), axis=-1) * src['fps']; m = src['stance'][1:] & src['stance'][:-1]
    src['v_thr_cm_s'] = float(np.percentile(sp[m], 95)) if m.any() else 15.0; src['src_foot_speed_in_contact_p50'] = float(np.percentile(sp[m], 50)) if m.any() else 0.0; return src
