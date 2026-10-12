"""Report 33: new ARDY-G1 motions -> Megatron -> {G1, H2, A3, GR3, Adam} -> R0 (raw) vs R2 (recursive refiner) [+ R2f falling-edge latch ablation] -> ruler + npz for rendering."""
import os, sys, json, time, numpy as np, torch
S = '/tmp/claude-1000/-home-arenalabs-Desktop--be-water--robot-/eff459e5-3969-4f6d-9ec8-eae021b0a07a/scratchpad'
R31 = '/home/arenalabs/Desktop/alphamotion video/resource/report/31_realtime_refiner_bakeoff/scripts'
sys.path.insert(0, R31)
from common import *                      # Pack, pack(), decode(), src_root_xz, to_ruler, finish_src, foot_low, ruler, megatron
import refiners
from refiners import r0, r2, Latch, sigmoid
from arena.refiner.robots import load_robot
from ardy_in import load_ardy, SPEC, DOF, REST, NAMES
GEN = f'{S}/ardy33/gen'; OUT = f'{S}/ardy33/out'; os.makedirs(f'{OUT}/npz', exist_ok=True)
KEYS = ['n1_walkturn', 'n2_jacks', 'n3_squat', 'n4_sidestep', 'n5_runstop']

# ---- extra targets: register their vendor MJCF + foot bodies in the ruler, then build bake-off packs on the codec's own descriptor
def pack_extra(name):
    xml = resolve_xml(name); rob = load_robot(name); sx, dx, rx, qn, _ = build_from_mjcf(xml, name)
    assert list(sx.joint_names) == list(rob.spec.joint_names) and float(np.abs(dx - rob.dof.numpy()).max()) == 0.0
    pf = profiles()[name]; ruler.XML[name] = xml; ruler.FEET[name] = (pf['left']['foot_name'], pf['right']['foot_name'])
    m, _ = ruler.model(name); assert m.jnt_type[0] == 0 and all(m.jnt_type[i] == 3 for i in range(1, m.njnt)), f'{name}: joint layout not free+hinges'
    return Pack(name, name, rob.spec, rob.dof.numpy(), rob.rest.numpy(), qn, xml, 0.5 * (pf['left']['leg_len_cm'] + pf['right']['leg_len_cm']))
PACKS = [pack('unitree_g1_29dof'), pack('unitree_h2')] + [pack_extra(n) for n in ('agibot_a3_t3d0', 'fourier_gr3', 'pnd_adam_lite')]
import foot_geom
ruler.foot_lowest_points = lambda m, d, robot: foot_geom.foot_lowest_points_subtree(m, d, robot, ruler.FEET)     # ruler: whole foot subtree
HULLS = {pk.name: foot_geom.collision_hulls(pk) for pk in PACKS}                                                     # latch: collision hulls, foot subtree
print('collision hulls:', json.dumps(HULLS, indent=0), flush=True)

def src_new(key):
    a = load_ardy(f'{GEN}/{key}.npz'); st = refiners.stance_from_ardy(a['contacts']); con = torch.zeros(len(st), SPEC.J)
    for side, col in (('left', 0), ('right', 1)):
        for n in (f'{side}_ankle_roll_link', f'{side}_toe_link'): con[:, NAMES.index(n)] = torch.as_tensor(st[:, col]).float()
    idx = {n: i for i, n in enumerate(a['ardy_names'])}; P = a['ardy_P']
    foot_h = np.stack([np.minimum(P[:, idx['left_ankle_roll_skel'], 1], P[:, idx['left_toe_base'], 1]), np.minimum(P[:, idx['right_ankle_roll_skel'], 1], P[:, idx['right_toe_base'], 1])], 1)
    return dict(name=f'ardyG1new/{key}', kind='ardy_g1', spec=SPEC, dof=DOF, rest=REST, gR=torch.as_tensor(a['gR']).float(), gp=a['gp'].float(), root_R=a['root_R'], root_cm=a['root_cm'], fps=a['fps'], stance=st, con=con,
                q_native=a['q'], leg_cm=80.0, foot_h_cm=foot_h, text=a['text'], fk_err_cm=a['fk_err_cm'], rot_fit_deg=a['rot_fit_deg'],
                foot_xz_cm=np.stack([P[:, idx['left_ankle_roll_skel']][:, [0, 2]], P[:, idx['right_ankle_roll_skel']][:, [0, 2]]], 1))

# ---- ablation: release the latch on the FALLING EDGE of the decoder contact probability (p below its 0.5 decision point while decreasing),
#      instead of waiting for the hysteresis floor (0.4). Everything else identical to R2.
class LatchFall(Latch):
    def __init__(self, *a, **k): super().__init__(*a, **k); self.prev_p = [None, None]
    def logits(self, logit_t):
        for i in range(2):
            p = sigmoid(max(float(logit_t[j]) for j in self.pk.contact_nodes[i])); pp = self.prev_p[i]
            if self.logit_on[i] and (p < self.off_thr or (p < 0.5 and pp is not None and p < pp)): self.logit_on[i] = False
            elif not self.logit_on[i] and p > self.on_thr: self.logit_on[i] = True
            self.prev_p[i] = p
def make_latch_fall(pk, dec, src, base):
    w = refiners.measured_weights(pk, dec, src, base); prov = src_root_xz(src, pk)
    rs = np.linalg.norm(np.diff(prov[:, [0, 2]], axis=0), axis=-1) * src['fps']; rs = np.concatenate([rs[:1], rs])
    return LatchFall(pk, src['fps'], refiners.band_cm(pk, dec, src, base), refiners.clearance(pk, src), src['v_thr_cm_s'], rs)
def r2_fall(pk, dec, src, base):
    orig = refiners.make_latch; refiners.make_latch = make_latch_fall
    try: o = r2(pk, dec, src, base)
    finally: refiners.make_latch = orig
    o['mode'] += ' + falling-edge latch release'; return o

def metrics(pk, src, dec, base, o, name):
    pos, quat, qj = to_ruler(pk, o['q'], o['root_R'], o['root_cm']); a = ruler.audit(pk.ruler_robot, pos, quat, qj, src['fps'], label=name)
    gR, gp = fk_from_angles(torch.as_tensor(o['q']), pk.spec, pk.dof, rest=pk.rest, root_R=torch.as_tensor(o['root_R']))
    gR0, gp0 = fk_from_angles(torch.as_tensor(base['q']), pk.spec, pk.dof, rest=pk.rest, root_R=torch.as_tensor(base['root_R']))
    a['posture_disp_vs_R0_cm'] = float((gp - gp0).norm(dim=-1).mean()); a['rot_vs_decoded_deg'] = float(geodesic_deg(gR, dec['R']).mean()); a['pos_vs_decoded_cm'] = float((gp - dec['gp_dec']).norm(dim=-1).mean())
    if list(src['spec'].joint_names) == list(pk.spec.joint_names):
        a['rot_vs_source_deg'] = float(geodesic_deg(gR, src['gR']).mean()); a['pos_vs_source_cm'] = float((gp - src['gp']).norm(dim=-1).mean())
        for nm, j in (('hand', pk.hands[0]), ('foot', pk.feet[0])): a[f'{nm}_vs_source_cm'] = float((gp[:, j] - src['gp'][:, j]).norm(dim=-1).mean())
    low, _ = foot_low(pk, gR, gp + torch.as_tensor(o['root_cm'], dtype=torch.float32)[:, None]); st = (low.numpy() < 1.0)
    tp = (st & src['stance']).sum(); fp = (st & ~src['stance']).sum(); fn = (~st & src['stance']).sum(); a['contact_f1_vs_source'] = float(2 * tp / max(2 * tp + fp + fn, 1))
    a['ms_per_frame'] = o['ms_per_frame']; a['mode'] = o['mode']; a['extra'] = o['extra']; return a

res = {}
for key in KEYS:
    src = finish_src(src_new(key)); T = len(src['gR']); fps = src['fps']
    v = np.linalg.norm(np.diff(src['gp'].numpy() + src['root_cm'][:, None], axis=0), axis=-1).mean(1) * fps / 100
    row = {'text': src['text'], 'fps': fps, 'T': T, 'ardy_fk_err_cm_max': max(src['fk_err_cm'].values()), 'ardy_rot_fit_deg': src['rot_fit_deg'], 'v_thr_cm_s': src['v_thr_cm_s'],
           'joint_speed_per_second_m_s': [float(v[int(s * fps):int((s + 1) * fps)].mean()) for s in range(int(T / fps))], 'stance_frac': src['stance'].mean(0).tolist(), 'targets': {}}
    pk = PACKS[0]; pos, quat, qj = to_ruler(pk, src['q_native'], src['root_R'], src['root_cm']); row['native_g1'] = ruler.audit('g1', pos, quat, qj, fps, label='native')
    for pk in PACKS:
        dec = decode(src, pk, cache_dir=f'{S}/ardy33/cache'); base = r0(pk, dec, src); cands = {'R0': base}
        for name, fn in [('R2', lambda: r2(pk, dec, src, base)), ('R2f', lambda: r2_fall(pk, dec, src, base))]:
            try: cands[name] = fn()
            except Exception as e: print('FAILED', key, pk.name, name, repr(e)[:300], flush=True); cands[name] = None
        tgt = {'enc_ms': dec['enc_ms'], 'dec_ms': dec['dec_ms'], 'decoded_root_h_cm': [float(dec['root_h_cm'].min()), float(dec['root_h_cm'].max())], 'cands': {}}
        for name, o in cands.items():
            if o is None: continue
            tgt['cands'][name] = metrics(pk, src, dec, base, o, f'{key}|{pk.name}|{name}')
            np.savez(f"{OUT}/npz/{key}__{pk.name}__{name}.npz", q=o['q'], root_R=o['root_R'], root_cm=o['root_cm'], fps=fps)
        row['targets'][pk.name] = tgt
        print(key, pk.name, ' '.join(f"{n}:slip95={m['slip_p95_cm_s']:.1f}/p50={m['slip_p50_cm_s']:.1f}/pen={m['pen_max_cm']:.1f}/low={m['sole_min_cm']:.1f}/disp={m['posture_disp_vs_R0_cm']:.1f}/{m['ms_per_frame']:.1f}ms" for n, m in tgt['cands'].items()), flush=True)
    res[key] = row; json.dump(res, open(f'{OUT}/results.json', 'w'), indent=1)
json.dump(HULLS, open(f'{OUT}/foot_hulls.json', 'w'), indent=1)
print('DONE')
