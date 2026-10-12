"""Arm A: ARDY's own post-processor applied to the source before the codec. Native+A ruler (G1), and A-source -> Megatron -> G1/H2 with R0/R2/E."""
import json, numpy as np, torch, common
from common import *; from refiners import *; from run_all import metrics, PACKS
import ardy_in, ardy_core_in as C
def src_g1_pp(key):
    a = ardy_in.load_ardy(f'{S}/bake/ardy_pp/{key}__g1.npz'); st = stance_from_ardy(a['contacts']); con = torch.zeros(len(st), ardy_in.SPEC.J)
    for side, col in (('left', 0), ('right', 1)):
        for n in (f'{side}_ankle_roll_link', f'{side}_toe_link'): con[:, ardy_in.NAMES.index(n)] = torch.as_tensor(st[:, col]).float()
    idx = {n: i for i, n in enumerate(a['ardy_names'])}; P = a['ardy_P']
    foot_h = np.stack([np.minimum(P[:, idx['left_ankle_roll_skel'], 1], P[:, idx['left_toe_base'], 1]), np.minimum(P[:, idx['right_ankle_roll_skel'], 1], P[:, idx['right_toe_base'], 1])], 1)
    return finish_src(dict(name=f'ardyG1pp/{key}', kind='ardy_g1_pp', spec=ardy_in.SPEC, dof=ardy_in.DOF, rest=ardy_in.REST, gR=torch.as_tensor(a['gR']).float(), gp=a['gp'].float(), root_R=a['root_R'], root_cm=a['root_cm'], fps=a['fps'], stance=st, con=con, q_native=a['q'], leg_cm=80.0, foot_h_cm=foot_h, text=a['text'],
                foot_xz_cm=np.stack([P[:, idx['left_ankle_roll_skel']][:, [0, 2]], P[:, idx['right_ankle_roll_skel']][:, [0, 2]]], 1)))
def src_core_pp(key):
    a = C.load_ardy(f'{S}/bake/ardy_pp/{key}__core.npz'); st = stance_from_ardy(a['contacts']); con = torch.zeros(len(st), C.SPEC.J)
    for side, col in (('left', 0), ('right', 1)):
        for j in C.FEET_IDX[side]: con[:, j] = torch.as_tensor(st[:, col]).float()
    P = a['ardy_P']; foot_h = np.stack([P[:, C.FEET_IDX['left'], 1].min(1), P[:, C.FEET_IDX['right'], 1].min(1)], 1)
    return finish_src(dict(name=f'ardyCorepp/{key}', kind='ardy_core_pp', spec=C.SPEC, dof=C.DOF, rest=C.REST, gR=a['gR'].float(), gp=a['gp'], root_R=a['root_R'], root_cm=a['root_cm'], fps=a['fps'], stance=st, con=con, q_native=None, leg_cm=C.LEG_CM, foot_h_cm=foot_h, text=a['text'],
                foot_xz_cm=np.stack([P[:, C.FEET_IDX['left'][0]][:, [0, 2]], P[:, C.FEET_IDX['right'][0]][:, [0, 2]]], 1)))
res = {}
for maker, tag in ((src_g1_pp, 'ardy_g1_pp'), (src_core_pp, 'ardy_core_pp')):
    for key in PROMPTS:
        src = maker(key); row = {'text': src['text'], 'fps': src['fps'], 'T': len(src['gR']), 'targets': {}}
        if src.get('q_native') is not None:
            pos, quat, qj = to_ruler(PACKS[0], src['q_native'], src['root_R'], src['root_cm']); row['native_g1'] = ruler.audit('g1', pos, quat, qj, src['fps'], label='native+A')
        for pk in PACKS:
            dec = decode(src, pk); base = r0(pk, dec, src); cands = {'R0': base, 'R2': r2(pk, dec, src, base), 'E': eric(pk, dec, src, base)}
            row['targets'][pk.name] = {'cands': {n: metrics(pk, src, dec, base, o, f'{src["name"]}|{pk.name}|{n}') for n, o in cands.items()}}
            print(src['name'], pk.name, ' '.join(f"{n}:slip95={m['slip_p95_cm_s']:.1f}/pen={m['pen_max_cm']:.1f}" for n, m in row['targets'][pk.name]['cands'].items()), flush=True)
        res[src['name']] = row
    json.dump({k: v for k, v in res.items() if k.startswith(tag.split('_')[1][:4])}, open(f'{S}/bake/out/results_{tag}.json', 'w'), indent=1)
json.dump(res, open(f'{S}/bake/out/results_pp.json', 'w'), indent=1); print('PP DONE')
