"""Bake-off driver: sources x targets x candidates -> ruler metrics + extras -> results.json; per-candidate npz for rendering."""
import sys, json, time, numpy as np, torch
from common import *; from refiners import *
OUT = f'{S}/bake/out'; os.makedirs(f'{OUT}/npz', exist_ok=True)
PACKS = [pack('unitree_g1_29dof'), pack('unitree_h2')]
def metrics(pk, src, dec, base, o, name):
    pos, quat, qj = to_ruler(pk, o['q'], o['root_R'], o['root_cm']); a = ruler.audit(pk.ruler_robot, pos, quat, qj, src['fps'], label=name)
    gR, gp = fk_from_angles(torch.as_tensor(o['q']), pk.spec, pk.dof, rest=pk.rest, root_R=torch.as_tensor(o['root_R']))
    gR0, gp0 = fk_from_angles(torch.as_tensor(base['q']), pk.spec, pk.dof, rest=pk.rest, root_R=torch.as_tensor(base['root_R']))
    a['posture_disp_vs_R0_cm'] = float((gp - gp0).norm(dim=-1).mean()); a['rot_vs_decoded_deg'] = float(geodesic_deg(gR, dec['R']).mean())
    a['pos_vs_decoded_cm'] = float((gp - dec['gp_dec']).norm(dim=-1).mean())
    if src['spec'].J == pk.spec.J and list(src['spec'].joint_names) == list(pk.spec.joint_names):
        a['rot_vs_source_deg'] = float(geodesic_deg(gR, src['gR']).mean()); a['pos_vs_source_cm'] = float((gp - src['gp']).norm(dim=-1).mean())
        for nm, j in (('hand', pk.hands[0]), ('foot', pk.feet[0])): a[f'{nm}_vs_source_cm'] = float((gp[:, j] - src['gp'][:, j]).norm(dim=-1).mean())
    low, _ = foot_low(pk, gR, gp + torch.as_tensor(o['root_cm'], dtype=torch.float32)[:, None]); st = (low.numpy() < 1.0)
    tp = (st & src['stance']).sum(); fp = (st & ~src['stance']).sum(); fn = (~st & src['stance']).sum(); a['contact_f1_vs_source'] = float(2 * tp / max(2 * tp + fp + fn, 1))
    a['ms_per_frame'] = o['ms_per_frame']; a['mode'] = o['mode']; a['extra'] = {k: v for k, v in o['extra'].items()}
    return a
def run(src_list, tag):
    res = {}
    for src in src_list:
        src = finish_src(src); row = {'text': src['text'], 'fps': src['fps'], 'T': len(src['gR']), 'v_thr_cm_s': src['v_thr_cm_s'], 'src_foot_speed_in_contact_p50': src['src_foot_speed_in_contact_p50'], 'stance_frac': src['stance'].mean(0).tolist(), 'targets': {}}
        if src.get('q_native') is not None:                             # native route (no codec), G1 only
            pk = PACKS[0]; pos, quat, qj = to_ruler(pk, src['q_native'], src['root_R'], src['root_cm']); row['native_g1'] = ruler.audit('g1', pos, quat, qj, src['fps'], label='native')
        for pk in PACKS:
            dec = decode(src, pk); base = r0(pk, dec, src); cands = {'R0': base}
            for name, fn in [('R1', lambda: r1(pk, dec, src, base)), ('R2', lambda: r2(pk, dec, src, base)), ('R2h', lambda: r2(pk, dec, src, base, hand_pin=True)), ('R3', lambda: r3(pk, dec, src, base)), ('E', lambda: eric(pk, dec, src, base))]:
                try: cands[name] = fn()
                except Exception as e: print('FAILED', src['name'], pk.name, name, repr(e)[:300], flush=True); cands[name] = None
            tgt = {'enc_ms': dec['enc_ms'], 'dec_ms': dec['dec_ms'], 'cands': {}}
            for name, o in cands.items():
                if o is None: continue
                tgt['cands'][name] = metrics(pk, src, dec, base, o, f'{src["name"]}|{pk.name}|{name}')
                np.savez(f"{OUT}/npz/{src['name'].replace('/', '__')}__{pk.name}__{name}.npz", q=o['q'], root_R=o['root_R'], root_cm=o['root_cm'], fps=src['fps'])
            row['targets'][pk.name] = tgt
            print(src['name'], pk.name, ' '.join(f"{n}:slip95={m['slip_p95_cm_s']:.1f}/pen={m['pen_max_cm']:.1f}/disp={m['posture_disp_vs_R0_cm']:.1f}/{m['ms_per_frame']:.1f}ms" for n, m in tgt['cands'].items()), flush=True)
        res[src['name']] = row
        json.dump(res, open(f'{OUT}/results_{tag}.json', 'w'), indent=1)
    return res
if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'ardy_g1': run([src_ardy_g1(p) for p in PROMPTS], 'ardy_g1')
    elif which == 'ardy_core': run([src_ardy_core(p) for p in PROMPTS], 'ardy_core')
    elif which == 'official': run([src_official_g1(s) for s in SOMA10], 'official')
    print('DONE', which)
