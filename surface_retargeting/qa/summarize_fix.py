"""Before/after table of the root-jump fix for the 12 parity runs (old = frame-parallel exports backed up in
results/export_old_frameparallel, new = results/parity/*.json of the sequential strict pass)."""
import json
from pathlib import Path

import numpy as np

OUT = Path('/var/tmp/alphamotion/umr_factory/umr_gpu')
CL = ['lafan1_smplx__dance1_subject2', 'hiphi_getup__getup0004_neutral_30fps', 'case3_kick_right__20260330000586_1_1775269234']
RB = ['unitree_g1_29dof', 'fourier_gr3', 'unitree_h2', 'agibot_a3_t3d0']
LLOYD = {('lafan1_smplx__dance1_subject2', 'unitree_g1_29dof'): dict(max=27.2, gt5=208)}
rows = []
print('| clip | robot | frames | UMR root step max cm / >5cm | OLD strict max / >5cm | NEW strict max / >5cm (kept max) | NEW root excess vs UMR p99 / max cm | NEW offset vs UMR max cm | kept OLD (OLD after continuity gate) -> NEW | kept: slip>1cm/s / ground pen / self pen / obj pen frames / human-contact solid |')
print('|---|---|---|---|---|---|---|---|---|---|')
for c in CL:
    for r in RB:
        n = f'{c}__{r}'
        jf = OUT / 'results' / 'parity' / f'{n}.json'
        if not jf.exists():
            continue
        J = json.load(open(jf))
        if 'root_step' not in J or 'continuity' not in J.get('export_qc', {}):
            continue
        old = OUT / 'results' / 'export_old_frameparallel' / f'{n}.npz'
        if old.exists():
            z = np.load(old); q = z['qpos'].astype(float); fps = float(z['fps'])
            st = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1) * fps / 30 * 100
            o = dict(max=round(float(st.max()), 1), gt5=int((st > 5).sum()), kept=round(float(((z['flags'] & 255) == 0).mean()), 3))
        else:
            o = dict(LLOYD.get((c, r), dict(max=None, gt5=None)), kept=None)
        kg = json.load(open(OUT / 'results' / 'export_old_frameparallel' / 'kept_with_gate.json')).get(n, {}) if (OUT / 'results' / 'export_old_frameparallel' / 'kept_with_gate.json').exists() else {}
        o['kept_gate'] = kg.get('kept_old_with_gate')
        rs = J['root_step']; cq = J['export_qc'].get('continuity', {}); g = J['gpu_strict']
        kept = round(1 - J['gpu_flagged'] / J['frames'], 3)
        row = dict(clip=c, robot=r, frames=J['frames'], fps=J['fps'], umr_max=rs['gpu_umr']['max_cm'], old=o,
                   new_max=rs['gpu_strict']['max_cm'], new_gt5=rs['gpu_strict']['frames_gt_5cm'], new_kept_max=rs['gpu_strict_kept']['max_cm'],
                   new_kept_gt5=rs['gpu_strict_kept']['frames_gt_5cm'], excess_p99=cq.get('root_excess_cm_p99'), excess_max=cq.get('root_excess_cm_max'),
                   offset_max=cq.get('root_offset_vs_umr_cm_max'), kept=kept, flags=J['gpu_flags_hist'],
                   kept_slip=g.get('kept_contact_slip_frac_gt_1cms'), kept_gpen=g.get('kept_ground_pen_frames_gt_0p5mm'),
                   kept_cgpen=g.get('kept_colground_pen_frames_gt_0p5mm'), kept_spen=g.get('kept_self_pen_frames_gt_0p5mm'),
                   kept_open=g.get('kept_obj_pen_frames_gt_0p5mm'), solid=g.get('kept_human_contact_robot_solid_frac'), strict_s=J['timings_s']['strict'])
        rows.append(row)
        print(f"| {c.split('__')[1][:22]} ({J['fps']:.0f} fps) | {r} | {J['frames']} | {row['umr_max']} / {rs['gpu_umr']['frames_gt_5cm']} | {o['max']} / {o['gt5']} | "
              f"{row['new_max']} / {row['new_gt5']} ({row['new_kept_max']}) | {row['excess_p99']} / {row['excess_max']} | {row['offset_max']} | "
              f"{o['kept']} ({o['kept_gate']}) -> {kept} | {row['kept_slip']} / {max(row['kept_gpen'] or 0, row['kept_cgpen'] or 0)} / {row['kept_spen']} / {row['kept_open']} / {row['solid']} |")
json.dump(rows, open(OUT / 'results' / 'fix_jump_summary.json', 'w'), indent=1)
