"""Markdown tables from results/parity/*.json (parity + strictness) -> stdout."""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import glob, json
from pathlib import Path
OUT = Path(f'{SRCFG.work}/results/parity')
CL = {'case3_kick_right__20260330000586_1_1775269234_s3': 'Omni soccer 30fps', 'lafan1_smplx__dance1_subject2': 'LAFAN dance', 'hiphi_getup__getup0004_neutral_30fps': 'HiPHI get-up', 'case3_kick_right__20260330000586_1_1775269234': 'Omni soccer'}
RB = {'unitree_g1_29dof': 'G1', 'fourier_gr3': 'GR3', 'unitree_h2': 'H2', 'agibot_a3_t3d0': 'A3'}
rows = []
for f in sorted(OUT.glob('*.json')):
    d = json.load(open(f)); rows.append(d)
rows.sort(key=lambda d: (list(CL).index(d['clip']), list(RB).index(d['robot'])))
print('### A. UMR-stage parity (GPU UMR vs CPU UMR, both after UMR\'s LQR filter, same inputs)\n')
print('| clip | robot | frames | joint max-abs/frame p50 (rad) | p99 | max | root max (mm) | surface pt mean (mm) | surface pt p99 of frame max (mm) | max (mm) |')
print('|---|---|---|---|---|---|---|---|---|---|')
for d in rows:
    p = d['parity_umr_stage']
    print(f"| {CL[d['clip']]} | {RB[d['robot']]} | {d['frames']} | {p['joint_maxabs_rad_p50']:.1e} | {p['p99']:.1e} | {p['max']:.3f} | {p['root_pos_max_mm']:.2f} | {p['surface_point_mm_mean']:.4f} | {p['surface_point_mm_p99_of_frame_max']:.3f} | {p['surface_point_mm_max']:.1f} |")
print('\n### B. Constraint satisfaction, one CPU ruler (eval/judge.py, MuJoCo + dense points), counts of frames\n')
print('| clip | robot | variant | ground pen >0.5mm (pts) | ground pen (MuJoCo col. geoms) | self pen >0.5mm | object pen >0.5mm | human-contact -> robot solid | human obj-contact -> robot solid | human off -> robot lifted | contact pt speed p50 / p95 (cm/s) | slip >1 cm/s frac | flagged (excluded) |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for d in rows:
    for v in ('cpu_umr', 'cpu_factory_strict', 'gpu_strict'):
        if v not in d:
            continue
        g = d[v]
        kept = v == 'gpu_strict'
        k = lambda a, b: g.get(b) if kept and b in g else g.get(a)
        name = {'cpu_umr': 'CPU UMR', 'cpu_factory_strict': 'CPU UMR + factory strict', 'gpu_strict': 'GPU strict (kept frames)'}[v]
        fl = g.get('flagged_excluded', 0 if v != 'cpu_factory_strict' else d.get('cpu_factory_flagged', 0))
        print(f"| {CL[d['clip']]} | {RB[d['robot']]} | {name} | {k('ground_pen_frames_gt_0p5mm','kept_ground_pen_frames_gt_0p5mm')} | {k('colground_pen_frames_gt_0p5mm','kept_colground_pen_frames_gt_0p5mm')} | {k('self_pen_frames_gt_0p5mm','kept_self_pen_frames_gt_0p5mm')} | {k('obj_pen_frames_gt_0p5mm','kept_obj_pen_frames_gt_0p5mm') if 'obj_pen_frames_gt_0p5mm' in g else '-'} | {k('human_contact_robot_solid_frac','kept_human_contact_robot_solid_frac'):.3f} | {(k('human_obj_contact_robot_solid_frac','kept_human_obj_contact_robot_solid_frac') if 'human_obj_contact_robot_solid_frac' in g else '-')} | {g['human_off_robot_lifted_frac']:.3f} | {g['contact_speed_cms_p50']} / {g['contact_speed_cms_p95']} | {g['contact_slip_frac_gt_1cms']} | {fl} |")
print('\n### B2. Slip on kept GPU frames (DATA_SPEC_v2 rule: contact point tangential speed < 1 cm/s)\n')
print('| clip | robot | fps | latched-point speed p50 / p95 (cm/s) | frac > 1 cm/s | contact-patch median speed p50 / p95 (cm/s) | patch frac > 1 cm/s | CPU UMR patch p50 / p95 |')
print('|---|---|---|---|---|---|---|---|')
for d in rows:
    g = d['gpu_strict']; c = d['cpu_umr']
    if 'kept_region_lowest_point_speed_cms_p50' not in g:
        continue
    print(f"| {CL[d['clip']]} | {RB[d['robot']]} | {d['fps']:.0f} | {g['kept_region_lowest_point_speed_cms_p50']} / {g['kept_region_lowest_point_speed_cms_p95']} | {g['kept_region_lowest_point_slip_frac_gt_1cms']} | {g['kept_contact_speed_cms_p50']} / {g['kept_contact_speed_cms_p95']} | {g['kept_contact_slip_frac_gt_1cms']} | {c['contact_speed_cms_p50']} / {c['contact_speed_cms_p95']} |")
print('\n### C. GPU flag breakdown (frames; a frame can carry several flags)\n')
print('| clip | robot | frames | flagged | of which human slips there | ground pen (1) | obj pen (2) | self (4) | ground contact (8) | obj contact (16) | slip (32) | lift (64) | GPU s (UMR / strict, B=1) | CPU UMR fps |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for d in rows:
    h = d['gpu_flags_hist']; t = d['timings_s']
    print(f"| {CL[d['clip']]} | {RB[d['robot']]} | {d['frames']} | {d['gpu_flagged']} ({100*d['gpu_flagged']/d['frames']:.1f}%) | {d.get('gpu_flagged_while_human_slips','-')} | {h['1']} | {h['2']} | {h['4']} | {h['8']} | {h['16']} | {h['32']} | {h['64']} | {t['umr_solve']} / {t['strict']} | {d['cpu_umr_fps']} |")
