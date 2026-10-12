"""Tables, G1 loss ledger, chart, README for report 31."""
import json, os, numpy as np, glob
S = '/tmp/claude-1000/-home-arenalabs-Desktop--be-water--robot-/eff459e5-3969-4f6d-9ec8-eae021b0a07a/scratchpad'; OUT = f'{S}/bake/out'
R = '/home/arenalabs/Desktop/alphamotion video/resource/report/31_realtime_refiner_bakeoff'; os.makedirs(R, exist_ok=True)
res = {k: json.load(open(f'{OUT}/results_{k}.json')) for k in ('ardy_g1', 'ardy_core', 'official', 'ardy_g1_pp', 'ardy_core_pp') if os.path.exists(f'{OUT}/results_{k}.json')}
duel = json.load(open(f'{S}/refiner_duel/ruler_results.json'))
CANDS = ['R0', 'R1', 'R2', 'R2h', 'R3', 'E']
KEYS = [('slip_p95_cm_s', 'slip95'), ('slip_p50_cm_s', 'slip50'), ('pen_p95_cm', 'pen95'), ('pen_max_cm', 'penmax'), ('pen_frames_over_0p5cm', 'pen%'), ('airborne_frames', 'air%'), ('root_step_p95_cm', 'rootstep95'),
        ('joint_jerk_p95_deg_s3', 'jerk95'), ('joint_limit_excess_max_deg', 'limit'), ('posture_disp_vs_R0_cm', 'disp_vs_R0'), ('rot_vs_decoded_deg', 'rot_vs_dec'), ('contact_f1_vs_source', 'F1'), ('ms_per_frame', 'ms/frame')]
def med(rows, k): v = [r[k] for r in rows if k in r and r[k] is not None]; return float(np.median(v)) if v else float('nan')
def table(src_kind, pkname):
    rows = {c: [] for c in CANDS}
    for srcname, row in res[src_kind].items():
        t = row['targets'].get(pkname)
        if not t: continue
        for c in CANDS:
            if c in t['cands']: rows[c].append(t['cands'][c])
    hdr = '| cand | mode | ' + ' | '.join(k[1] for k in KEYS) + ' | hand EE cm | src rot° / pos cm |\n|' + '---|' * (len(KEYS) + 4) + '\n'
    body = ''
    for c in CANDS:
        if not rows[c]: continue
        cells = []
        for k, _ in KEYS:
            v = med(rows[c], k); cells.append(f'{v:.2f}' if k in ('pen_p95_cm', 'pen_max_cm', 'pen_frames_over_0p5cm', 'airborne_frames', 'contact_f1_vs_source', 'limit') else (f'{v/1e3:.0f}k' if k == 'joint_jerk_p95_deg_s3' else f'{v:.1f}'))
        hand = med([r['extra'] for r in rows[c]], 'hand_ee_err_cm_mean'); src = (med(rows[c], 'rot_vs_source_deg'), med(rows[c], 'pos_vs_source_cm'))
        body += f'| {c} | {rows[c][0]["mode"]} | ' + ' | '.join(cells) + f' | {"" if np.isnan(hand) else f"{hand:.2f}"} | {"" if np.isnan(src[0]) else f"{src[0]:.1f} / {src[1]:.1f}"} |\n'
    return hdr + body
def native_rows(src_kind):
    out = []
    for srcname, row in res[src_kind].items():
        if 'native_g1' in row: out.append(row['native_g1'])
    return out
md = ['# 31 · 实时精修器对决（R0–R3 vs Eric 离线）+ G1 全路线损失账本', '',
      '尺子：`refiner_duel/ruler.py`（厂商 MJCF FK；每脚碰撞几何最低点；站立相 = 最低点 < 1 cm 且前后各 2 帧腐蚀；滑移 = 站立相脚身原点水平速度）。每格是各段的中位数。',
      '姿态位移 = 候选相对 R0 的关节位置平均变化（cm）；rot_vs_dec = 投影后旋转与解码目标的测地误差；F1 = 候选脚底 < 1 cm 对源接触标签；src = 同骨架源可比时相对源的旋转/位置误差（G1 源→G1）。', '']
for kind, title in (('ardy_g1', 'ARDY-G1 源，5 条 prompt'), ('ardy_core', 'ARDY-Core 人体源，5 条 prompt'), ('official', 'BoneSeed 官方 G1 源（10 段 SOMA），Megatron 自往返')):
    if kind not in res: continue
    for pk in ('unitree_g1_29dof', 'unitree_h2'):
        md += [f'## {title} → {pk}', '', table(kind, pk), '']
# ---- G1 loss ledger
md += ['## G1 损失账本（同内容、不同路线，同一把尺子，中位数）', '', '| 路线 | slip95 | slip50 | pen95 | penmax | pen% | air% | jerk95 | rot° vs src | pos cm vs src | ms/frame |', '|---|---|---|---|---|---|---|---|---|---|---|']
def ledger_row(name, rows, ms=''):
    if not rows: return
    g = lambda k: med(rows, k)
    md.append(f'| {name} | {g("slip_p95_cm_s"):.1f} | {g("slip_p50_cm_s"):.1f} | {g("pen_p95_cm"):.2f} | {g("pen_max_cm"):.2f} | {g("pen_frames_over_0p5cm"):.2f} | {g("airborne_frames"):.2f} | {g("joint_jerk_p95_deg_s3")/1e3:.0f}k | ' + (f'{g("rot_vs_source_deg"):.1f}' if not np.isnan(g('rot_vs_source_deg')) else '') + ' | ' + (f'{g("pos_vs_source_cm"):.1f}' if not np.isnan(g('pos_vs_source_cm')) else '') + f' | {ms if ms != "" else (f"{g(chr(109)+chr(115)+chr(95)+chr(112)+chr(101)+chr(114)+chr(95)+chr(102)+chr(114)+chr(97)+chr(109)+chr(101)):.1f}" if not np.isnan(g("ms_per_frame")) else "")} |')
def cand_rows(kind, pk, c): return [row['targets'][pk]['cands'][c] for row in res[kind].values() if pk in row['targets'] and c in row['targets'][pk]['cands']]
ledger_row('ARDY-G1 native (no codec)', native_rows('ardy_g1'), '—')
for c in CANDS: ledger_row(f'ARDY-G1 → Megatron → G1, {c}', cand_rows('ardy_g1', 'unitree_g1_29dof', c))
if 'ardy_core' in res:
    for c in CANDS: ledger_row(f'ARDY-Core → Megatron → G1, {c}', cand_rows('ardy_core', 'unitree_g1_29dof', c))
if 'ardy_g1_pp' in res:
    ledger_row('A: ARDY-G1 native + ARDY post-process (offline ADMM, NVIDIA disables it for G1)', native_rows('ardy_g1_pp'), '—')
    for c in ('R0', 'R2', 'E'): ledger_row(f'A + Megatron → G1, {c}', cand_rows('ardy_g1_pp', 'unitree_g1_29dof', c))
if 'ardy_core_pp' in res:
    for c in ('R0', 'R2', 'E'): ledger_row(f'A(Core) + Megatron → G1, {c}', cand_rows('ardy_core_pp', 'unitree_g1_29dof', c))
ledger_row('BoneSeed official G1 (Bones retarget)', [a for a in duel['bones']], '—')
ledger_row('NVIDIA soma-retargeter G1', [a for a in duel['nvidia'] if a['robot'] == 'g1'], '—')
ledger_row('Eric Prime + 100+100 refiner G1', [a for a in duel['teammate'] if a['robot'] == 'g1'], '—')
if 'official' in res:
    ledger_row('official G1 native (ruler on the same 10 clips)', native_rows('official'), '—')
    for c in CANDS: ledger_row(f'official G1 → Megatron → G1 (self round trip), {c}', cand_rows('official', 'unitree_g1_29dof', c))
md += ['']
# ---- chart
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
COL = {'native': '#52514e', 'R0': '#2a78d6', 'R1': '#eb6834', 'R2': '#1baf7a', 'R2h': '#eda100', 'R3': '#e87ba4', 'E': '#4a3aa7'}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor='#fcfcfb')
stages = [('source', native_rows('ardy_g1'))] + [(c, cand_rows('ardy_g1', 'unitree_g1_29dof', c)) for c in CANDS]
for ax, (k, title, unit) in zip(axes, [('slip_p95_cm_s', 'Stance slip p95', 'cm/s'), ('pen_max_cm', 'Floor penetration max', 'cm'), ('posture_disp_vs_R0_cm', 'Posture displacement vs R0', 'cm')]):
    ax.set_facecolor('#fcfcfb'); names = [s[0] for s in stages]; vals = [med(s[1], k) if s[1] else np.nan for s in stages]
    ax.bar(range(len(names)), [0 if np.isnan(v) else v for v in vals], color=[COL.get(n, COL['native']) for n in names], width=0.62)
    for i, v in enumerate(vals):
        if not np.isnan(v): ax.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8.5, color='#0b0b0b')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=9); ax.set_title(title, loc='left', fontsize=11, fontweight='bold'); ax.set_ylabel(unit, color='#52514e'); ax.grid(axis='y', color='#e6e5e1', lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
fig.suptitle('ARDY-G1 → Megatron → G1: median over 5 prompts, per stage (source = ARDY output without the codec)', x=0.01, ha='left', fontsize=10, color='#52514e')
fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f'{R}/loss_ledger_g1.png', dpi=150)
json.dump(res, open(f'{R}/results.json', 'w'), indent=1)
open(f'{R}/README_tables.md', 'w').write('\n'.join(md)); print('\n'.join(md))
