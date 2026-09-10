"""汇总 out/*.json -> 表 (markdown + json). 每台机器人 × 条件: 全身旋转误差 / 腿 / 骨盆高 / 手残差 / 脚残差 / 速度."""
import json, glob, os, sys, numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/tmp/refiner_suite/out')
files = sorted(glob.glob(f'{OUT}/*.json')); files = [f for f in files if 'smoke' not in f and 'summary' not in f]
R = {os.path.basename(f)[:-5]: json.load(open(f)) for f in files}
CONDS = ['zero_ik_hands', 'zero_ik_point', 'zero_ik_plane', 'zero_ik_axis', 'zero_ik_full', 'soft', 'hands',
         'feet_point', 'feet_plane', 'feet_axis', 'feet_full', 'encfeet_hands', 'encfeet_full', 'codec_axis', 'codec_full', 'feet_axis_ref0', 'feet_full_ref0']
MET = ['rot_all', 'rot_arm', 'rot_leg', 'pos_all', 'pelvis_h_err', 'hand_mm', 'hand_deg', 'foot_mm', 'foot_deg', 'foot_normal_deg', 'sec_per_frame_ms']
def g(r, c, m, k='mean'):
    x = R[r]['conds'].get(c, {}).get(m); 
    if x is None: return np.nan
    return x if isinstance(x, (int, float)) else x.get(k, np.nan)
lines = []
# 表 1: 每台机器人, 主要条件的全身旋转误差 (°, 均值)
main = ['zero_ik_hands', 'zero_ik_point', 'zero_ik_plane', 'zero_ik_axis', 'zero_ik_full', 'hands', 'feet_point', 'feet_plane', 'feet_axis', 'feet_full', 'encfeet_full', 'codec_full', 'feet_axis_ref0', 'feet_full_ref0']
lines.append('## 表1 全身旋转误差 rot_all (°, 均值; 零样本机身没有真值, 只报残差)\n')
lines.append('| robot | frames | ' + ' | '.join(main) + ' |'); lines.append('|' + '---|' * (len(main) + 2))
for r in R:
    if R[r]['zero_shot']: continue
    lines.append(f'| {r} | {R[r]["n_frames"]} | ' + ' | '.join(f'{g(r,c,"rot_all"):.1f}' for c in main) + ' |')
cached = [r for r in R if not R[r]['zero_shot']]
lines.append('| **mean** | | ' + ' | '.join(f'**{np.nanmean([g(r,c,"rot_all") for r in cached]):.1f}**' for c in main) + ' |')
for m, t in (('rot_leg', '腿旋转误差 (°)'), ('pelvis_h_err', '骨盆高误差 (cm)'), ('rot_arm', '臂旋转误差 (°)')):
    lines.append(f'\n## {t} — {m}\n'); lines.append('| robot | ' + ' | '.join(main) + ' |'); lines.append('|' + '---|' * (len(main) + 1))
    for r in cached: lines.append(f'| {r} | ' + ' | '.join(f'{g(r,c,m):.1f}' for c in main) + ' |')
    lines.append('| **mean** | ' + ' | '.join(f'**{np.nanmean([g(r,c,m) for r in cached]):.1f}**' for c in main) + ' |')
# 表 2: 约束满足度 (所有机身含零样本)
lines.append('\n## 表2 约束残差 (均值): 手 mm / 手 ° / 脚 mm / 脚 ° / 脚底法线 ° ; 速度 ms/帧\n')
sub = ['zero_ik_full', 'hands', 'feet_point', 'feet_plane', 'feet_axis', 'feet_full']
lines.append('| robot | ' + ' | '.join(sub) + ' |'); lines.append('|' + '---|' * (len(sub) + 1))
for r in R:
    cells = []
    for c in sub:
        cells.append(f'{g(r,c,"hand_mm"):.1f}mm {g(r,c,"hand_deg"):.1f}° / {g(r,c,"foot_mm"):.0f}mm {g(r,c,"foot_deg"):.0f}° n{g(r,c,"foot_normal_deg"):.0f}° / {g(r,c,"sec_per_frame_ms"):.0f}ms')
    lines.append(f'| {r}{" (0-shot)" if R[r]["zero_shot"] else ""} | ' + ' | '.join(cells) + ' |')
# 表 3: p95
lines.append('\n## 表3 rot_all p95 (°)\n'); lines.append('| robot | ' + ' | '.join(main) + ' |'); lines.append('|' + '---|' * (len(main) + 1))
for r in cached: lines.append(f'| {r} | ' + ' | '.join(f'{g(r,c,"rot_all","p95"):.1f}' for c in main) + ' |')
md = '\n'.join(lines); open(f'{OUT}/TABLE.md', 'w').write(md); print(md)
json.dump({r: {c: {m: g(r, c, m) for m in MET} for c in CONDS} for r in R}, open(f'{OUT}/summary.json', 'w'), indent=1)
