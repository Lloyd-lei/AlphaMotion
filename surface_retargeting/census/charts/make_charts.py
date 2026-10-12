# -*- coding: utf-8 -*-
"""Five synthesis charts for REPORT.md. Read-only inputs; writes PNGs into charts/.
Run: OMP_NUM_THREADS=2 taskset -c 0-15 nice -n 10 python3 make_charts.py
"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LogNorm, LinearSegmentedColormap

D = f'{SRCFG.data_root}/diversity'
C = f'{D}/charts'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT)
fam = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({'font.family': fam, 'font.size': 10.5, 'axes.unicode_minus': False})

# ---- tokens (dataviz reference palette, light) ----
SURF, INK, INK2, MUTED, GRID, BASE = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SRC = ['HiPHI', 'BONES-SEED', 'AMASS(xwbc)', 'GRAIL', 'OmniContact', 'LAFAN/Mixamo']
COL = {'HiPHI': '#2a78d6', 'BONES-SEED': '#eb6834', 'AMASS(xwbc)': '#1baf7a', 'GRAIL': '#4a3aa7',
       'OmniContact': '#e87ba4', 'LAFAN/Mixamo': MUTED}
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
CLS = ['locomotion', 'dynamic', 'ground_contact', 'sitting_on_object', 'upper_body_manipulation',
       'whole_body_object', 'gesture_social', 'idle_posture', 'other']
CLS_ZH = {'locomotion': '走路移动', 'dynamic': '高动态（跳/踢/舞/武/摔/翻）', 'ground_contact': '地面接触（跪/爬/躺/坐地）',
          'sitting_on_object': '坐物体（椅/箱/凳）', 'upper_body_manipulation': '上肢操作物体',
          'whole_body_object': '全身+物体（搬/推/踢）', 'gesture_social': '手势/社交/情绪',
          'idle_posture': '站姿/调整/拉伸', 'other': '其他'}

N = json.load(open(f'{C}/synthesis_numbers.json'))
core = json.load(open(f'{D}/core_sources/summary.json'))
og = json.load(open(f'{D}/omnicontact_grail/summary.json'))
hs = json.load(open(f'{D}/hiphi/hiphi_diversity_summary.json'))


def style(ax, xgrid=True):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(BASE); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, length=0)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)


def fig_(w, h, **kw):
    f = plt.figure(figsize=(w, h), facecolor=SURF, **kw)
    return f


def legend_row(fig, names, y=0.955, extra=None):
    hs_ = [plt.Rectangle((0, 0), 1, 1, color=COL[n]) for n in names]
    lab = [extra.get(n, n) if extra else n for n in names]
    fig.legend(hs_, lab, loc='upper center', bbox_to_anchor=(0.5, y), ncol=len(names), frameon=False,
               fontsize=10, labelcolor=INK2, handlelength=1.0, handleheight=1.0, columnspacing=1.6)


# =====================================================================
# 1) hours by class x source, split: converted-in-hand | xwbc-derivative-only | pending download
# =====================================================================
H = N['hiphi']['classes_primary']
G = N['grail']
p_bones = core['primary_hours_by_source']['BONES-SEED 原始']
p_amass = core['primary_hours_by_source']['AMASS(xwbc)']
p_lafan = core['primary_hours_by_source']['LAFAN(xwbc)']
p_mix = core['primary_hours_by_source']['Mixamo']
z = {c: 0.0 for c in CLS}
gr_now = dict(z); gr_now['locomotion'] = sum(G['converted_now'][k] for k in ('curb', 'slope', 'stair_p1', 'stair_p2'))
gr_now['sitting_on_object'] = G['converted_now']['sitting']
gr_now['ground_contact'] = 0.653  # 155 kneel/deep-squat clips of pickup_ground (all converted)
gr_now['upper_body_manipulation'] = G['converted_now']['pickup_table'] + G['converted_now']['pickup_ground'] - 0.653
gr_pend = dict(z); gr_pend['locomotion'] = G['pending_est']['stair_p1'] + G['pending_est']['stair_p2']
gr_pend['upper_body_manipulation'] = G['pending_est']['pickup_table']
omni = dict(z); omni['whole_body_object'] = 11.119; omni['upper_body_manipulation'] = 0.907
panels = {
    'A': {'HiPHI': {c: H[c].get('in_hand', 0) for c in CLS}, 'GRAIL': gr_now, 'OmniContact': omni},
    'B': {'AMASS(xwbc)': {c: p_amass[c] for c in CLS}, 'LAFAN/Mixamo': {c: p_lafan[c] for c in CLS}},
    'C': {'HiPHI': {c: H[c]['total'] - H[c].get('in_hand', 0) for c in CLS}, 'BONES-SEED': {c: p_bones[c] for c in CLS},
          'GRAIL': gr_pend, 'LAFAN/Mixamo': {c: p_mix[c] for c in CLS}},
}
tot = {k: sum(sum(v.values()) for v in P.values()) for k, P in panels.items()}
titles = {'A': f'① 在盘且已转 SMPL-X（UMR 可直接用）\n合计 {tot["A"]:.1f} h',
          'B': f'② 在盘但只有 xwbc 22 刚体派生\n（进 UMR 要重下原始）合计 {tot["B"]:.1f} h',
          'C': f'③ 待下载（只知道元数据）\n合计 {tot["C"]:.1f} h'}
f = fig_(16, 7.4)
axs = f.subplots(1, 3, sharey=True, gridspec_kw={'wspace': 0.08, 'left': 0.17, 'right': 0.985, 'top': 0.80, 'bottom': 0.09})
xmax = 0
for k, P in panels.items():
    for c in CLS:
        xmax = max(xmax, sum(P[s][c] for s in P))
y = np.arange(len(CLS))[::-1]
for ax, (k, P) in zip(axs, panels.items()):
    style(ax)
    left = np.zeros(len(CLS))
    for s in SRC:
        if s not in P:
            continue
        v = np.array([P[s][c] for c in CLS])
        ax.barh(y, v, left=left, height=0.62, color=COL[s], edgecolor=SURF, linewidth=1.0)
        left += v
    for yi, tv in zip(y, left):
        ax.text(tv + xmax * 0.012, yi, f'{tv:.1f}', va='center', ha='left', fontsize=9.5, color=INK)
    ax.set_xlim(0, xmax * 1.13)
    ax.set_title(titles[k], fontsize=11.5, color=INK, loc='left')
    ax.set_xlabel('小时（原片，不含镜像；每条只归一个主类）', color=INK2, fontsize=9.5)
axs[0].set_yticks(y); axs[0].set_yticklabels([CLS_ZH[c] for c in CLS], color=INK)
f.suptitle('人体动作时长：按动作大类 × 数据源，分“现在能用 / 只有派生格式 / 还没下”', x=0.012, ha='left', y=0.985, fontsize=14, color=INK)
legend_row(f, SRC, y=0.925, extra={'LAFAN/Mixamo': 'LAFAN（②）/ Mixamo（③）'})
f.text(0.012, 0.012, '注：GRAIL ①=今天 18:06 已转部分，③=HF 上有人体但没下的估计（楼梯 23.9 h + 桌面取物 3.4 h）；GRAIL 地面取物另有 6.4 h 官方没给人体，永远拿不到，不计。'
       'HiPHI ①=已在盘的 9 个包（全部已转）。BONES 镜像 144.1 h、HiPHI 镜像 308.7 h 未计入。',
       fontsize=8.8, color=INK2)
f.savefig(f'{C}/01_hours_by_class_source.png', dpi=150, facecolor=SURF)
plt.close(f)

# =====================================================================
# 2) contact coverage heatmap (source x contact type)
# =====================================================================
cont_cols = ['feet_only', 'hands_ground', 'knees', 'hips_back', 'object_hands', 'object_feet']
cont_zh = ['只用脚着地\n（不碰物体）', '手撑地', '膝着地', '臀/背\n（坐物体、坐地、躺）', '手-物体', '脚-物体\n（踢/踩/推）']
gav = G['human_available_est']
rows = [
    ('HiPHI ★', 308.73, N['hiphi']['contacts6']),
    ('BONES-SEED', 144.22, N['bones_contacts6']),
    ('AMASS(xwbc)', 43.83, N['amass_contacts6']),
    ('GRAIL ★（合成）', sum(gav.values()), {'feet_only': gav['curb'] + gav['slope'] + gav['stair_p1'] + gav['stair_p2'],
                                       'knees': 0.653, 'hips_back': gav['sitting'],
                                       'object_hands': gav['pickup_table'] + gav['pickup_ground']}),
    ('OmniContact ★', 12.026, {'object_hands': 3.484 + 2.929 + 0.907 + 1.051, 'object_feet': 2.064 + 0.535 + 0.533 + 0.523 + 1.051}),
    ('LAFAN', 4.599, N['lafan_contacts6']),
    ('Mixamo', 2.408, N['mixamo_contacts6']),
]
M = np.array([[r[2].get(c, 0.0) for c in cont_cols] for r in rows])
cmap = LinearSegmentedColormap.from_list('seqblue', SEQ)
f = fig_(13, 6.6)
ax = f.add_axes([0.16, 0.10, 0.70, 0.72])
style(ax, xgrid=False)
Mm = np.ma.masked_where(M <= 0.0, M)
im = ax.imshow(Mm, cmap=cmap, norm=LogNorm(vmin=0.05, vmax=200), aspect='auto')
ax.set_facecolor('#f0efec')
for i, r in enumerate(rows):
    for j, c in enumerate(cont_cols):
        v = M[i, j]
        if v <= 0:
            ax.text(j, i, '0', ha='center', va='center', fontsize=10, color=MUTED)
            continue
        rgba = cmap(LogNorm(vmin=0.05, vmax=200)(v))
        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        tc = 'white' if lum < 0.55 else INK
        ax.text(j, i, f'{v:.2f} h\n{100 * v / r[1]:.1f}%', ha='center', va='center', fontsize=9.5, color=tc)
ax.set_xticks(range(len(cont_cols))); ax.set_xticklabels(cont_zh, color=INK, fontsize=10)
ax.xaxis.tick_top()
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{r[0]}\n共 {r[1]:.1f} h' for r in rows], color=INK, fontsize=10)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-.5, len(cont_cols), 1), minor=True); ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
ax.grid(which='minor', color=SURF, linewidth=2.5); ax.tick_params(which='minor', length=0)
cax = f.add_axes([0.885, 0.10, 0.018, 0.72])
cb = f.colorbar(im, cax=cax); cb.outline.set_visible(False); cb.ax.tick_params(colors=INK2, labelsize=9)
cb.set_label('小时（对数色阶）', color=INK2)
f.text(0.012, 0.955, '接触类型覆盖：每个源有多少小时带这种接触（原片，一条可算多种）', fontsize=14, color=INK)
f.text(0.012, 0.915, '★ = 有物体 6DoF 位姿+网格；其余源的“物体”只是文字推断。百分比 = 占该源总时长。', fontsize=9.5, color=INK2)
f.text(0.012, 0.012, '注：GRAIL 楼梯/斜坡/路沿 44.0 h 记在“只用脚”（踩的是地形网格）。GRAIL 按 HF 上有人体的 55.6 h 计；OmniContact 每段都带物体，所以“只用脚”为 0。\n'
       'BONES/AMASS 的“臀/背”包含坐椅子。接触都是从文字/类别推出来的，不是从动作数据算的。', fontsize=8.8, color=INK2, linespacing=1.5)
f.savefig(f'{C}/02_contact_coverage.png', dpi=150, facecolor=SURF)
plt.close(f)

# =====================================================================
# 3) object categories: with pose+mesh | text only
# =====================================================================
ho = hs['hoi_by_object']
pose = [
    ('椅子', {'HiPHI': ho['chair']['hours_orig'], 'GRAIL': gav['sitting']}),
    ('楼梯（地形）', {'GRAIL': gav['stair_p1'] + gav['stair_p2']}),
    ('桌子（约 60 cm 小边桌）', {'HiPHI': ho['table']['hours_orig']}),
    ('箱子', {'HiPHI': ho['box']['hours_orig'], 'OmniContact': 3.484 + 2.064 + 2.929}),
    ('厨房/食物小物件（GRAIL 75 类）', {'GRAIL': gav['pickup_table'] + gav['pickup_ground'], 'HiPHI': ho['bottle']['hours_orig']}),
    ('球 / 足球', {'HiPHI': ho['ball']['hours_orig'] + ho['soccerball']['hours_orig'], 'OmniContact': 0.535 + 0.533 + 0.523 + 0.907 + 1.051}),
    ('斜坡（地形）', {'GRAIL': gav['slope']}),
    ('路沿（地形）', {'GRAIL': gav['curb']}),
    ('水桶', {'HiPHI': ho['bucket']['hours_orig']}),
    ('踏凳', {'HiPHI': ho['stepstool']['hours_orig']}),
    ('衣架', {'HiPHI': ho['clothrack']['hours_orig']}),
    ('长凳', {'HiPHI': ho['bench']['hours_orig']}),
    ('拖把', {'HiPHI': ho['mop']['hours_orig']}),
    ('垃圾桶', {'HiPHI': ho['trashbin']['hours_orig']}),
]
bo = json.load(open(f'{D}/bones_seed/bones_summary.json'))['objects']['prop_category_hours_originals']
am = json.load(open(f'{D}/xwbc_amass_lafan/xwbc_summary.json'))['amass']['objects']['implied_object_category_hours']
tmap = [('box_crate_generic', '箱子/板条箱'), ('terrain_env', '环境/障碍（墙/架/台阶）'), ('furniture', '家具（椅/桌/架）'),
        ('articulated_fixture', '门/冰箱/抽屉'), ('tool', '工具'), ('kitchenware_food', '厨房/食物'),
        ('handheld_device', '手机等手持设备'), ('small_geometric_toy', '小几何体（GRAB）'), ('ball_sport_gear', '球/运动器材'),
        ('assistive', '拐杖等辅具'), ('instrument', '乐器'), ('person_or_doll', '人/玩偶（沙袋婴儿）')]
text = [(zh, {'BONES-SEED': bo.get(k, 0.0), 'AMASS(xwbc)': am.get(k, 0.0)}) for k, zh in tmap]
pose.sort(key=lambda r: -sum(r[1].values())); text.sort(key=lambda r: -sum(r[1].values()))
f = fig_(15, 8.2)
axL = f.add_axes([0.20, 0.09, 0.33, 0.76]); axR = f.add_axes([0.70, 0.09, 0.28, 0.76])
for ax, data, srcs, ttl in ((axL, pose, ['HiPHI', 'GRAIL', 'OmniContact'], f'有物体 6DoF 位姿+网格（合计 {sum(sum(r[1].values()) for r in pose):.1f} h）'),
                            (axR, text, ['BONES-SEED', 'AMASS(xwbc)'], f'只有文字/推断，没有物体数据（可重复计）')):
    style(ax)
    yy = np.arange(len(data))[::-1]
    left = np.zeros(len(data))
    for s in srcs:
        v = np.array([r[1].get(s, 0.0) for r in data])
        ax.barh(yy, v, left=left, height=0.62, color=COL[s], edgecolor=SURF, linewidth=1.0)
        left += v
    mx = left.max()
    for yi, tv in zip(yy, left):
        ax.text(tv + mx * 0.012, yi, f'{tv:.2f}', va='center', fontsize=9.2, color=INK)
    ax.set_yticks(yy); ax.set_yticklabels([r[0] for r in data], color=INK, fontsize=10)
    ax.set_xlim(0, mx * 1.14)
    ax.set_title(ttl, loc='left', fontsize=11.5, color=INK)
    ax.set_xlabel('小时（原片）', color=INK2, fontsize=9.5)
f.text(0.012, 0.965, '物体类别时长（共 26 类，按小时排序）', fontsize=14, color=INK)
legend_row(f, ['HiPHI', 'GRAIL', 'OmniContact', 'BONES-SEED', 'AMASS(xwbc)'], y=0.945)
f.text(0.012, 0.012, '注：GRAIL 按 HF 上有人体的小时计（楼梯已转 9.97 h，其余 23.9 h 待下）；GRAIL 物体是程序化/生成资产，人是合成角色。HiPHI 每条只有 1 个物体。'
       'BONES/AMASS 类别来自文字字段，一条可属多类。Mixamo 名字带道具 0.24 h 未画。', fontsize=8.8, color=INK2)
f.savefig(f'{C}/03_object_categories.png', dpi=150, facecolor=SURF)
plt.close(f)

# =====================================================================
# 4) status funnel
# =====================================================================
stages = ['元数据已知\n（原片）', '原始数据在盘', '已转 SMPL-X', '已重定向\nUMR', '已重定向\n旧管线（非 UMR）']
umr_h = (5771 / 90 + 1500 / 90) / 3600 + 3945 / 30 / 3600  # Omni kick full + carry window, LAFAN dance1 full
vals = {
    'HiPHI': [308.73, N['hiphi']['on_disk_hours'], N['hiphi']['converted_hours'], 0, 0],
    'BONES-SEED': [144.22, 0, 0, 0, 144.22],
    'AMASS(xwbc)': [43.83, 0, 0, 0, 43.83],
    'GRAIL': [sum(gav.values()), G['converted_total'], G['converted_total'], 0, 0],
    'OmniContact': [12.026, 12.026, 12.026, (5771 / 90 + 1500 / 90) / 3600, 0],
    'LAFAN/Mixamo': [4.599 + 2.408, 0, 3945 / 30 / 3600, 3945 / 30 / 3600, 4.599],
}
f = fig_(13, 8)
ax = f.add_axes([0.07, 0.25, 0.90, 0.60])
style(ax, xgrid=False); ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
x = np.arange(len(stages)); bottom = np.zeros(len(stages))
for s in SRC:
    v = np.array(vals[s])
    ax.bar(x, v, bottom=bottom, width=0.56, color=COL[s], edgecolor=SURF, linewidth=1.0)
    bottom += v
for xi, tv in zip(x, bottom):
    lab = f'{tv:.1f} h' if tv >= 1 else f'≈{tv:.2f} h'
    ax.text(xi, tv + 8, lab, ha='center', va='bottom', fontsize=11, color=INK)
ax.set_xticks(x); ax.set_xticklabels(stages, color=INK, fontsize=10.5)
ax.set_ylabel('小时', color=INK2)
ax.set_ylim(0, bottom.max() * 1.12)
notes = ['HiPHI 308.7 · BONES 144.2\nGRAIL 55.6 · AMASS 43.8\nOmni 12.0 · LAFAN/Mixamo 7.0',
         'HiPHI 9/32 包 86.0\nGRAIL 28.4 · Omni 12.0\n（另有 xwbc 派生 48.4 h）',
         'HiPHI 4889 条 86.0\nGRAIL 10943 条 28.4\nOmni 690 条 12.0',
         '2 段完整 + 4 个 QA 窗口\n完整段跑了 3 台\n（目标 21 台）',
         'AMASS/LAFAN → 18 台（xwbc/GMR）\nBONES → 只有 G1（厂商自带）']
for xi, t in zip(x, notes):
    ax.text(xi, -0.12, t, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=8.6, color=INK2, linespacing=1.4)
f.text(0.012, 0.955, '数据状态漏斗：知道有 → 下到本机 → 转成 SMPL-X → 重定向到机器人', fontsize=14, color=INK)
legend_row(f, SRC, y=0.935)
f.text(0.012, 0.012, f'口径：今天 18:10 实读。UMR 已重定向的人体时长只有约 {umr_h * 3600:.0f} 秒（OmniContact 踢球整段 + 搬箱窗口、LAFAN 舞蹈 1 段），所以柱子看不见。\n'
       'BONES 的“旧管线”是厂商给的 G1 版本，人体原始数据本机还没有；AMASS/LAFAN 的旧管线结果来自 xwbc/GMR，不是 UMR。', fontsize=8.8, color=INK2, linespacing=1.5)
f.savefig(f'{C}/04_status_funnel.png', dpi=150, facecolor=SURF)
plt.close(f)

# =====================================================================
# 5) actor diversity
# =====================================================================
act = [
    ('BONES-SEED', 522, '男 269 / 女 253 · 年龄 17–71 · 体重 38–145 kg'),
    ('AMASS(xwbc)', 428, '428 个人/场次文件夹（不等于人数），无身高/性别'),
    ('HiPHI', 132, '男 76 / 女 56 · 体重 40–82 kg · 女性时长 34%'),
    ('OmniContact', 6, '约 6 人（按骨架身高聚类估计），无性别'),
    ('LAFAN/Mixamo', 5, 'LAFAN 5 人；Mixamo 只有 1 个角色'),
    ('GRAIL', 0, '0 个真人：全是合成角色，体型参数全 0，身高 1.66 m'),
]
f = fig_(15, 7.2)
axA = f.add_axes([0.10, 0.12, 0.36, 0.70]); axB = f.add_axes([0.58, 0.12, 0.40, 0.70])
style(axA)
yy = np.arange(len(act))[::-1]
for yi, (s, n, t) in zip(yy, act):
    axA.barh(yi, n, height=0.55, color=COL[s], edgecolor=SURF, alpha=0.45 if s == 'AMASS(xwbc)' else 1.0)
    axA.text(n + 8, yi + 0.02, f'{n}', va='center', fontsize=10.5, color=INK)
    axA.text(n + 8, yi - 0.36, t, va='center', fontsize=8.6, color=INK2)
axA.set_yticks(yy); axA.set_yticklabels([a[0] for a in act], color=INK)
axA.set_xlim(0, 1050); axA.set_xlabel('演员数', color=INK2)
axA.set_title('演员人数（AMASS 浅色 = 只是文件夹数）', loc='left', fontsize=11.5, color=INK)
style(axB)
hsrc = [('BONES-SEED', N['bones_actor_heights_cm']), ('HiPHI', N['hiphi']['actor_heights_cm']),
        ('OmniContact', [173.2, 176.5, 180.1, 184.3, 187.7, 189.0]), ('GRAIL', [166.0])]
rng = np.random.default_rng(0)
yb = np.arange(len(hsrc))[::-1]
for yi, (s, hv) in zip(yb, hsrc):
    hv = np.array(hv, float)
    j = rng.uniform(-0.22, 0.22, len(hv)) if len(hv) > 20 else np.zeros(len(hv))
    axB.scatter(hv, yi + j, s=14 if len(hv) > 20 else 46, color=COL[s], edgecolor=SURF, linewidth=0.8, alpha=0.75 if len(hv) > 20 else 1, zorder=3)
    if len(hv) > 1:
        p10, p50, p90 = np.percentile(hv, [10, 50, 90])
        axB.plot([p10, p90], [yi - 0.36, yi - 0.36], color=INK2, lw=2, solid_capstyle='round')
        axB.plot([p50], [yi - 0.36], 'o', color=INK, ms=5)
        axB.text(hv.max() + 1.2, yi, f'{hv.min():.0f}–{hv.max():.0f} cm\n中位 {p50:.0f}', va='center', fontsize=9, color=INK2)
    else:
        axB.text(hv[0] + 1.5, yi, '1 个合成体型', va='center', fontsize=9, color=INK2)
axB.set_yticks(yb); axB.set_yticklabels([h[0] for h in hsrc], color=INK)
axB.set_xlim(140, 212); axB.set_xlabel('身高 cm（每个点 = 1 个演员；横线 = 10%–90% 分位，黑点 = 中位）', color=INK2, fontsize=9.5)
axB.set_title('演员身高分布（只画有身高数据的源）', loc='left', fontsize=11.5, color=INK)
f.text(0.012, 0.955, '演员多样性：有多少人、多高、男女比例', fontsize=14, color=INK)
f.text(0.012, 0.012, '注：现在已转 SMPL-X 的 HiPHI 只覆盖 57 个演员（9 个带物体的包）；OmniContact 身高来自 26 套骨架校准的聚类，是估计。LAFAN、AMASS、Mixamo 无身高数据。', fontsize=8.8, color=INK2)
f.savefig(f'{C}/05_actor_diversity.png', dpi=150, facecolor=SURF)
plt.close(f)

json.dump({'panel_totals_fig1': tot, 'umr_retargeted_seconds': umr_h * 3600, 'funnel': vals,
           'contact_matrix': {r[0]: dict(zip(cont_cols, map(float, M[i]))) for i, r in enumerate(rows)},
           'objects_pose': pose, 'objects_text': text},
          open(f'{C}/chart_data.json', 'w'), ensure_ascii=False, indent=1)
print('ok', tot, umr_h * 3600)
