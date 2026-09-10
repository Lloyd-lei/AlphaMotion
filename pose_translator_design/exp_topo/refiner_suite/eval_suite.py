"""EE SE(3) 约束下的全身精度评估 (每台机器人一进程).
有真值机身 (缓存 15 台): 真值 = GMR 重定向 (语料外 600 段抽样). 输入只给 根旋转 + 双手 SE(3) (+ 可选双脚), 模型软补全 -> 精修器硬钉.
零样本机身 (X2 / A3): 无真值, 钉点由 G1 真值按臂长/腿长比缩放而来; 只报钉残差 + 姿态自洽, 精度靠视频.
条件:
  zero_ik      : 不用模型, 从零姿态起 IK 钉 根+双手 (对照: 精修器本身有多强)
  soft         : 编码器只见 根+双手, 解码, 不精修
  hands        : soft + 精修器钉双手 (full), 脚不管
  feet_point / feet_plane / feet_axis / feet_full : hands + 双脚按四档约束
  encfeet_hands: 编码器也见双脚 (只软), 精修器只钉双手  (脚信息给模型 vs 给精修器)
  encfeet_full : 编码器见双脚 + 精修器钉双手双脚 full (上界)
用法: eval_suite.py <robot> <out.json> [--n_clips 40 --stride 3 --max_per_clip 200 --bs 64 --conds a,b,c --ck FZ5c...]"""
import os, sys, json, time, argparse, numpy as np, torch
E = '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'; sys.path.insert(0, E)
sys.path.insert(0, os.path.expanduser('~/Desktop/AlphaMotion/src')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_grad_enabled(False)
from common import SkeletonSpec, fk_global, matrix_to_rot6d, rot6d_to_matrix
from dual_common import build_model, geom, static_cond
from manifold_project import fit_angles, fk_from_angles, load_dof
from semantics import build_name_embeddings, embed_matrix
from refine import refine, geo_deg
import profiles as PF

ap = argparse.ArgumentParser()
ap.add_argument('robot'); ap.add_argument('out')
ap.add_argument('--ck', default='FZ5c_f32_slipw2_hpair01_rp3_0902')
ap.add_argument('--n_clips', type=int, default=40); ap.add_argument('--stride', type=int, default=3)
ap.add_argument('--max_per_clip', type=int, default=200); ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--conds', default='zero_ik_hands,zero_ik_point,zero_ik_plane,zero_ik_axis,zero_ik_full,soft,hands,feet_point,feet_plane,feet_axis,feet_full,encfeet_hands,encfeet_full,codec_axis,codec_full')
ap.add_argument('--lm_iters', type=int, default=60); ap.add_argument('--w_ref', type=float, default=1e-2)
ap.add_argument('--src', default='unitree_g1_29dof', help='零样本机身的钉点来源 (有真值机身)')
ap.add_argument('--threads', type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(a.threads)
RUNS = '/var/tmp/alphamotion/topo_exp/runs'; CK = f'{RUNS}/{a.ck}'
C = os.environ.get('AM_CACHE', '/var/tmp/alphamotion/topo_exp/cache_amass_lconv_tfix')
SET = os.environ.get('AM_SET', '/var/tmp/alphamotion/topo_exp/cache_amass_heldout/sequences.json')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
PROF = json.load(open(os.path.expanduser('~/tmp/refiner_suite/profiles.json')))
ZS = a.robot in PF.XML                      # 零样本机身
t0 = time.time()

# ---------------- 片段抽样 (确定性: 均匀跨 600 段)
seqs_all = {q['name']: q for q in json.load(open(f'{C}/sequences.json'))}
names = [q['name'] for q in json.load(open(SET))]
clips = [seqs_all[n] for n in names if n in seqs_all]
step = max(1, len(clips) // a.n_clips); clips = clips[::step][:a.n_clips]
rows, cid = [], []
for k, q in enumerate(clips):
    r = torch.arange(q['start'], q['start'] + q['len'], a.stride)[:a.max_per_clip]; rows.append(r); cid += [k] * len(r)
rows = torch.cat(rows); cid = np.array(cid); B = len(rows)
print(f'[{a.robot}] {len(clips)} clips, {B} frames (stride {a.stride})', flush=True)

# ---------------- 机身 + 真值
def load_spec(n):
    d = np.load(f'{C}/{n}_spec.npz', allow_pickle=True)
    return SkeletonSpec(str(d['name']), d['parents'], d['rest_offsets'], [str(x) for x in d['joint_names']])
cfg = json.load(open(f'{CK}/config.json'))
src = a.src if ZS else a.robot
spec_s = load_spec(src)
l6 = torch.load(f'{C}/{src}_paired_local6d.pt', mmap=True)
gR_s, gp_s = fk_global(l6[rows].float().to(dev), spec_s); gR_s, gp_s = gR_s.double().cpu(), gp_s.double().cpu()   # 真值 (源机身), 根相对 cm
_, gpf = fk_global(l6[::50].float().to(dev), spec_s); reach_s = float(gpf.norm(dim=-1).max()); del gpf
if ZS:
    spec, dof, rest, _ = PF.load(a.robot)
    pr = np.zeros((spec.J, 3), np.float32)
    for j in range(spec.J):
        p_ = spec.parents[j]; pr[j] = (pr[p_] if p_ >= 0 else 0) + np.asarray(spec.rest_offsets[j])
    reach = float(np.linalg.norm(pr, axis=1).max())
else:
    spec = spec_s; dof, rest = load_dof(C, a.robot, device='cpu'); dof, rest = dof.double(), rest.double(); reach = reach_s
J = spec.J; jn = spec.joint_names; root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
P = PROF[a.robot]; Ps = PROF[src]
HANDS = [P['left']['hand'], P['right']['hand']]; FEET = [P['left']['foot'], P['right']['foot']]
def chain(side, kind):   # 手/脚到肩/髋的关节链 (含两端)
    j = P[side][kind]; top = P[side]['shoulder' if kind == 'hand' else 'hip']; out = [j]
    while out[-1] != top: out.append(int(spec.parents[out[-1]]))
    return out
ARM = chain('left', 'hand') + chain('right', 'hand'); LEG = chain('left', 'foot') + chain('right', 'foot')
NONROOT = [j for j in range(J) if j != root]

# 真值 (投到机械流形上, 与 indist_test 同): 有真值机身 = 自身 GMR; 零样本 = 无
if not ZS:
    q_gt, _ = fit_angles(gR_s, spec, dof, rest=rest, clamp=True, method='global', root_R=gR_s[:, root])
    Rg, Pg = fk_from_angles(q_gt, spec, dof, rest=rest, root_R=gR_s[:, root])
    root_R = gR_s[:, root]
    tgt_pos = {j: Pg[:, j] for j in HANDS + FEET}; tgt_rot = {j: Rg[:, j] for j in HANDS + FEET}
else:
    # 钉点从源机身按 肩/髋 起算, 臂长/腿长比缩放; 旋转照抄 (所有机身 rest 帧都是世界对齐的, 见 profiles)
    dof_s, rest_s = load_dof(C, src, device='cpu'); dof_s, rest_s = dof_s.double(), rest_s.double()
    root_s = int(np.where(np.asarray(spec_s.parents) < 0)[0][0]); root_R = gR_s[:, root_s]
    q_gs, _ = fit_angles(gR_s, spec_s, dof_s, rest=rest_s, clamp=True, method='global', root_R=root_R)
    Rg_s, Pg_s = fk_from_angles(q_gs, spec_s, dof_s, rest=rest_s, root_R=root_R)
    _, restp = fk_from_angles(torch.zeros(1, J, 3, dtype=torch.float64), spec, dof, rest=rest); restp = restp[0]
    _, restp_s = fk_from_angles(torch.zeros(1, spec_s.J, 3, dtype=torch.float64), spec_s, dof_s, rest=rest_s); restp_s = restp_s[0]
    tgt_pos, tgt_rot = {}, {}
    for side in ('left', 'right'):
        for kind, top, L in (('hand', 'shoulder', 'arm_len_cm'), ('foot', 'hip', 'leg_len_cm')):
            j, js = P[side][kind], Ps[side][kind]; t, ts = P[side][top], Ps[side][top]
            s = P[side][L] / Ps[side][L]
            tgt_pos[j] = restp[t][None] + (Pg_s[:, js] - restp_s[ts][None]) * s
            tgt_rot[j] = Rg_s[:, js]
    print(f'[{a.robot}] zero-shot: targets from {src} scaled arm x{P["left"]["arm_len_cm"]/Ps["left"]["arm_len_cm"]:.3f} leg x{P["left"]["leg_len_cm"]/Ps["left"]["leg_len_cm"]:.3f}', flush=True)
print(f'[{a.robot}] GT ready {time.time()-t0:.0f}s', flush=True)

# ---------------- 模型
lut = build_name_embeddings(jn, device=dev, encoder=cfg.get('sem_encoder', 'siglip'))
data = {a.robot: {'spec': spec, 'st': static_cond(spec, dev), 'reach': reach, 'sem': embed_matrix(jn, lut).to(dev), 'dof': dof.float().to(dev)}}
from dof_head import DofKin
data[a.robot]['kin'] = DofKin(spec, dof.float().to(dev), rest.float().to(dev), device=dev)
model = build_model(cfg, dev); model.load_state_dict(torch.load(f'{CK}/model.pt', map_location=dev), strict=False); model.eval()

def enc_input(vis_feet):
    """编码器输入 [B,J,9]: 可见关节 = 真值 (或缩放钉点), 其余 = 静止姿态占位 (会被 mask 掉)."""
    if not ZS: x_R, x_p = Rg.clone(), Pg.clone()
    else:
        x_R = torch.eye(3, dtype=torch.float64).expand(B, J, 3, 3).clone(); x_p = restp[None].expand(B, J, 3).clone()
        x_R[:, root] = root_R
        for j in HANDS + FEET: x_R[:, j] = tgt_rot[j]; x_p[:, j] = tgt_pos[j]
    vis = [root] + HANDS + (FEET if vis_feet else [])
    mask = torch.ones(B, J, dtype=torch.bool); mask[:, vis] = False
    x = torch.cat([matrix_to_rot6d(x_R.float()), x_p.float() / reach], -1)
    return x, mask

def soft_fill(vis_feet):
    x, mask = enc_input(vis_feet); outs = []
    for i in range(0, B, 1024):
        s = slice(i, min(i + 1024, B)); n = s.stop - s.start
        zp, zr = model.encode(x[s].to(dev), geom(data, a.robot, n), joint_mask=mask[s].to(dev))
        prot, _ = model.decode(zp, zr, geom(data, a.robot, n)); outs.append(prot.cpu())
    Rp = rot6d_to_matrix(torch.cat(outs).double())
    q, _ = fit_angles(Rp, spec, dof, rest=rest, clamp=True, method='global', root_R=root_R)
    return q
q_soft = {False: soft_fill(False), True: soft_fill(True)}
print(f'[{a.robot}] soft fill done {time.time()-t0:.0f}s', flush=True)

# ---------------- 精修
WP = 1e4 / reach ** 2; WR = 1e2; WD = 1e2
def pins_for(hand_mode, foot_mode):
    pins = []
    for j in HANDS:
        if hand_mode: pins.append(dict(j=j, pos=tgt_pos[j], rot=tgt_rot[j], mode='full', w_pos=WP, w_rot=WR, w_dir=WD))
    if foot_mode:
        for side, j in (('left', FEET[0]), ('right', FEET[1])):
            pins.append(dict(j=j, pos=tgt_pos[j], rot=tgt_rot[j], mode=foot_mode, w_pos=WP, w_rot=WR, w_dir=WD,
                             sole_normal_local=P[side]['sole_normal_local'], forward_local=P[side]['foot_forward_local']))
    return pins
# (起点, 手钉, 脚档): 起点 zero=零姿态(不用模型) / soft=编码器见根+双手 / encfeet=编码器见根+双手+双脚 / codec=零姿态IK全钉结果过一遍编解码(全可见)再钉
COND = {'zero_ik_hands': ('zero', True, None), 'zero_ik_point': ('zero', True, 'point'), 'zero_ik_plane': ('zero', True, 'plane'),
        'zero_ik_axis': ('zero', True, 'axis'), 'zero_ik_full': ('zero', True, 'full'),
        'soft': ('soft', False, None), 'hands': ('soft', True, None),
        'feet_point': ('soft', True, 'point'), 'feet_plane': ('soft', True, 'plane'), 'feet_axis': ('soft', True, 'axis'), 'feet_full': ('soft', True, 'full'),
        'encfeet_hands': ('encfeet', True, None), 'encfeet_full': ('encfeet', True, 'full'),
        'codec_axis': ('codec', True, 'axis'), 'codec_full': ('codec', True, 'full')}

def do_refine(q0, pins):
    Q, GR, GP = [], [], []
    for i in range(0, B, a.bs):
        s = slice(i, min(i + a.bs, B))
        pb = [{k: (v[s] if torch.is_tensor(v) and v.dim() > 1 else v) for k, v in c.items()} for c in pins]
        q, gR, gp = refine(q0[s], spec, dof, rest, root_R[s], pb, w_ref=a.w_ref, lm_iters=a.lm_iters)
        Q.append(q); GR.append(gR); GP.append(gp)
    return torch.cat(Q), torch.cat(GR), torch.cat(GP)

def codec_roundtrip(gR, gp):
    """全身可见过一遍编解码 (codec 当先验/去噪器), 回 fit_angles 角度."""
    x = torch.cat([matrix_to_rot6d(gR.float()), gp.float() / reach], -1); outs = []
    for i in range(0, B, 1024):
        s = slice(i, min(i + 1024, B)); n = s.stop - s.start
        zp, zr = model.encode(x[s].to(dev), geom(data, a.robot, n)); prot, _ = model.decode(zp, zr, geom(data, a.robot, n)); outs.append(prot.cpu())
    Rp = rot6d_to_matrix(torch.cat(outs).double())
    q, _ = fit_angles(Rp, spec, dof, rest=rest, clamp=True, method='global', root_R=root_R)
    return q
_ik_full = None
def run_cond(name):
    global _ik_full
    start, hand_mode, foot_mode = COND[name]
    if start == 'zero': q0 = torch.zeros(B, J, 3, dtype=torch.float64)
    elif start == 'soft': q0 = q_soft[False]
    elif start == 'encfeet': q0 = q_soft[True]
    else:
        if _ik_full is None: _ik_full = do_refine(torch.zeros(B, J, 3, dtype=torch.float64), pins_for(True, 'full'))
        q0 = codec_roundtrip(_ik_full[1], _ik_full[2])
    if not hand_mode and not foot_mode:
        gR, gp = fk_from_angles(q0, spec, dof, rest=rest, root_R=root_R); return q0, gR, gp
    return do_refine(q0, pins_for(hand_mode, foot_mode))

def metrics(q, gR, gp):
    m = {}
    hp = torch.stack([(gp[:, j] - tgt_pos[j]).norm(dim=-1) for j in HANDS], 1) * 10   # mm
    hr = torch.stack([geo_deg(gR[:, j], tgt_rot[j]) for j in HANDS], 1)
    fp = torch.stack([(gp[:, j] - tgt_pos[j]).norm(dim=-1) for j in FEET], 1) * 10
    fy = torch.stack([(gp[:, j, 1] - tgt_pos[j][:, 1]).abs() for j in FEET], 1) * 10
    fr = torch.stack([geo_deg(gR[:, j], tgt_rot[j]) for j in FEET], 1)
    def axdeg(j, side, key):
        a_ = torch.as_tensor(P[side][key], dtype=gR.dtype); u = gR[:, j] @ a_; v = tgt_rot[j] @ a_
        return torch.rad2deg(torch.arccos((u * v).sum(-1).clamp(-1, 1)))
    fn = torch.stack([axdeg(FEET[0], 'left', 'sole_normal_local'), axdeg(FEET[1], 'right', 'sole_normal_local')], 1)
    ff = torch.stack([axdeg(FEET[0], 'left', 'foot_forward_local'), axdeg(FEET[1], 'right', 'foot_forward_local')], 1)
    m['hand_mm'] = hp.mean(1); m['hand_deg'] = hr.mean(1); m['foot_mm'] = fp.mean(1); m['foot_h_mm'] = fy.mean(1); m['foot_deg'] = fr.mean(1)
    m['foot_normal_deg'] = fn.mean(1); m['foot_fwd_deg'] = ff.mean(1)
    m['pelvis_h'] = -torch.minimum(gp[:, FEET[0], 1], gp[:, FEET[1], 1])
    if not ZS:
        m['rot_all'] = geo_deg(gR[:, NONROOT], Rg[:, NONROOT]).mean(1)
        m['rot_arm'] = geo_deg(gR[:, ARM], Rg[:, ARM]).mean(1); m['rot_leg'] = geo_deg(gR[:, LEG], Rg[:, LEG]).mean(1)
        m['pos_all'] = (gp[:, NONROOT] - Pg[:, NONROOT]).norm(dim=-1).mean(1)
        m['pos_arm'] = (gp[:, ARM] - Pg[:, ARM]).norm(dim=-1).mean(1); m['pos_leg'] = (gp[:, LEG] - Pg[:, LEG]).norm(dim=-1).mean(1)
        m['q_deg'] = torch.rad2deg((q - q_gt).abs()).flatten(1).mean(1)
        m['pelvis_h_err'] = (m['pelvis_h'] - (-torch.minimum(Pg[:, FEET[0], 1], Pg[:, FEET[1], 1]))).abs()
    return {k: v.numpy() for k, v in m.items()}

res = {'robot': a.robot, 'zero_shot': ZS, 'ck': a.ck, 'n_clips': len(clips), 'n_frames': B, 'stride': a.stride, 'reach': reach,
       'hands': [jn[j] for j in HANDS], 'feet': [jn[j] for j in FEET], 'arm_chain': [jn[j] for j in ARM], 'leg_chain': [jn[j] for j in LEG],
       'w_pos': WP, 'w_rot': WR, 'w_dir': WD, 'w_ref': a.w_ref, 'lm_iters': a.lm_iters, 'conds': {}}
if not ZS:
    res['gt_pelvis_h'] = float((-torch.minimum(Pg[:, FEET[0], 1], Pg[:, FEET[1], 1])).mean())
QS = {}
for name in a.conds.split(','):
    t1 = time.time(); q, gR, gp = run_cond(name); m = metrics(q, gR, gp); dt = time.time() - t1
    summ = {k: {'mean': float(v.mean()), 'p95': float(np.percentile(v, 95))} for k, v in m.items()}
    summ['per_clip'] = {k: [float(v[cid == c].mean()) for c in range(len(clips))] for k, v in m.items() if k in ('rot_all', 'rot_arm', 'rot_leg', 'hand_mm', 'foot_mm', 'pelvis_h_err')}
    summ['sec_per_frame_ms'] = dt / B * 1000
    res['conds'][name] = summ; QS[name] = q.float().numpy()
    line = ' '.join(f'{k}={summ[k]["mean"]:.2f}' for k in ('rot_all', 'rot_arm', 'rot_leg', 'pos_all', 'pelvis_h_err', 'hand_mm', 'hand_deg', 'foot_mm', 'foot_deg') if k in summ)
    print(f'[{a.robot}] {name:14s} {line}  ({dt:.0f}s)', flush=True)
res['clips'] = [q['name'] for q in clips]
json.dump(res, open(a.out, 'w'), indent=1)
np.savez_compressed(a.out.replace('.json', '_q.npz'), rows=rows.numpy(), cid=cid, **QS, q_gt=(q_gt.float().numpy() if not ZS else np.zeros(0)))
print(f'[{a.robot}] done {time.time()-t0:.0f}s -> {a.out}', flush=True)
