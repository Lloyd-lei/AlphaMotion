"""精修套件公共部分 (eval_suite / render_rollout 共用): 机身加载, 真值, 钉点, 模型软补全, 条件表.
与 eval_suite.py 中的逻辑一字不差 (eval 先跑, 这里是抽出来给渲染用的)."""
import os, sys, json, time, numpy as np, torch
E = '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'; sys.path.insert(0, E)
sys.path.insert(0, os.path.expanduser('~/Desktop/AlphaMotion/src')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SkeletonSpec, fk_global, matrix_to_rot6d, rot6d_to_matrix
from dual_common import build_model, geom, static_cond
from manifold_project import fit_angles, fk_from_angles, load_dof
from semantics import build_name_embeddings, embed_matrix
from refine import refine, geo_deg
import profiles as PF

RUNS = '/var/tmp/alphamotion/topo_exp/runs'
C = os.environ.get('AM_CACHE', '/var/tmp/alphamotion/topo_exp/cache_amass_lconv_tfix')
SET = os.environ.get('AM_SET', '/var/tmp/alphamotion/topo_exp/cache_amass_heldout/sequences.json')
PROF = json.load(open(os.path.expanduser('~/tmp/refiner_suite/profiles.json')))
COND = {'zero_ik_hands': ('zero', True, None), 'zero_ik_point': ('zero', True, 'point'), 'zero_ik_plane': ('zero', True, 'plane'),
        'zero_ik_axis': ('zero', True, 'axis'), 'zero_ik_full': ('zero', True, 'full'),
        'soft': ('soft', False, None), 'hands': ('soft', True, None),
        'feet_point': ('soft', True, 'point'), 'feet_plane': ('soft', True, 'plane'), 'feet_axis': ('soft', True, 'axis'), 'feet_full': ('soft', True, 'full'),
        'encfeet_hands': ('encfeet', True, None), 'encfeet_full': ('encfeet', True, 'full'),
        'codec_axis': ('codec', True, 'axis'), 'codec_full': ('codec', True, 'full'),
        # 拆开"模型起点"和"模型参考": 起点=模型软补全, 但零空间拉向零姿态 (直立) 而不是模型姿态
        'feet_axis_ref0': ('soft', True, 'axis', 'zero'), 'feet_full_ref0': ('soft', True, 'full', 'zero'),
        'best_axis': ('best', True, 'axis'), 'best_full': ('best', True, 'full')}   # best = 两起点 (模型 / 零姿态) 逐帧取钉点残差小的

def load_spec(n):
    d = np.load(f'{C}/{n}_spec.npz', allow_pickle=True)
    return SkeletonSpec(str(d['name']), d['parents'], d['rest_offsets'], [str(x) for x in d['joint_names']])

class Robot:
    def __init__(self, robot, rows, ck='FZ5c_f32_slipw2_hpair01_rp3_0902', src='unitree_g1_29dof', dev=None,
                 bs=64, lm_iters=60, w_ref=1e-2, log=print):
        self.name = robot; self.rows = rows; self.B = B = len(rows); self.bs = bs; self.lm_iters = lm_iters; self.w_ref = w_ref
        self.dev = dev = dev or ('cuda' if torch.cuda.is_available() else 'cpu'); self.log = log
        self.ZS = ZS = robot in PF.XML; self.CK = CK = f'{RUNS}/{ck}'; self.cfg = cfg = json.load(open(f'{CK}/config.json'))
        self.src = src = src if ZS else robot
        spec_s = load_spec(src); self.spec_s = spec_s
        l6 = torch.load(f'{C}/{src}_paired_local6d.pt', mmap=True)
        gR_s, gp_s = fk_global(l6[rows].float().to(dev), spec_s); gR_s = gR_s.double().cpu()
        _, gpf = fk_global(l6[::50].float().to(dev), spec_s); reach_s = float(gpf.norm(dim=-1).max()); del gpf
        if ZS:
            spec, dof, rest, _ = PF.load(robot)
            pr = np.zeros((spec.J, 3), np.float32)
            for j in range(spec.J):
                p_ = spec.parents[j]; pr[j] = (pr[p_] if p_ >= 0 else 0) + np.asarray(spec.rest_offsets[j])
            reach = float(np.linalg.norm(pr, axis=1).max())
        else:
            spec = spec_s; dof, rest = load_dof(C, robot, device='cpu'); dof, rest = dof.double(), rest.double(); reach = reach_s
        self.spec, self.dof, self.rest, self.reach = spec, dof, rest, reach
        self.J = J = spec.J; self.jn = jn = spec.joint_names; self.root = root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
        self.P = P = PROF[robot]; Ps = PROF[src]
        self.HANDS = HANDS = [P['left']['hand'], P['right']['hand']]; self.FEET = FEET = [P['left']['foot'], P['right']['foot']]
        def chain(side, kind):
            j = P[side][kind]; top = P[side]['shoulder' if kind == 'hand' else 'hip']; out = [j]
            while out[-1] != top: out.append(int(spec.parents[out[-1]]))
            return out
        self.ARM = chain('left', 'hand') + chain('right', 'hand'); self.LEG = chain('left', 'foot') + chain('right', 'foot')
        self.NONROOT = [j for j in range(J) if j != root]
        if not ZS:
            self.q_gt, _ = fit_angles(gR_s, spec, dof, rest=rest, clamp=True, method='global', root_R=gR_s[:, root])
            self.Rg, self.Pg = fk_from_angles(self.q_gt, spec, dof, rest=rest, root_R=gR_s[:, root])
            self.root_R = gR_s[:, root]
            self.tgt_pos = {j: self.Pg[:, j] for j in HANDS + FEET}; self.tgt_rot = {j: self.Rg[:, j] for j in HANDS + FEET}
        else:
            dof_s, rest_s = load_dof(C, src, device='cpu'); dof_s, rest_s = dof_s.double(), rest_s.double()
            root_s = int(np.where(np.asarray(spec_s.parents) < 0)[0][0]); self.root_R = root_R = gR_s[:, root_s]
            q_gs, _ = fit_angles(gR_s, spec_s, dof_s, rest=rest_s, clamp=True, method='global', root_R=root_R)
            Rg_s, Pg_s = fk_from_angles(q_gs, spec_s, dof_s, rest=rest_s, root_R=root_R)
            _, restp = fk_from_angles(torch.zeros(1, J, 3, dtype=torch.float64), spec, dof, rest=rest); self.restp = restp = restp[0]
            _, restp_s = fk_from_angles(torch.zeros(1, spec_s.J, 3, dtype=torch.float64), spec_s, dof_s, rest=rest_s); restp_s = restp_s[0]
            self.tgt_pos, self.tgt_rot = {}, {}
            for side in ('left', 'right'):
                for kind, top, L in (('hand', 'shoulder', 'arm_len_cm'), ('foot', 'hip', 'leg_len_cm')):
                    j, js = P[side][kind], Ps[side][kind]; t, ts = P[side][top], Ps[side][top]
                    s = P[side][L] / Ps[side][L]
                    self.tgt_pos[j] = restp[t][None] + (Pg_s[:, js] - restp_s[ts][None]) * s
                    self.tgt_rot[j] = Rg_s[:, js]
            self.src_q, self.src_spec, self.src_dof, self.src_rest = q_gs, spec_s, dof_s, rest_s
        # 模型
        lut = build_name_embeddings(jn, device=dev, encoder=cfg.get('sem_encoder', 'siglip'))
        self.data = {robot: {'spec': spec, 'st': static_cond(spec, dev), 'reach': reach, 'sem': embed_matrix(jn, lut).to(dev), 'dof': dof.float().to(dev)}}
        from dof_head import DofKin
        self.data[robot]['kin'] = DofKin(spec, dof.float().to(dev), rest.float().to(dev), device=dev)
        self.model = build_model(cfg, dev); self.model.load_state_dict(torch.load(f'{CK}/model.pt', map_location=dev), strict=False); self.model.eval()
        self.WP = 1e4 / reach ** 2; self.WR = 1e2; self.WD = 1e2
        self.q_soft = {}; self._ik_full = None

    # ---- 编码器输入 / 软补全
    def enc_input(self, vis_feet):
        B, J, root = self.B, self.J, self.root
        if not self.ZS: x_R, x_p = self.Rg.clone(), self.Pg.clone()
        else:
            x_R = torch.eye(3, dtype=torch.float64).expand(B, J, 3, 3).clone(); x_p = self.restp[None].expand(B, J, 3).clone()
            x_R[:, root] = self.root_R
            for j in self.HANDS + self.FEET: x_R[:, j] = self.tgt_rot[j]; x_p[:, j] = self.tgt_pos[j]
        vis = [root] + self.HANDS + (self.FEET if vis_feet else [])
        mask = torch.ones(B, J, dtype=torch.bool); mask[:, vis] = False
        return torch.cat([matrix_to_rot6d(x_R.float()), x_p.float() / self.reach], -1), mask

    def codec(self, x, mask=None):
        outs = []; dev = self.dev
        for i in range(0, self.B, 1024):
            s = slice(i, min(i + 1024, self.B)); n = s.stop - s.start
            with torch.no_grad():
                zp, zr = self.model.encode(x[s].to(dev), geom(self.data, self.name, n), joint_mask=None if mask is None else mask[s].to(dev))
                prot, _ = self.model.decode(zp, zr, geom(self.data, self.name, n)); outs.append(prot.cpu())
        Rp = rot6d_to_matrix(torch.cat(outs).double())
        q, _ = fit_angles(Rp, self.spec, self.dof, rest=self.rest, clamp=True, method='global', root_R=self.root_R)
        return q

    def soft(self, vis_feet):
        if vis_feet not in self.q_soft: self.q_soft[vis_feet] = self.codec(*self.enc_input(vis_feet))
        return self.q_soft[vis_feet]

    # ---- 钉点 / 精修
    def pins_for(self, hand_mode, foot_mode):
        P = self.P; pins = []
        for j in self.HANDS:
            if hand_mode: pins.append(dict(j=j, pos=self.tgt_pos[j], rot=self.tgt_rot[j], mode='full', w_pos=self.WP, w_rot=self.WR, w_dir=self.WD))
        if foot_mode:
            for side, j in (('left', self.FEET[0]), ('right', self.FEET[1])):
                pins.append(dict(j=j, pos=self.tgt_pos[j], rot=self.tgt_rot[j], mode=foot_mode, w_pos=self.WP, w_rot=self.WR, w_dir=self.WD,
                                 sole_normal_local=P[side]['sole_normal_local'], forward_local=P[side]['foot_forward_local']))
        return pins

    def do_refine(self, q0, pins, q_ref=None):
        Q, GR, GP = [], [], []
        for i in range(0, self.B, self.bs):
            s = slice(i, min(i + self.bs, self.B))
            pb = [{k: (v[s] if torch.is_tensor(v) and v.dim() > 1 else v) for k, v in c.items()} for c in pins]
            q, gR, gp = refine(q0[s], self.spec, self.dof, self.rest, self.root_R[s], pb, w_ref=self.w_ref, lm_iters=self.lm_iters,
                               q_ref=None if q_ref is None else q_ref[s])
            Q.append(q); GR.append(gR); GP.append(gp)
        return torch.cat(Q), torch.cat(GR), torch.cat(GP)

    def pin_cost(self, gR, gp, pins):
        """精修器目标里的钉点项 (不含 ref 项), 逐帧. 用来在两个起点之间选."""
        c = torch.zeros(self.B, dtype=torch.float64)
        for p in pins:
            j = p['j']; m = p.get('mode', 'full'); d = gp[:, j] - p['pos']
            c = c + p['w_pos'] * ((d[:, 1] ** 2) if m == 'plane' else (d ** 2).sum(-1))
            if m == 'full': c = c + p['w_rot'] * ((gR[:, j] - p['rot']) ** 2).flatten(1).sum(-1)
            else:
                a = torch.as_tensor(p['sole_normal_local' if m == 'plane' else 'forward_local'], dtype=torch.float64)
                c = c + p['w_dir'] * (((gR[:, j] @ a) - (p['rot'] @ a)) ** 2).sum(-1)
        return c

    def run_cond(self, name):
        start, hand_mode, foot_mode = COND[name][:3]; ref = COND[name][3] if len(COND[name]) > 3 else None; B, J = self.B, self.J
        if start == 'best':
            qa, Ra, Pa = self.run_cond('feet_' + foot_mode); qb, Rb, Pb = self.run_cond('zero_ik_' + foot_mode)
            pins = self.pins_for(True, foot_mode); ca, cb = self.pin_cost(Ra, Pa, pins), self.pin_cost(Rb, Pb, pins)
            use_a = ca <= cb; self.best_frac_model = float(use_a.double().mean())
            print(f'[best_{foot_mode}] model start picked on {100 * self.best_frac_model:.1f}% frames (median cost model {ca.median():.3g} zero {cb.median():.3g})', flush=True)
            w = use_a[:, None, None]
            return torch.where(w, qa, qb), torch.where(w[..., None], Ra, Rb), torch.where(w, Pa, Pb)
        if start == 'zero': q0 = torch.zeros(B, J, 3, dtype=torch.float64)
        elif start == 'soft': q0 = self.soft(False)
        elif start == 'encfeet': q0 = self.soft(True)
        else:
            if self._ik_full is None: self._ik_full = self.do_refine(torch.zeros(B, J, 3, dtype=torch.float64), self.pins_for(True, 'full'))
            gR, gp = self._ik_full[1], self._ik_full[2]
            q0 = self.codec(torch.cat([matrix_to_rot6d(gR.float()), gp.float() / self.reach], -1))
        if not hand_mode and not foot_mode:
            gR, gp = fk_from_angles(q0, self.spec, self.dof, rest=self.rest, root_R=self.root_R); return q0, gR, gp
        return self.do_refine(q0, self.pins_for(hand_mode, foot_mode), q_ref=torch.zeros_like(q0) if ref == 'zero' else None)

    def gt(self):
        return self.q_gt, self.Rg, self.Pg
