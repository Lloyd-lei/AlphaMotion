"""三具身地面动作对照 (0831) —— 同一段人体动作, 同一个生产 Greenwich,
解码到 fourier_gr3 / pnd_adam_lite / agibot_x2, 四联面板并排给 lloyd 看。

目的: "脚不贴地、姿势不对" 到底是码本/表征的通病 (→ 接触标签有理),
还是某几台机器人的关节墙 (→ 具身墙)。

路径 (照抄 ctrl_verify/adam_tracks_lafan.py + ctrl_verify/x2_family.py):
  human_smpl (LAFAN cache, 人体侧) -> build_global -> pose9 (gp/reach, reach 用
  训练全量口径 121.92cm) -> Greenwich.encode (PT_c1dec_reenc_0812) ->
  decode_full(目标 spec/dof) -> 只取旋转头 (w_pos=0) ->
  constraints.project(method="global", lm_iters=25) -> hinge 角 -> MuJoCo qpos ->
  运动学回放 (棋盘地面 + 平行光)。

没有 GT (对解码本身): 源是人体, 解码是零样本; X2 连缓存/原生 retarget 都没有。
所以判断 = 肉眼 + 下面这些无需真值的物理量。

root 约定 (x2_family 的约定, 只把 reach 比换成腿长比):
  位置 = 人体根轨迹 × 站立高度比 (机器人 qpos0 的骨盆->脚底距离 / 人体站立骨盆高)。
        x2_family 用整身 reach 比, 实测对 GR3 会把骨盆压低 3cm (reach 比 0.714),
        对 Adam 压低 12cm; 骨盆高度是"腿 + 脚厚"撑起来的, 所以按站立高度比。
        人体站立高 = rest 腿长 + 走路对照窗实测踝高 p10 (脚踝到脚底)。
  朝向 = 解码骨盆。实测人体 rest 骨盆系与机器人 rest 骨盆系差一个近似常量
        (rpy ~ 90,0,90; 人体根旋转直接套到机器人上是 123deg 的错), 所以不能透传
        人体朝向; 解码骨盆本来就在机器人自己的系里。
不做任何"把最低脚顶到地面"的事后平移 —— 那会把穿地深度定义成 0。

对照臂 (LAFAN 是配对缓存, GR3/Adam 有 xwbc 原生 retarget, X2 没有):
  native      = 原生 retarget 整条 qpos (raw npz) —— 这把尺子的地板在哪。
  decode_nat  = Greenwich 关节角 + 原生根 (位置+朝向) —— 隔离"根约定"的贡献。

usage:
  python tri_embod_ground.py                       # 三个地面窗 + 走路对照窗
  python tri_embod_ground.py --no_render           # 只出数字
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, "/home/arenalabs/Desktop/AlphaMotion/src")
os.environ.setdefault("MUJOCO_GL", "egl")
torch.set_grad_enabled(False)
import mujoco as mj
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as Rot

from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.spatial import SkeletonSpec, build_global
from alphamotion.engine.descriptor import build_from_mjcf
from alphamotion.engine import constraints as MP
from alphamotion.engine.nets.rotations import rot6d_to_matrix

C = "/var/tmp/alphamotion/topo_exp/cache_xwbc_lafan/"
RAW = "/var/tmp/alphamotion/xwbc_lafan/extracted/"
CKPT = "/var/tmp/alphamotion/topo_exp/runs/PT_c1dec_reenc_0812"
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])        # v_ardy(Y-up) = AX @ v_mj(Z-up)
FPS = 30
T = 90
HUMAN_REACH = 121.91773223876953          # v2_reach_cache: max over ckpt 的两个缓存
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PANEL_W, PANEL_H = 560, 600
GRID_IDX = np.linspace(0, T - 1, 8).round().astype(int)

WINDOWS = {
    "crawl_ground1": ("ground1_subject1", 154517),        # 趴地爬行, 骨盆 13-15cm
    "fallgetup": ("fallAndGetUp1_subject1", 90800),       # 摔倒->爬起, 骨盆 9-96cm
    "ground_aiming": ("aiming1_subject4", 11023),         # 判官固定窗, 骨盆 40-46cm
    "walk_ctrl": ("walk1_subject1", 413653),              # 对照臂: 正常走路
}

EMB = {
    "fourier_gr3": dict(
        xml="/var/tmp/alphamotion/config_harvest/GMR/assets/fourier_gr3v2_1_1/mjcf/gr3v2_1_1.xml",
        foot={"left": ["left_foot_pitch_link", "left_foot_roll_link"],
              "right": ["right_foot_pitch_link", "right_foot_roll_link"]},
        leg={"left": ("left_thigh_pitch_link", "left_shank_pitch_link", "left_foot_pitch_link"),
             "right": ("right_thigh_pitch_link", "right_shank_pitch_link", "right_foot_pitch_link")},
        torso="torso_link", native="fourier_gr3", label="GR3 (Fourier GR3v2)"),
    "pnd_adam_lite": dict(
        xml="/var/tmp/alphamotion/config_harvest/GMR/assets/pnd_adam_lite/adam_lite.xml",
        foot={"left": ["anklePitchLeft", "toeLeft"], "right": ["anklePitchRight", "toeRight"]},
        leg={"left": ("hipPitchLeft", "shinLeft", "anklePitchLeft"),
             "right": ("hipPitchRight", "shinRight", "anklePitchRight")},
        torso="torso", native="pnd_adam_lite", label="ADAM (PND Adam Lite)"),
    "agibot_x2": dict(
        xml="/var/tmp/alphamotion/robot_models/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml",
        foot={"left": ["left_ankle_pitch_link", "left_ankle_roll_link"],
              "right": ["right_ankle_pitch_link", "right_ankle_roll_link"]},
        leg={"left": ("left_hip_pitch_link", "left_knee_link", "left_ankle_pitch_link"),
             "right": ("right_hip_pitch_link", "right_knee_link", "right_ankle_pitch_link")},
        torso="torso_link", native=None, label="X2 (Agibot X2-Ultra)"),
}
HUM_LEG = {"left": ("L_Hip", "L_Knee", "L_Ankle"), "right": ("R_Hip", "R_Knee", "R_Ankle")}


# ------------------------------------------------------------------ 场景
def add_checker_floor(sp):
    """高对比棋盘地面 (照抄 hpos_hybrid_viz/pilot_decode_viz2; lloyd 要求看得见地面)。"""
    tx = sp.add_texture()
    tx.name = "_checker"; tx.type = mj.mjtTexture.mjTEXTURE_2D
    tx.builtin = mj.mjtBuiltin.mjBUILTIN_CHECKER
    tx.rgb1 = [0.2, 0.3, 0.45]; tx.rgb2 = [0.85, 0.85, 0.9]
    tx.mark = mj.mjtMark.mjMARK_EDGE; tx.markrgb = [0.1, 0.1, 0.1]
    tx.width = 512; tx.height = 512
    mt = sp.add_material()
    mt.name = "_checker_mat"; mt.texrepeat = [1, 1]; mt.texuniform = True; mt.reflectance = 0
    mt.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "_checker"
    planes = [g for g in sp.geoms if g.type == mj.mjtGeom.mjGEOM_PLANE]
    if planes:
        for g in planes:
            g.material = "_checker_mat"; g.rgba = [1, 1, 1, 1]
            g.contype = 1; g.conaffinity = 1
        if not planes[0].name:
            planes[0].name = "_floor"
        name = planes[0].name
    else:
        g = sp.worldbody.add_geom(name="_floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[30, 30, 0.1])
        g.material = "_checker_mat"; g.contype = 1; g.conaffinity = 1
        name = "_floor"
    sky = sp.add_texture()
    sky.name = "_sky"; sky.type = mj.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mj.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.55, 0.7, 0.9]; sky.rgb2 = [0.9, 0.93, 0.97]
    sky.width = 512; sky.height = 512
    sun = sp.worldbody.add_light(name="_sun", pos=[0, -1.5, 3.5], dir=[0, 0.3, -1],
                                 diffuse=[0.7, 0.7, 0.7], castshadow=True)
    sun.type = mj.mjtLightType.mjLIGHT_DIRECTIONAL
    sp.visual.headlight.ambient = [0.35, 0.35, 0.35]
    sp.visual.headlight.diffuse = [0.5, 0.5, 0.5]
    return name


def label(img, text, sub="", sub2=""):
    im = Image.new("RGB", (img.shape[1], img.shape[0]), (10, 10, 14))
    im.paste(Image.fromarray(img), (0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 21); f2 = ImageFont.truetype(FONT, 14)
    d.rectangle([0, 0, im.width, 26 + (18 if sub else 0) + (18 if sub2 else 0)], fill=(15, 15, 20))
    d.text((8, 3), text, font=f, fill=(255, 255, 255))
    if sub:
        d.text((8, 27), sub, font=f2, fill=(190, 200, 210))
    if sub2:
        d.text((8, 45), sub2, font=f2, fill=(255, 210, 120))
    return np.asarray(im)


# ------------------------------------------------------------------ 几何/度量
def rest_pos(spec):
    p = np.zeros((spec.J, 3))
    for j in range(spec.J):
        if spec.parents[j] >= 0:
            p[j] = p[spec.parents[j]] + spec.rest_offsets[j]
    return p


def rest_reach(spec):
    return float(np.linalg.norm(rest_pos(spec), axis=1).max())


def rest_leg_len(spec, chain):
    """rest 姿态下 hip->knee->ankle 的折线长度 (spec 单位 = cm)。"""
    p = rest_pos(spec)
    idx = [list(spec.joint_names).index(n) for n in chain]
    return float(np.linalg.norm(p[idx[1]] - p[idx[0]]) + np.linalg.norm(p[idx[2]] - p[idx[1]]))


def flexion(a, b, c):
    """[T,3]x3 -> b 处的屈曲角 (deg), 伸直 = 0。"""
    u = a - b; v = c - b
    cs = (u * v).sum(-1) / np.maximum(np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1), 1e-9)
    return 180.0 - np.degrees(np.arccos(np.clip(cs, -1, 1)))


def q_stats(x):
    return dict(mean=round(float(np.mean(x)), 1), p95=round(float(np.percentile(x, 95)), 1),
                max=round(float(np.max(x)), 1))


# ------------------------------------------------------------------ 单具身
class Robot:
    def __init__(self, name, cfg):
        self.name, self.cfg = name, cfg
        self.spec, self.dof, self.rest, self.qnames, _ = build_from_mjcf(cfg["xml"], name)
        sp = mj.MjSpec.from_file(cfg["xml"])
        if not any(j.type == mj.mjtJoint.mjJNT_FREE for j in sp.joints):
            sp.bodies[1].add_freejoint()                 # GMR 的 GR3 是固定底座
        self.floor_name = add_checker_floor(sp)
        sp.visual.global_.offwidth = PANEL_W; sp.visual.global_.offheight = PANEL_H
        m = sp.compile()
        self.m, self.d = m, mj.MjData(m)
        assert m.jnt_type[0] == mj.mjtJoint.mjJNT_FREE, name
        self.hinge = list(range(1, m.njnt))
        self.hnames = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) for j in self.hinge]
        self.hadr = np.array([m.jnt_qposadr[j] for j in self.hinge])
        self.jrange = m.jnt_range[self.hinge]
        self.jlim = m.jnt_limited[self.hinge].astype(bool)
        self.bnames = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)]
        self.bid = {n: i for i, n in enumerate(self.bnames)}
        self.floor_gid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, self.floor_name)
        assert self.floor_gid >= 0, name
        self.foot_geoms, self.foot_bodies = {}, {}
        for s, bl in cfg["foot"].items():
            ids = [self.bid[b] for b in bl if b in self.bid]
            assert ids, (name, s, bl)
            self.foot_bodies[s] = ids
            self.foot_geoms[s] = [g for g in range(m.ngeom) if m.geom_bodyid[g] in ids]
            assert any(m.geom_contype[g] > 0 for g in self.foot_geoms[s]), (name, s)
        self.foot_body_mask = np.zeros(m.nbody, bool)
        for ids in self.foot_bodies.values():
            self.foot_body_mask[ids] = True
        self.reach = rest_reach(self.spec)
        self.leg = rest_leg_len(self.spec, cfg["leg"]["left"])
        self._mesh = {}
        # 站立高度: qpos0 (rest 关节) 下 root body 原点到最低脚底顶点的垂直距离
        self.d.qpos[:] = m.qpos0
        mj.mj_forward(m, self.d)
        self.stand = float(self.d.xpos[1][2] -
                           min(self.geom_low(g) for gs in self.foot_geoms.values() for g in gs)) * 100
        self.dof_t = torch.as_tensor(self.dof, dtype=torch.float64)
        self.rest_t = torch.as_tensor(self.rest, dtype=torch.float64)
        self.root_j = int(np.where(np.asarray(self.spec.parents) < 0)[0][0])

    def geom_low(self, g):
        m, d = self.m, self.d
        R = d.geom_xmat[g].reshape(3, 3); p = d.geom_xpos[g]
        if m.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            if mid not in self._mesh:
                a = m.mesh_vertadr[mid]; n = m.mesh_vertnum[mid]
                self._mesh[mid] = m.mesh_vert[a:a + n].copy()
            return float((self._mesh[mid] @ R.T + p)[:, 2].min())
        return float(p[2] - np.abs(R[2]).dot(m.geom_size[g]))

    def solve(self, codes, G):
        rot, _ = G.decode_full(codes, self.spec, self.dof)          # 只用旋转头 (w_pos=0)
        Rg = rot6d_to_matrix(rot).double().cpu()
        _, _, q = MP.project(Rg, self.spec, self.dof_t, rest=self.rest_t,
                             method="global", lm_iters=25)
        return Rg, q.detach().numpy()

    def qpos_from_q(self, q, root_p, root_q_wxyz):
        n = root_p.shape[0]
        jp = {}
        for j, names3 in enumerate(self.qnames):
            for k, nm in enumerate(names3):
                if nm:
                    jp[nm] = q[:, j, k]
        missing = [h for h in self.hnames if h not in jp]
        qp = np.tile(self.m.qpos0, (n, 1)).astype(np.float64)
        qp[:, 0:3] = root_p; qp[:, 3:7] = root_q_wxyz
        for a, nm in zip(self.hadr, self.hnames):
            if nm in jp:
                qp[:, a] = jp[nm]
        return qp, missing

    def qpos_native(self, clip, i0, n):
        """xwbc 原生 retarget 的整条 qpos (raw npz, Isaac schema)。"""
        p = f"{RAW}{self.cfg['native']}/{clip}.npz"
        if not self.cfg["native"] or not os.path.exists(p):
            return None
        d = np.load(p, allow_pickle=True)
        jn = [str(x) for x in d["joint_names"]]
        sl = slice(i0, i0 + n)
        qp = np.tile(self.m.qpos0, (n, 1)).astype(np.float64)
        qp[:, 0:3] = d["body_pos_w"][sl, 0].astype(np.float64)
        qp[:, 3:7] = d["body_quat_w"][sl, 0].astype(np.float64)
        for a, nm in zip(self.hadr, self.hnames):
            if nm in jn:
                qp[:, a] = d["joint_pos"][sl, jn.index(nm)].astype(np.float64)
        return qp

    def probe(self, qp):
        """逐帧 mj_forward: 真碰撞穿地 + 脚几何最低点 + 脚水平位置 + 关键 body 位置。"""
        m, d = self.m, self.d
        n = len(qp)
        pen_foot = np.zeros(n); pen_all = np.zeros(n)
        low = {s: np.zeros(n) for s in self.foot_geoms}
        fxy = {s: np.zeros((n, 2)) for s in self.foot_geoms}
        keys = sorted({b for tri in self.cfg["leg"].values() for b in tri} | {self.cfg["torso"]})
        bpos = {b: np.zeros((n, 3)) for b in keys}
        root_z = np.zeros(n)
        for t in range(n):
            d.qpos[:] = qp[t]; d.qvel[:] = 0
            mj.mj_forward(m, d)
            for i in range(d.ncon):
                c = d.contact[i]
                g1, g2 = int(c.geom1), int(c.geom2)
                if self.floor_gid not in (g1, g2):
                    continue
                other = g2 if g1 == self.floor_gid else g1
                dep = max(-float(c.dist), 0.0) * 100.0
                pen_all[t] = max(pen_all[t], dep)
                if self.foot_body_mask[m.geom_bodyid[other]]:
                    pen_foot[t] = max(pen_foot[t], dep)
            for s, gs in self.foot_geoms.items():
                low[s][t] = min(self.geom_low(g) for g in gs)
                fxy[s][t] = np.mean([d.geom_xpos[g, :2] for g in gs], 0)
            for b in keys:
                bpos[b][t] = d.xpos[self.bid[b]]
            root_z[t] = d.xpos[1][2]
        return dict(pen_foot=pen_foot, pen_all=pen_all, low=low, fxy=fxy,
                    bpos=bpos, root_z=root_z)


def phys_metrics(rb, qp, pr, tol_deg=0.5):
    m = {}
    m["pen_foot_max_cm"] = round(float(pr["pen_foot"].max()), 2)
    m["pen_foot_p95_cm"] = round(float(np.percentile(pr["pen_foot"], 95)), 2)
    m["pen_body_max_cm"] = round(float(pr["pen_all"].max()), 2)
    m["pen_body_p95_cm"] = round(float(np.percentile(pr["pen_all"], 95)), 2)
    zz = np.stack([pr["low"][s] for s in ["left", "right"]], 1) * 100.0    # cm, 地面 z=0
    m["foot_low_min_cm"] = round(float(zz.min()), 2)
    m["foot_low_mean_cm"] = round(float(zz.mean()), 2)
    m["foot_contact_rate"] = round(float((zz < 2.0).any(1).mean()), 3)
    sk = []
    for i, s in enumerate(["left", "right"]):
        z = zz[:, i]; v = np.linalg.norm(np.diff(pr["fxy"][s], axis=0), axis=1) * FPS * 100
        st = (z[:-1] < 2.0) & (z[1:] < 2.0)
        if st.any():
            sk.append(v[st])
    sk = np.concatenate(sk) if sk else np.zeros(1)
    m["stance_skate_mean_cm_s"] = round(float(sk.mean()), 1)
    m["stance_skate_p95_cm_s"] = round(float(np.percentile(sk, 95)), 1)
    m["stance_frames"] = int(sk.size)
    jq = qp[:, rb.hadr]
    tol = np.radians(tol_deg)
    lo, hi = rb.jrange[:, 0], rb.jrange[:, 1]
    sat = ((jq <= lo + tol) | (jq >= hi - tol)) & rb.jlim[None, :]
    m["sat_frac_all_joints"] = round(float(sat.mean()), 4)
    m["sat_frame_frac_any"] = round(float(sat.any(1).mean()), 3)
    per = sat.mean(0)
    m["sat_top_joints"] = {rb.hnames[i]: round(float(per[i]), 3)
                           for i in np.argsort(-per)[:6] if per[i] > 0.02}
    m["sat_n_joints_gt50pct"] = int((per > 0.5).sum())
    fl_k, fl_h = [], []
    for s, (h, k, a) in rb.cfg["leg"].items():
        P = pr["bpos"]
        fl_k.append(flexion(P[h], P[k], P[a]))
        fl_h.append(flexion(P[rb.cfg["torso"]], P[h], P[k]))
    m["knee_flex_deg"] = q_stats(np.concatenate(fl_k))
    m["hip_flex_deg"] = q_stats(np.concatenate(fl_h))
    m["root_height_mean_cm"] = round(float(pr["root_z"].mean() * 100), 1)
    return m


def human_sole(hs, L, HRW, clip_start=413653, n=300):
    """人体踝关节到脚底的距离: 走路对照窗里踝高的 p10 (脚平放在地上的那些帧)。"""
    rows = torch.arange(clip_start, clip_start + n)
    _, gp, _ = build_global(L[rows].float(), hs, "cpu")
    w = gp.numpy() + HRW[rows].numpy()[:, None]
    nm = list(hs.joint_names)
    z = np.concatenate([w[:, nm.index("L_Ankle"), 1], w[:, nm.index("R_Ankle"), 1]])
    return float(np.percentile(z, 10))


# ------------------------------------------------------------------ 人体面板
def human_world(gp_cm, root_cm):
    """[T,J,3] 根相对 cm Y-up + [T,3] 根世界 cm Y-up -> [T,J,3] m Z-up。"""
    return (gp_cm + root_cm[:, None]) @ AX / 100.0


def human_panel(hw, parents, head, sub):
    """棍状人体 (照抄 spatial5.capsule_tile 的 mjv_connector 做法), 真实地面关系。"""
    sp = mj.MjSpec()
    sp.visual.global_.offwidth = PANEL_W; sp.visual.global_.offheight = PANEL_H
    add_checker_floor(sp)
    m = sp.compile(); d = mj.MjData(m); mj.mj_forward(m, d)
    ren = mj.Renderer(m, PANEL_H, PANEL_W)
    cam = mj.MjvCamera(); cam.distance, cam.elevation, cam.azimuth = 2.6, -8, 135
    par = np.asarray(parents)
    out = []
    for t in range(hw.shape[0]):
        cam.lookat[:] = [hw[t, 0, 0], hw[t, 0, 1], 0.5]
        ren.update_scene(d, cam)
        sc = ren.scene
        for j in range(len(par)):
            if par[j] < 0 or sc.ngeom >= sc.maxgeom:
                continue
            gg = sc.geoms[sc.ngeom]; sc.ngeom += 1
            mj.mjv_initGeom(gg, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.array([0.95, 0.45, 0.15, 1], np.float32))
            mj.mjv_connector(gg, mj.mjtGeom.mjGEOM_CAPSULE, 0.028, hw[t, par[j]], hw[t, j])
        out.append(label(ren.render().copy(), head, sub,
                         f"ankle z {min(hw[t, 7, 2], hw[t, 8, 2]) * 100:5.1f} cm"))
    del ren
    return out


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default=",".join(WINDOWS))
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default="/home/arenalabs/tmp")
    ap.add_argument("--no_render", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(C + "human_smpl_spec.npz", allow_pickle=True)
    hs = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                      [str(n) for n in d["joint_names"]])
    hd = np.load(C + "human_smpl_dof.npz", allow_pickle=True)["dof"]
    L = torch.load(C + "human_smpl_paired_local6d.pt", mmap=True)
    HRW = torch.load(C + "human_smpl_root_world.pt", mmap=True)
    hname = list(hs.joint_names)
    SEQS = {s["name"]: s for s in json.load(open(C + "sequences.json"))}

    G = Greenwich.load(args.ckpt, device=args.device)
    print("[greenwich]", args.ckpt, "device", G.device)
    robots = {n: Robot(n, c) for n, c in EMB.items()}
    hreach = rest_reach(hs)
    hleg = rest_leg_len(hs, HUM_LEG["left"])
    hsole = human_sole(hs, L, HRW)                 # 踝到脚底 (走路对照窗 p10 踝高)
    hstand = hleg + hsole
    print(f"[human] rest_reach {hreach:.1f}cm leg {hleg:.1f}cm ankle->sole {hsole:.1f}cm "
          f"standing {hstand:.1f}cm")
    for n, rb in robots.items():
        print(f"[{n}] J {rb.spec.J} hinges {len(rb.hinge)} reach {rb.reach:.1f}cm "
              f"(x{rb.reach / hreach:.3f}) leg {rb.leg:.1f}cm (x{rb.leg / hleg:.3f}) "
              f"standing {rb.stand:.1f}cm (x{rb.stand / hstand:.3f}) "
              f"native={rb.cfg['native'] or 'NONE'}")

    report = {"ckpt": os.path.basename(args.ckpt), "T": T, "fps": FPS,
              "note": "human source, zero-shot decode, rotation-only (w_pos=0); "
                      "no GT for the decode. native arm = xwbc retarget (gr3/adam only).",
              "human": {"rest_reach_cm": round(hreach, 1), "leg_cm": round(hleg, 1),
                        "ankle_to_sole_cm": round(hsole, 1), "standing_h_cm": round(hstand, 1)},
              "embodiments": {n: {"rest_reach_cm": round(rb.reach, 1),
                                  "reach_ratio": round(rb.reach / hreach, 3),
                                  "leg_cm": round(rb.leg, 1),
                                  "leg_ratio": round(rb.leg / hleg, 3),
                                  "standing_h_cm": round(rb.stand, 1),
                                  "stand_ratio": round(rb.stand / hstand, 3),
                                  "hinges": len(rb.hinge),
                                  "native": rb.cfg["native"]} for n, rb in robots.items()},
              "windows": {}}

    for tag in [w for w in args.windows.split(",") if w]:
        clip, st = WINDOWS[tag]
        i0 = st - SEQS[clip]["start"]
        rows = torch.arange(st, st + T)
        l6 = L[rows].float()
        g6, gp, win_reach = build_global(l6, hs, "cpu")
        pose9 = torch.cat([g6, gp / HUMAN_REACH], -1)
        codes = G.encode(pose9.to(G.device), hs, hd)
        root_yup = HRW[rows].numpy().astype(np.float64)                # cm Y-up
        hw = human_world(gp.numpy().astype(np.float64), root_yup)      # m Z-up

        hi = {n: hname.index(n) for n in ["Pelvis", "Chest", "L_Hip", "L_Knee", "L_Ankle",
                                          "R_Hip", "R_Knee", "R_Ankle"]}
        hk = np.concatenate([flexion(hw[:, hi[f"{s}_Hip"]], hw[:, hi[f"{s}_Knee"]],
                                     hw[:, hi[f"{s}_Ankle"]]) for s in "LR"])
        hh = np.concatenate([flexion(hw[:, hi["Chest"]], hw[:, hi[f"{s}_Hip"]],
                                     hw[:, hi[f"{s}_Knee"]]) for s in "LR"])
        ank_z = np.stack([hw[:, hi["L_Ankle"], 2], hw[:, hi["R_Ankle"], 2]], 1) * 100
        wrep = {"clip": clip, "start": st,
                "human": {"knee_flex_deg": q_stats(hk), "hip_flex_deg": q_stats(hh),
                          "pelvis_h_cm": [round(float(root_yup[:, 1].min()), 1),
                                          round(float(root_yup[:, 1].max()), 1)],
                          "ankle_z_min_cm": round(float(ank_z.min()), 1),
                          "ankle_z_mean_cm": round(float(ank_z.mean()), 1),
                          "ankle_contact_rate": round(float((ank_z < 12.0).any(1).mean()), 3),
                          "window_reach_cm": round(float(win_reach), 1)},
                "robots": {}}
        print(f"\n=== {tag} {clip}@{st}  human knee {wrep['human']['knee_flex_deg']} "
              f"hip {wrep['human']['hip_flex_deg']} pelvis {wrep['human']['pelvis_h_cm']}cm")

        panels = {}
        for n, rb in robots.items():
            Rg, q = rb.solve(codes, G)
            root_p = (root_yup @ AX) / 100.0 * (rb.stand / hstand)
            quat_dec = Rot.from_matrix(AX.T @ Rg[:, rb.root_j].numpy() @ AX).as_quat()[:, [3, 0, 1, 2]]
            arms = {}
            qp, missing = rb.qpos_from_q(q, root_p, quat_dec)
            pr = rb.probe(qp)
            arms["decode"] = phys_metrics(rb, qp, pr)
            arms["decode"]["missing_mjcf_joints"] = missing
            panels[n] = (qp, pr)
            qn = rb.qpos_native(clip, i0, T)
            if qn is not None:
                arms["native"] = phys_metrics(rb, qn, rb.probe(qn))
                qh, _ = rb.qpos_from_q(q, qn[:, 0:3], qn[:, 3:7])
                arms["decode_nativeroot"] = phys_metrics(rb, qh, rb.probe(qh))
                geo = np.degrees((Rot.from_quat(quat_dec[:, [1, 2, 3, 0]]) *
                                  Rot.from_quat(qn[:, 3:7][:, [1, 2, 3, 0]]).inv()).magnitude())
                arms["root_ori_decoded_vs_native_deg"] = round(float(geo.mean()), 1)
                arms["root_h_decode_vs_native_cm"] = [arms["decode"]["root_height_mean_cm"],
                                                      arms["native"]["root_height_mean_cm"]]
            wrep["robots"][n] = arms
            for k in ["decode", "decode_nativeroot", "native"]:
                if k not in arms:
                    continue
                a = arms[k]
                print(f"  [{n:14s} {k:18s}] pen_foot {a['pen_foot_max_cm']:6.1f}/"
                      f"{a['pen_foot_p95_cm']:6.1f} body {a['pen_body_max_cm']:6.1f} | "
                      f"low {a['foot_low_min_cm']:6.1f} contact {a['foot_contact_rate']:.2f} | "
                      f"skate {a['stance_skate_mean_cm_s']:6.1f} | sat {a['sat_frac_all_joints']:.3f}"
                      f"/{a['sat_frame_frac_any']:.2f} | knee {a['knee_flex_deg']['mean']:5.1f} "
                      f"hip {a['hip_flex_deg']['mean']:5.1f} | rootz {a['root_height_mean_cm']:5.1f}")
                if a["sat_top_joints"]:
                    print(f"        sat: {a['sat_top_joints']}")
            if "root_ori_decoded_vs_native_deg" in arms:
                print(f"        root ori decoded vs native: "
                      f"{arms['root_ori_decoded_vs_native_deg']:.1f} deg")
        report["windows"][tag] = wrep
        json.dump(report, open(f"{args.out}/tri_embod_metrics.json", "w"), indent=1,
                  ensure_ascii=False)

        if args.no_render:
            continue
        cols = [human_panel(hw, hs.parents, "HUMAN SOURCE (SMPL)",
                            f"{clip} @{st} | NO GT: zero-shot decode")]
        for n, rb in robots.items():
            qp, pr = panels[n]
            a = wrep["robots"][n]["decode"]
            ren = mj.Renderer(rb.m, PANEL_H, PANEL_W)
            cam = mj.MjvCamera()
            sc = rb.stand / hstand
            cam.distance, cam.elevation, cam.azimuth = 2.6 * sc, -8, 135
            F = []
            for t in range(len(qp)):
                rb.d.qpos[:] = qp[t]; mj.mj_forward(rb.m, rb.d)
                cam.lookat[:] = [qp[t, 0], qp[t, 1], 0.5 * sc]
                ren.update_scene(rb.d, cam)
                F.append(label(ren.render().copy(), rb.cfg["label"],
                               f"rot-only decode | sat {a['sat_frac_all_joints']*100:.1f}% "
                               f"| skate {a['stance_skate_mean_cm_s']:.0f} cm/s",
                               f"foot below floor {pr['pen_foot'][t]:5.1f} cm | "
                               f"foot z {min(pr['low']['left'][t], pr['low']['right'][t])*100:5.1f} cm"))
            del ren
            cols.append(F)
        comp = [np.concatenate([c[t] for c in cols], axis=1) for t in range(T)]
        p = f"{args.out}/tri_embod_{tag}.mp4"
        w = iio.get_writer(p, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p")
        for f in comp:
            w.append_data(f)
        for f in comp:                                   # 0.5x 慢放重播
            w.append_data(f); w.append_data(f)
        w.close()
        print("WROTE", p)
        grid = np.concatenate([np.concatenate([comp[i] for i in GRID_IDX[r * 2:r * 2 + 2]], axis=1)
                               for r in range(4)], axis=0)
        gpath = f"{args.out}/tri_embod_{tag}_grid.png"
        iio.imwrite(gpath, grid[::2, ::2])
        print("WROTE", gpath)

    json.dump(report, open(f"{args.out}/tri_embod_metrics.json", "w"), indent=1, ensure_ascii=False)
    print("WROTE", f"{args.out}/tri_embod_metrics.json")


if __name__ == "__main__":
    main()
