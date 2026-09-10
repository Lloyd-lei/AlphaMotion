"""视频 rollout: 一段语料外人体动作, 同场并排: 人体源(棍状) | 真值机器人(GMR, 有真值机身才有) | 各条件精修结果.
每台精修机器人身上画红点 = 钉点目标 (手/脚 SE(3) 的位置部分). 世界系: 水平位移透传真值根轨迹, 竖直 = 真值根高 (零样本机身按骨盆高比缩放 G1 的).
用法: render_rollout.py <robot> <clip_regex> <out_dir> [--conds zero_ik_full,feet_axis,... --T 300 --at 0]"""
import os, sys, re, json, math, argparse, subprocess, numpy as np, torch
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suite_core as SC
from suite_core import C, load_spec, PF
from common import fk_global
from manifold_project import fk_from_angles

ap = argparse.ArgumentParser()
ap.add_argument('robot'); ap.add_argument('clip'); ap.add_argument('out')
ap.add_argument('--conds', default='zero_ik_full,hands,feet_axis'); ap.add_argument('--T', type=int, default=300); ap.add_argument('--at', type=int, default=-1)
ap.add_argument('--W', type=int, default=1920); ap.add_argument('--H', type=int, default=1080); ap.add_argument('--gap', type=float, default=1.25)
ap.add_argument('--grid', type=int, default=12); ap.add_argument('--tag', default='')
a = ap.parse_args()
W, H, T = a.W, a.H, a.T
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])   # ardy (Y-up) -> mujoco (Z-up)
os.makedirs(a.out, exist_ok=True)
FONT_B = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'; FONT_R = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
NICE = {'zero_ik_hands': 'IK only · hands', 'zero_ik_full': 'IK only · hands+feet', 'zero_ik_axis': 'IK only · feet axis',
        'soft': 'model soft-fill', 'hands': 'model + hands', 'feet_point': 'model + feet point', 'feet_plane': 'model + feet plane',
        'feet_axis': 'model + feet axis', 'feet_full': 'model + feet full', 'encfeet_full': 'model(sees feet) + full', 'codec_full': 'IK→codec→refine', 'feet_axis_ref0': 'model + feet axis (ref zero)', 'feet_full_ref0': 'model + feet full (ref zero)', 'best_axis': 'best of 2 starts · feet axis', 'best_full': 'best of 2 starts · feet full'}

# ---------------- 片段
seqs = json.load(open(f'{C}/sequences.json'))
m = [q for q in seqs if re.search(a.clip, q['name'], re.I) and q['len'] >= T]
assert m, f'no clip {a.clip}'; q = m[0]
at = a.at if a.at >= 0 else max(0, (q['len'] - T) // 2)
rows = torch.arange(q['start'] + at, q['start'] + at + T)
print(f'[clip] {q["name"]} len {q["len"]} at {at}', flush=True)
R = SC.Robot(a.robot, rows)
conds = a.conds.split(',')
Q = {}; GP = {}
for c in conds:
    qq, gR, gp = R.run_cond(c); Q[c] = qq.float().numpy(); GP[c] = gp.numpy(); print(f'[{c}] done', flush=True)
BODIES = (['gt'] if not R.ZS else []) + conds
if not R.ZS: Q['gt'] = R.q_gt.float().numpy(); GP['gt'] = R.Pg.numpy()
TGT = {j: R.tgt_pos[j].numpy() for j in R.HANDS + R.FEET}

# ---------------- 世界系: 根轨迹 (真值透传)
src = R.src
RW = torch.load(f'{C}/{src}_root_world.pt', mmap=True).float()[rows].numpy() / 100.0      # ardy m
if R.ZS:
    RW = RW.copy(); RW[:, 1] *= R.P['pelvis_height_rest_cm'] / SC.PROF[src]['pelvis_height_rest_cm']
RWh = torch.load(f'{C}/human_smpl_root_world.pt', mmap=True).float()[rows].numpy() / 100.0
hs = load_spec('human_smpl'); hl6 = torch.load(f'{C}/human_smpl_paired_local6d.pt', mmap=True)[rows].float()
hp = fk_global(hl6, hs)[1].numpy() / 100.0                                                 # 人体关节 (全局 FK, 局部6d 须先 build_global), 根相对 ardy m
root_R = R.root_R.numpy()                                                                    # ardy 世界朝向
RRm = np.einsum('ij,bjk,kl->bil', AX.T, root_R, AX)
xy = ((RW - RW[0]) @ AX)[:, :2]; xyh = ((RWh - RWh[0]) @ AX)[:, :2]
R0 = RRm[0]; yaw = math.atan2(R0[1, 0], R0[0, 0])
if np.linalg.norm(xy[-1] - xy[0]) > 1.5: d_ = xy[-1] - xy[0]; yaw = math.atan2(d_[1], d_[0])
c, s = math.cos(-yaw), math.sin(-yaw); Rz = np.array([[c, -s], [s, c]]); Rz3 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
TRJ = xy @ Rz.T; TRJh = xyh @ Rz.T; RRm = np.einsum('ij,bjk->bik', Rz3, RRm)
ROT_A = AX @ Rz3 @ AX.T                                                                      # ardy 系内的转正
hp = np.einsum('ij,bnj->bni', ROT_A, hp)
HZ = RWh[:, 1].copy(); HZ -= (hp[:, :, 1] + HZ[:, None]).min()                             # 人体贴地
ZR = RW[:, 1]                                                                                # 机器人根高 (真值)
ALL = ['human'] + BODIES; n_all = len(ALL)
OFF = {n: np.array([0.0, (i - (n_all - 1) / 2) * a.gap]) for i, n in enumerate(ALL)}

# ---------------- 场景
def resolve_xml(emb):
    if emb in PF.XML: return PF.XML[emb]
    EXTRA = {'unitree_h2': '/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml',
             'unitree_r1': '/var/tmp/alphamotion/robot_models/unitree_ros/robots/r1_description/unitree_r1_mjcf.xml'}   # 关节名与缓存 slot_joints 全对上 (31/31, 26/26)
    if emb in EXTRA: return EXTRA[emb]
    import importlib.util
    alias = {'unitree_g1_29dof': 'unitree_g1', 'unitree_g1_23dof': 'unitree_g1'}   # 23dof 缓存也是 29 个槽 (锁死的 ≈0), 用 29dof 网格
    gp_ = '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/external/GMR/general_motion_retargeting/params.py'
    s2 = importlib.util.spec_from_file_location('gp', gp_); m2 = importlib.util.module_from_spec(s2); s2.loader.exec_module(m2)
    return str(m2.ROBOT_XML_DICT[alias.get(emb, emb)])
root = mj.MjSpec(); root.modelname = 'rollout'
root.visual.global_.offwidth = W; root.visual.global_.offheight = H; root.visual.global_.fovy = 32
root.visual.quality.shadowsize = 8192; root.visual.map.shadowclip = 2.0; root.visual.map.shadowscale = 0.9
root.visual.headlight.ambient = [0.34, 0.34, 0.36]; root.visual.headlight.diffuse = [0.12, 0.12, 0.12]; root.visual.headlight.specular = [0, 0, 0]
sky = root.add_texture(); sky.name = '_sky'; sky.type = mj.mjtTexture.mjTEXTURE_SKYBOX; sky.builtin = mj.mjtBuiltin.mjBUILTIN_GRADIENT
sky.rgb1 = [0.93, 0.94, 0.96]; sky.rgb2 = [0.80, 0.83, 0.88]; sky.width = sky.height = 512
tx = root.add_texture(); tx.name = '_grid'; tx.type = mj.mjtTexture.mjTEXTURE_2D; tx.builtin = mj.mjtBuiltin.mjBUILTIN_CHECKER; tx.width = tx.height = 512
tx.rgb1 = [0.885, 0.89, 0.90]; tx.rgb2 = [0.862, 0.867, 0.882]; tx.mark = mj.mjtMark.mjMARK_EDGE; tx.markrgb = [0.835, 0.84, 0.855]
mt = root.add_material(); mt.name = '_gridm'; mt.texrepeat = [120, 120]; mt.reflectance = 0; mt.specular = 0; mt.shininess = 0
mt.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = '_grid'
root.worldbody.add_geom(name='_floor', type=mj.mjtGeom.mjGEOM_PLANE, size=[60, 60, .1], material='_gridm')
sun = root.worldbody.add_light(name='_sun', pos=[3, -4, 8], dir=[-0.3, 0.45, -1], diffuse=[.5, .5, .5], specular=[0, 0, 0], castshadow=True)
sun.type = mj.mjtLightType.mjLIGHT_DIRECTIONAL
fill = root.worldbody.add_light(name='_fill', pos=[-4, 6, 5], dir=[0.4, -0.6, -0.6], diffuse=[.16, .16, .18], specular=[0, 0, 0], castshadow=False)
fill.type = mj.mjtLightType.mjLIGHT_DIRECTIONAL
SKIN = [0.93, 0.80, 0.62, 1]; H_CAPS = [j for j in range(hs.J) if hs.parents[j] >= 0]
for j in H_CAPS:
    b = root.worldbody.add_body(name=f'_hc{j}', mocap=True)
    b.add_geom(name=f'_hg{j}', type=mj.mjtGeom.mjGEOM_CAPSULE, size=[0.03, 0.1, 0], rgba=SKIN, contype=0, conaffinity=0)
for j in range(hs.J):
    b = root.worldbody.add_body(name=f'_hj{j}', mocap=True)
    b.add_geom(name=f'_hs{j}', type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.036 if hs.parents[j] < 0 else 0.03, 0, 0], rgba=SKIN, contype=0, conaffinity=0)
PREFIX = {}
xml = resolve_xml(a.robot)
for n in BODIES:
    ch = mj.MjSpec.from_file(xml)
    for l in list(ch.lights): ch.delete(l)
    for c_ in list(ch.cameras): ch.delete(c_)
    for g in list(ch.worldbody.geoms): ch.delete(g)
    fr = root.worldbody.add_frame(pos=[0, 0, 0]); PREFIX[n] = f'{n}/'; root.attach(ch, prefix=PREFIX[n], frame=fr)
    for k, j in enumerate(R.HANDS + R.FEET):
        b = root.worldbody.add_body(name=f'_tg{n}_{k}', mocap=True, pos=[0, 0, -5])
        b.add_geom(name=f'_tgg{n}_{k}', type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.028, 0, 0], rgba=[0.9, 0.12, 0.12, 0.9] if n != 'gt' else [0.2, 0.5, 0.95, 0.9], contype=0, conaffinity=0)
sc = root.compile(); d = mj.MjData(sc)
print(f'[scene] nbody={sc.nbody} ngeom={sc.ngeom} nq={sc.nq}', flush=True)
# 关节地址
if R.ZS:
    from alphamotion.embodiment import registry as reg
    sj = reg.load_from_xml(xml, a.robot).qnames
else:
    sj = [str(s_) for s_ in np.load(f'{C}/{a.robot}_dof.npz', allow_pickle=True)['slot_joints']]
ADR, FREE, JB = {}, {}, {}
for n in BODIES:
    adr = {}
    for k, names in enumerate(sj):
        lst = list(names) if isinstance(names, (list, tuple)) else [x for x in str(names).split(',') if x]
        for s_, jn_ in enumerate(lst):
            jid = mj.mj_name2id(sc, mj.mjtObj.mjOBJ_JOINT, PREFIX[n] + jn_)
            if jid >= 0 and sc.jnt_type[jid] == mj.mjtJoint.mjJNT_HINGE: adr[(k, s_)] = int(sc.jnt_qposadr[jid])
    ADR[n] = adr
    fj = [i for i in range(sc.njnt) if sc.jnt_type[i] == mj.mjtJoint.mjJNT_FREE and (mj.mj_id2name(sc, mj.mjtObj.mjOBJ_BODY, sc.jnt_bodyid[i]) or '').startswith(PREFIX[n])]
    assert len(fj) == 1, (n, fj); FREE[n] = int(sc.jnt_qposadr[fj[0]])
print(f'[{a.robot}] hinge={len(ADR[BODIES[0]])} of {sum(len(x) if isinstance(x,(list,tuple)) else 1 for x in sj)} slots', flush=True)
HCM = {j: int(sc.body_mocapid[mj.mj_name2id(sc, mj.mjtObj.mjOBJ_BODY, f'_hc{j}')]) for j in H_CAPS}
HJM = {j: int(sc.body_mocapid[mj.mj_name2id(sc, mj.mjtObj.mjOBJ_BODY, f'_hj{j}')]) for j in range(hs.J)}
HCG = {j: mj.mj_name2id(sc, mj.mjtObj.mjOBJ_GEOM, f'_hg{j}') for j in H_CAPS}
TGM = {n: [int(sc.body_mocapid[mj.mj_name2id(sc, mj.mjtObj.mjOBJ_BODY, f'_tg{n}_{k}')]) for k in range(4)] for n in BODIES}
FOVY = math.radians(32)
def cam_basis(cam):
    az, el = math.radians(cam.azimuth), math.radians(cam.elevation)
    fwd = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    pos = np.asarray(cam.lookat) - cam.distance * fwd
    right = np.cross(fwd, [0, 0, 1]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    return pos, fwd, right, up
def project(cam, p):
    pos, fwd, right, up = cam_basis(cam); v = np.asarray(p) - pos
    z = v @ fwd; f = (H / 2) / math.tan(FOVY / 2)
    return W / 2 + f * (v @ right) / z, H / 2 - f * (v @ up) / z, z
def _smooth(x, k=20):
    x = np.asarray(x, np.float64); out = np.empty_like(x)
    for i in range(len(x)): out[i] = x[max(0, i - k):i + k + 1].mean(0)
    return out
ren = mj.Renderer(sc, H, W, max_geom=20000); cam = mj.MjvCamera()
f_title = ImageFont.truetype(FONT_B, 36); f_sub = ImageFont.truetype(FONT_R, 21); f_lab = ImageFont.truetype(FONT_B, 19); f_small = ImageFont.truetype(FONT_R, 18)

def frame_at(t):
    d.qpos[:] = sc.qpos0
    P_ = hp[t] @ AX; P_[:, 2] += HZ[t]; P_[:, :2] += TRJh[t] + OFF['human']
    for j in HJM: d.mocap_pos[HJM[j]] = P_[j]
    for j in H_CAPS:
        a_, b_ = P_[hs.parents[j]], P_[j]; v = b_ - a_; L = np.linalg.norm(v); d.mocap_pos[HCM[j]] = (a_ + b_) / 2
        if L > 1e-6:
            u_ = v / L; ax = np.cross([0, 0, 1], u_); sn = np.linalg.norm(ax); ang = math.atan2(sn, u_[2])   # 完整 0..pi (asin 会把朝下的骨头镜像到上面)
            rv = ax / sn * ang if sn > 1e-9 else (np.array([math.pi, 0, 0]) if u_[2] < 0 else np.zeros(3))
            d.mocap_quat[HCM[j]] = Rotation.from_rotvec(rv).as_quat(scalar_first=True); sc.geom_size[HCG[j]][1] = L / 2
    for n in BODIES:
        a0 = FREE[n]; d.qpos[a0:a0 + 2] = TRJ[t] + OFF[n]; d.qpos[a0 + 2] = ZR[t]
        d.qpos[a0 + 3:a0 + 7] = Rotation.from_matrix(RRm[n if False else t] if False else RRm[t]).as_quat(scalar_first=True)
        for (k, s_), ad in ADR[n].items(): d.qpos[ad] = Q[n][t, k, s_]
        for k, j in enumerate(R.HANDS + R.FEET):
            p_ = (ROT_A @ TGT[j][t] / 100.0) @ AX; p_[2] += ZR[t]; p_[:2] += TRJ[t] + OFF[n]
            d.mocap_pos[TGM[n][k]] = p_
    mj.mj_forward(sc, d)

cen = _smooth(TRJ)
tag = a.tag or re.sub('[^A-Za-z0-9]+', '_', q['name'])[:36]
out = f'{a.out}/{a.robot}_{tag}_at{at}.mp4'
pr = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '30', '-i', '-',
                       '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
ROW_W = (n_all - 1) * a.gap
dist = max(4.5, ROW_W / (2 * math.tan(FOVY / 2) * (W / H)) * 1.06 + 1.8)
grid_idx = set(np.linspace(0, T - 1, a.grid).astype(int).tolist()); grid_frames = []
title = f'{a.robot}   ·   EE SE(3) refiner   ·   {"zero-shot (no GT)" if R.ZS else "vs GMR ground truth"}'
for i in range(T):
    frame_at(i); u = i / max(1, T - 1)
    cam.azimuth = 200 - 40 * (0.5 - 0.5 * math.cos(math.pi * u)); cam.elevation = -10; cam.distance = dist
    cam.lookat[:] = [cen[i][0], cen[i][1], 0.62]
    ren.update_scene(d, cam); im = Image.fromarray(ren.render().copy()); dr = ImageDraw.Draw(im, 'RGBA')
    for n in ALL:
        xy_ = (TRJh[i] if n == 'human' else TRJ[i]) + OFF[n]; px, py, z = project(cam, [xy_[0], xy_[1], -0.02])
        if z <= 0: continue
        lab = {'human': 'human source', 'gt': 'GMR ground truth'}.get(n, NICE.get(n, n)); tw = dr.textlength(lab, font=f_lab)
        dr.rounded_rectangle([px - tw / 2 - 8, py + 8, px + tw / 2 + 8, py + 34], radius=5, fill=(20, 22, 28, 150))
        dr.text((px - tw / 2, py + 10), lab, font=f_lab, fill=(245, 245, 248, 255))
    dr.text((44, 30), title, font=f_title, fill=(24, 26, 32, 255))
    dr.text((46, 76), 'red dots = pinned targets (hands + feet SE(3), from the retarget / scaled from G1)   ·   input to the model: root + hands only', font=f_sub, fill=(70, 74, 84, 255))
    src_ = f'source: {q["name"]}  @{at}'; tw = dr.textlength(src_, font=f_small); dr.text((W - 44 - tw, H - 62), src_, font=f_small, fill=(70, 74, 84, 255))
    dr.rectangle([44, H - 30, W - 44, H - 26], fill=(0, 0, 0, 40)); dr.rectangle([44, H - 30, 44 + (W - 88) * u, H - 26], fill=(24, 26, 32, 200))
    pr.stdin.write(np.ascontiguousarray(np.asarray(im.convert('RGB')), np.uint8).tobytes())
    if i in grid_idx: grid_frames.append(im.convert('RGB').resize((W // 3, H // 3)))
    if i % 60 == 0: print(f'[render] {i}/{T}', flush=True)
pr.stdin.close(); assert pr.wait() == 0; print('WROTE', out, flush=True)
cols = 3; rws = math.ceil(len(grid_frames) / cols)
G = Image.new('RGB', (cols * (W // 3), rws * (H // 3)), (255, 255, 255))
for k, f_ in enumerate(grid_frames): G.paste(f_, ((k % cols) * (W // 3), (k // cols) * (H // 3)))
G.save(out.replace('.mp4', '_grid.png')); print('WROTE', out.replace('.mp4', '_grid.png'), flush=True)
ren.close()
