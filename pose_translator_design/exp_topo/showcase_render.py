"""对外展示渲染 (0903): 一段人体动作 -> N 台机器人同台同步执行, 单场景单机位, 无调试叠字。
与 crossrig_amass.py 同一条推理链 (单帧 codec, 水平位移透传, 竖直=模型预测根高, 朝向=解码根旋转),
只是把每台机器人 attach 进同一个 MjSpec, 一起渲 1080p, 跟拍 + 缓慢环绕机位。
用法: showcase_render.py <ckpt> <out_dir> <clip_regex@start> ...
环境: AM_CACHE / AM_SMPL_CACHE / AM_LBL 同 crossrig; AM_T 帧数; AM_BODIES 逗号列表 (默认 hero 8 台);
      AM_W/AM_H 分辨率 (默认 1920x1080); AM_TITLE 左上角小标; AM_SRC 右上角来源字样; AM_ROWS 排数 (默认 1)
      AM_HUMAN=mesh|stick (默认 mesh: SMPL-X 网格, 顶点每帧 mjr_uploadMesh 进同一 MuJoCo 世界; 无 SMPL 参数的源用 stick)
      AM_ARROWS=1 (默认开): 接触矢量 —— 人体脚下橙色箭头 = 喂给编码器的接触标签 (输入), 机器人绿色箭头 = 解码几何的接触读数 (输出)"""
import os, sys, json, re, subprocess, math
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from common import SkeletonSpec, fk_global, matrix_to_rot6d, rot6d_to_matrix
from dual_common import build_model, geom, static_cond, fk_pos_torch
from manifold_project import fit_angles, load_dof
from semantics import build_name_embeddings, embed_matrix
from contact_label import sole_offsets
sys.path.insert(0, "/home/arenalabs/Desktop/AlphaMotion/src")

RUNS = "/var/tmp/alphamotion/topo_exp/runs"
C = os.environ.get("AM_CACHE", "/var/tmp/alphamotion/topo_exp/cache_lafan_v3")
C_SMPL = os.environ.get("AM_SMPL_CACHE", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")   # SMPL 序旋转 (网格用)
HUMAN = os.environ.get("AM_HUMAN", "mesh"); ARROWS = os.environ.get("AM_ARROWS", "1") == "1"
MASK = os.environ.get("AM_MASK", "")        # "ee" = 只喂根+末端; "L_Wrist,R_Wrist" = 只喂根+这些关节; 空 = 全身
CODES = os.environ.get("AM_CODES", "0") == "1"   # 左下角画离散码板 (FSQ 整数码, 逐帧)
LBL = os.environ.get("AM_LBL", "/home/arenalabs/tmp/cl_v3/lafan")
X2_XML = "/var/tmp/alphamotion/robot_models/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
CK = sys.argv[1] if sys.argv[1].startswith("/") else f"{RUNS}/{sys.argv[1]}"
OUT = sys.argv[2]; PATS = sys.argv[3:]
T = int(os.environ.get("AM_T", 300))
W = int(os.environ.get("AM_W", 1920)); H = int(os.environ.get("AM_H", 1080))
HERO = "unitree_g1_29dof,pnd_adam_lite,fourier_gr3,x2,unitree_h1_2,booster_t1_29dof,fourier_n1,kuavo_s45"
BODIES = os.environ.get("AM_BODIES", HERO).split(",")
ROWS = int(os.environ.get("AM_ROWS", 2))
TITLE = os.environ.get("AM_TITLE", "AlphaMotion")
SRC = os.environ.get("AM_SRC", "")
GAP = float(os.environ.get("AM_GAP", 1.35))      # 横向间距 (m)
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
NICE = {"human_smpl": "Human", "unitree_g1_29dof": "Unitree G1", "pnd_adam_lite": "PND Adam",
        "fourier_gr3": "Fourier GR-3", "x2": "AgiBot X2 · 0-shot", "unitree_h1_2": "Unitree H1-2",
        "booster_t1_29dof": "Booster T1", "fourier_n1": "Fourier N1", "kuavo_s45": "Kuavo S45",
        "engineai_pm01": "EngineAI PM01", "hightorque_hi": "HighTorque Hi", "tienkung": "TienKung",
        "unitree_h1": "Unitree H1", "booster_k1": "Booster K1", "booster_t1": "Booster T1 · 23dof",
        "unitree_g1_23dof": "Unitree G1 · 23dof"}
os.makedirs(OUT, exist_ok=True)
dev = os.environ.get("AM_DEV", "cpu"); torch.set_grad_enabled(False)
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])   # ardy (Y-up) -> mujoco (Z-up)

# ------------------------------------------------------------------ 片段
seqs = json.load(open(f"{C}/sequences.json"))
picks = []
for p in PATS:
    p, _, at = p.partition("@")
    m = [q for q in seqs if re.search(p, q["name"], re.I) and q["len"] >= T + (0 if at else 20)]   # 指定 @起点 时允许整段刚好 T 帧 (GenMo 短片)
    assert m, f"没有匹配 {p} 的片段"
    q = dict(m[len(m) // 2]); q["at"] = int(at) if at else max(0, (q["len"] - T) // 2)   # 默认取片段正中一窗
    picks.append((p, q)); print(f"[pick] {p}: {q['name']} len={q['len']} at={q['at']}", flush=True)
rows = torch.cat([torch.arange(q["start"] + q["at"], q["start"] + q["at"] + T) for _, q in picks])
B = len(rows)

# ------------------------------------------------------------------ SMPL-X 网格 (骨盆相对, ardy Y-up)
VREL = None; SMPL_F = None
if HUMAN == "mesh":
    import tempfile
    l6s = torch.load(f"{C_SMPL}/human_smpl_paired_local6d.pt", mmap=True)[rows].double()
    aa = Rotation.from_matrix(rot6d_to_matrix(l6s.float()).numpy().reshape(-1, 3, 3)).as_rotvec().reshape(B, 22, 3)
    tmp = tempfile.mkdtemp()
    np.savez(f"{tmp}/params.npz", go=aa[:, 0].astype(np.float32), bp=aa[:, 1:22].reshape(B, 63).astype(np.float32))
    code = f"""
import numpy as np, torch, smplx
z=np.load('{tmp}/params.npz'); n=len(z['go'])
m=smplx.create('/home/arenalabs/Downloads/SMPLX_NEUTRAL.npz', model_type='smplx', use_pca=False, batch_size=n)
with torch.no_grad(): o=m(global_orient=torch.tensor(z['go']), body_pose=torch.tensor(z['bp']))
v=o.vertices.numpy()-o.joints[:,0:1].numpy()
np.savez('{tmp}/verts.npz', v=v.astype(np.float32), f=m.faces.astype(np.int32)); print('verts ok')
"""
    r_ = subprocess.run(["/home/arenalabs/miniconda3/envs/genmo-gem/bin/python", "-c", code], capture_output=True, text=True)
    assert "verts ok" in r_.stdout, r_.stderr[-800:]
    z_ = np.load(f"{tmp}/verts.npz"); VREL, SMPL_F = z_["v"], z_["f"]
    print(f"[smpl] verts {VREL.shape} faces {SMPL_F.shape}", flush=True)

def load_spec_npz(name):
    d = np.load(f"{C}/{name}_spec.npz", allow_pickle=True)
    return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], [str(x) for x in d["joint_names"]])

cfg = json.load(open(f"{CK}/config.json"))
specs, names_all = {}, []
for n in ["human_smpl"] + [b for b in BODIES if b != "x2"]:
    specs[n] = load_spec_npz(n); names_all += specs[n].joint_names
x2 = None
if "x2" in BODIES:
    from alphamotion.embodiment import registry as _reg
    x2 = _reg.load_from_xml(X2_XML, "agibot_x2")
    specs["x2"] = SkeletonSpec(x2.spec.name, np.asarray(x2.spec.parents), np.asarray(x2.spec.rest_offsets),
                               list(x2.spec.joint_names))
    names_all += specs["x2"].joint_names
lut = build_name_embeddings(names_all, device=dev, encoder=cfg.get("sem_encoder", "siglip"))

def entry(n):
    spec = specs[n]
    if n == "x2":
        pr = np.zeros((spec.J, 3), np.float32)
        for j in range(spec.J):
            p_ = spec.parents[j]; pr[j] = (pr[p_] if p_ >= 0 else 0) + np.asarray(spec.rest_offsets[j])
        e = {"spec": spec, "st": static_cond(spec, dev), "reach": float(np.linalg.norm(pr, axis=1).max()),
             "sem": embed_matrix(spec.joint_names, lut).to(dev),
             "dof": torch.as_tensor(np.asarray(x2.dof), dtype=torch.float32, device=dev)}
        from dof_head import DofKin
        e["kin"] = DofKin(spec, e["dof"], torch.as_tensor(np.asarray(x2.rest), dtype=torch.float32, device=dev), device=dev)
        return e
    l6 = torch.load(f"{C}/{n}_paired_local6d.pt", mmap=True)[rows].float().to(dev)
    gR, gp = fk_global(l6, spec); reach = float(gp.norm(dim=-1).max())
    e = {"spec": spec, "st": static_cond(spec, dev), "reach": reach,
         "sem": embed_matrix(spec.joint_names, lut).to(dev),
         "pose": torch.cat([matrix_to_rot6d(gR), gp / reach], -1)}
    if os.path.exists(f"{C}/{n}_dof.npz"):
        from dof_head import build_kin
        e["dof"] = torch.as_tensor(np.load(f"{C}/{n}_dof.npz")["dof"], dtype=torch.float32, device=dev)
        e["kin"] = build_kin(C, n, spec, device=dev)
    return e

data = {n: entry(n) for n in ["human_smpl"] + BODIES}
model = build_model(cfg, dev); model.load_state_dict(torch.load(f"{CK}/model.pt", map_location=dev)); model.eval()
con = None
if cfg.get("contact_chan"):
    z = np.load(f"{LBL}/human_smpl_contact.npz", allow_pickle=True)
    h_ = json.loads(str(z["header"])); thr_, D_ = h_["thr_m"], h_["D_used_m"]
    s_ = thr_ - z["gap"][rows.numpy()].astype(np.float32)
    con = torch.as_tensor((np.sign(s_) * np.log1p(np.abs(s_) / thr_) / np.log1p(D_ / thr_)).astype(np.float32), device=dev)
HCON = None
if cfg.get("contact_chan"):
    HCON = z["contact_soft"][rows.numpy()].astype(np.float32)     # [B,J] 喂给编码器的软接触标签 (输入侧)
JM = None
if MASK:
    from masking import leaf_mask
    hs_ = specs["human_smpl"]; keep = np.zeros(hs_.J, bool); keep[int(np.where(np.asarray(hs_.parents) < 0)[0][0])] = True
    if MASK == "ee": keep |= leaf_mask(hs_).numpy()
    else:
        for nm in MASK.split(","): keep[list(hs_.joint_names).index(nm.strip())] = True
    JM = torch.as_tensor(~keep, device=dev)[None].expand(B, -1)
    VIS_J = [j for j in range(hs_.J) if keep[j]]
    print(f"[mask] encoder sees {len(VIS_J)}/{hs_.J} joints: {[hs_.joint_names[j] for j in VIS_J]}", flush=True)
zp_, zr_, (CP, CR) = model.encode(data["human_smpl"]["pose"], geom(data, "human_smpl", B), con=con, joint_mask=JM, return_codes=True)
codes = (zp_, zr_)
CR = CR.detach().cpu().numpy(); CP = CP.detach().cpu().numpy(); NLEV = len(np.unique(np.round(CR, 4)))
print(f"[codes] rot {CR.shape} levels {NLEV} ({CR.min():.2f}..{CR.max():.2f})  pos {CP.shape}", flush=True)
dec, dpos = {}, {}
for tgt in BODIES:
    dec[tgt], dpos[tgt] = model.decode(*codes, geom(data, tgt, B))
if os.environ.get("AM_DBG_H"):
    _ri = int(np.where(np.asarray(specs[BODIES[0]].parents) < 0)[0][0])
    print("[dbg] pred root h", BODIES[0], np.round(dpos[BODIES[0]][::10, _ri, 0].cpu().numpy(), 3).tolist(), flush=True)
    print("[dbg] human root h", np.round(torch.load(f"{C}/human_smpl_root_world.pt", mmap=True).float()[rows][::10, 1].numpy() / 100, 3).tolist(), flush=True)
print("[decode] done; root height (m) per body:", {n: round(float(dpos[n][:, int(np.where(np.asarray(specs[n].parents) < 0)[0][0]), 0].mean()), 3) for n in BODIES}, flush=True)

# ------------------------------------------------------------------ 世界系: 位移透传, 朝向=解码根, 整体转正
RWh = torch.load(f"{C}/human_smpl_root_world.pt", mmap=True).float()[rows].numpy() / 100.0
XY = np.zeros((B, 2))
RR = {t: rot6d_to_matrix(dec[t][:, 0, :6]).cpu().numpy() for t in BODIES}
RRm = {t: np.einsum("ij,bjk,kl->bil", AX.T, RR[t], AX) for t in BODIES}     # mujoco 系根旋转
def root_idx(t): return int(np.where(np.asarray(specs[t].parents) < 0)[0][0])
hp = fk_pos_torch(data["human_smpl"]["pose"][..., :6], specs["human_smpl"]).cpu().numpy() / 100.0  # 根相对 ardy m
YAW = np.zeros(len(picks)); TRJ = np.zeros((B, 2))
for wi, (_, q) in enumerate(picks):
    sl = slice(wi * T, (wi + 1) * T)
    xy = ((RWh[sl] - RWh[sl.start]) @ AX)[:, :2]
    # 朝向: 第一台机器人首帧解码根 yaw (行进 >1.5m 时用行进方向), 转到 +X, 横向排队沿 Y
    R0 = RRm[BODIES[0]][sl.start]; yaw = math.atan2(R0[1, 0], R0[0, 0])
    if np.linalg.norm(xy[-1] - xy[0]) > 1.5:
        d_ = xy[-1] - xy[0]; yaw = math.atan2(d_[1], d_[0])
    YAW[wi] = -yaw
    c, s = math.cos(-yaw), math.sin(-yaw); Rz = np.array([[c, -s], [s, c]])
    TRJ[sl] = xy @ Rz.T
    Rz3 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for t in BODIES: RRm[t][sl] = np.einsum("ij,bjk->bik", Rz3, RRm[t][sl])
    hp[sl] = np.einsum("ij,bnj->bni", AX @ Rz3 @ AX.T, hp[sl])    # 人体关节也转正 (ardy 系内)
    if VREL is not None: VREL[sl] = np.einsum("ij,bnj->bni", (AX @ Rz3 @ AX.T).astype(np.float32), VREL[sl])

# ------------------------------------------------------------------ 排队位置: 人体前排正中, 机器人分排站后面, 后排错半格
ALL = ["human_smpl"] + BODIES
n_all = len(ALL); per_row = math.ceil(n_all / ROWS)
ROW_D = float(os.environ.get("AM_ROWD", 2.6))                      # 排间深度
rows = [[] for _ in range(ROWS)]
front = ["human_smpl"] + BODIES[:per_row - 1]; mid = len(front) // 2
front = front[1:mid + 1] + ["human_smpl"] + front[mid + 1:]         # 人体放中间
rows[0] = front; rest = BODIES[per_row - 1:]
for r_ in range(1, ROWS): rows[r_] = rest[(r_ - 1) * per_row:r_ * per_row]
OFF = {}; ROWOF = {}
for r_, row in enumerate(rows):
    m = len(row)
    for c_, n in enumerate(row):
        OFF[n] = np.array([-r_ * ROW_D, (c_ - (m - 1) / 2) * GAP]); ROWOF[n] = r_
ROW_W = max((len(row) - 1) * GAP for row in rows)

# ------------------------------------------------------------------ 场景
def resolve_xml(emb):
    import importlib.util
    alias = {"unitree_g1_29dof": "unitree_g1"}
    gp = '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/external/GMR/general_motion_retargeting/params.py'
    s2 = importlib.util.spec_from_file_location("gp", gp); m2 = importlib.util.module_from_spec(s2); s2.loader.exec_module(m2)
    return str(m2.ROBOT_XML_DICT[alias.get(emb, emb)])

root = mj.MjSpec(); root.modelname = "showcase"
root.visual.global_.offwidth = W; root.visual.global_.offheight = H
root.visual.global_.fovy = 32
root.visual.quality.shadowsize = 8192
root.visual.map.shadowclip = 2.0; root.visual.map.shadowscale = 0.9
root.visual.headlight.ambient = [0.34, 0.34, 0.36]; root.visual.headlight.diffuse = [0.12, 0.12, 0.12]
root.visual.headlight.specular = [0.0, 0.0, 0.0]
sky = root.add_texture(); sky.name = "_sky"; sky.type = mj.mjtTexture.mjTEXTURE_SKYBOX
sky.builtin = mj.mjtBuiltin.mjBUILTIN_GRADIENT; sky.rgb1 = [0.93, 0.94, 0.96]; sky.rgb2 = [0.80, 0.83, 0.88]
sky.width = sky.height = 512
tx = root.add_texture(); tx.name = "_grid"; tx.type = mj.mjtTexture.mjTEXTURE_2D
tx.builtin = mj.mjtBuiltin.mjBUILTIN_CHECKER; tx.width = tx.height = 512
tx.rgb1 = [0.885, 0.89, 0.90]; tx.rgb2 = [0.862, 0.867, 0.882]; tx.mark = mj.mjtMark.mjMARK_EDGE; tx.markrgb = [0.835, 0.84, 0.855]
mt = root.add_material(); mt.name = "_gridm"; mt.texrepeat = [120, 120]; mt.reflectance = 0.0; mt.specular = 0.0; mt.shininess = 0.0
mt.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "_grid"
root.worldbody.add_geom(name="_floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[60, 60, .1], material="_gridm")
sun = root.worldbody.add_light(name="_sun", pos=[3, -4, 8], dir=[-0.3, 0.45, -1], diffuse=[.5, .5, .5],
                               specular=[0.0, 0.0, 0.0], castshadow=True); sun.type = mj.mjtLightType.mjLIGHT_DIRECTIONAL
fill = root.worldbody.add_light(name="_fill", pos=[-4, 6, 5], dir=[0.4, -0.6, -0.6], diffuse=[.16, .16, .18], specular=[0, 0, 0], castshadow=False)
fill.type = mj.mjtLightType.mjLIGHT_DIRECTIONAL
# 人体: SMPL-X 网格 (顶点每帧改写) 或 胶囊 + 关节球 (mocap)
hs = specs["human_smpl"]; H_CAPS = [j for j in range(hs.J) if hs.parents[j] >= 0] if VREL is None else []
SKIN = [0.93, 0.80, 0.62, 1]
if VREL is not None:
    me_ = root.add_mesh(); me_.name = "_smpl"
    v0_ = VREL[0] @ AX; v0_[:, 2] -= v0_[:, 2].min()
    me_.uservert = v0_.astype(np.float64).flatten().tolist(); me_.userface = SMPL_F.flatten().tolist()
    root.worldbody.add_geom(name="_hmesh", type=mj.mjtGeom.mjGEOM_MESH, meshname="_smpl", rgba=[0.90, 0.76, 0.60, 1] if not MASK else [0.62, 0.64, 0.70, 0.38],
                            contype=0, conaffinity=0)
for j in H_CAPS:
    b = root.worldbody.add_body(name=f"_hc{j}", mocap=True)
    b.add_geom(name=f"_hg{j}", type=mj.mjtGeom.mjGEOM_CAPSULE, size=[0.03, 0.1, 0], rgba=SKIN, contype=0, conaffinity=0)
for j in (range(hs.J) if VREL is None else []):
    b = root.worldbody.add_body(name=f"_hj{j}", mocap=True)
    b.add_geom(name=f"_hs{j}", type=mj.mjtGeom.mjGEOM_SPHERE, size=[0.036 if hs.parents[j] < 0 else 0.03, 0, 0],
               rgba=SKIN, contype=0, conaffinity=0)
# 脚印: 每台 4 个候选盘 (mocap), 模型自感踩地时亮
NFP = 4
for n in BODIES:
    for k in range(NFP):
        b = root.worldbody.add_body(name=f"_fp{n}_{k}", mocap=True, pos=[0, 0, -5])
        b.add_geom(name=f"_fpg{n}_{k}", type=mj.mjtGeom.mjGEOM_CYLINDER, size=[0.075, 0.004, 0],
                   rgba=[0.16, 0.62, 0.42, 0.55], contype=0, conaffinity=0)
PREFIX = {}
for n in BODIES:
    ch = mj.MjSpec.from_file(X2_XML if n == "x2" else resolve_xml(n))
    for g in list(ch.worldbody.geoms): ch.delete(g)
    for l in list(ch.lights): ch.delete(l)          # X2 的 MJCF 自带一盏聚光灯 (挂在 body 下), 会在地上打出光斑
    for c in list(ch.cameras): ch.delete(c)
    fr = root.worldbody.add_frame(pos=[0, 0, 0]); PREFIX[n] = f"{n}/"
    root.attach(ch, prefix=PREFIX[n], frame=fr)
sc = root.compile(); d = mj.MjData(sc)
print(f"[scene] nbody={sc.nbody} ngeom={sc.ngeom} nq={sc.nq}", flush=True)
MESH = None
if VREL is not None:
    gid_ = mj.mj_name2id(sc, mj.mjtObj.mjOBJ_GEOM, "_hmesh"); mid_ = int(sc.geom_dataid[gid_])
    sc.geom_pos[gid_] = 0; sc.geom_quat[gid_] = [1, 0, 0, 0]; sc.geom_rbound[gid_] = 100
    sc.geom_aabb[gid_] = [0, 0, 0, 100, 100, 100]
    MESH = dict(mid=mid_, va=int(sc.mesh_vertadr[mid_]), na=int(sc.mesh_normaladr[mid_]), n=int(sc.mesh_vertnum[mid_]))
    assert int(sc.mesh_normalnum[mid_]) == MESH["n"], "期望逐顶点法线"
def vnormals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3): np.add.at(n, f[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)

# ------------------------------------------------------------------ 关节角 + 地址
Q = {}; ADR = {}; FREE = {}; JB = {}; SOLE = {}; THR = {}
for n in BODIES:
    spec = specs[n]
    if n == "x2":
        dof = torch.as_tensor(np.asarray(x2.dof), dtype=torch.float32, device=dev)
        rest = torch.as_tensor(np.asarray(x2.rest), dtype=torch.float32, device=dev); sj = x2.qnames
    else:
        dof, rest = load_dof(C, n, device=dev)
        sj = [str(s) for s in np.load(f"{C}/{n}_dof.npz", allow_pickle=True)["slot_joints"]]
    q, _ = fit_angles(dec[n], spec, dof, rest=rest, clamp=True, method="global"); Q[n] = q.cpu().numpy()
    adr = {}
    for k, names in enumerate(sj):
        lst = list(names) if isinstance(names, (list, tuple)) else [x for x in str(names).split(",") if x]
        for s_, jn in enumerate(lst):
            jid = mj.mj_name2id(sc, mj.mjtObj.mjOBJ_JOINT, PREFIX[n] + jn)
            if jid >= 0 and sc.jnt_type[jid] == mj.mjtJoint.mjJNT_HINGE: adr[(k, s_)] = int(sc.jnt_qposadr[jid])
    ADR[n] = adr
    fj = [i for i in range(sc.njnt) if sc.jnt_type[i] == mj.mjtJoint.mjJNT_FREE
          and (mj.mj_id2name(sc, mj.mjtObj.mjOBJ_JOINT, i) or "").startswith(PREFIX[n])]
    if not fj:   # 匿名 free joint: 按 body 树找该前缀下的根 body
        fj = [i for i in range(sc.njnt) if sc.jnt_type[i] == mj.mjtJoint.mjJNT_FREE
              and (mj.mj_id2name(sc, mj.mjtObj.mjOBJ_BODY, sc.jnt_bodyid[i]) or "").startswith(PREFIX[n])]
    assert len(fj) == 1, (n, fj); FREE[n] = int(sc.jnt_qposadr[fj[0]])
    JB[n] = [mj.mj_name2id(sc, mj.mjtObj.mjOBJ_BODY, PREFIX[n] + jn) for jn in spec.joint_names]
    SOLE[n] = sole_offsets("agibot_x2" if n == "x2" else n, spec)
    THR[n] = 0.04
    if os.path.exists(f"{LBL}/{n}_contact.npz"):
        THR[n] = float(json.loads(str(np.load(f"{LBL}/{n}_contact.npz", allow_pickle=True)["header"]))["thr_m"])
    print(f"[{n}] hinge={len(adr)} free@{FREE[n]} sole={len(SOLE[n])} thr={THR[n]:.3f}", flush=True)
HZ = RWh[:, 1].copy()   # 人体根高 (m, 世界 Y-up)
for wi in range(len(picks)):
    sl = slice(wi * T, (wi + 1) * T)
    low = (hp[sl][:, :, 1] + HZ[sl, None]).min()          # 每窗常数贴地
    if VREL is not None: low = (VREL[sl][:, :, 1] + HZ[sl, None]).min()
    HZ[sl] -= low
FPG = {n: [mj.mj_name2id(sc, mj.mjtObj.mjOBJ_GEOM, f"_fpg{n}_{k}") for k in range(NFP)] for n in BODIES}
FPM = {n: [int(sc.body_mocapid[mj.mj_name2id(sc, mj.mjtObj.mjOBJ_BODY, f"_fp{n}_{k}")]) for k in range(NFP)] for n in BODIES}
HCM = {j: int(sc.body_mocapid[mj.mj_name2id(sc, mj.mjtObj.mjOBJ_BODY, f"_hc{j}")]) for j in H_CAPS}
HJM = {j: int(sc.body_mocapid[mj.mj_name2id(sc, mj.mjtObj.mjOBJ_BODY, f"_hj{j}")]) for j in (range(hs.J) if VREL is None else [])}
HCG = {j: mj.mj_name2id(sc, mj.mjtObj.mjOBJ_GEOM, f"_hg{j}") for j in H_CAPS}

# ------------------------------------------------------------------ 机位 + 投影
FOVY = math.radians(root.visual.global_.fovy)
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
ARR = []          # 本帧接触矢量 [(xyz, root_xy, length, rgba)], update_scene 后画进 scene
def arrow_dir(base, root_xy):
    """从接触点斜向外上 (水平离开根 0.7 + 竖直 1.0), 免得箭头藏在腿里"""
    h = np.asarray(base[:2], np.float64) - np.asarray(root_xy, np.float64); nh = np.linalg.norm(h)
    h = h / nh if nh > 1e-6 else np.array([1.0, 0.0])
    v = np.array([0.7 * h[0], 0.7 * h[1], 1.0]); return v / np.linalg.norm(v)
ORANGE = np.array([0.96, 0.56, 0.12, 0.95], np.float32); GREEN = np.array([0.14, 0.64, 0.42, 0.95], np.float32)
BLUE = np.array([0.16, 0.45, 0.98, 1.0], np.float32); SPH = []   # 被观测关节 (遮挡模式)
def draw_arrows():
    scn = ren.scene
    for p_ in SPH:
        if scn.ngeom >= scn.maxgeom: break
        g_ = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(g_, mj.mjtGeom.mjGEOM_SPHERE, np.array([0.05, 0, 0]), np.asarray(p_, np.float64), np.eye(3).ravel(), BLUE)
        scn.ngeom += 1
    for p_, r_, L_, c_ in ARR:
        if scn.ngeom >= scn.maxgeom: break
        g_ = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(g_, mj.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), c_)
        b_ = np.asarray(p_, np.float64)
        mj.mjv_connector(g_, mj.mjtGeom.mjGEOM_ARROW, 0.032, b_, b_ + arrow_dir(b_, r_) * L_)
        scn.ngeom += 1
f_title = ImageFont.truetype(FONT_B, 40); f_sub = ImageFont.truetype(FONT_R, 22); f_lab = ImageFont.truetype(FONT_B, 19)
f_small = ImageFont.truetype(FONT_R, 18)

def frame_at(t, wi):
    d.qpos[:] = sc.qpos0
    # 人体
    P = (hp[t] @ AX); P[:, 2] += HZ[t]; P[:, :2] += TRJ[t] + OFF["human_smpl"]
    ARR.clear(); SPH.clear()
    if MASK: SPH.extend(P[j].copy() for j in VIS_J)
    if MESH is not None:
        vw = (VREL[t] @ AX).astype(np.float32); vw[:, 2] += HZ[t]; vw[:, :2] += TRJ[t] + OFF["human_smpl"]
        a_, n_ = MESH["va"], MESH["n"]
        sc.mesh_vert[a_:a_ + n_] = vw; sc.mesh_normal[MESH["na"]:MESH["na"] + n_] = vnormals(vw, SMPL_F)
        mj.mjr_uploadMesh(sc, ren._mjr_context, MESH["mid"])
    if ARROWS and HCON is not None:
        for j in range(hs.J):
            pr_ = float(HCON[t, j])
            if pr_ > 0.5:
                base = P[j].copy(); base[2] = 0.006 if base[2] < 0.14 else base[2]
                ARR.append((base, P[0, :2], 0.14 + 0.22 * pr_, ORANGE))
    for j in HJM: d.mocap_pos[HJM[j]] = P[j]
    for j in H_CAPS:
        a, b_ = P[hs.parents[j]], P[j]; v = b_ - a; L = np.linalg.norm(v)
        d.mocap_pos[HCM[j]] = (a + b_) / 2
        if L > 1e-6:
            ax = np.cross([0, 0, 1], v / L); sn = np.linalg.norm(ax)
            rv = ax / sn * math.asin(min(1, sn)) if sn > 1e-9 else np.zeros(3)
            if (v / L)[2] < 0 and sn <= 1e-9: rv = np.array([math.pi, 0, 0])
            d.mocap_quat[HCM[j]] = Rotation.from_rotvec(rv).as_quat(scalar_first=True)
            sc.geom_size[HCG[j]][1] = L / 2
    # 机器人
    for n in BODIES:
        a0 = FREE[n]; ri = root_idx(n)
        d.qpos[a0:a0 + 2] = TRJ[t] + OFF[n]
        d.qpos[a0 + 2] = float(dpos[n][t, ri, 0])
        d.qpos[a0 + 3:a0 + 7] = Rotation.from_matrix(RRm[n][t]).as_quat(scalar_first=True)
        for (k, s_), a in ADR[n].items(): d.qpos[a] = Q[n][t, k, s_]
    mj.mj_forward(sc, d)
    # 脚印: 脚底点离地 < thr 的关节 (与训练 cgeo 同函数), 只画最低 NFP 个
    for n in BODIES:
        spec = specs[n]; cand = []
        for j in range(spec.J):
            bid = JB[n][j]
            if bid < 0: continue
            pz = d.xpos[bid].copy()
            if j in SOLE[n]: pz = pz + d.xmat[bid].reshape(3, 3) @ (AX.T @ np.asarray(SOLE[n][j]))
            if pz[2] < 0.35: cand.append((pz[2], pz))
            if ARROWS:
                pr_ = 1 / (1 + math.exp(-(THR[n] - pz[2]) / (THR[n] / 3)))
                if pr_ > 0.5:
                    base = pz.copy(); base[2] = 0.006 if base[2] < 0.14 else base[2]
                    ARR.append((base, TRJ[t] + OFF[n], 0.14 + 0.22 * pr_, GREEN))
        cand.sort(key=lambda x: x[0])
        for k in range(NFP):
            if k < len(cand) and 1 / (1 + math.exp(-(THR[n] - cand[k][0]) / (THR[n] / 3))) > 0.5:
                d.mocap_pos[FPM[n][k]] = [cand[k][1][0], cand[k][1][1], 0.004]
            else:
                d.mocap_pos[FPM[n][k]] = [0, 0, -5]
    mj.mj_forward(sc, d)

for wi, (pat, q) in enumerate(picks):
    sl = slice(wi * T, (wi + 1) * T)
    cen = _smooth(TRJ[sl]);
    out = f"{OUT}/showcase_{re.sub('[^A-Za-z0-9]+', '_', q['name'])[:40]}_at{q['at']}.mp4"
    pr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-pix_fmt", "yuv420p", out],
                          stdin=subprocess.PIPE)
    # 机位: 环绕 (方位 205° -> 155° 缓慢摆), 俯 13°, 距离按队伍宽度定
    dist = max(5.5, ROW_W / (2 * math.tan(FOVY / 2) * (W / H)) * 1.06 + 1.6 + 0.5 * (ROWS - 1) * ROW_D) * float(os.environ.get("AM_ZOOM", 1))
    ELEV = float(os.environ.get("AM_ELEV", -9 if ROWS == 1 else -17))
    for i, t in enumerate(range(sl.start, sl.stop)):
        frame_at(t, wi)
        u = i / max(1, T - 1)
        cam.azimuth = 200 - 40 * (0.5 - 0.5 * math.cos(math.pi * u))
        cam.elevation = ELEV; cam.distance = dist
        cam.lookat[:] = [cen[i][0] - 0.5 * (ROWS - 1) * ROW_D, cen[i][1], 0.62 + 0.06 * (ROWS - 1)]
        ren.update_scene(d, cam); draw_arrows(); im = Image.fromarray(ren.render().copy())
        dr = ImageDraw.Draw(im, "RGBA")
        # 标签: 每台脚下
        for li, n in enumerate(ALL):
            xy = TRJ[t] + OFF[n]; px, py, z = project(cam, [xy[0], xy[1] - 0.0, -0.02])
            if z <= 0: continue
            py += 8 + (30 * (li % 2) if ROWS == 1 else 0)
            lab = NICE.get(n, n); tw = dr.textlength(lab, font=f_lab)
            dr.rounded_rectangle([px - tw / 2 - 8, py, px + tw / 2 + 8, py + 26], radius=5, fill=(20, 22, 28, 150))
            dr.text((px - tw / 2, py + 2), lab, font=f_lab, fill=(245, 245, 248, 255))
        # 左上: 标题
        dr.text((44, 34), TITLE, font=f_title, fill=(24, 26, 32, 255))
        sub = "one motion  ·  every body  ·  single-frame codec, no post-processing"
        if MASK: sub = f"encoder sees {len(VIS_J)} of {hs.J} joints  ·  the rest is inferred by the network, not retargeted"
        dr.text((46, 84), sub, font=f_sub, fill=(70, 74, 84, 255))
        ly = 122
        if MASK:
            dr.ellipse([52, ly + 1, 72, ly + 21], fill=(41, 115, 250, 255))
            dr.text((84, ly), f"observed joints (encoder input, {len(VIS_J)}/{hs.J})", font=f_small, fill=(70, 74, 84, 255)); ly += 30
            dr.ellipse([52, ly + 1, 72, ly + 21], fill=(158, 163, 178, 160))
            dr.text((84, ly), "hidden joints (ghost = ground truth, never shown to the model)", font=f_small, fill=(70, 74, 84, 255)); ly += 30
        if CODES:
            cs_ = 5; cr_ = CR[t]; n_lat, n_dim = cr_.shape; bx, by = 44, H - 60 - n_dim * cs_
            lv = (cr_.T - CR.min()) / max(1, CR.max() - CR.min())          # [dim, latent] 0..1
            board = np.zeros((n_dim, n_lat, 3), np.uint8)
            board[..., 0] = (30 + 200 * lv).astype(np.uint8); board[..., 1] = (40 + 120 * lv).astype(np.uint8); board[..., 2] = (120 - 90 * lv).astype(np.uint8)
            bi = Image.fromarray(board).resize((n_lat * cs_, n_dim * cs_), Image.NEAREST)
            dr.rounded_rectangle([bx - 6, by - 30, bx + n_lat * cs_ + 6, by + n_dim * cs_ + 6], radius=6, fill=(255, 255, 255, 170))
            im.paste(bi, (bx, by))
            dr.text((bx, by - 26), f"latent codes this frame  ·  {n_lat} tokens × {n_dim} dims, {NLEV} levels (FSQ)  ·  the robots decode from these, nothing else", font=f_small, fill=(70, 74, 84, 255))
        if ARROWS:
            dr.polygon([(52, ly + 14), (62, ly - 4), (72, ly + 14)], fill=(245, 143, 31, 255)); dr.rectangle([59, ly + 14, 65, ly + 24], fill=(245, 143, 31, 255))
            dr.text((84, ly), "contact label in (encoder input)", font=f_small, fill=(70, 74, 84, 255)); ly += 30
            dr.polygon([(52, ly + 14), (62, ly - 4), (72, ly + 14)], fill=(36, 163, 107, 255)); dr.rectangle([59, ly + 14, 65, ly + 24], fill=(36, 163, 107, 255))
            dr.text((84, ly), "contact predicted (decoded geometry)", font=f_small, fill=(70, 74, 84, 255))
        # 右上: 来源
        src = SRC or f"source: {q['name']}"
        tw = dr.textlength(src, font=f_small); dr.text((W - 44 - tw, 40), src, font=f_small, fill=(70, 74, 84, 255))
        # 底部: 进度条
        dr.rectangle([44, H - 30, W - 44, H - 26], fill=(0, 0, 0, 40)); dr.rectangle([44, H - 30, 44 + (W - 88) * u, H - 26], fill=(24, 26, 32, 200))
        pr.stdin.write(np.ascontiguousarray(np.asarray(im.convert("RGB")), np.uint8).tobytes())
        if i % 50 == 0: print(f"[{wi}] {i}/{T}", flush=True)
    pr.stdin.close(); assert pr.wait() == 0; print("WROTE", out, flush=True)
ren.close()
