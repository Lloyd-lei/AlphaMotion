"""AMASS 蹲/跪动作的跨具身对比 (lloyd 0901): smpl源 | g1 | adam | gr3 | X2。
X2 不在语料里 (无 GT), spec 由 MJCF 现场构建, 纯零样本+肉眼判。
用法: crossrig_amass.py <ckpt> <out_dir> <clip_regex> [clip_regex ...]"""
import os, sys, json, re, subprocess
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
from PIL import Image, ImageDraw, ImageFont
from common import SkeletonSpec, fk_global, matrix_to_rot6d, rot6d_to_matrix
from dual_common import build_model, geom, static_cond, fk_pos_torch
from manifold_project import fit_angles, load_dof
from semantics import build_name_embeddings, embed_matrix
# X2 用产品侧 registry (与 ctrl_verify/x2_corpus.py 同源): 带 spec/dof/rest/qnames
sys.path.insert(0, "/home/arenalabs/Desktop/AlphaMotion/src")

RUNS = "/var/tmp/alphamotion/topo_exp/runs"
C = os.environ.get("AM_CACHE", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
LBL = os.environ.get("AM_LBL", "/home/arenalabs/tmp/cl_v3/amass")
CV3 = "/var/tmp/alphamotion/topo_exp/cache_amass_v3"  # unused
# 0901 方言改写: 模型吃 AM_CACHE (可能是 LAFAN 方言改写版), 但 SMPL-X 网格面板必须吃
# 原始 SMPL 序的旋转, 所以单独指一份 (默认原始 AMASS 缓存)。
C_SMPL = os.environ.get("AM_SMPL_CACHE", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
X2_XML = "/var/tmp/alphamotion/robot_models/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
CK, OUT = sys.argv[1] if sys.argv[1].startswith("/") else f"{RUNS}/{sys.argv[1]}", sys.argv[2]
PATS = sys.argv[3:]
TGTS = ["unitree_g1_29dof", "pnd_adam_lite", "fourier_gr3"]
T = int(os.environ.get("AM_T", 150))
FOLLOW = os.environ.get("AM_FOLLOW","0")=="1"   # 跟拍: 机位/画框跟着当前根走 (长位移片段用)
def _smooth(x, k=15):
    x=np.asarray(x,np.float64); out=np.empty_like(x)
    for i in range(len(x)): out[i]=x[max(0,i-k):i+k+1].mean(0)
    return out
NM = 8
W = H = 460
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
os.makedirs(OUT, exist_ok=True)
dev = os.environ.get("AM_DEV", "cuda"); torch.set_grad_enabled(False)

seqs = json.load(open(f"{C}/sequences.json"))
picks = []
for p in PATS:
    p, _, at = p.partition("@")          # 0902: "name@起始帧" 指定窗口 (匍匐/起身段不在片段中点)
    m = [q for q in seqs if re.search(p, q["name"], re.I) and q["len"] >= T + 20]
    assert m, f"没有匹配 {p} 的片段"
    q = dict(m[len(m)//2]); q["at"] = int(at) if at else q["len"] // 2
    picks.append((p, q))
    print(f"[pick] {p}: {q['name']} len={q['len']} at={q['at']}")
rows = torch.cat([torch.arange(q["start"]+q["at"], q["start"]+q["at"]+T) for _, q in picks])

def load_spec_npz(name):
    d = np.load(f"{C}/{name}_spec.npz", allow_pickle=True)
    return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], [str(x) for x in d["joint_names"]])

cfg = json.load(open(f"{CK}/config.json"))
# --- 语料内具身 (含 human) ---
names_all = []
specs = {}
for n in ["human_smpl"] + TGTS:
    specs[n] = load_spec_npz(n); names_all += specs[n].joint_names
# --- X2: MJCF 现场构建 ---
from alphamotion.embodiment import registry as _reg
x2 = _reg.load_from_xml(X2_XML, "agibot_x2")
from common import SkeletonSpec as _SS
_xsp = x2.spec
specs["x2"] = _SS(_xsp.name, np.asarray(_xsp.parents), np.asarray(_xsp.rest_offsets),
                  list(_xsp.joint_names))
names_all += specs["x2"].joint_names
lut = build_name_embeddings(names_all, device=dev, encoder=cfg.get("sem_encoder", "siglip"))

def entry(n):
    spec = specs[n]
    if n == "x2":
        pr = np.zeros((spec.J, 3), np.float32)
        for j in range(spec.J):
            p_ = spec.parents[j]
            pr[j] = (pr[p_] if p_ >= 0 else 0) + np.asarray(spec.rest_offsets[j])
        reach = float(np.linalg.norm(pr, axis=1).max())
        e = {"spec": spec, "st": static_cond(spec, dev), "reach": reach,
             "sem": embed_matrix(spec.joint_names, lut).to(dev),
             "dof": torch.as_tensor(np.asarray(x2.dof), dtype=torch.float32, device=dev)}
        from dof_head import DofKin
        e["kin"] = DofKin(spec, e["dof"],
                          torch.as_tensor(np.asarray(x2.rest), dtype=torch.float32, device=dev),
                          device=dev)
    else:
        l6 = torch.load(f"{C}/{n}_paired_local6d.pt", mmap=True)[rows].float().to(dev)
        gR, gp = fk_global(l6, spec)
        reach = float(gp.norm(dim=-1).max())
        e = {"spec": spec, "st": static_cond(spec, dev), "reach": reach,
             "sem": embed_matrix(spec.joint_names, lut).to(dev),
             "pose": torch.cat([matrix_to_rot6d(gR), gp / reach], -1)}
        dz = f"{C}/{n}_dof.npz"
        if os.path.exists(dz):
            from dof_head import build_kin
            e["dof"] = torch.as_tensor(np.load(dz)["dof"], dtype=torch.float32, device=dev)
            e["kin"] = build_kin(C, n, spec, device=dev)
    print(f"[{n}] J={spec.J} reach={reach:.1f}")
    return e

data = {n: entry(n) for n in ["human_smpl"] + TGTS + ["x2"]}
model = build_model(cfg, dev)
model.load_state_dict(torch.load(f"{CK}/model.pt", map_location=dev)); model.eval()
B = len(rows)
con = None
if cfg.get("contact_chan"):
    z = np.load(f"{LBL}/human_smpl_contact.npz", allow_pickle=True)
    h_ = json.loads(str(z["header"])); thr_, D_ = h_["thr_m"], h_["D_used_m"]
    s_ = thr_ - z["gap"][rows.numpy()].astype(np.float32)
    con = torch.as_tensor((np.sign(s_)*np.log1p(np.abs(s_)/thr_)/np.log1p(D_/thr_)).astype(np.float32), device=dev)
codes = model.encode(data["human_smpl"]["pose"], geom(data, "human_smpl", B), con=con)
dec = {}; dpos = {}
for tgt in TGTS + ["x2"]:
    dec[tgt], dpos[tgt] = model.decode(*codes, geom(data, tgt, B))

if os.environ.get("AM_DUMP"):   # 只导出模型根高 + 各机身 GT 根高, 不渲染 (0902 根高抖动诊断)
    out_={}
    for tgt in TGTS:
        ri_=int(np.where(np.asarray(specs[tgt].parents)<0)[0][0])
        out_[f"pred_{tgt}"]=dpos[tgt][:,ri_,0].detach().cpu().numpy()
        out_[f"gt_{tgt}"]=torch.load(f"{C}/{tgt}_root_world.pt",mmap=True).float()[rows].numpy()[:,1]/100.0
    out_["gt_human"]=torch.load(f"{C}/human_smpl_root_world.pt",mmap=True).float()[rows].numpy()[:,1]/100.0
    out_["names"]=np.array([qq["name"]+"@"+str(qq.get("at",0)) for _,qq in picks]); out_["T"]=T
    np.savez(os.environ["AM_DUMP"],**out_); print("DUMPED",os.environ["AM_DUMP"]); sys.exit(0)

# 根轨迹: human 的世界根透传, 贴地由渲染端做
RWh = torch.load(f"{C}/human_smpl_root_world.pt", mmap=True).float()[rows].numpy() / 100.0
AX = np.array([[0.,1,0],[0,0,1],[1,0,0]])
# 根朝向: 用**解码出来的该目标自己的根旋转** —— LAFAN human_smpl 的 rest 约定特殊
# (骨骼局部 +Z), 它的全局根旋转搬到机器人身上会歪 90° (0809 立过案的 gauge 陷阱)。
def pred_traj(tgt):
    """Y3: 位移 = root 槽航向速度积分 (AM_ODO=pred), 每窗首帧为原点。"""
    ri = int(np.where(np.asarray(specs[tgt].parents) < 0)[0][0])
    slot = dpos[tgt][:, ri].detach().cpu().numpy()
    R = rot6d_to_matrix(dec[tgt][:, 0, :6]).cpu().numpy()
    yaw = np.arctan2(R[:, 0, 2], R[:, 2, 2])
    # Y5b: 水平 = 透传 (实测姿态可学部分只有 1.5cm, 没东西可学); 竖直 = 预测高度。
    return (RWh - RWh[0]) @ AX
USE_PRED_ODO = os.environ.get("AM_ODO") == "pred" and bool(cfg.get("root_pos"))

def root_R_of(tgt):
    return rot6d_to_matrix(dec[tgt][:, 0, :6]).cpu().numpy()
from scipy.spatial.transform import Rotation

def render_robot(tgt, xml):
    spec = specs[tgt]
    if tgt == "x2":
        dof = torch.as_tensor(np.asarray(x2.dof), dtype=torch.float32, device=dev)
        rest = torch.as_tensor(np.asarray(x2.rest), dtype=torch.float32, device=dev)
        sj = x2.qnames
    else:
        dof, rest = load_dof(C, tgt, device=dev)
        sj = [str(s) for s in np.load(f"{C}/{tgt}_dof.npz", allow_pickle=True)["slot_joints"]]
    q, _ = fit_angles(dec[tgt], spec, dof, rest=rest, clamp=True, method="global")
    q = q.cpu().numpy()
    sp = mj.MjSpec.from_file(xml)
    sky=sp.add_texture(); sky.name="_sky"; sky.type=mj.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin=mj.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1=[0.5,0.63,0.8]; sky.rgb2=[0.92,0.94,0.97]; sky.width=sky.height=256
    tx = sp.add_texture(); tx.name="_ck"; tx.type=mj.mjtTexture.mjTEXTURE_2D
    tx.builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER; tx.width=tx.height=300
    tx.rgb1=[0.22,0.24,0.28]; tx.rgb2=[0.55,0.58,0.62]
    mt=sp.add_material(); mt.name="_ckm"; mt.texrepeat=[10,10]
    mt.textures[mj.mjtTextureRole.mjTEXROLE_RGB]="_ck"
    sp.worldbody.add_geom(name="_floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[12,12,.1], material="_ckm")
    sp.worldbody.add_light(name="_sun", pos=[0,-1,3], dir=[0,.25,-1], diffuse=[.75,.75,.75], castshadow=False)
    for k in range(NM):
        _b=sp.worldbody.add_body(name=f"_m{k}",mocap=True,pos=[0,0,-50])
        _b.add_geom(name=f"_ms{k}",type=mj.mjtGeom.mjGEOM_SPHERE,size=[0.06,0,0],
                    rgba=[0.1,0.95,0.2,0.65],contype=0,conaffinity=0)
    sp.visual.global_.offheight=max(int(sp.visual.global_.offheight),H)
    sp.visual.global_.offwidth=max(int(sp.visual.global_.offwidth),W)
    sc = sp.compile(); d = mj.MjData(sc)
    adr={}
    for k,names in enumerate(sj):
        _lst = list(names) if isinstance(names,(list,tuple)) else [x for x in str(names).split(",") if x]
        for s_,jn in enumerate(_lst):
            jid=mj.mj_name2id(sc,mj.mjtObj.mjOBJ_JOINT,jn)
            if jid>=0 and sc.jnt_type[jid]==mj.mjtJoint.mjJNT_HINGE:
                adr[(k,s_)]=int(sc.jnt_qposadr[jid])
    ren=mj.Renderer(sc,H,W); cam=mj.MjvCamera()
    cam.distance,cam.elevation,cam.azimuth=2.9,-12,135
    sd_m=[int(sc.body_mocapid[mj.mj_name2id(sc,mj.mjtObj.mjOBJ_BODY,f"_m{k}")]) for k in range(NM)]
    gid_m=[mj.mj_name2id(sc,mj.mjtObj.mjOBJ_GEOM,f"_ms{k}") for k in range(NM)]
    jb=[mj.mj_name2id(sc,mj.mjtObj.mjOBJ_BODY,n) for n in spec.joint_names]
    if tgt=="x2":
        pr=np.zeros((spec.J,3),np.float32)
        for j in range(spec.J):
            p_=spec.parents[j]; pr[j]=(pr[p_] if p_>=0 else 0)+np.asarray(spec.rest_offsets[j])
        d_=float(-pr[:,1].min()); d_=d_/100.0 if d_>3 else d_       # cm 或 m 自适应
        thr_r=max(0.05*d_,0.03)
        lab_r=None
        import json as _j
        _tbl=_j.load(open("/home/arenalabs/tmp/sole_offsets3.json"))
        _ent=_tbl.get("agibot_x2",{}); _sm={}
        _names=list(spec.joint_names)
        for _side in ("a","b"):
            for _c in _ent.get(_side,[]):
                if _c["body"] in _names:
                    _sm[_names.index(_c["body"])]=np.asarray(_c["off_ardy"],np.float32); break
    else:
        _z=np.load(f"{LBL}/{tgt}_contact.npz",allow_pickle=True)
        thr_r=float(json.loads(str(_z["header"]))["thr_m"])
        lab_r=(_z["contact"][rows.numpy()]|(_z["gap"][rows.numpy()]<0)); del _z
    from contact_label import sole_offsets as _sole
    _sm = _sole(tgt, spec)          # {joint: off_ardy(m)}; X2/缺 MJCF -> {}
    RR = root_R_of(tgt)
    has_free = sc.njnt>0 and sc.jnt_type[0]==mj.mjtJoint.mjJNT_FREE
    TRJ = pred_traj(tgt) if USE_PRED_ODO else (RWh - RWh[0]) @ AX
    # 第一遍干跑: 每帧最低点 (不含绿球), 每窗取最小当常数地面偏移 ——
    # 逐帧贴地会把每一帧的最低点都摁到 0: 永远有脚"贴地", 腾空相被摁没 (lloyd 0901)
    zoff=np.zeros(B,np.float32)
    if has_free:
        zmins=np.zeros(B,np.float32)
        for t in range(B):
            d.qpos[:]=sc.qpos0
            d.qpos[0:2]=TRJ[t][:2]
            d.qpos[3:7]=Rotation.from_matrix(AX.T@RR[t]@AX).as_quat(scalar_first=True)
            for (k,s_),a in adr.items(): d.qpos[a]=q[t,k,s_]
            mj.mj_forward(sc,d)
            zmins[t]=min(d.geom_xpos[g][2] for g in range(sc.ngeom)
                         if sc.geom_bodyid[g]>0 and not (mj.mj_id2name(sc,mj.mjtObj.mjOBJ_GEOM,g) or "").startswith("_ms"))
        for wi in range(B//T):
            zoff[wi*T:(wi+1)*T]=zmins[wi*T:(wi+1)*T].min()
    F=[]; stats=[]
    for t in range(B):
        d.qpos[:]=sc.qpos0
        if has_free:
            pm = TRJ[t]
            d.qpos[0:2] = pm[:2]
            d.qpos[3:7]=Rotation.from_matrix(AX.T@RR[t]@AX).as_quat(scalar_first=True)
        for (k,s_),a in adr.items(): d.qpos[a]=q[t,k,s_]
        mj.mj_forward(sc,d)
        if has_free:
            if USE_PRED_ODO:
                ri_=int(np.where(np.asarray(spec.parents)<0)[0][0])
                d.qpos[2]=float(dpos[tgt][t,ri_,0])       # 模型自己的根高
            else:
                d.qpos[2]-=zoff[t]
            mj.mj_forward(sc,d)
        # 接触自感: 关节世界高 < 该机身自己的阈值 -> 绿球
        def _jz(j):
            z_ = d.xpos[jb[j]][2]
            if j in _sm:            # 脚底点: joint + R_world @ off_mj, off_mj = AX.T @ off_ardy
                z_ = z_ + float((d.xmat[jb[j]].reshape(3,3) @ (AX.T @ np.asarray(_sm[j])))[2])
            return z_
        # 连续读数 c = sigmoid((thr - h)/tau), tau=thr/3 —— 与训练的 cgeo 同一个函数。
        # 球色随 c 连续变: 红(悬空) -> 黄(临界) -> 绿(踩实); 只画离地 < 0.35m 的候选。
        tau_r = thr_r / 3.0
        cand = sorted((j for j in range(spec.J) if jb[j]>=0 and _jz(j) < 0.35),
                      key=_jz)[:NM]
        cvals = {j: 1.0/(1.0+np.exp(-(thr_r-_jz(j))/tau_r)) for j in cand}
        act=[j for j in cand if cvals[j] > 0.5]
        for k in range(NM):
            if k < len(cand):
                j = cand[k]; c = cvals[j]
                d.mocap_pos[sd_m[k]] = d.xpos[jb[j]]
                # 红(0) -> 黄(0.5) -> 绿(1)
                r_ = 1.0-max(0.0,(c-0.5)*2); g_ = min(1.0,c*2)
                sc.geom_rgba[gid_m[k]] = [r_, g_, 0.12, 0.6]
            else:
                d.mocap_pos[sd_m[k]] = [0,0,-50]
        mj.mj_forward(sc,d)
        # 左右脚数值 (脚底两关节, 缺则取最低两关节)
        _fj = sorted(_sm) if len(_sm)>=2 else cand[:2]
        vec = "  ".join("%s%+.2f"%(("L" if i==0 else "R"), cvals.get(j, 1.0/(1.0+np.exp(-(thr_r-_jz(j))/tau_r))))
                        for i,j in enumerate(_fj[:2]))
        if lab_r is not None:
            _rd=set(act); _lb=set(int(x) for x in np.flatnonzero(lab_r[t]))
            stats.append(("%s | touch %d/%d miss %d"%(vec,len(_rd&_lb),len(_lb),len(_lb-_rd)), len(_lb-_rd)==0))
        else:
            stats.append(("%s | self %d"%(vec,len(act)), True))
        # 固定机位: 每窗一个, 覆盖整条轨迹
        wi=t//T; seg=TRJ[wi*T:(wi+1)*T]
        if FOLLOW:
            if t%T==0: _sm_trj=_smooth(seg)
            c_=_sm_trj[t-wi*T]; cam.lookat[:]=[c_[0],c_[1],0.55]; cam.distance=3.2
        else:
            c_=seg.mean(0); span=float(np.linalg.norm(seg.max(0)-seg.min(0)))
            cam.lookat[:]=[c_[0],c_[1],0.55]; cam.distance=max(3.2,span*1.1+2.4)
        ren.update_scene(d,cam); F.append(ren.render().copy())
    ren.close()
    return F, stats

def render_human():
    spec=specs["human_smpl"]
    p=fk_pos_torch(data["human_smpl"]["pose"][...,:6],spec).cpu().numpy()/100.0  # 米, 根相对
    sp=mj.MjSpec.from_file  # 用空场景+胶囊
    spx=mj.MjSpec()
    tx=spx.add_texture(); tx.name="_ck"; tx.type=mj.mjtTexture.mjTEXTURE_2D
    tx.builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER; tx.width=tx.height=300
    tx.rgb1=[0.22,0.24,0.28]; tx.rgb2=[0.55,0.58,0.62]
    mt=spx.add_material(); mt.name="_ckm"; mt.texrepeat=[10,10]
    mt.textures[mj.mjtTextureRole.mjTEXROLE_RGB]="_ck"
    spx.worldbody.add_geom(name="_floor",type=mj.mjtGeom.mjGEOM_PLANE,size=[30,30,.1],material="_ckm")
    spx.worldbody.add_light(name="_sun",pos=[0,-1,3],dir=[0,.25,-1],diffuse=[.75,.75,.75],castshadow=False)
    caps=[]
    for j in range(spec.J):
        if spec.parents[j]<0: continue
        b=spx.worldbody.add_body(name=f"_c{j}",mocap=True)
        b.add_geom(name=f"_g{j}",type=mj.mjtGeom.mjGEOM_CAPSULE,size=[0.022,0.1,0],rgba=[0.9,0.85,0.6,1])
        caps.append(j)
    sc=spx.compile(); d=mj.MjData(sc)
    ren=mj.Renderer(sc,H,W); cam=mj.MjvCamera()
    cam.distance,cam.elevation,cam.azimuth=2.9,-12,135
    gid={j:mj.mj_name2id(sc,mj.mjtObj.mjOBJ_GEOM,f"_g{j}") for j in caps}
    mid={j:int(sc.body_mocapid[mj.mj_name2id(sc,mj.mjtObj.mjOBJ_BODY,f"_c{j}")]) for j in caps}
    F=[]
    for t in range(B):
        w=(p[t]@AX)  # ardy->mj Z-up, 根相对
        w[:,2]-=w[:,2].min()
        w[:,:2]+=((RWh[t]-RWh[0])@AX)[:2]
        for j in caps:
            a=w[spec.parents[j]]; b_=w[j]; c=(a+b_)/2; v=b_-a; L=np.linalg.norm(v)
            d.mocap_pos[mid[j]]=c
            if L>1e-6:
                zaxis=v/L
                d.mocap_quat[mid[j]]=Rotation.from_rotvec(
                    np.cross([0,0,1],zaxis)*np.arcsin(np.clip(np.linalg.norm(np.cross([0,0,1],zaxis)),-1,1))
                    /max(np.linalg.norm(np.cross([0,0,1],zaxis)),1e-9)).as_quat(scalar_first=True)
                sc.geom_size[gid[j]][1]=L/2
        mj.mj_forward(sc,d)
        cam.lookat[:]=[*(((RWh[t]-RWh[0])@AX)[:2]),0.6]
        ren.update_scene(d,cam); F.append(ren.render().copy())
    ren.close(); return F

def resolve_xml(emb):
    import importlib.util
    alias={"unitree_g1_29dof":"unitree_g1"}
    gp='/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/external/GMR/general_motion_retargeting/params.py'
    s2=importlib.util.spec_from_file_location("gp",gp); m2=importlib.util.module_from_spec(s2); s2.loader.exec_module(m2)
    return str(m2.ROBOT_XML_DICT[alias.get(emb,emb)])

def render_human_mesh():
    """SMPL-X 网格面板: AMASS human_smpl 的 local6d 就是 SMPL 序 (export_smpl_params 已验)。
    顶点在 genmo-gem 环境算 (rot 没有 smplx), painter 光栅化在本进程。"""
    import subprocess, tempfile, cv2
    from scipy.spatial.transform import Rotation as _R
    l6 = torch.load(f"{C_SMPL}/human_smpl_paired_local6d.pt", mmap=True)[rows].double()
    Rm = rot6d_to_matrix(l6.float()).numpy()
    aa = _R.from_matrix(Rm.reshape(-1,3,3)).as_rotvec().reshape(len(rows),22,3)
    tmp = tempfile.mkdtemp()
    np.savez(f"{tmp}/params.npz", global_orient=aa[:,0].astype(np.float32),
             body_pose=aa[:,1:22].reshape(len(rows),63).astype(np.float32),
             transl=(RWh-RWh[0]).astype(np.float32))
    code = f"""
import numpy as np, torch, smplx
z=np.load('{tmp}/params.npz')
m=smplx.create('/home/arenalabs/Downloads/SMPLX_NEUTRAL.npz', model_type='smplx', use_pca=False, batch_size=len(z['transl']))
with torch.no_grad():
    o=m(global_orient=torch.tensor(z['global_orient']), body_pose=torch.tensor(z['body_pose']),
        transl=torch.tensor(z['transl']))
np.savez('{tmp}/verts.npz', v=o.vertices.numpy().astype(np.float32), f=m.faces.astype(np.int32))
print('verts ok')
"""
    r = subprocess.run(["/home/arenalabs/miniconda3/envs/genmo-gem/bin/python","-c",code],
                       capture_output=True, text=True)
    assert "verts ok" in r.stdout, r.stderr[-800:]
    z = np.load(f"{tmp}/verts.npz"); verts, faces = z["v"], z["f"]
    light = np.array([0.35,0.75,0.55]); light/=np.linalg.norm(light)
    F=[]
    for t in range(len(rows)):
        wi=t//T; seg=verts[wi*T:(wi+1)*T].reshape(-1,3)
        if FOLLOW:
            if t%T==0:
                _cen=_smooth(verts[wi*T:(wi+1)*T].mean(1))
                _spn=max(float(np.abs(verts[wi*T+i]-_cen[i]).max()) for i in range(min(T,len(verts)-wi*T)))
            center=_cen[t-wi*T]; span=_spn
        else:
            center=seg.mean(0); span=float(np.abs(seg-center).max())
        scale=0.42*min(W,H)/max(span,1e-6)
        v=verts[t]-center
        pts=np.empty((len(v),2),np.int32)
        # AM_SMPL_UP=z: LAFAN 缓存的 SMPL 旋转是 Z-up (0902 实测), 画面竖轴用 z, 深度用 y
        _up=2 if os.environ.get("AM_SMPL_UP","y")=="z" else 1
        pts[:,0]=(W/2+v[:,0]*scale).astype(np.int32)
        pts[:,1]=(H*0.54-v[:,_up]*scale).astype(np.int32)
        depth=v[:,3-_up]
        fr=np.full((H,W,3),38,np.uint8)
        tri=v[faces]
        n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); n/=np.linalg.norm(n,axis=1,keepdims=True)+1e-9
        shade=np.clip(n@light,0,1)
        order=np.argsort(depth[faces].mean(1))
        for fi in order:
            c=60+165*shade[fi]
            cv2.fillConvexPoly(fr,pts[faces[fi]],(int(c*0.75),int(c*0.85),int(c)))
        F.append(fr)
    return F

# AM_HUMAN=stick: 用与模型同一套 fk 的胶囊人 (LAFAN 缓存的 SMPL 根旋转与世界系不一致, 网格面板会翻转)
panels={"smpl":render_human() if os.environ.get("AM_HUMAN","mesh")=="stick" else render_human_mesh()}; pstats={}
for tgt in TGTS: panels[tgt],pstats[tgt]=render_robot(tgt,resolve_xml(tgt))
panels["x2"],pstats["x2"]=render_robot("x2",X2_XML)
f_=ImageFont.truetype(FONT,20)
ORDER=["smpl"]+TGTS+["x2"]
for i,(pat,qq) in enumerate(picks):
    sl=slice(i*T,(i+1)*T)
    comp=[]
    for t in range(sl.start,sl.stop):
        row=[]
        for nm in ORDER:
            im=Image.fromarray(panels[nm][t]); dr=ImageDraw.Draw(im)
            dr.rectangle([0,0,W,54],fill=(0,0,0)); dr.text((8,2),nm,font=f_,fill=(255,255,255))
            if nm in pstats:
                txt,ok_=pstats[nm][t]
                dr.text((8,28),txt,font=f_,fill=(120,255,140) if ok_ else (255,120,120))
            row.append(np.array(im))
        comp.append(np.concatenate(row,axis=1))
    h_,w_=comp[0].shape[:2]; h_-=h_%2; w_-=w_%2
    out=f"{OUT}/crossrig_{re.sub('[^A-Za-z0-9]+','_',qq['name'])[:36]}_at{qq.get('at',0)}.mp4"
    pr=subprocess.Popen(["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24",
        "-s",f"{w_}x{h_}","-r","30","-i","-","-c:v","libx264","-crf","22","-pix_fmt","yuv420p",out],stdin=subprocess.PIPE)
    for fme in comp+comp: pr.stdin.write(np.ascontiguousarray(fme[:h_,:w_],np.uint8).tobytes())
    pr.stdin.close(); assert pr.wait()==0
    print("WROTE",out)
