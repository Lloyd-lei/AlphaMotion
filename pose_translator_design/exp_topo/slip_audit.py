"""脚滑审计 (lloyd 0901): 脚步运动和身体位移一致吗? 滑动多严重? 几何读出的接触标签能训 WBC 吗?

对每个目标机身, 五条臂同一把尺 (脚底点世界系水平速度, 只在"接触中"的帧上量):
  M_h  模型姿态 + 人体根轨迹透传 (= 视频里放的那条)
  M_g  模型姿态 + GMR 机器人自己的根轨迹透传 (机器人尺寸的位移)
  M_a  模型姿态 + 接触反推里程计 (踩地脚钉死, 身体跟着脚走)
  GT   GMR 重定向姿态 + GMR 根轨迹 (数据地板: 训练数据自己滑多少)
  H    人体源 + 自己的根轨迹 + 自己的标签 (mocap 地板)
接触口径: 机器人 = 视频同一函数 sigmoid((thr-h)/tau)>0.5 (thr 取该机身标签阈值, 脚底点),
另与 GMR 数据标签 (cl_v3) 比一致性。
用法: AM_CACHE=... AM_LBL=... slip_audit.py <ckpt> <out.json> [fam:n ...]  默认 walk:4 run:3 jump:3 dance:3 turn:2"""
import os, sys, json, re
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
from scipy.spatial.transform import Rotation
from common import SkeletonSpec, fk_global, matrix_to_rot6d, rot6d_to_matrix
from dual_common import build_model, geom, static_cond, fk_pos_torch
from manifold_project import fit_angles, load_dof
from semantics import build_name_embeddings, embed_matrix
from contact_label import family, sole_offsets, V_THR, FPS
sys.path.insert(0, "/home/arenalabs/Desktop/AlphaMotion/src")

RUNS = "/var/tmp/alphamotion/topo_exp/runs"
C = os.environ.get("AM_CACHE", "/var/tmp/alphamotion/topo_exp/cache_amass_lconv")
LBL = os.environ.get("AM_LBL", "/home/arenalabs/tmp/cl_v3/amass_lconv")
X2_XML = "/var/tmp/alphamotion/robot_models/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
CK = sys.argv[1] if sys.argv[1].startswith("/") else f"{RUNS}/{sys.argv[1]}"
OUT = sys.argv[2]
FAMS = dict(a.split(":") for a in sys.argv[3:]) or {"walk": 4, "run": 3, "jump": 3, "dance": 3, "turn": 2}
FAMS = {k: int(v) for k, v in FAMS.items()}
TGTS = os.environ.get("AM_TGTS", "unitree_g1_29dof,pnd_adam_lite,fourier_gr3").split(",")
SKIP_X2 = os.environ.get("AM_SKIP_X2") == "1"
T = 150
dev = "cuda"; torch.set_grad_enabled(False)
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])

# ---------------------------------------------------------------- 选片
seqs = json.load(open(f"{C}/sequences.json"))
rng = np.random.RandomState(3)
picks = []
for fam, n in FAMS.items():
    m = [q for q in seqs if family(q["name"]) == fam and q["len"] >= T + 20]
    for q in [m[i] for i in rng.choice(len(m), min(n, len(m)), replace=False)]:
        picks.append((fam, q))
for fam, q in picks:
    print(f"[pick] {fam}: {q['name'][:70]} len={q['len']}")
rows = torch.cat([torch.arange(q["start"] + q["len"] // 2, q["start"] + q["len"] // 2 + T) for _, q in picks])
B = len(rows); NW = B // T
fam_of_win = [f for f, _ in picks]

def load_spec_npz(name):
    d = np.load(f"{C}/{name}_spec.npz", allow_pickle=True)
    return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], [str(x) for x in d["joint_names"]])

cfg = json.load(open(f"{CK}/config.json"))
names_all, specs = [], {}
for n in ["human_smpl"] + TGTS:
    specs[n] = load_spec_npz(n); names_all += specs[n].joint_names
from alphamotion.embodiment import registry as _reg
x2 = _reg.load_from_xml(X2_XML, "agibot_x2")
specs["x2"] = SkeletonSpec(x2.spec.name, np.asarray(x2.spec.parents), np.asarray(x2.spec.rest_offsets), list(x2.spec.joint_names))
names_all += specs["x2"].joint_names
lut = build_name_embeddings(names_all, device=dev, encoder=cfg.get("sem_encoder", "siglip"))

def entry(n):
    spec = specs[n]
    if n == "x2":
        pr = np.zeros((spec.J, 3), np.float32)
        for j in range(spec.J):
            p_ = spec.parents[j]; pr[j] = (pr[p_] if p_ >= 0 else 0) + np.asarray(spec.rest_offsets[j])
        reach = float(np.linalg.norm(pr, axis=1).max())
        e = {"spec": spec, "st": static_cond(spec, dev), "reach": reach,
             "sem": embed_matrix(spec.joint_names, lut).to(dev),
             "dof": torch.as_tensor(np.asarray(x2.dof), dtype=torch.float32, device=dev)}
        from dof_head import DofKin
        e["kin"] = DofKin(spec, e["dof"], torch.as_tensor(np.asarray(x2.rest), dtype=torch.float32, device=dev), device=dev)
    else:
        l6 = torch.load(f"{C}/{n}_paired_local6d.pt", mmap=True)[rows].float().to(dev)
        gR, gp = fk_global(l6, spec)
        reach = float(gp.norm(dim=-1).max())
        e = {"spec": spec, "st": static_cond(spec, dev), "reach": reach,
             "sem": embed_matrix(spec.joint_names, lut).to(dev),
             "pose": torch.cat([matrix_to_rot6d(gR), gp / reach], -1), "gR": gR}
        dz = f"{C}/{n}_dof.npz"
        if os.path.exists(dz):
            from dof_head import build_kin
            e["dof"] = torch.as_tensor(np.load(dz)["dof"], dtype=torch.float32, device=dev)
            e["kin"] = build_kin(C, n, spec, device=dev)
    return e

data = {n: entry(n) for n in ["human_smpl"] + TGTS + ["x2"]}
model = build_model(cfg, dev)
model.load_state_dict(torch.load(f"{CK}/model.pt", map_location=dev)); model.eval()
con = None
if cfg.get("contact_chan"):
    z = np.load(f"{LBL}/human_smpl_contact.npz", allow_pickle=True)
    h_ = json.loads(str(z["header"])); thr_, D_ = h_["thr_m"], h_["D_used_m"]
    s_ = thr_ - z["gap"][rows.numpy()].astype(np.float32)
    con = torch.as_tensor((np.sign(s_) * np.log1p(np.abs(s_) / thr_) / np.log1p(D_ / thr_)).astype(np.float32), device=dev)
    H_LAB = z["contact"][rows.numpy()]
# ---------------------------------------------------------------- 0901 诊断变体 (AM_VARIANT)
#  quant  : 正常 (量化码)                 —— 基线
#  cont   : 跳过 FSQ 取整, 连续特征直接解码 —— 差异 = 量化噪声
#  smoothK: 量化前特征窗内 K 帧滑动平均再取整 —— 差异 = 编码器逐帧时序噪声
#  nocon  : 输入接触通道置 NaN (无接触信息)  —— 接触通道对水平脚步的贡献
#  self   : 机器人自编码 (G1 姿态+G1 标签 -> G1), 不经人体翻译 —— 码本精度地板
VAR = os.environ.get("AM_VARIANT", "quant")
def _con_of(name):
    z = np.load(f"{LBL}/{name}_contact.npz", allow_pickle=True)
    h_ = json.loads(str(z["header"])); thr_, D_ = h_["thr_m"], h_["D_used_m"]
    s_ = thr_ - z["gap"][rows.numpy()].astype(np.float32)
    return torch.as_tensor((np.sign(s_) * np.log1p(np.abs(s_) / thr_) / np.log1p(D_ / thr_)).astype(np.float32), device=dev)
def _smooth(f, k):
    f = f.clone(); out = f.clone()
    for w in range(NW):
        seg = f[w*T:(w+1)*T]
        pad = torch.cat([seg[:1].expand(k//2, *seg.shape[1:]), seg, seg[-1:].expand(k//2, *seg.shape[1:])], 0)
        out[w*T:(w+1)*T] = pad.unfold(0, k, 1).mean(-1)
    return out
def _codes_from(src, con_src):
    gsrc = geom(data, src, B)
    if VAR == "quant" or VAR == "self":
        return model.encode(data[src]["pose"], gsrc, con=con_src)
    if VAR == "nocon":
        return model.encode(data[src]["pose"], gsrc, con=torch.full_like(con_src, float("nan")))
    fp, fr = model.encode_features(data[src]["pose"], gsrc, con=con_src)
    if VAR == "cont":
        zp = model.fsqp.proj_out(torch.tanh(fp)); zr = (model.fsqp if model.single_codebook else model.fsqr).proj_out(torch.tanh(fr))
        return zp, zr
    k = int(VAR.replace("smooth", ""))
    ip, ir = model.ints_from_features(_smooth(fp, k), _smooth(fr, k))
    return model.fsqp.from_ints(ip), (model.fsqp if model.single_codebook else model.fsqr).from_ints(ir)
dec, dpos = {}, {}
with torch.no_grad():
    if VAR == "self":
        for tgt in TGTS + ["x2"]:
            if tgt == "x2": continue
            dec[tgt], dpos[tgt] = model.decode(*_codes_from(tgt, _con_of(tgt)), geom(data, tgt, B))
    else:
        codes = _codes_from("human_smpl", con)
        for tgt in TGTS + ["x2"]:
            dec[tgt], dpos[tgt] = model.decode(*codes, geom(data, tgt, B))

RWh = torch.load(f"{C}/human_smpl_root_world.pt", mmap=True).float()[rows].numpy() / 100.0
def win_rel(P):
    """每窗首帧为原点的轨迹 (mj Z-up)."""
    P = P @ AX
    out = P.copy()
    for w in range(NW):
        out[w*T:(w+1)*T] -= P[w*T]
    return out
TRJ_h = win_rel(RWh)

def resolve_xml(emb):
    import importlib.util
    alias = {"unitree_g1_29dof": "unitree_g1"}
    gp = '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/external/GMR/general_motion_retargeting/params.py'
    s2 = importlib.util.spec_from_file_location("gp", gp); m2 = importlib.util.module_from_spec(s2); s2.loader.exec_module(m2)
    return str(m2.ROBOT_XML_DICT[alias.get(emb, emb)])

# ---------------------------------------------------------------- 尺子
def seg_runs(mask):
    """bool[T] -> [(s,e)] 连续 True 段 (e 不含)."""
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None: s = i
        if not v and s is not None: out.append((s, i)); s = None
    if s is not None: out.append((s, len(mask)))
    return out

def foot_metrics(F, con, trj, fam_w):
    """F: [B,2,3] 脚底点世界 (m); con: [B,2] bool; trj: [B,3] 根世界.
    每窗算: 接触中脚水平速度, 站立段漂移, 反推里程 vs 透传里程."""
    per_fam = {}
    for w in range(NW):
        sl = slice(w*T, (w+1)*T)
        Fw, cw, tw = F[sl], con[sl], trj[sl]
        v_c = []; drift = []; n_c = 0; n_fast = 0
        for f in range(2):
            v = np.linalg.norm(np.diff(Fw[:, f, :2], axis=0), axis=1) * FPS   # m/s, 帧 t->t+1
            both = cw[:-1, f] & cw[1:, f]
            v_c += list(v[both]); n_c += int(both.sum()); n_fast += int((v[both] > V_THR).sum())
            for s, e in seg_runs(cw[:, f]):
                if e - s >= 5:
                    drift.append(float(np.linalg.norm(Fw[e-1, f, :2] - Fw[s, f, :2])))
        # 接触反推里程: 有脚踩地就把身体拉回来 (脚钉死), 没脚踩地用透传增量
        ra = np.zeros((T, 2)); ra[0] = tw[0, :2]; anchor = None
        for t in range(T - 1):
            cands = [f for f in range(2) if cw[t, f] and cw[t+1, f]]
            if cands:
                f = anchor if anchor in cands else cands[0]; anchor = f
                rel0 = Fw[t, f, :2] - tw[t, :2]; rel1 = Fw[t+1, f, :2] - tw[t+1, :2]
                ra[t+1] = ra[t] - (rel1 - rel0)
            else:
                anchor = None; ra[t+1] = ra[t] + (tw[t+1, :2] - tw[t, :2])
        path_pass = float(np.linalg.norm(np.diff(tw[:, :2], axis=0), axis=1).sum())
        path_anch = float(np.linalg.norm(np.diff(ra, axis=0), axis=1).sum())
        disp_pass = float(np.linalg.norm(tw[-1, :2] - tw[0, :2]))
        disp_anch = float(np.linalg.norm(ra[-1] - ra[0]))
        gap_end = float(np.linalg.norm(ra[-1] - tw[-1, :2]))
        flight = float((~cw.any(1)).mean())
        d = per_fam.setdefault(fam_w[w], dict(v=[], drift=[], n_c=0, n_fast=0, path_pass=[], path_anch=[],
                                              disp_pass=[], disp_anch=[], gap_end=[], flight=[], contact_frac=[]))
        d["v"] += v_c; d["drift"] += drift; d["n_c"] += n_c; d["n_fast"] += n_fast
        d["path_pass"].append(path_pass); d["path_anch"].append(path_anch)
        d["disp_pass"].append(disp_pass); d["disp_anch"].append(disp_anch); d["gap_end"].append(gap_end)
        d["flight"].append(flight); d["contact_frac"].append(float(cw.mean()))
    rep = {}
    for k, d in per_fam.items():
        v = np.array(d["v"]) * 100; dr = np.array(d["drift"]) * 100
        rep[k] = dict(
            slip_v_med_cms=float(np.median(v)) if len(v) else None,
            slip_v_p95_cms=float(np.percentile(v, 95)) if len(v) else None,
            slip_frac_over_vthr=float(d["n_fast"] / max(d["n_c"], 1)),
            stance_drift_med_cm=float(np.median(dr)) if len(dr) else None,
            stance_drift_p95_cm=float(np.percentile(dr, 95)) if len(dr) else None,
            n_stance=len(dr),
            disp_pass_m=float(np.mean(d["disp_pass"])), disp_anch_m=float(np.mean(d["disp_anch"])),
            anch_over_pass=float(np.sum(d["disp_anch"]) / max(np.sum(d["disp_pass"]), 1e-6)),
            end_gap_cm=float(np.mean(d["gap_end"]) * 100),
            end_gap_over_disp=float(np.sum(d["gap_end"]) / max(np.sum(d["disp_pass"]), 1e-6)),
            flight_frac=float(np.mean(d["flight"])), contact_frac=float(np.mean(d["contact_frac"])),
        )
    return rep

# ---------------------------------------------------------------- 机器人: MuJoCo 正解取脚底点
def robot_arms(tgt, xml):
    spec = specs[tgt]
    if tgt == "x2":
        dof = torch.as_tensor(np.asarray(x2.dof), dtype=torch.float32, device=dev)
        rest = torch.as_tensor(np.asarray(x2.rest), dtype=torch.float32, device=dev)
        sj = x2.qnames
    else:
        dof, rest = load_dof(C, tgt, device=dev)
        sj = [str(s) for s in np.load(f"{C}/{tgt}_dof.npz", allow_pickle=True)["slot_joints"]]
    sc = mj.MjModel.from_xml_path(xml); d = mj.MjData(sc)
    adr = {}
    for k, names in enumerate(sj):
        lst = list(names) if isinstance(names, (list, tuple)) else [x for x in str(names).split(",") if x]
        for s_, jn in enumerate(lst):
            jid = mj.mj_name2id(sc, mj.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0 and sc.jnt_type[jid] == mj.mjtJoint.mjJNT_HINGE:
                adr[(k, s_)] = int(sc.jnt_qposadr[jid])
    jb = [mj.mj_name2id(sc, mj.mjtObj.mjOBJ_BODY, n) for n in spec.joint_names]
    sm = sole_offsets(tgt, spec)
    if tgt == "x2":
        tbl = json.load(open("/home/arenalabs/tmp/sole_offsets3.json"))["agibot_x2"]; sm = {}
        for side in ("a", "b"):
            for c in tbl[side]:
                if c["body"] in spec.joint_names: sm[spec.joint_names.index(c["body"])] = np.asarray(c["off_ardy"], np.float32); break
    assert len(sm) == 2, (tgt, sm)
    feet = sorted(sm)   # 两个脚底点关节 (左右按名字排)
    feet = sorted(feet, key=lambda j: 0 if re.search(r"left|_l\b|^l_", spec.joint_names[j], re.I) else 1)
    ri = int(np.where(np.asarray(spec.parents) < 0)[0][0])
    if tgt != "x2":
        z = np.load(f"{LBL}/{tgt}_contact.npz", allow_pickle=True)
        thr = float(json.loads(str(z["header"]))["thr_m"])
        lab_gt = (z["contact"][rows.numpy()] | (z["gap"][rows.numpy()] < 0))[:, feet]; del z
    else:
        pr = np.zeros((spec.J, 3), np.float32)
        for j in range(spec.J):
            p_ = spec.parents[j]; pr[j] = (pr[p_] if p_ >= 0 else 0) + np.asarray(spec.rest_offsets[j])
        d_ = float(-pr[:, 1].min()); d_ = d_ / 100.0 if d_ > 3 else d_
        thr = max(0.05 * d_, 0.03); lab_gt = None
    tau = thr / 3

    def run(q, RR, xy, zmode, zval=None):
        """-> F [B,2,3] 脚底世界, c [B,2] 读出接触, root [B,3]"""
        F = np.zeros((B, 2, 3)); root = np.zeros((B, 3)); zmins = np.zeros(B)
        for t in range(B):
            d.qpos[:] = sc.qpos0; d.qpos[0:2] = xy[t]
            d.qpos[3:7] = Rotation.from_matrix(AX.T @ RR[t] @ AX).as_quat(scalar_first=True)
            for (k, s_), a in adr.items(): d.qpos[a] = q[t, k, s_]
            mj.mj_forward(sc, d)
            for i, j in enumerate(feet):
                F[t, i] = d.xpos[jb[j]] + d.xmat[jb[j]].reshape(3, 3) @ (AX.T @ sm[j])
            zmins[t] = F[t, :, 2].min(); root[t] = d.qpos[0:3]
        if zmode == "pred":
            dz = zval - root[:, 2]
        else:   # 每窗常数贴地 (脚底最低点 = 0), 与 crossrig 同规则
            dz = np.zeros(B)
            for w in range(NW): dz[w*T:(w+1)*T] = -zmins[w*T:(w+1)*T].min()
        F[:, :, 2] += dz[:, None]; root[:, 2] += dz
        c = 1 / (1 + np.exp(-(thr - F[:, :, 2]) / tau)) > 0.5
        return F, c, root

    out = {}
    qM = fit_angles(dec[tgt], spec, dof, rest=rest, clamp=True, method="global")[0].cpu().numpy()
    RM = rot6d_to_matrix(dec[tgt][:, 0, :6]).cpu().numpy()
    zM = dpos[tgt][:, ri, 0].cpu().numpy()
    F, c, root = run(qM, RM, TRJ_h[:, :2], "pred", zM)
    out["M_h"] = foot_metrics(F, c, root, fam_of_win)
    ARR[f"{tgt}/M_h"] = (F, c, root, RM); cM = c
    if tgt != "x2":
        RWr = torch.load(f"{C}/{tgt}_root_world.pt", mmap=True).float()[rows].numpy() / 100.0
        TRJ_r = win_rel(RWr)
        F, c, root = run(qM, RM, TRJ_r[:, :2], "pred", zM)
        out["M_g"] = foot_metrics(F, c, root, fam_of_win)
        # 数据地板: GMR 姿态 + GMR 根 (高度=GMR 自己的, 每窗贴地)
        gR = data[tgt]["gR"]
        qG = fit_angles(matrix_to_rot6d(gR), spec, dof, rest=rest, clamp=True, method="global")[0].cpu().numpy()
        RG = gR[:, ri].cpu().numpy()
        F, c, root = run(qG, RG, TRJ_r[:, :2], "floor")
        out["GT"] = foot_metrics(F, c, root, fam_of_win)
        ARR[f"{tgt}/GT"] = (F, c, root, RG); ARR[f"{tgt}/lab_gt"] = lab_gt
        out["GT_lab"] = foot_metrics(F, lab_gt, root, fam_of_win)   # 同姿态, 用 cl_v3 标签
        # 标签一致性: 模型读出 vs GMR 数据标签 (逐帧逐脚)
        a, b = cM.ravel(), lab_gt.ravel()
        out["label_vs_gt"] = dict(agree=float((a == b).mean()),
                                  prec=float((a & b).sum() / max(a.sum(), 1)), rec=float((a & b).sum() / max(b.sum(), 1)),
                                  model_contact_frac=float(a.mean()), gt_contact_frac=float(b.mean()))
        # 位移比: GMR 根位移 / 人体根位移 (每窗)
        out["gmr_over_human_disp"] = float(np.sum([np.linalg.norm(TRJ_r[w*T+T-1, :2] - TRJ_r[w*T, :2]) for w in range(NW)]) /
                                           max(np.sum([np.linalg.norm(TRJ_h[w*T+T-1, :2] - TRJ_h[w*T, :2]) for w in range(NW)]), 1e-6))
    out["thr_m"] = thr
    return out

# ---------------------------------------------------------------- 人体源
def human_arm():
    spec = specs["human_smpl"]
    p = fk_pos_torch(data["human_smpl"]["pose"][..., :6], spec).cpu().numpy() / 100.0
    jn = spec.joint_names
    LT, RT, LA, RA = [jn.index(x) for x in ("L_Toe", "R_Toe", "L_Ankle", "R_Ankle")]
    F = np.zeros((B, 2, 3))
    for t in range(B):
        w = p[t] @ AX + TRJ_h[t]
        F[t, 0] = w[LT]; F[t, 1] = w[RT]
    c = np.stack([H_LAB[:, LT] | H_LAB[:, LA], H_LAB[:, RT] | H_LAB[:, RA]], 1)
    return foot_metrics(F, c, TRJ_h, fam_of_win)

ARR = {}
res = {"picks": [(f, q["name"]) for f, q in picks], "human": human_arm()}
for tgt in TGTS:
    res[tgt] = robot_arms(tgt, resolve_xml(tgt)); print("done", tgt, flush=True)
if not SKIP_X2: res["x2"] = robot_arms("x2", X2_XML)
json.dump(res, open(OUT, "w"), indent=1)
np.savez(OUT.replace(".json", "_arrays.npz"), fam=np.array(fam_of_win), TRJ_h=TRJ_h, H_LAB=H_LAB,
         **{k.replace("/", "__") + "__" + n: v for k, tup in ARR.items() if isinstance(tup, tuple) for n, v in zip(("F", "c", "root", "R"), tup)},
         **{k.replace("/", "__"): v for k, v in ARR.items() if not isinstance(v, tuple)})

# ---------------------------------------------------------------- 表
fams = list(FAMS)
def row(name, rep):
    cells = []
    for f in fams:
        r = rep.get(f)
        if not r or r["slip_v_med_cms"] is None: cells.append("      -       "); continue
        cells.append(f"{r['slip_v_med_cms']:4.1f}/{r['slip_v_p95_cms']:5.1f} {r['slip_frac_over_vthr']*100:3.0f}% {r['stance_drift_med_cm']:4.1f}")
    print(f"{name:26s}| " + " | ".join(cells))
print("\n接触中脚底水平速度 中位/p95 cm/s, >10cm/s 占比, 站立段漂移中位 cm")
print(f"{'arm':26s}| " + " | ".join(f"{f:^22s}" for f in fams))
row("human (mocap)", res["human"])
for tgt in TGTS + ([] if SKIP_X2 else ["x2"]):
    for arm in ("M_h", "M_g", "M_a", "GT", "GT_lab"):
        if arm in res[tgt]: row(f"{tgt[:12]} {arm}", res[tgt][arm])
print("\n身体位移 vs 脚步反推位移 (M_h 臂): 反推/透传 比, 窗末差 cm, 差/位移")
for tgt in TGTS + ([] if SKIP_X2 else ["x2"]):
    for f in fams:
        r = res[tgt]["M_h"].get(f)
        if r: print(f"{tgt[:12]:12s} {f:6s} pass={r['disp_pass_m']:.2f}m anch={r['disp_anch_m']:.2f}m ratio={r['anch_over_pass']:.2f} end_gap={r['end_gap_cm']:.0f}cm ({r['end_gap_over_disp']*100:.0f}%) flight={r['flight_frac']*100:.0f}%")
    if "gmr_over_human_disp" in res[tgt]:
        print(f"{tgt[:12]:12s} GMR根位移/人体根位移 = {res[tgt]['gmr_over_human_disp']:.2f}   读出标签 vs GMR标签: {res[tgt]['label_vs_gt']}")
