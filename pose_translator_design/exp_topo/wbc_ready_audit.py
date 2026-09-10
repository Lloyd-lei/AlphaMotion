"""WBC 可训性系统验收 (lloyd 0901): 转换出来的数据够不够格拿去训 tracking WBC?
四张考卷, 每条测试片段逐条打分, 全程在 **语料外** 片段上 (Y5b 只训 LAFAN; 本集 = AMASS heldout 600 条):
  1 姿态准不准   vs GMR 配对真值: 全局旋转 geodesic 角 (°), 手腕/脚底 根系位置误差 (cm), 根高误差 (cm)
  2 脚滑不滑     真站立相 (GMR 速度门标签) 内脚底水平速度 中位/p95 (cm/s), 每步站立漂移 (cm)
                 三种根轨迹臂: M_h 人体根透传 | M_g GMR 根 | M_a 接触锚定里程计 (只用模型自己的读出)
  3 接触标签对不对 模型几何读出 vs GMR 速度门标签: 逐帧 F1, 触地时机误差 (帧), 触地位置误差 (cm), 漏检率
  4 物理自洽     穿地占比 (脚底低于标定地板 thr), 标签踩地但脚悬空占比
高度口径: 脚 = 脚底点离标定地板 (与 contact_label 的 gap 同口径: 脚用 sole_offsets, 地板 = 语料 h0), 模型根高 = 预测 gap_root, GT 根高 = 标签 gap_root。
尺子 = GT 臂 (GMR 姿态+GMR 根+同一套算法) 的逐片段分布: 每项指标 "通过" = 不劣于 GT 臂的 p95。
用法: AM_CACHE=<tfix 缓存> AM_LBL=<标签目录> AM_SET=<heldout sequences.json> wbc_ready_audit.py <ckpt> <out.json>"""
import os, sys, json, re, time
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
from scipy.spatial.transform import Rotation
from common import SkeletonSpec, fk_global, matrix_to_rot6d, rot6d_to_matrix
from dual_common import build_model, geom, static_cond
from manifold_project import fit_angles, load_dof
from semantics import build_name_embeddings, embed_matrix
from contact_label import family, sole_offsets, V_THR, FPS

RUNS = "/var/tmp/alphamotion/topo_exp/runs"
C = os.environ["AM_CACHE"]; LBL = os.environ["AM_LBL"]
SET = os.environ.get("AM_SET", "/var/tmp/alphamotion/topo_exp/cache_amass_heldout/sequences.json")
CK = sys.argv[1] if sys.argv[1].startswith("/") else f"{RUNS}/{sys.argv[1]}"
OUT = sys.argv[2]
TGTS = os.environ.get("AM_TGTS", "unitree_g1_29dof,pnd_adam_lite,fourier_gr3").split(",")
MAXC = int(os.environ.get("AM_MAXCLIPS", "0"))
dev = os.environ.get("AM_DEV", "cuda"); torch.set_grad_enabled(False)
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])
TELE = 0.3        # 根单帧跳 >0.3m = 数据拼接/瞬移, 该帧不计 (与 slip_audit 同)
ONSET_WIN = 15    # 触地时机匹配窗 (帧, 0.5s)

# ---------------------------------------------------------------- 测试集: heldout 片段名 -> tfix 缓存里的行
seqs_all = {q["name"]: q for q in json.load(open(f"{C}/sequences.json"))}
names = [q["name"] for q in json.load(open(SET))]
clips = [seqs_all[n] for n in names if n in seqs_all]
if MAXC: clips = clips[:MAXC]
ONLY = os.environ.get("AM_ONLY")          # 0902 WBC 试训导出: 只跑这份名单里的片段 (一行一个名字)
if ONLY:
    # 0903: 名单行可写 name@start:len (片段内偏移, 帧) 只导出一个窗口 (地面动作片段几分钟长, 只要其中一段)
    keep = {}
    for l in open(ONLY):
        l = l.strip()
        if not l: continue
        n, _, w = l.partition("@"); keep[n] = w
    clips2 = []
    for q in clips:
        if q["name"] not in keep: continue
        w = keep[q["name"]]
        if w:
            a, _, ln = w.partition(":"); a = int(a); ln = int(ln) if ln else q["len"] - a
            assert a + ln <= q["len"], f"{q['name']}@{a}:{ln} 超出片段长度 {q['len']}"
            q = dict(q, start=q["start"] + a, len=ln, name=f"{q['name']}_at{a}")
        clips2.append(q)
    clips = clips2
EXP = os.environ.get("AM_EXPORT")         # 0902: 导出 A(模型)/B(GMR 原生) 两臂 qpos csv (root xyz, quat xyzw, 关节 MJCF 顺序) 给 mjlab
if EXP: os.makedirs(EXP, exist_ok=True)
rows = torch.cat([torch.arange(q["start"], q["start"] + q["len"]) for q in clips])
B = len(rows); print(f"[set] {len(clips)} clips, {B} rows", flush=True)
cl_sl = []; o = 0
for q in clips: cl_sl.append(slice(o, o + q["len"])); o += q["len"]
def dataset_of(n): return n.split("__")[0]

def load_spec_npz(name):
    d = np.load(f"{C}/{name}_spec.npz", allow_pickle=True)
    return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], [str(x) for x in d["joint_names"]])
cfg = json.load(open(f"{CK}/config.json"))
names_all, specs = [], {}
for n in ["human_smpl"] + TGTS:
    specs[n] = load_spec_npz(n); names_all += specs[n].joint_names
lut = build_name_embeddings(names_all, device=dev, encoder=cfg.get("sem_encoder", "siglip"))
def entry(n):
    spec = specs[n]
    l6 = torch.load(f"{C}/{n}_paired_local6d.pt", mmap=True)[rows].float().to(dev)
    gR, gp = fk_global(l6, spec); reach = float(gp.norm(dim=-1).max())
    e = {"spec": spec, "st": static_cond(spec, dev), "reach": reach, "sem": embed_matrix(spec.joint_names, lut).to(dev),
         "pose": torch.cat([matrix_to_rot6d(gR), gp / reach], -1), "gR": gR}
    dz = f"{C}/{n}_dof.npz"
    if os.path.exists(dz):
        from dof_head import build_kin
        e["dof"] = torch.as_tensor(np.load(dz)["dof"], dtype=torch.float32, device=dev)
        e["kin"] = build_kin(C, n, spec, device=dev)
    return e
data = {n: entry(n) for n in ["human_smpl"] + TGTS}
from tdec import wrap_if_tdec, decode_clip_windows
model = wrap_if_tdec(build_model(cfg, dev), cfg); model.load_state_dict(torch.load(f"{CK}/model.pt", map_location=dev)); model.eval()
TK = int(cfg.get("tdec", 0) or 0)
con = None
if cfg.get("contact_chan"):
    z = np.load(f"{LBL}/human_smpl_contact.npz", allow_pickle=True)
    h_ = json.loads(str(z["header"])); thr_, D_ = h_["thr_m"], h_["D_used_m"]
    s_ = thr_ - z["gap"][rows.numpy()].astype(np.float32)
    con = torch.as_tensor((np.sign(s_) * np.log1p(np.abs(s_) / thr_) / np.log1p(D_ / thr_)).astype(np.float32), device=dev)
    if os.environ.get("AM_NOCON"): con = torch.full_like(con, float("nan")); print("[con] AM_NOCON: 接触通道全 NaN (对照臂: 不给标签)")
BS = int(os.environ.get("AM_BS", "4096"))
def batched(fn, n, cpu=False):
    """cpu=True: 每批算完立刻挪 CPU 再拼 (0902: 全集码 zp/zr 各 ~12GB, 拼接瞬时 36GB, GPU 共享时 OOM); 数值逐比特不变"""
    outs = []
    for i in range(0, n, BS):
        o = fn(slice(i, min(i + BS, n)))
        outs.append((tuple(x.cpu() for x in o) if isinstance(o, tuple) else o.cpu()) if cpu else o)
    return tuple(torch.cat([o[k] for o in outs]) for k in range(len(outs[0]))) if isinstance(outs[0], tuple) else torch.cat(outs)
t0 = time.time()
enc_in = lambda s: (data["human_smpl"]["pose"][s], geom(data, "human_smpl", s.stop - s.start))
if TK > 1:
    codes = batched(lambda s: model.encode(*enc_in(s), con=None if con is None else con[s]), B, cpu=True)   # z 驻 CPU
else:
    # 0902 显存/内存: 全集 z (zp,zr = proj_out(q)) 193k x K x d 太大 (d=256 时 50GB), 只存 FSQ 量化格值 q [B,K,f] (~2GB),
    # 解码前逐批重算 z = proj_out(q) —— 与 encode 里同一个 Linear 同一批形状, 逐位一致.
    qcodes = batched(lambda s: model.encode(*enc_in(s), con=None if con is None else con[s], return_codes=True)[2], B, cpu=True)
    def z_of(s):
        acc = lambda c: c.sum(0) if c.dim() == 4 else c          # ResidualFSQ: 各级之和 (forward 里 acc), 单级 FSQ: 原样
        cp, cr = qcodes[0][s].to(dev), qcodes[1][s].to(dev)
        fp = model.fsqp; fr = fp if model.single_codebook else model.fsqr
        if getattr(model, "rot_only", False): z = fr.proj_out(acc(cr)); return z, z
        zp = fp.proj_out(acc(cp)); zr = zp if model.single_codebook else fr.proj_out(acc(cr))
        return zp, zr
dec, dpos = {}, {}
for tgt in TGTS:
    if TK > 1:      # 时序解码器: 逐片段、逐帧中心窗 (片段头尾复制补齐), 不跨片段
        outs = [decode_clip_windows(model, codes[0][sl].to(dev), codes[1][sl].to(dev), lambda n: geom(data, tgt, n), TK) for sl in cl_sl]
        dec[tgt] = torch.cat([o[0] for o in outs]); dpos[tgt] = torch.cat([o[1] for o in outs])
    else:
        dec[tgt], dpos[tgt] = batched(lambda s: model.decode(*z_of(s), geom(data, tgt, s.stop - s.start)), B)
print(f"[codec] {time.time()-t0:.0f}s", flush=True)
RWh = torch.load(f"{C}/human_smpl_root_world.pt", mmap=True).float()[rows].numpy() / 100.0

def resolve_xml(emb):
    import importlib.util
    alias = {"unitree_g1_29dof": "unitree_g1"}
    gp = '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/external/GMR/general_motion_retargeting/params.py'
    s2 = importlib.util.spec_from_file_location("gp", gp); m2 = importlib.util.module_from_spec(s2); s2.loader.exec_module(m2)
    return str(m2.ROBOT_XML_DICT[alias.get(emb, emb)])

def seg_runs(mask):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None: s = i
        if not v and s is not None: out.append((s, i)); s = None
    if s is not None: out.append((s, len(mask)))
    return out

def geo_deg(Ra, Rb):
    """[T,J,3,3] x2 -> [T,J] geodesic deg"""
    tr = np.einsum("tjii->tj", np.einsum("tjki,tjkl->tjil", Ra, Rb))
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))

def slip_stats(F, lab, tele):
    """F [T,2,3] 脚底世界 m; lab [T,2] 真站立; tele [T] bool 瞬移帧 -> dict"""
    v_c, drift = [], []
    for f in range(2):
        v = np.linalg.norm(np.diff(F[:, f, :2], axis=0), axis=1) * FPS
        both = lab[:-1, f] & lab[1:, f] & ~tele[:-1] & ~tele[1:]
        v_c += list(v[both])
        for s, e in seg_runs(lab[:, f]):
            if e - s >= 5 and not tele[s:e].any():
                drift.append(float(np.linalg.norm(F[e-1, f, :2] - F[s, f, :2])))
    v_c = np.array(v_c) * 100; drift = np.array(drift) * 100
    return dict(v_med=float(np.median(v_c)) if len(v_c) else np.nan, v_p95=float(np.percentile(v_c, 95)) if len(v_c) else np.nan,
                drift_med=float(np.median(drift)) if len(drift) else np.nan, drift_p95=float(np.percentile(drift, 95)) if len(drift) else np.nan,
                n_stance=int(len(drift)))

def onset_stats(c, lab, Fm, Fg):
    """接触时机/位置: 模型读出 c vs 真标签 lab [T,2]; Fm/Fg [T,2,3] (同一根轨迹) 用于触地位置误差"""
    a, b = c.ravel(), lab.ravel()
    tp = (a & b).sum(); f1 = 2 * tp / max(a.sum() + b.sum(), 1)
    offs, pos_err, miss, n = [], [], 0, 0
    for f in range(2):
        on_g = [s for s, e in seg_runs(lab[:, f]) if e - s >= 3]
        on_m = np.array([s for s, e in seg_runs(c[:, f]) if e - s >= 3])
        for s in on_g:
            n += 1
            if len(on_m) == 0: miss += 1; continue
            k = int(np.argmin(np.abs(on_m - s)))
            if abs(on_m[k] - s) > ONSET_WIN: miss += 1; continue
            offs.append(int(on_m[k] - s))
            pos_err.append(float(np.linalg.norm(Fm[on_m[k], f, :2] - Fg[s, f, :2])) * 100)
    return dict(agree=float((a == b).mean()), f1=float(f1),
                onset_abs_med=float(np.median(np.abs(offs))) if offs else np.nan,
                onset_bias=float(np.mean(offs)) if offs else np.nan,
                onset_pos_med=float(np.median(pos_err)) if pos_err else np.nan,
                onset_miss=float(miss / max(n, 1)), n_onset=n)

# ---------------------------------------------------------------- 逐机身
def audit(tgt):
    spec = specs[tgt]; xml = resolve_xml(tgt)
    dof, rest = load_dof(C, tgt, device=dev)
    sj = [str(s) for s in np.load(f"{C}/{tgt}_dof.npz", allow_pickle=True)["slot_joints"]]
    sc = mj.MjModel.from_xml_path(xml); d = mj.MjData(sc)
    adr = {}
    for k, nms in enumerate(sj):
        lst = list(nms) if isinstance(nms, (list, tuple)) else [x for x in str(nms).split(",") if x]
        for s_, jn in enumerate(lst):
            jid = mj.mj_name2id(sc, mj.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0 and sc.jnt_type[jid] == mj.mjtJoint.mjJNT_HINGE: adr[(k, s_)] = int(sc.jnt_qposadr[jid])
    jb = [mj.mj_name2id(sc, mj.mjtObj.mjOBJ_BODY, n) for n in spec.joint_names]
    sm = sole_offsets(tgt, spec); assert len(sm) == 2, (tgt, sm)
    feet = sorted(sm, key=lambda j: 0 if re.search(r"left|_l\b|^l_", spec.joint_names[j], re.I) else 1)
    hands = [j for j, n in enumerate(spec.joint_names) if re.search(r"wrist|hand", n, re.I) and jb[j] >= 0]
    hands = sorted(hands, key=lambda j: (0 if re.search(r"left|_l\b|^l_", spec.joint_names[j], re.I) else 1, -len(spec.joint_names[j])))
    hands = [h for h in hands if re.search(r"left", spec.joint_names[h], re.I)][-1:] + [h for h in hands if not re.search(r"left", spec.joint_names[h], re.I)][-1:]
    ri = int(np.where(np.asarray(spec.parents) < 0)[0][0])
    z = np.load(f"{LBL}/{tgt}_contact.npz", allow_pickle=True)
    thr = float(json.loads(str(z["header"]))["thr_m"]); tau = thr / 3
    gap_all = z["gap"][rows.numpy()].astype(np.float32)
    lab_all = (z["contact"][rows.numpy()] | (gap_all < 0))[:, feet]
    zG = gap_all[:, ri]; gapF = gap_all[:, feet]; del z, gap_all      # 真根高 = 标签同一口径 (根关节中心离标定地板)
    print(f"[{tgt}] feet={[spec.joint_names[j] for j in feet]} hands={[spec.joint_names[j] for j in hands]} thr={thr*100:.1f}cm", flush=True)
    t0 = time.time()
    qM = batched(lambda s: fit_angles(dec[tgt][s], spec, dof, rest=rest, clamp=True, method="global")[0], B).cpu().numpy()
    RM_all = rot6d_to_matrix(dec[tgt]).cpu().numpy()
    gR = data[tgt]["gR"]
    qG = batched(lambda s: fit_angles(matrix_to_rot6d(gR[s]), spec, dof, rest=rest, clamp=True, method="global")[0], B).cpu().numpy()
    RG_all = gR.cpu().numpy()
    zM = dpos[tgt][:, ri, 0].cpu().numpy()
    if not cfg.get("root_pos"):
        # 0902 雷达图兼容: 老 checkpoint (C1/P3V2 等) 没有位置流/根高槽, dpos[root] 只是根相对零点 (~0),
        # 硬用会把整臂放到地上. 退化为 GT 根高透传 (rooth 误差恒 0, 穿地/悬空/接触只反映姿态).
        # 只在 config 无 root_pos 时触发, 所有 root_pos=True 的既往判卷逐比特不变.
        zM = zG.copy(); print(f"[{tgt}] 无 root_pos 位置流 -> 根高 GT 透传 (rooth/穿地/悬空口径见 method)", flush=True)
    RWr = torch.load(f"{C}/{tgt}_root_world.pt", mmap=True).float()[rows].numpy() / 100.0
    print(f"[{tgt}] fit {time.time()-t0:.0f}s", flush=True)

    def fk(q, RR, xy):
        """-> F [T,2,3] 脚底点, Hd [T,2,3] 手, root [T,3], Rroot [T,3,3]  (z 未定地板)"""
        T_ = len(q); F = np.zeros((T_, 2, 3)); Hd = np.zeros((T_, 2, 3)); root = np.zeros((T_, 3)); Rr = np.zeros((T_, 3, 3))
        for t in range(T_):
            d.qpos[:] = sc.qpos0; d.qpos[0:2] = xy[t]
            d.qpos[3:7] = Rotation.from_matrix(AX.T @ RR[t] @ AX).as_quat(scalar_first=True)
            for (k, s_), a in adr.items(): d.qpos[a] = q[t, k, s_]
            mj.mj_forward(sc, d)
            for i, j in enumerate(feet): F[t, i] = d.xpos[jb[j]] + d.xmat[jb[j]].reshape(3, 3) @ (AX.T @ sm[j])   # 脚底点, 与标签同口径
            for i, j in enumerate(hands): Hd[t, i] = d.xpos[jb[j]]
            root[t] = d.qpos[0:3]; Rr[t] = d.xmat[1].reshape(3, 3) if sc.nbody > 1 else np.eye(3)
        return F, Hd, root, Rr
    # 0903 导出用的仿真资产 (GMR/mjlab XML) 与标签尺子 (sole_offsets3, 按 xwbc 资产量的) 的脚可能不是
    # 同一只脚: GR3 尺子说脚底在踝下 16.8cm, 仿真 XML 的脚只有 7.0cm -> 按尺子摆根高会整台悬空 9.8cm。
    # 两边都是量出来的 (静止姿态下脚 body 局部系里最低网格顶点), 差值逐资产打印, 不是手调常数; G1 两边同一只脚, 差 ~0。
    def _asset_sole_drop():
        d.qpos[:] = sc.qpos0; mj.mj_forward(sc, d); drops = []
        for j in feet:
            bid = jb[j]; lo = None
            for g in range(sc.ngeom):
                b = sc.geom_bodyid[g]
                # 该脚 body 及其子树 (脚踝 pitch/roll 之下的连杆)
                while b >= 0 and b != bid: b = sc.body_parentid[b] if b != 0 else -1
                if b != bid: continue
                if sc.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
                    mid = sc.geom_dataid[g]; v = sc.mesh_vert[sc.mesh_vertadr[mid]:sc.mesh_vertadr[mid] + sc.mesh_vertnum[mid]]
                    zmin = ((d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g])[:, 2].min()
                else:
                    zmin = d.geom_xpos[g][2] - sc.geom_size[g][0]
                lo = zmin if lo is None else min(lo, zmin)
            if lo is not None: drops.append(d.xpos[bid][2] - lo)
        return float(np.mean(drops)) if drops else None
    if EXP:
        _sim_drop = _asset_sole_drop(); _gauge_drop = float(np.mean([-sm[j][1] for j in feet]))
        _dz_export = (_gauge_drop - _sim_drop) if _sim_drop is not None else 0.0
        print(f"[export] {tgt}: 尺子脚底在脚 body 下 {_gauge_drop*100:.1f}cm, 仿真资产 {_sim_drop*100 if _sim_drop is not None else float('nan'):.1f}cm -> 导出根高 {-_dz_export*100:+.1f}cm", flush=True)
    def qpos_csv(path, q, RR, xy, z):
        """与 fk 同一套 qpos 摆法, 根高 = 标签口径真根高 (换算到仿真资产的脚); 写 GMR/mjlab csv"""
        rows_ = []
        for t in range(len(q)):
            d.qpos[:] = sc.qpos0; d.qpos[0:2] = xy[t]; d.qpos[2] = z[t] - _dz_export
            d.qpos[3:7] = Rotation.from_matrix(AX.T @ RR[t] @ AX).as_quat(scalar_first=True)
            for (k, s_), a in adr.items(): d.qpos[a] = q[t, k, s_]
            qp = d.qpos.copy(); rows_.append(np.concatenate([qp[:3], qp[3:7][[1, 2, 3, 0]], qp[7:]]))
        np.savetxt(path, np.array(rows_), delimiter=",", fmt="%.6f")
    per = []
    t0 = time.time()
    for ci, (q, sl) in enumerate(zip(clips, cl_sl)):
        T_ = sl.stop - sl.start
        trj_h = (RWh[sl] - RWh[sl.start]) @ AX; trj_r = (RWr[sl] - RWr[sl.start]) @ AX
        tele = np.zeros(T_, bool); tele[1:] = np.linalg.norm(np.diff(trj_h[:, :2], axis=0), axis=1) > TELE
        lab = lab_all[sl]
        # GT 臂: GMR 姿态 + GMR 根 xy + 标签口径的真根高 (gap_root)
        Fg, Hg, rg, Rg = fk(qG[sl], RG_all[sl, ri], trj_r[:, :2])
        dz = zG[sl] - rg[:, 2]; Fg[:, :, 2] += dz[:, None]; rg[:, 2] += dz
        if os.environ.get("AM_DEBUG") and ci < 3:  # 口径自检: GT 臂脚底高应与标签 gap 逐帧一致
            print("  dbg", q["name"][:40], "Fg feet z med", np.median(Fg[:, :, 2], 0), "label gap feet med", np.median(gapF[sl], 0), "root", np.median(zG[sl]), flush=True)
        # 模型臂: 姿态相同, 根高 = 预测; 根 xy 三种
        Fm, Hm, rm, Rm = fk(qM[sl], RM_all[sl, 0], trj_h[:, :2])
        dzm = zM[sl] - rm[:, 2]; Fm[:, :, 2] += dzm[:, None]; rm[:, 2] += dzm
        cM = 1 / (1 + np.exp(-(thr - Fm[:, :, 2]) / tau)) > 0.5
        if EXP:
            tag = os.path.basename(CK.rstrip("/"))
            qpos_csv(f"{EXP}/{q['name']}__{tgt}__A_{tag}.csv", qM[sl], RM_all[sl, 0], trj_h[:, :2], zM[sl])
            qpos_csv(f"{EXP}/{q['name']}__{tgt}__B_native.csv", qG[sl], RG_all[sl, ri], trj_r[:, :2], zG[sl])
        # M_g: 同姿态换 GMR 根 xy (脚底 xy 平移即可, 姿态/高度不变)
        off = (trj_r - trj_h)[:, None, :2]
        Fmg = Fm.copy(); Fmg[:, :, :2] += off
        # M_a: 接触锚定里程计 (只用模型读出 cM): 累积平移 S, 踩地脚 f 的位移被抵消 (脚钉死身体走)
        S = np.zeros((T_, 2))
        anchor = None
        for t in range(T_ - 1):
            cands = [f for f in range(2) if cM[t, f] and cM[t+1, f]]
            if cands and not tele[t+1]:
                f = anchor if anchor in cands else cands[0]; anchor = f
                S[t+1] = S[t] - (Fm[t+1, f, :2] - Fm[t, f, :2])
            else:
                anchor = None; S[t+1] = S[t]
        Fma = Fm.copy(); Fma[:, :, :2] += S[:, None]
        # ---- 1 姿态
        rot = geo_deg(RM_all[sl], RG_all[sl])
        def rootrel(P, r, R): return np.einsum("tji,tkj->tki", R, P - r[:, None])   # R^T (P - root)
        hand_err = np.linalg.norm(rootrel(Hm, rm, Rm) - rootrel(Hg, rg, Rg), axis=-1) * 100
        foot_err = np.linalg.norm(rootrel(Fm, rm, Rm) - rootrel(Fg, rg, Rg), axis=-1) * 100
        rooth_err = np.abs(rm[:, 2] - rg[:, 2]) * 100
        rec = dict(name=q["name"], dataset=dataset_of(q["name"]), family=family(q["name"]), T=T_, tele=int(tele.sum()),
                   rot_deg=float(rot.mean()), rot_p95=float(np.percentile(rot.mean(1), 95)),
                   hand_cm=float(hand_err.mean()), hand_p95=float(np.percentile(hand_err, 95)),
                   foot_cm=float(foot_err.mean()), foot_p95=float(np.percentile(foot_err, 95)),
                   rooth_cm=float(rooth_err.mean()), rooth_bias_cm=float(((rm[:, 2] - rg[:, 2]) * 100).mean()),   # 0902 带符号: 负 = 模型根比标签低
                   gmr_over_human=float(np.linalg.norm(trj_r[-1, :2] - trj_r[0, :2]) / max(np.linalg.norm(trj_h[-1, :2] - trj_h[0, :2]), 1e-3)),
                   disp_h=float(np.linalg.norm(trj_h[-1, :2] - trj_h[0, :2])))
        # ---- 2 脚滑 (真站立 = GMR 速度门标签)
        for arm, F in (("M_h", Fm), ("M_g", Fmg), ("M_a", Fma), ("GT", Fg)):
            for k, v in slip_stats(F, lab, tele).items(): rec[f"{arm}.{k}"] = v
        # ---- 3 接触标签: 模型读出 vs 真标签; 位置在 M_g (同 GMR 根) 上比
        cG = 1 / (1 + np.exp(-(thr - Fg[:, :, 2]) / tau)) > 0.5      # 同一读出函数用在 GMR 姿态上 = 读出函数自身的天花板
        for k, v in onset_stats(cM, lab, Fmg, Fg).items(): rec[f"M.{k}"] = v
        for k, v in onset_stats(cG, lab, Fg, Fg).items(): rec[f"GT.{k}"] = v
        # ---- 4 物理
        rec["M.pen_frac"] = float((Fm[:, :, 2].min(1) < -thr).mean())
        rec["M.float_frac"] = float((lab & (Fm[:, :, 2] > thr)).sum() / max(lab.sum(), 1))
        rec["GT.float_frac"] = float((lab & (Fg[:, :, 2] > thr)).sum() / max(lab.sum(), 1))
        rec["GT.pen_frac"] = float((Fg[:, :, 2].min(1) < -thr).mean())
        rec["M.contact_frac"] = float(cM.mean()); rec["GT.contact_frac"] = float(lab.mean())
        per.append(rec)
        if ci % 50 == 0: print(f"[{tgt}] clip {ci}/{len(clips)} {time.time()-t0:.0f}s", flush=True)
    return dict(thr_m=thr, feet=[spec.joint_names[j] for j in feet], hands=[spec.joint_names[j] for j in hands],
                root_h_src="model" if cfg.get("root_pos") else "gt_passthrough", clips=per)

res = {"ckpt": CK, "cache": C, "set": SET, "n_clips": len(clips), "rows": B}
for tgt in TGTS:
    res[tgt] = audit(tgt); json.dump(res, open(OUT, "w"), indent=1); print("done", tgt, flush=True)

# ---------------------------------------------------------------- 汇总表
def agg(per, key):
    v = np.array([r.get(key, np.nan) for r in per], float); v = v[~np.isnan(v)]
    return (float(np.median(v)), float(np.percentile(v, 95))) if len(v) else (np.nan, np.nan)
print("\n每项: 逐片段中位 / p95   (通过率 = 片段值 <= GT 臂 p95 的比例)")
for tgt in TGTS:
    per = res[tgt]["clips"]
    print(f"\n== {tgt}  ({len(per)} clips)")
    print("  1 姿态  rot %.1f°/%.1f  hand %.1f/%.1fcm  foot %.1f/%.1fcm  rooth %.1f/%.1fcm" % (
        *agg(per, "rot_deg"), *agg(per, "hand_cm"), *agg(per, "foot_cm"), *agg(per, "rooth_cm")))
    for arm in ("M_h", "M_g", "M_a", "GT"):
        vm, vp = agg(per, f"{arm}.v_med"); dm, dp = agg(per, f"{arm}.drift_med")
        gt95_v = agg(per, "GT.v_med")[1]; gt95_d = agg(per, "GT.drift_med")[1]
        pv = np.nanmean([r.get(f"{arm}.v_med", np.nan) <= gt95_v for r in per]); pd = np.nanmean([r.get(f"{arm}.drift_med", np.nan) <= gt95_d for r in per])
        print(f"  2 脚滑 {arm:4s} 接触中脚速 {vm:5.1f}/{vp:5.1f} cm/s (通过 {pv*100:3.0f}%)  站立漂移 {dm:4.1f}/{dp:4.1f} cm (通过 {pd*100:3.0f}%)")
    for who in ("M", "GT"):
        print("  3 标签 %-2s F1 %.2f/%.2f  时机|误差| %.1f/%.1f 帧 (偏 %+.1f)  触地位置 %.1f/%.1f cm  漏检 %.0f%%/%.0f%%" % (
            who, *agg(per, f"{who}.f1"), *agg(per, f"{who}.onset_abs_med"), agg(per, f"{who}.onset_bias")[0],
            *agg(per, f"{who}.onset_pos_med"), *(np.array(agg(per, f"{who}.onset_miss")) * 100)))
    print("  4 物理 穿地 M %.1f%%/%.1f%% GT %.1f%%/%.1f%%   标签踩地却悬空 M %.1f%%/%.1f%% GT %.1f%%/%.1f%%" % (
        *(np.array(agg(per, "M.pen_frac")) * 100), *(np.array(agg(per, "GT.pen_frac")) * 100),
        *(np.array(agg(per, "M.float_frac")) * 100), *(np.array(agg(per, "GT.float_frac")) * 100)))
