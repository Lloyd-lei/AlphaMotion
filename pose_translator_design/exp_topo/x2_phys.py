"""免真值物理判官 (lloyd 0903: "x2 根跳不跳 脚滑不滑 和 fz5c 可以有个对照吧").
X2 是零样本机身 (缓存里没有它的重定向真值), 所以不能用 wbc_ready_audit 的姿态/脚位置误差; 这里只量**不需要真值**的东西:
  1 根高抖不抖   根高逐帧速度 p95 (cm/s), 二阶差分 (抖动) p95, 单帧跳 >3cm 的帧占比
  2 姿态抖不抖   根旋转角速度 p95 (°/s), 全身关节角速度 p95 (°/s), 及各自二阶差分 (抖动)
  3 脚滑不滑     (a) 模型自己读出的接触 (脚底 < thr) 内脚底水平速度 中位/p95 (cm/s)
                 (b) 人体源标签站立相 (时机外借自人体接触标签, 与机身无关) 内脚底水平速度, 站立漂移 (cm)
  4 物理自洽     穿地占比 (脚底 < -thr), 人体站立却脚悬空占比 (> thr), 模型读出接触 vs 人体标签 F1
尺子 = 人体源自己 (同一片段的人体根高速度/关节角速度) —— 机器人不该比源动作抖; 对照臂 = 另一个 ckpt 同口径。
机身: x2 (零样本, 从 XML 现载, thr=4cm 地板=0) + unitree_g1_29dof (有真值的机身当校验: 此判官排名应与 wbc_ready_audit 一致)。
用法: AM_CACHE=<tfix> AM_LBL=<标签目录> x2_phys.py <ckpt> <out.json>     或   x2_phys.py --cmp a.json b.json ..."""
import os, sys, json, re, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
FPS = 30.0

def agg(per, key):
    v = np.array([r.get(key, np.nan) for r in per], float); v = v[~np.isnan(v)]
    return (float(np.median(v)), float(np.percentile(v, 95))) if len(v) else (np.nan, np.nan)

ROWS = [("根高速度 p95 cm/s", "rooth_v_p95"), ("根高抖动(二阶) p95 cm/s", "rooth_jit_p95"), ("根高单帧跳>3cm %", "rooth_jump_frac", 100),
        ("根旋转角速度 p95 °/s", "rootR_v_p95"), ("根旋转抖动 p95 °/s", "rootR_jit_p95"),
        ("关节角速度 p95 °/s", "joint_v_p95"), ("关节抖动 p95 °/s", "joint_jit_p95"),
        ("脚滑@自读接触 中位 cm/s", "slipM_v_med"), ("脚滑@自读接触 p95 cm/s", "slipM_v_p95"),
        ("脚滑@人体站立 中位 cm/s", "slipH_v_med"), ("脚滑@人体站立 p95 cm/s", "slipH_v_p95"), ("站立漂移@人体站立 cm", "slipH_drift_med"),
        ("穿地 %", "pen_frac", 100), ("人体站立却悬空 %", "float_frac", 100), ("自读接触 vs 人体标签 F1", "contact_f1"), ("自读接触占比 %", "contact_frac", 100)]

def cmp_table(paths):
    R = [json.load(open(p)) for p in paths]; tags = [os.path.basename(r["ckpt"].rstrip("/")) for r in R]
    tg = [t for t in R[0] if isinstance(R[0][t], dict) and "clips" in R[0][t]]
    for t in tg:
        print(f"\n== {t}   (逐片段中位数, {len(R[0][t]['clips'])} 条; 括号内 = p95 片段)   " + "  |  ".join(tags))
        for row in ROWS:
            lab, key = row[0], row[1]; m = row[2] if len(row) > 2 else 1
            cells = []
            for r in R:
                a, b = agg(r[t]["clips"], key); cells.append(f"{a*m:7.2f} ({b*m:6.2f})")
            print(f"  {lab:28s} " + "  |  ".join(cells))
    if "human" in R[0]:
        print("\n== 人体源 (尺子, 同 600 片段)")
        for lab, key in [("根高速度 p95 cm/s", "rooth_v_p95"), ("根高抖动(二阶) p95 cm/s", "rooth_jit_p95"), ("根高单帧跳>3cm %", "rooth_jump_frac"),
                         ("根旋转角速度 p95 °/s", "rootR_v_p95"), ("根旋转抖动 p95 °/s", "rootR_jit_p95"), ("关节角速度 p95 °/s", "joint_v_p95"), ("关节抖动 p95 °/s", "joint_jit_p95")]:
            a, b = agg(R[0]["human"]["clips"], key); m = 100 if key.endswith("frac") else 1
            print(f"  {lab:28s} {a*m:7.2f} ({b*m:6.2f})")

if sys.argv[1] == "--cmp":
    cmp_table(sys.argv[2:]); sys.exit(0)

import torch
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
from scipy.spatial.transform import Rotation
from common import SkeletonSpec, fk_global, matrix_to_rot6d, rot6d_to_matrix
from dual_common import build_model, geom, static_cond
from manifold_project import fit_angles, load_dof
from semantics import build_name_embeddings, embed_matrix
from contact_label import sole_offsets
sys.path.insert(0, "/home/arenalabs/Desktop/AlphaMotion/src")

RUNS = "/var/tmp/alphamotion/topo_exp/runs"
C = os.environ["AM_CACHE"]; LBL = os.environ["AM_LBL"]
SET = os.environ.get("AM_SET", "/var/tmp/alphamotion/topo_exp/cache_amass_heldout/sequences.json")
CK = sys.argv[1] if sys.argv[1].startswith("/") else f"{RUNS}/{sys.argv[1]}"
OUT = sys.argv[2]
TGTS = os.environ.get("AM_TGTS", "x2,unitree_g1_29dof").split(",")
MAXC = int(os.environ.get("AM_MAXCLIPS", "0"))
dev = os.environ.get("AM_DEV", "cuda"); torch.set_grad_enabled(False)
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])
TELE = 0.3; JUMP = 0.03
X2_XML = "/var/tmp/alphamotion/robot_models/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"

seqs_all = {q["name"]: q for q in json.load(open(f"{C}/sequences.json"))}
names = [q["name"] for q in json.load(open(SET))]
clips = [seqs_all[n] for n in names if n in seqs_all]
if MAXC: clips = clips[:MAXC]
rows = torch.cat([torch.arange(q["start"], q["start"] + q["len"]) for q in clips])
B = len(rows); print(f"[set] {len(clips)} clips, {B} rows", flush=True)
cl_sl = []; o = 0
for q in clips: cl_sl.append(slice(o, o + q["len"])); o += q["len"]

def load_spec_npz(name):
    d = np.load(f"{C}/{name}_spec.npz", allow_pickle=True)
    return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], [str(x) for x in d["joint_names"]])
cfg = json.load(open(f"{CK}/config.json"))
names_all, specs = [], {}
for n in ["human_smpl"] + [t for t in TGTS if t != "x2"]:
    specs[n] = load_spec_npz(n); names_all += specs[n].joint_names
x2 = None
if "x2" in TGTS:
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
        e = {"spec": spec, "st": static_cond(spec, dev), "reach": float(np.linalg.norm(pr, axis=1).max()),
             "sem": embed_matrix(spec.joint_names, lut).to(dev),
             "dof": torch.as_tensor(np.asarray(x2.dof), dtype=torch.float32, device=dev)}
        from dof_head import DofKin
        e["kin"] = DofKin(spec, e["dof"], torch.as_tensor(np.asarray(x2.rest), dtype=torch.float32, device=dev), device=dev)
        return e
    l6 = torch.load(f"{C}/{n}_paired_local6d.pt", mmap=True)[rows].float().to(dev)
    gR, gp = fk_global(l6, spec); reach = float(gp.norm(dim=-1).max())
    e = {"spec": spec, "st": static_cond(spec, dev), "reach": reach, "sem": embed_matrix(spec.joint_names, lut).to(dev),
         "pose": torch.cat([matrix_to_rot6d(gR), gp / reach], -1), "gR": gR}
    if os.path.exists(f"{C}/{n}_dof.npz"):
        from dof_head import build_kin
        e["dof"] = torch.as_tensor(np.load(f"{C}/{n}_dof.npz")["dof"], dtype=torch.float32, device=dev)
        e["kin"] = build_kin(C, n, spec, device=dev)
    return e
data = {n: entry(n) for n in ["human_smpl"] + TGTS}
from tdec import wrap_if_tdec
model = wrap_if_tdec(build_model(cfg, dev), cfg); model.load_state_dict(torch.load(f"{CK}/model.pt", map_location=dev)); model.eval()
assert not int(cfg.get("tdec", 0) or 0), "只支持逐帧解码 (与 FZ5c/阶梯同)"
assert cfg.get("root_pos"), "没有根高位置流的 ckpt 量不了根高"
zH = np.load(f"{LBL}/human_smpl_contact.npz", allow_pickle=True)
hh = json.loads(str(zH["header"])); jn_h = hh["joint_names"]
con = None
if cfg.get("contact_chan"):
    thr_, D_ = hh["thr_m"], hh["D_used_m"]; s_ = thr_ - zH["gap"][rows.numpy()].astype(np.float32)
    con = torch.as_tensor((np.sign(s_) * np.log1p(np.abs(s_) / thr_) / np.log1p(D_ / thr_)).astype(np.float32), device=dev)
cH = zH["contact"][rows.numpy()]; gH = zH["gap"][rows.numpy()].astype(np.float32)
fl = [jn_h.index("L_Ankle"), jn_h.index("L_Toe")]; fr_ = [jn_h.index("R_Ankle"), jn_h.index("R_Toe")]
labH = np.stack([cH[:, fl].any(1), cH[:, fr_].any(1)], 1)      # 人体站立相 [B,2] (时机尺子, 与机身无关)
zHroot = gH[:, jn_h.index("Pelvis")]                            # 人体根高 (标签口径)
del zH
BS = int(os.environ.get("AM_BS", "2048"))
def batched(fn, n, cpu=False):
    outs = []
    for i in range(0, n, BS):
        o = fn(slice(i, min(i + BS, n)))
        outs.append((tuple(x.cpu() for x in o) if isinstance(o, tuple) else o.cpu()) if cpu else o)
    return tuple(torch.cat([o[k] for o in outs]) for k in range(len(outs[0]))) if isinstance(outs[0], tuple) else torch.cat(outs)
t0 = time.time()
enc_in = lambda s: (data["human_smpl"]["pose"][s], geom(data, "human_smpl", s.stop - s.start))
qcodes = batched(lambda s: model.encode(*enc_in(s), con=None if con is None else con[s], return_codes=True)[2], B, cpu=True)
def z_of(s):
    acc = lambda c: c.sum(0) if c.dim() == 4 else c
    cp, cr = qcodes[0][s].to(dev), qcodes[1][s].to(dev)
    fp = model.fsqp; fr = fp if model.single_codebook else model.fsqr
    if getattr(model, "rot_only", False): z = fr.proj_out(acc(cr)); return z, z
    zp = fp.proj_out(acc(cp)); zr = zp if model.single_codebook else fr.proj_out(acc(cr))
    return zp, zr
dec, dpos = {}, {}
for tgt in TGTS:
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

def geo_step(R):
    """R [T,J,3,3] -> [T-1,J] 相邻帧 geodesic 度"""
    tr = np.einsum("tjii->tj", np.einsum("tjki,tjkl->tjil", R[:-1], R[1:]))
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))

def motion_stats(z, Rroot, Rall, tele, pre):
    """根高 z [T] m, 根旋转 [T,3,3], 全身旋转 [T,J,3,3] -> 速度/抖动 p95 (免真值)"""
    ok = ~tele[1:]
    vz = np.abs(np.diff(z)) * FPS * 100; jz = np.abs(np.diff(z, 2)) * FPS * 100
    ok2 = ok[:-1] & ok[1:]
    wr = geo_step(Rroot[:, None])[:, 0] * FPS; wj = geo_step(Rall).mean(1) * FPS
    r = {f"{pre}rooth_v_p95": float(np.percentile(vz[ok], 95)) if ok.any() else np.nan,
         f"{pre}rooth_jit_p95": float(np.percentile(jz[ok2], 95)) if ok2.any() else np.nan,
         f"{pre}rooth_jump_frac": float((np.abs(np.diff(z))[ok] > JUMP).mean()) if ok.any() else np.nan,
         f"{pre}rootR_v_p95": float(np.percentile(wr[ok], 95)) if ok.any() else np.nan,
         f"{pre}rootR_jit_p95": float(np.percentile(np.abs(np.diff(wr))[ok2], 95)) if ok2.any() else np.nan,
         f"{pre}joint_v_p95": float(np.percentile(wj[ok], 95)) if ok.any() else np.nan,
         f"{pre}joint_jit_p95": float(np.percentile(np.abs(np.diff(wj))[ok2], 95)) if ok2.any() else np.nan}
    return r

def slip(F, lab, tele, pre):
    v_c, drift = [], []
    for f in range(2):
        v = np.linalg.norm(np.diff(F[:, f, :2], axis=0), axis=1) * FPS
        both = lab[:-1, f] & lab[1:, f] & ~tele[:-1] & ~tele[1:]
        v_c += list(v[both])
        for s, e in seg_runs(lab[:, f]):
            if e - s >= 5 and not tele[s:e].any(): drift.append(float(np.linalg.norm(F[e-1, f, :2] - F[s, f, :2])))
    v_c = np.array(v_c) * 100; drift = np.array(drift) * 100
    return {f"{pre}v_med": float(np.median(v_c)) if len(v_c) else np.nan, f"{pre}v_p95": float(np.percentile(v_c, 95)) if len(v_c) else np.nan,
            f"{pre}drift_med": float(np.median(drift)) if len(drift) else np.nan, f"{pre}n": int(len(drift))}

# ---------------------------------------------------------------- 人体源尺子
gRh = data["human_smpl"]["gR"].cpu().numpy(); ri_h = jn_h.index("Pelvis")
per_h = []
for q, sl in zip(clips, cl_sl):
    trj_h = (RWh[sl] - RWh[sl.start]) @ AX
    tele = np.zeros(sl.stop - sl.start, bool); tele[1:] = np.linalg.norm(np.diff(trj_h[:, :2], axis=0), axis=1) > TELE
    per_h.append(dict(name=q["name"], **motion_stats(zHroot[sl], gRh[sl, ri_h], gRh[sl], tele, "")))
res = {"ckpt": CK, "cache": C, "set": SET, "n_clips": len(clips), "rows": B, "human": {"clips": per_h}}
print("[human] 尺子: 根高速度 p95 中位 %.1f cm/s, 关节角速度 p95 中位 %.1f °/s" % (agg(per_h, "rooth_v_p95")[0], agg(per_h, "joint_v_p95")[0]), flush=True)

# ---------------------------------------------------------------- 逐机身
def judge(tgt):
    spec = specs[tgt]
    if tgt == "x2":
        xml = X2_XML; dof = torch.as_tensor(np.asarray(x2.dof), dtype=torch.float32, device=dev)
        rest = torch.as_tensor(np.asarray(x2.rest), dtype=torch.float32, device=dev); sj = x2.qnames; thr = 0.04
        sole_name = "agibot_x2"
    else:
        xml = resolve_xml(tgt); dof, rest = load_dof(C, tgt, device=dev)
        sj = [str(s) for s in np.load(f"{C}/{tgt}_dof.npz", allow_pickle=True)["slot_joints"]]
        thr = float(json.loads(str(np.load(f"{LBL}/{tgt}_contact.npz", allow_pickle=True)["header"]))["thr_m"]); sole_name = tgt
    tau = thr / 3
    sc = mj.MjModel.from_xml_path(xml); d = mj.MjData(sc)
    adr = {}
    for k, nms in enumerate(sj):
        lst = list(nms) if isinstance(nms, (list, tuple)) else [x for x in str(nms).split(",") if x]
        for s_, jn in enumerate(lst):
            jid = mj.mj_name2id(sc, mj.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0 and sc.jnt_type[jid] == mj.mjtJoint.mjJNT_HINGE: adr[(k, s_)] = int(sc.jnt_qposadr[jid])
    fj = [i for i in range(sc.njnt) if sc.jnt_type[i] == mj.mjtJoint.mjJNT_FREE]; assert len(fj) == 1, (tgt, fj)
    a0 = int(sc.jnt_qposadr[fj[0]])
    jb = [mj.mj_name2id(sc, mj.mjtObj.mjOBJ_BODY, n) for n in spec.joint_names]
    sm = sole_offsets(sole_name, spec); assert len(sm) == 2, (tgt, sm)
    feet = sorted(sm, key=lambda j: 0 if re.search(r"left|_l\b|^l_", spec.joint_names[j], re.I) else 1)
    ri = int(np.where(np.asarray(spec.parents) < 0)[0][0])
    print(f"[{tgt}] hinge={len(adr)} free@{a0} feet={[spec.joint_names[j] for j in feet]} thr={thr*100:.1f}cm", flush=True)
    t0 = time.time()
    qM = batched(lambda s: fit_angles(dec[tgt][s], spec, dof, rest=rest, clamp=True, method="global")[0], B).cpu().numpy()
    RM = rot6d_to_matrix(dec[tgt]).cpu().numpy(); zM = dpos[tgt][:, ri, 0].cpu().numpy()
    print(f"[{tgt}] fit {time.time()-t0:.0f}s", flush=True)
    per = []
    for ci, (q, sl) in enumerate(zip(clips, cl_sl)):
        T_ = sl.stop - sl.start
        trj_h = (RWh[sl] - RWh[sl.start]) @ AX
        tele = np.zeros(T_, bool); tele[1:] = np.linalg.norm(np.diff(trj_h[:, :2], axis=0), axis=1) > TELE
        F = np.zeros((T_, 2, 3)); Rr = np.zeros((T_, 3, 3)); rz = np.zeros(T_)
        for t in range(T_):
            i = sl.start + t
            d.qpos[:] = sc.qpos0; d.qpos[a0:a0+2] = trj_h[t, :2]; d.qpos[a0+2] = zM[i]
            d.qpos[a0+3:a0+7] = Rotation.from_matrix(AX.T @ RM[i, ri] @ AX).as_quat(scalar_first=True)
            for (k, s_), a in adr.items(): d.qpos[a] = qM[i, k, s_]
            mj.mj_forward(sc, d)
            for f, j in enumerate(feet): F[t, f] = d.xpos[jb[j]] + d.xmat[jb[j]].reshape(3, 3) @ (AX.T @ sm[j])
            Rr[t] = d.xmat[jb[ri]].reshape(3, 3); rz[t] = d.qpos[a0+2]
        cM = 1 / (1 + np.exp(-(thr - F[:, :, 2]) / tau)) > 0.5
        lab = labH[sl]
        rec = dict(name=q["name"], T=T_, tele=int(tele.sum()))
        rec.update(motion_stats(rz, Rr, RM[sl], tele, ""))
        rec.update(slip(F, cM, tele, "slipM_")); rec.update(slip(F, lab, tele, "slipH_"))
        rec["pen_frac"] = float((F[:, :, 2].min(1) < -thr).mean())
        rec["float_frac"] = float((lab & (F[:, :, 2] > thr)).sum() / max(lab.sum(), 1))
        a, b = cM.ravel(), lab.ravel(); tp = (a & b).sum()
        rec["contact_f1"] = float(2 * tp / max(a.sum() + b.sum(), 1)); rec["contact_frac"] = float(cM.mean())
        rec["rooth_med"] = float(np.median(rz))
        per.append(rec)
        if ci % 100 == 0: print(f"[{tgt}] clip {ci}/{len(clips)} {time.time()-t0:.0f}s", flush=True)
    return dict(thr_m=thr, feet=[spec.joint_names[j] for j in feet], clips=per)

for tgt in TGTS:
    res[tgt] = judge(tgt); json.dump(res, open(OUT, "w"), indent=1); print("done", tgt, flush=True)
cmp_table([OUT])
