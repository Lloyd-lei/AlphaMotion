"""0903 训练前数据 gauge 自检 (只读, 几分钟)。

对每个 (缓存, 具身) 报:
  rows        姿态 / 根轨迹 / 接触标签 三者行数是否一致
  h0_cm       标签器量出的地面 (语料 x 具身), 训练侧 --root_pos 会把根高减去它 (地面=0)
  root_cm     根高中位 (减 h0 后), 人形应在 40-120
  low_p2/p50  最低关节离地 (减 h0 后) p2 / p50, cm
  pen_frac    最低关节 < -3 cm 的帧占比 (穿地)
并对 AMASS 抽 200 段 (KIT/CMU/TotalCapture 为主) 量人体 vs g1 根水平速度的最佳互相关滞后:
  修好的缓存中位 |lag| 应 ~0 帧; 老缓存 (未修时间轴) 用来对照。
用法: python gauge_check.py [--out ~/tmp/gauge_check_0903.json]
"""
import os, sys, json, argparse
import numpy as np, torch

R = "/var/tmp/alphamotion/topo_exp"
ap = argparse.ArgumentParser()
ap.add_argument("--caches", default=f"{R}/cache_lafan_v3,{R}/cache_amass_lconv_tfix")
ap.add_argument("--old_amass", default=f"{R}/cache_amass_lconv")     # 时间轴未修, 只做对照
ap.add_argument("--exclude", default=os.environ.get("AM_EXCLUDE_EMB", "pal_talos,engineai_pm01"))
ap.add_argument("--out", default=os.path.expanduser("~/tmp/gauge_check_0903.json"))
a = ap.parse_args()
ex = {x for x in a.exclude.split(",") if x}
warn = []
rep = {"tables": {}, "timebase": {}, "warnings": warn}

for c in a.caches.split(","):
    man = json.load(open(f"{c}/manifest.json"))
    names = [n for n in man["embodiments"] if n not in ex]
    tab = {}
    print(f"\n== {c}  ({len(names)} 具身)")
    print(f"{'emb':20s} {'rows':>9s} {'h0_cm':>7s} {'root_cm':>8s} {'low_p2':>7s} {'low_p50':>8s} {'pen%':>6s}")
    for n in names:
        l6 = torch.load(f"{c}/{n}_paired_local6d.pt", mmap=True)
        rw = torch.load(f"{c}/{n}_root_world.pt", mmap=True)
        cf = f"{c}/{n}_contact.npz"
        row = {"rows_pose": int(l6.shape[0]), "rows_root": int(rw.shape[0])}
        if not os.path.exists(cf):
            row["contact"] = None; warn.append(f"{os.path.basename(c)}/{n}: 没有接触标签")
            print(f"{n:20s} {l6.shape[0]:9d}   (无接触标签)")
            tab[n] = row; continue
        z = np.load(cf); hdr = json.loads(str(z["header"]))
        h0 = float(hdr["h0_m"]); gap = z["gap"]
        row["rows_contact"] = int(gap.shape[0]); row["h0_cm"] = round(h0 * 100, 2)
        if not (l6.shape[0] == rw.shape[0] == gap.shape[0]):
            warn.append(f"{os.path.basename(c)}/{n}: 行数不一致 pose={l6.shape[0]} root={rw.shape[0]} contact={gap.shape[0]}")
        # gap 已是 "离该语料地面 (h0) 的高度", 直接用; 抽样 40 万帧省内存
        N = gap.shape[0]; idx = np.linspace(0, N - 1, min(N, 400_000)).astype(int)
        g = gap[idx].astype(np.float32); low = g.min(1) * 100
        rooty = (rw[idx, 1].float().numpy() - h0 * 100)
        row.update(root_cm=round(float(np.median(rooty)), 1),
                   low_p2=round(float(np.percentile(low, 2)), 1),
                   low_p50=round(float(np.percentile(low, 50)), 1),
                   pen_frac=round(float((low < -3).mean()), 4))
        if not (40 <= row["root_cm"] <= 120):
            warn.append(f"{os.path.basename(c)}/{n}: 根高中位 {row['root_cm']} cm 不在 40-120")
        if row["low_p2"] < -3:
            warn.append(f"{os.path.basename(c)}/{n}: 最低关节 p2 = {row['low_p2']} cm (穿地)")
        print(f"{n:20s} {N:9d} {row['h0_cm']:7.2f} {row['root_cm']:8.1f} {row['low_p2']:7.1f} {row['low_p50']:8.1f} {100*row['pen_frac']:6.2f}")
        tab[n] = row
    rep["tables"][c] = tab

# ---- 时间轴: 人体 vs g1 根水平速度互相关滞后 -------------------------------------
def lag_stats(c, seqs, robot="unitree_g1_29dof", maxlag=60):
    rh = torch.load(f"{c}/human_smpl_root_world.pt", mmap=True)
    rg = torch.load(f"{c}/{robot}_root_world.pt", mmap=True)
    lags = []
    for q in seqs:
        s, L = q["start"], q["len"]
        if L < 150: continue
        h = rh[s:s + L].float().numpy(); g = rg[s:s + L].float().numpy()
        vh = np.linalg.norm(np.diff(h[:, [0, 2]], axis=0), axis=1)
        vg = np.linalg.norm(np.diff(g[:, [0, 2]], axis=0), axis=1)
        if vh.std() < 1e-4 or vg.std() < 1e-4: continue
        vh = (vh - vh.mean()) / vh.std(); vg = (vg - vg.mean()) / vg.std()
        # 只看片段后半段 (时间轴误差随时间累积, 中段/尾段才看得出来)
        m = len(vh) // 2; vh2 = vh[m:]; vg2 = vg
        best, bl = -9, 0
        for lag in range(-maxlag, maxlag + 1):
            a0, a1 = m, m + len(vh2)
            b0, b1 = a0 + lag, a1 + lag
            if b0 < 0 or b1 > len(vg2): continue
            cc = float(np.dot(vh2, vg2[b0:b1]) / len(vh2))
            if cc > best: best, bl = cc, lag
        lags.append(bl)
    lags = np.abs(np.array(lags))
    return {"n": int(len(lags)), "median_abs_lag": float(np.median(lags)) if len(lags) else None,
            "p90_abs_lag": float(np.percentile(lags, 90)) if len(lags) else None,
            "frac_lag_le2": float((lags <= 2).mean()) if len(lags) else None}

am = [c for c in a.caches.split(",") if "amass" in c]
if am:
    c = am[0]
    seqs = json.load(open(f"{c}/sequences.json"))
    rng = np.random.default_rng(0)
    pick = [q for q in seqs if q["name"].split("__")[0] in ("KIT", "CMU", "TotalCapture")]
    pick = [pick[i] for i in rng.choice(len(pick), min(200, len(pick)), replace=False)]
    for tag, cc in (("fixed", c), ("old_unfixed", a.old_amass)):
        if not os.path.exists(f"{cc}/unitree_g1_29dof_root_world.pt"): continue
        st = lag_stats(cc, pick); rep["timebase"][tag] = st
        print(f"[timebase] {tag:12s} {os.path.basename(cc)}: n={st['n']} |lag| 中位 {st['median_abs_lag']} 帧, "
              f"p90 {st['p90_abs_lag']}, ≤2 帧占比 {st['frac_lag_le2']:.2f}")
    if rep["timebase"].get("fixed", {}).get("median_abs_lag", 0) and rep["timebase"]["fixed"]["median_abs_lag"] > 3:
        warn.append(f"时间轴: 修后 |lag| 中位 {rep['timebase']['fixed']['median_abs_lag']} 帧 > 3")

print("\n== 报警", len(warn))
for w in warn: print("  !", w)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(rep, open(a.out, "w"), indent=1)
print("WROTE", a.out)
