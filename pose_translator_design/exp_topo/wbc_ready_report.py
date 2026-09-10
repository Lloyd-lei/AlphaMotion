"""wbc_ready_audit 结果 -> 按家族/数据集分表 + 挑最差/中位片段 (供 slipvid_amass 渲染)。 用法: wbc_ready_report.py <json> [tgt]"""
import sys, json, numpy as np
R = json.load(open(sys.argv[1])); tgts = [t for t in R if isinstance(R[t], dict) and "clips" in R[t]]
if len(sys.argv) > 2: tgts = [sys.argv[2]]
def med95(v): v = np.array(v, float); v = v[~np.isnan(v)]; return (np.median(v), np.percentile(v, 95)) if len(v) else (np.nan, np.nan)
for tgt in tgts:
    per = R[tgt]["clips"]; print(f"\n=== {tgt} {len(per)} clips  thr={R[tgt]['thr_m']*100:.1f}cm")
    gt95 = {k: med95([r[f"GT.{k}"] for r in per])[1] for k in ("v_med", "drift_med")}
    gt95["onset"] = med95([r["GT.onset_abs_med"] for r in per])[1]; gt95["f1"] = np.percentile([r["GT.f1"] for r in per if not np.isnan(r["GT.f1"])], 5)
    print("  GT 尺子 (p95): 接触脚速 %.1f cm/s  站立漂移 %.1f cm  时机 %.1f 帧  F1(p5) %.2f" % (gt95["v_med"], gt95["drift_med"], gt95["onset"], gt95["f1"]))
    def row(sub, name):
        n = len(sub)
        if n == 0: return
        p_slip = np.mean([r["M_h.drift_med"] <= gt95["drift_med"] for r in sub if not np.isnan(r["M_h.drift_med"])] or [np.nan])
        p_v = np.mean([r["M_h.v_med"] <= gt95["v_med"] for r in sub if not np.isnan(r["M_h.v_med"])] or [np.nan])
        p_on = np.mean([r["M.onset_abs_med"] <= gt95["onset"] for r in sub if not np.isnan(r["M.onset_abs_med"])] or [np.nan])
        p_f1 = np.mean([r["M.f1"] >= gt95["f1"] for r in sub])
        allp = np.mean([(r["M_h.drift_med"] <= gt95["drift_med"] if not np.isnan(r["M_h.drift_med"]) else True) and r["M.f1"] >= gt95["f1"]
                        and (r["M.onset_abs_med"] <= gt95["onset"] if not np.isnan(r["M.onset_abs_med"]) else True) for r in sub])
        print("  %-14s n=%3d | rot %4.1f° hand %4.1fcm foot %4.1fcm rootH %4.1fcm | 漂移 %4.1fcm(过%3.0f%%) 脚速 %4.1f(过%3.0f%%) | F1 %.2f(过%3.0f%%) 时机 %3.1f帧(过%3.0f%%) 触地位 %4.1fcm 漏 %3.0f%% | 穿地 %4.1f%% 悬空 %4.1f%% | 全过 %3.0f%%" % (
            name, n, med95([r["rot_deg"] for r in sub])[0], med95([r["hand_cm"] for r in sub])[0], med95([r["foot_cm"] for r in sub])[0], med95([r["rooth_cm"] for r in sub])[0],
            med95([r["M_h.drift_med"] for r in sub])[0], p_slip * 100, med95([r["M_h.v_med"] for r in sub])[0], p_v * 100,
            med95([r["M.f1"] for r in sub])[0], p_f1 * 100, med95([r["M.onset_abs_med"] for r in sub])[0], p_on * 100, med95([r["M.onset_pos_med"] for r in sub])[0],
            med95([r["M.onset_miss"] for r in sub])[0] * 100, med95([r["M.pen_frac"] for r in sub])[0] * 100, med95([r["M.float_frac"] for r in sub])[0] * 100, allp * 100))
    row(per, "ALL")
    print("  -- 按家族"); [row([r for r in per if r["family"] == f], f) for f in sorted(set(r["family"] for r in per))]
    print("  -- 按数据集"); [row([r for r in per if r["dataset"] == d], d) for d in sorted(set(r["dataset"] for r in per))]
    # 挑片段: 只看有位移的走路类 (disp_h>1m), 综合分 = 漂移/GT95 + rot/中位
    loc = [r for r in per if r["disp_h"] > 1.0 and not np.isnan(r["M_h.drift_med"])]
    sc = lambda r: r["M_h.drift_med"] / gt95["drift_med"] + (1 - r["M.f1"]) * 3
    loc.sort(key=sc)
    print("  -- 有位移片段 %d 条, 最好/中位/最差 (综合=漂移/GT95 + 3*(1-F1)):" % len(loc))
    for tag, r in (("best", loc[0]), ("median", loc[len(loc) // 2]), ("worst", loc[-1]), ("worst-1", loc[-2]), ("worst-2", loc[-3])):
        print("   %-7s %-60s rot %4.1f drift %4.1f v %4.1f F1 %.2f onset %3.1f miss %.0f%% pen %.0f%%" % (tag, r["name"][:60], r["rot_deg"], r["M_h.drift_med"], r["M_h.v_med"], r["M.f1"], r["M.onset_abs_med"], r["M.onset_miss"]*100, r["M.pen_frac"]*100))
