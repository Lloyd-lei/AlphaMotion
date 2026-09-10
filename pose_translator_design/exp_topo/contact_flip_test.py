"""Greenwich-v2 接触通道的"聋子测试" (0831)。

    python contact_flip_test.py --ckpt <run 目录或 runs/ 下的名字> [--tgt unitree_g1_29dof]

问的是一个问题: **模型到底有没有在读第 10 通道?**

为什么必须单独考这一项 (我们被这个坑烧过一次): XW 那条臂当初"接上了"时序码, 训练损失
和验收数字都在动, 直到有人做了一次归零测试 —— 把时序码整块置零, 输出几乎不变 ——
才发现那条臂对自己的时序码是聋的, 之前所有解读全部作废。接触通道是同一类风险:
接触状态是姿态的确定性函数 (spec §7 纠正四), 模型完全可以从姿态自己推出接触、
把输入通道当装饰品晾着, 而 loss 曲线一切正常。所以通道有没有被读, 只能靠反事实来问。

做法 (三条臂, 都在同一批帧上):
  flip_off  真接触的关节, 把输入接触位从软标签翻成 0 ("你没踩地")
  flip_on   明确腾空的关节 (gap 大), 把接触位翻成 1 ("你必须踩地")
  perm      接触向量在关节维上随机置换 (接触**数量**不变、**位置**全错)
            —— 这条是对照臂: 只看总量的模型在 perm 上不会动, 真读了模式的会动。

看四个量:
  delta_cm      被翻关节的解码 FK 位置变化 (cm) —— 主判据
  bleed_cm      **没被翻**的关节的位置变化 (cm) —— 全局漂移本底, delta 要显著高于它
  dh_cm         被翻关节离地高的**带符号**变化: flip_off 应为正 (被告知没踩 -> 抬起来),
                flip_on 应为负 (被告知要踩 -> 压下去)。方向对不对比幅度更能说明"读懂了"
  sign_rate     上面那个符号在逐 (帧,关节) 上成立的比例
  code_chg      FSQ 整数码有变化的格子占比 —— 通道若连码都改不动, 后面全免谈

判据 (先立线, 跑之前写死):
  FAIL  delta_cm < 0.5 或 delta_cm < 3 x bleed_cm      通道是装饰品, 整个方案作废
  WEAK  sign_rate < 0.60                                读到了但没读懂物理方向
  PASS  两条都过
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import CACHE_ROOT                                          # noqa: E402
from dual_common import build_model, geom, fk_pos_torch, CONTACT_UP    # noqa: E402
from contact_label import limb_groups                                  # noqa: E402
import v2_judge as VJ                                                  # noqa: E402

RUNS = f"{CACHE_ROOT}/runs"


def sub_contact(cache, emb, rows):
    """只取需要的行 -> (soft [R,J], hard [R,J] bool, gap_root [R])。"""
    f = f"{cache}/{emb}_contact.npz"
    if not os.path.exists(f):
        return None
    z = np.load(f)
    r = np.asarray(rows)
    # 0831 "距离进、置信出": 通道喂符号距离, 与训练/判官同一口径。
    hdr_ = json.loads(str(z["header"]))
    thr_, D_ = float(hdr_["thr_m"]), float(hdr_["D_used_m"])
    g_ = z["gap"][r].astype(np.float32); s_ = thr_ - g_
    soft = torch.as_tensor((np.sign(s_) * np.log1p(np.abs(s_) / thr_)
                            / np.log1p(D_ / thr_)).astype(np.float32))
    hard = torch.as_tensor(z["contact"][r] | (g_ < 0))
    gap = torch.as_tensor(g_)
    return soft, hard, gap


def decode_with(model, data, src, tgt, pose, con, dev, chunk=256):
    """给定接触通道 con [R,J] -> (FK 位置 cm [R,J,3], 离地高 m [R,J], 码整数)。"""
    outs, ints = [], []
    R = pose.shape[0]
    for i in range(0, R, chunk):
        sl = slice(i, min(i + chunk, R))
        B = pose[sl].shape[0]
        x = torch.cat([pose[sl], con[sl][..., None]], -1)
        zp, zr, cc = model.encode(x, geom(data, src, B), return_codes=True)
        prot, _ = model.decode(zp, zr, geom(data, tgt, B))
        outs.append(fk_pos_torch(prot, data[tgt]["spec"]))
        ints.append(torch.cat([cc[0].reshape(B, -1), cc[1].reshape(B, -1)], -1))
    return torch.cat(outs), torch.cat(ints)


def group_xfer(src_spec, tgt_spec, m_src):
    """源侧被翻的关节 [R,J_src] -> 目标侧"同肢体组"的关节 [R,J_tgt]。

    为什么要这一步: 通道是**源具身**的输入, 解码出来的却是**目标具身**的身体, 两边
    关节数不同、也没有一一对应。但"我们告诉它左脚没踩地, 左脚有没有动"这个问题是
    有意义的 —— 用标注器自己那套 limb_groups (contact_label.limb_groups, 不另立口径)
    把源关节映到肢体组、再落到目标关节上。src==tgt 时它退化成关节级掩码的组级版本。
    """
    gs = limb_groups(list(src_spec.joint_names))
    gt = limb_groups(list(tgt_spec.joint_names))
    uniq = sorted(set(gs) | set(gt))
    gi_s = torch.as_tensor([uniq.index(g) for g in gs], device=m_src.device)
    gi_t = torch.as_tensor([uniq.index(g) for g in gt], device=m_src.device)
    R = m_src.shape[0]
    hit = torch.zeros(R, len(uniq), dtype=torch.bool, device=m_src.device)
    hit.scatter_(1, gi_s[None].expand(R, -1), m_src)      # 组内任一关节被翻 -> 该组被翻
    return hit[:, gi_t]                                    # [R,J_tgt]


def arm(model, data, src, tgt, pose, con_base, con_new, flip_m, gap_root, dev, base=None):
    """一条反事实臂 -> 指标 dict。flip_m [R,J_tgt] bool = 目标侧受影响的关节。"""
    p0, i0 = base if base is not None else decode_with(model, data, src, tgt, pose, con_base, dev)
    p1, i1 = decode_with(model, data, src, tgt, pose, con_new, dev)
    d = (p1 - p0).norm(dim=-1)                                    # [R,J] cm
    dh = (p1 - p0)[..., CONTACT_UP]                               # [R,J] cm, 带符号
    fm = flip_m
    if int(fm.sum()) == 0:
        return None
    return dict(n_flip=int(fm.sum()),
                delta_cm=round(float(d[fm].mean()), 3),
                delta_p95_cm=round(float(d[fm].quantile(.95)), 3),
                bleed_cm=round(float(d[~fm].mean()), 3),
                dh_cm=round(float(dh[fm].mean()), 3),
                code_chg=round(float((i0 != i1).float().mean()), 4)), dh[fm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tgt", default=None, help="默认考 v2_judge 的两个目标具身")
    ap.add_argument("--n_rows", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--append", default=None)
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(0)
    ck = a.ckpt if os.path.isdir(a.ckpt) else f"{RUNS}/{a.ckpt}"
    cfg = json.load(open(f"{ck}/config.json"))
    if not cfg.get("contact_chan"):
        print(f"N/A: {ck} 不是 --contact_chan 训的, 没有第 10 通道可翻。")
        return
    caches = [c for c in str(cfg.get("cache", VJ.LAFAN)).split(",") if c]
    cache = caches[0]
    tgts = [a.tgt] if a.tgt else VJ.TGTS
    src = VJ.SRC

    # 行的选法: 从**有接触**的帧里随机抽 (全腾空帧上这个测试没有内容)。
    z = np.load(f"{cache}/{src}_contact.npz")
    anyc = z["contact"].any(1)
    pool = np.flatnonzero(anyc)
    rng = np.random.default_rng(0)
    rows = np.sort(rng.choice(pool, min(a.n_rows, len(pool)), replace=False))
    del z, anyc, pool

    model = build_model(cfg, dev)
    model.load_state_dict(torch.load(f"{ck}/model.pt", map_location=dev))
    model.eval()
    from semantics import build_name_embeddings
    names_all = []
    for n in [src] + tgts:
        d = np.load(f"{cache}/{n}_spec.npz", allow_pickle=True)
        names_all += [str(x) for x in d["joint_names"]]
    lut = build_name_embeddings(names_all, device=dev, encoder=cfg.get("sem_encoder", "siglip"))
    rows_t = torch.as_tensor(rows)
    data = {n: VJ.load_emb(n, rows_t, caches, cfg, lut, dev) for n in [src] + tgts}

    sc = sub_contact(cache, src, rows)
    assert sc is not None, f"缺 {src}_contact.npz"
    soft, hard, gap = [t.to(dev) for t in sc]
    gap_root = gap[:, 0]
    R, J = soft.shape

    res = {"ckpt": ck, "n_rows": int(R), "src": src, "arms": {}}
    with torch.no_grad():
        for tgt in tgts:
            pose = data[src]["pose"]
            base = decode_with(model, data, src, tgt, pose, soft, dev)
            out = {}
            # --- flip_off: 真接触 -> 0 ---
            sp, tp = data[src]["spec"], data[tgt]["spec"]
            m_off = hard
            c_off = torch.where(m_off, torch.full_like(soft, -1.0), soft)   # "你没踩地"=远
            r_off = arm(model, data, src, tgt, pose, soft, c_off,
                        group_xfer(sp, tp, m_off), gap_root, dev, base)
            # --- flip_on: 明确腾空 (gap > 3x 接触阈, 这里用 15cm) -> 1 ---
            m_on = (~hard) & (gap > 0.15)
            c_on = torch.where(m_on, torch.full_like(soft, 0.5), soft)      # "你必须踩地"
            r_on = arm(model, data, src, tgt, pose, soft, c_on,
                       group_xfer(sp, tp, m_on), gap_root, dev, base)
            # --- perm 对照臂: 接触数量不变、位置全错 ---
            perm = torch.stack([torch.randperm(J, device=dev) for _ in range(R)])
            c_pm = torch.gather(soft, 1, perm)
            m_pm = hard | (torch.gather(hard, 1, perm))
            r_pm = arm(model, data, src, tgt, pose, soft, c_pm,
                       group_xfer(sp, tp, m_pm), gap_root, dev, base)
            for nm, r in (("flip_off", r_off), ("flip_on", r_on), ("perm", r_pm)):
                if r is None:
                    continue
                st, dh = r
                # 方向: flip_off 说"没踩" -> 应抬起 (dh>0); flip_on 说"要踩" -> 应压下 (dh<0)
                if nm == "flip_off":
                    st["sign_rate"] = round(float((dh > 0).float().mean()), 3)
                elif nm == "flip_on":
                    st["sign_rate"] = round(float((dh < 0).float().mean()), 3)
                out[nm] = st
            res["arms"][f"{src}->{tgt}"] = out

    # ---- 判决 ----
    print(f"\n== contact_flip_test [{ck}]  {R} 帧 (均为有接触帧)")
    hdr = f"{'direction / arm':44s} {'n_flip':>8s} {'delta_cm':>9s} {'bleed_cm':>9s} {'dh_cm':>8s} {'sign':>6s} {'code_chg':>9s}"
    print(hdr); print("-" * len(hdr))
    verdicts = {}
    for k, arms in res["arms"].items():
        for nm, st in arms.items():
            print(f"{k+' / '+nm:44s} {st['n_flip']:8d} {st['delta_cm']:9.3f} {st['bleed_cm']:9.3f} "
                  f"{st['dh_cm']:8.3f} {st.get('sign_rate', float('nan')):6.3f} {st['code_chg']:9.4f}")
        o = arms.get("flip_off")
        if o is None:
            continue
        if o["delta_cm"] < 0.5 or o["delta_cm"] < 3 * max(o["bleed_cm"], 1e-6):
            v = "FAIL"
        elif o.get("sign_rate", 0) < 0.60:
            v = "WEAK"
        else:
            v = "PASS"
        verdicts[k] = v
        print(f"  -> {k}: {v}  (delta {o['delta_cm']:.2f}cm vs bleed {o['bleed_cm']:.2f}cm, "
              f"方向命中 {o.get('sign_rate', 0)*100:.0f}%)")
    res["verdict"] = verdicts
    res["criteria"] = ("FAIL: delta_cm<0.5 或 delta_cm<3*bleed_cm; "
                       "WEAK: sign_rate<0.60; 否则 PASS")
    json.dump(res, open(f"{ck}/contact_flip.json", "w"), indent=1)
    if a.append:
        allr = json.load(open(a.append)) if os.path.exists(a.append) else {}
        allr[os.path.basename(ck.rstrip("/"))] = res
        json.dump(allr, open(a.append, "w"), indent=1)
    print(f"\n判决 {verdicts} -> {ck}/contact_flip.json")
    print("FAIL = 模型对接触通道是聋的, 接触方案不成立, 不要再往下解读任何接触数字。")


if __name__ == "__main__":
    main()
