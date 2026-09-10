"""探针 B —— 码里到底有没有手臂信息 (0831)。冻结码本, 训一个小 MLP 读出。

三条输入, 同一批行、同一个 MLP 家族、同一个目标:
  raw   人体 pose9 拉平 (22x9) —— 信息天花板 (编码器输入本身)
  code  FSQ 整数格码全量拉平 (K x f x 2 流, 归一到 [-1,1]) —— 解码器实际拿到的东西
  pool  proj_out(tanh(features)).mean(1) 拼两流 —— zspace_battery 那条池化特征
目标: 目标具身 (g1 = 留出 / gr3 = 训练内) 的上肢 FK 位置 (cm) 与腿 FK 位置 (cm),
      关节掩码直接用 v2_judge.kw_mask + UL_KW / 腿关键词, 和判官同一把尺子。
留出: 训练行取自 v2_judge 的非 val 片段; 测试行 = 判官那 24 x 90 帧固定窗口。

判读:
  E_code ≈ E_raw << E_dec  -> 信息在码里, 解码器没用上 (病 A)
  E_code >> E_raw          -> 码把手臂丢了 (病 B)
顺手补一个判官缺的口径: 腿的 FK 位置误差 (cm), 因为判官腿用度、手臂用厘米,
不同尺子没法回答"手臂是不是特别差"。
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hand_probe_common as H
from dual_common import fk_pos_torch
import v2_judge as VJ

LEG_KW = VJ.HIP_KW + VJ.KNEE_KW + ("ankle", "foot", "toe")


@torch.no_grad()
def feats_and_targets(model, src, tgts, rows, dev, bs=512):
    """一次前向拿齐三种输入 + 每个目标的 GT/解码 上肢&腿位置。"""
    F = {"raw": [], "code": [], "pool": []}
    T = {}
    GRP = {}
    for t in tgts:
        GRP[t.name] = {"ul": VJ.kw_mask(t.spec, VJ.UL_KW, dev),
                       "hand": H.chain_leaf_mask(t.spec, ("shoulder", "upper_arm", "clavicle"), dev),
                       "leg": VJ.kw_mask(t.spec, LEG_KW, dev)}
        T[t.name] = {}
        for p_ in GRP[t.name]:
            T[t.name][f"gt_{p_}"] = []; T[t.name][f"dec_{p_}"] = []
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]; B = len(r)
        p9 = src.pose9(r)
        g = src.geom(B)
        F["raw"].append(p9.reshape(B, -1).half().cpu())
        fp, fr = model.encode_features(p9, g)
        ip, ir = model.ints_from_features(fp, fr)
        lv = torch.as_tensor(model.fsqp.levels, device=dev, dtype=torch.float32)
        half = (lv - 1) / 2
        cp = (ip[0].float() - torch.ceil(half)) / half
        cr = (ir[0].float() - torch.ceil(half)) / half
        F["code"].append(torch.cat([cp.reshape(B, -1), cr.reshape(B, -1)], -1).half().cpu())
        F["pool"].append(torch.cat([model.fsqp.proj_out(torch.tanh(fp)).mean(1),
                                    model.fsqr.proj_out(torch.tanh(fr)).mean(1)],
                                   -1).half().cpu())
        zp, zr = model.encode(p9, g)
        for t in tgts:
            gt_ = t.geom(B)
            drot, _ = model.decode(zp, zr, gt_)
            fd = fk_pos_torch(drot, t.spec)
            fg = fk_pos_torch(t.pose9(r)[..., :6], t.spec)
            for p_, m_ in GRP[t.name].items():
                T[t.name][f"gt_{p_}"].append(fg[:, m_].half().cpu())
                T[t.name][f"dec_{p_}"].append(fd[:, m_].half().cpu())
    F = {k: torch.cat(v) for k, v in F.items()}
    T = {n: {k: torch.cat(v) for k, v in d.items()} for n, d in T.items()}
    return F, T


class MLP(nn.Module):
    def __init__(self, din, dout, h=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, dout))

    def forward(self, x):
        return self.net(x)


def train_probe(Xtr, Ytr, Xte, Yte, dev, steps=4000, bs=1024, lr=1e-3, seed=0):
    """Y [N,J,3] cm. 返回测试集逐关节平均 L2 (cm)。输入按训练集统计标准化。"""
    torch.manual_seed(seed)
    mu, sd = Xtr.float().mean(0), Xtr.float().std(0).clamp_min(1e-3)
    ymu = Ytr.float().reshape(len(Ytr), -1).mean(0)
    J = Ytr.shape[1]
    net = MLP(Xtr.shape[1], J * 3).to(dev)
    with torch.no_grad():
        net.net[-1].bias.copy_(ymu.to(dev)); nn.init.zeros_(net.net[-1].weight)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    N = len(Xtr)
    mu_d, sd_d = mu.to(dev), sd.to(dev)
    for s in range(steps):
        idx = torch.randint(0, N, (bs,))
        x = ((Xtr[idx].to(dev).float() - mu_d) / sd_d)
        y = Ytr[idx].to(dev).float().reshape(bs, -1)
        loss = ((net(x) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    net.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(Xte), 2048):
            x = ((Xte[i:i + 2048].to(dev).float() - mu_d) / sd_d)
            p = net(x).reshape(-1, J, 3)
            errs.append((p - Yte[i:i + 2048].to(dev).float()).norm(dim=-1).cpu())
    return float(torch.cat(errs).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n_train", type=int, default=120000)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(0); np.random.seed(0)
    ck, cfg, model, Bd = H.load_ckpt(a.ckpt, dev)
    tag = a.tag or os.path.basename(ck.rstrip("/"))
    src = Bd[H.SRC]; tgts = [Bd[t] for t in H.TGTS]
    N = src.L.shape[0]
    val_ids, seqs = H.val_clip_ids()
    clip, _ = H.clip_of_row(N)
    tr_pool = np.flatnonzero(~np.isin(clip, list(val_ids)))
    g = np.random.default_rng(3)
    tr_rows = torch.as_tensor(np.sort(g.choice(tr_pool, min(a.n_train, len(tr_pool)),
                                               replace=False)))
    te_rows = H.judge_rows()
    print(f"[B] train {len(tr_rows)} 行 (非 val 片段) / test {len(te_rows)} 行 (判官固定窗口)",
          flush=True)
    Ftr, Ttr = feats_and_targets(model, src, tgts, tr_rows, dev)
    Fte, Tte = feats_and_targets(model, src, tgts, te_rows, dev)
    print("[B] 特征维度:", {k: v.shape[1] for k, v in Ftr.items()}, flush=True)

    res = {"ckpt": ck, "tag": tag, "n_train": int(len(tr_rows)), "n_test": int(len(te_rows)),
           "feat_dim": {k: int(v.shape[1]) for k, v in Ftr.items()}, "rows": {}}
    for t in tgts:
        for part in ("ul", "hand", "leg"):
            dec = float((Tte[t.name][f"dec_{part}"].float()
                         - Tte[t.name][f"gt_{part}"].float()).norm(dim=-1).mean())
            rad = float(Tte[t.name][f"gt_{part}"].float().norm(dim=-1).mean())
            row = {"decoder_cm": round(dec, 2), "radius_cm": round(rad, 1),
                   "decoder_rel": round(dec / max(rad, 1e-9), 4),
                   "n_joints": int(Tte[t.name][f"gt_{part}"].shape[1])}
            for fk in ("raw", "code", "pool"):
                e = train_probe(Ftr[fk], Ttr[t.name][f"gt_{part}"],
                                Fte[fk], Tte[t.name][f"gt_{part}"], dev, steps=a.steps)
                row[f"probe_{fk}_cm"] = round(e, 2)
            res["rows"][f"{t.name}/{part}"] = row
            print(f"[B] {t.name:20s} {part:4s} J={row['n_joints']:2d} r={row['radius_cm']:5.1f} "
                  f"decoder={row['decoder_cm']:5.2f} rel={row['decoder_rel']:.3f}  "
                  f"raw={row['probe_raw_cm']:5.2f} code={row['probe_code_cm']:5.2f} "
                  f"pool={row['probe_pool_cm']:5.2f}", flush=True)
    out = a.out or f"{H.RUNS}/HANDPROBE_B_{tag}.json"
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print("->", out)


if __name__ == "__main__":
    main()
