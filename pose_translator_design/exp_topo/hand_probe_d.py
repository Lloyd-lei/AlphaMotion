"""探针 D —— 掩码配方对上肢的影响 (0831)。

先说清一件事实 (masking.py 读出来的, 不是推测): mask_mix 里的 `ee` 家族是
"只留树叶可见, 其余全遮", 树叶在 human_smpl (22 关节, 无手指) 上 =
L_Toe/R_Toe/Head/L_Wrist/R_Wrist —— 也就是说 ee 那 25% 里手腕是**唯一可见**的
上肢关节, 不是被遮的。任务描述里"ee 遮末端让模型幻觉手"是反的。

真正要量的是: 训练时有多大比例的样本, 编码器根本看不到手臂, 却仍要全身重建。
  root   (15%): 只看根 -> 手臂 100% 靠先验
  sparse (30%): 只看根 + 1..3 个随机关节 -> 手臂大概率看不到
  rand   (30%): 保留 U(0.2,0.8)
  ee     (25%): 只看根 + 5 个树叶 (含双腕)
本脚本在判官那 24x90 固定窗口上, 分别用每个掩码家族喂编码器, 量目标具身的
上肢/手/腿 FK 位置误差 (cm), 并和不遮的基线比。
"root 遮蔽下误差几乎不涨" = 该部位本来就来自先验而不是码 -> 病 B 的直接证据。
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hand_probe_common as H
from dual_common import fk_pos_torch
from masking import sample_joint_mask, leaf_mask
import v2_judge as VJ

LEG_KW = VJ.HIP_KW + VJ.KNEE_KW + ("ankle", "foot", "toe")
FAMS = ["none", "rand", "ee", "root", "sparse"]


@torch.no_grad()
def run_family(model, src, tgts, rows, fam, dev, bs=256, seed=0):
    torch.manual_seed(seed)
    acc = {t.name: {} for t in tgts}
    GRP = {t.name: {"ul": VJ.kw_mask(t.spec, VJ.UL_KW, dev),
                    "hand": H.chain_leaf_mask(t.spec, ("shoulder", "upper_arm", "clavicle"), dev),
                    "leg": VJ.kw_mask(t.spec, LEG_KW, dev)} for t in tgts}
    for t in tgts:
        for p in GRP[t.name]:
            acc[t.name][p] = []
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]; B = len(r)
        p9 = src.pose9(r)
        jm = None
        if fam != "none":
            jm = sample_joint_mask(src.spec, B, {fam: 1.0}, device=dev)
        zp, zr = model.encode(p9, src.geom(B), joint_mask=jm)
        for t in tgts:
            drot, _ = model.decode(zp, zr, t.geom(B))
            e = (fk_pos_torch(drot, t.spec) - fk_pos_torch(t.pose9(r)[..., :6], t.spec)).norm(dim=-1)
            for p, m in GRP[t.name].items():
                acc[t.name][p].append(e[:, m].cpu())
    return {n: {p: round(float(torch.cat(v).mean()), 2) for p, v in d.items()}
            for n, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    ck, cfg, model, Bd = H.load_ckpt(a.ckpt, dev)
    tag = a.tag or os.path.basename(ck.rstrip("/"))
    src = Bd[H.SRC]; tgts = [Bd[t] for t in H.TGTS]
    rows = H.judge_rows()

    lm = leaf_mask(src.spec)
    vis_ee = [src.spec.joint_names[i] for i in range(src.spec.J) if lm[i]]
    res = {"ckpt": ck, "tag": tag, "mask_mix": cfg.get("mask_mix"),
           "mask_prob": cfg.get("mask_prob"),
           "human_leaves_visible_under_ee": vis_ee, "fams": {}}
    print(f"[D] mask_mix={cfg.get('mask_mix')} mask_prob={cfg.get('mask_prob')}")
    print(f"[D] ee 家族下 human_smpl 可见关节 = 根 + {vis_ee}")
    for fam in FAMS:
        r = run_family(model, src, tgts, rows, fam, dev)
        res["fams"][fam] = r
        for n, d in r.items():
            print(f"[D] {fam:7s} {n:20s} ul={d['ul']:6.2f} hand={d['hand']:6.2f} leg={d['leg']:6.2f}",
                  flush=True)
    out = a.out or f"{H.RUNS}/HANDPROBE_D_{tag}.json"
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    print("->", out)


if __name__ == "__main__":
    main()
