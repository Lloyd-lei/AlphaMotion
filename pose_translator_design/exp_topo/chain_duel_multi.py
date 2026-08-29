"""Numeric half of the chain duel, over MANY clips.

    chain_duel_multi.py --cache /var/tmp/alphamotion/genmo_cache

chain_duel.py renders one clip for the eye; this one answers whether the gap is
real.  One clip proved nothing three times on 0813, so every number here is a
per-clip distribution, and the verdict line reports how many clips agree, not
just the mean.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                            # noqa: E402
import eval_newbody as E                                         # noqa: E402
import manifold_project as MP                                    # noqa: E402
import greenwich_carnival as GC                                  # noqa: E402
from bridge_a import load_bridge_a                               # noqa: E402
from bridge_b import tokens_of                                   # noqa: E402
from manifold_project import _limits, fit_angles                 # noqa: E402
from common import rot6d_to_matrix                               # noqa: E402
from chain_duel import REACH, KEYS, dir_series, amp_series       # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/var/tmp/alphamotion/genmo_cache")
    ap.add_argument("--source", default="genmo", choices=("genmo", "amass"),
                    help="amass = the corpus A3/B3 were actually trained on; "
                         "the control that separates an intrinsic cost of the "
                         "token layer from GEM motion being out of distribution")
    ap.add_argument("--n_clips", type=int, default=24)
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--robots", nargs="+",
                    default=["Unitree G1", "Booster T1", "Fourier GR3"])
    ap.add_argument("--T", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, args.T

    sp = np.load(f"{os.environ['V5_AMASS']}/human_smpl_spec.npz", allow_pickle=True)
    sspec = E.SkeletonSpec(str(sp["name"]), sp["parents"], sp["rest_offsets"],
                           [str(x) for x in sp["joint_names"]])
    if args.source == "genmo":
        seqs = [s for s in json.load(open(f"{args.cache}/sequences.json"))
                if s["len"] >= T]
        R6 = torch.load(f"{args.cache}/human_smpl_global6d.pt", map_location="cpu")
    else:
        C = os.environ["V5_AMASS"]
        allseq = [s for s in json.load(open(f"{C}/sequences.json"))
                  if s["len"] >= T]
        g0 = np.random.default_rng(0)
        pick = [allseq[i] for i in g0.choice(len(allseq),
                                             min(args.n_clips, len(allseq)),
                                             replace=False)]
        loc = torch.load(f"{C}/human_smpl_paired_local6d.pt", map_location="cpu")
        chunks, seqs, at = [], [], 0
        for s in pick:                       # cache is LOCAL; compose first
            gr6, _gp, _r = V.build_global(loc[s["start"]:s["start"] + T],
                                          sspec, "cpu")
            chunks.append(gr6.cpu())
            seqs.append({"name": s["name"], "start": at, "len": T})
            at += T
        R6 = torch.cat(chunks)
    print(f"来源 {args.source}  片段 {len(seqs)} (>= {T} 帧)  "
          f"机器人 {len(args.robots)}", flush=True)

    class S:
        parents = sp["parents"]; rest_offsets = sp["rest_offsets"]
    model, cfg = E.load_run(f"{V.RUNS}/{args.spatial}", dev)
    enc = cfg.get("sem_encoder")
    sdof = np.load(f"{os.environ['V5_AMASS']}/human_smpl_dof.npz")["dof"]
    sg = E.target_geom(sspec, sdof, dev, sem_encoder=enc)
    ma, _ = load_bridge_a(f"{V.RUNS}/{args.stage_a}", dev)
    W = 5 ** torch.arange(6, device=dev)
    kj_h, _n, _t = V.key_joints(sspec)
    kidx_h = dict(zip(("root",) + KEYS, kj_h))

    bodies = {}
    for rn in args.robots:
        _n2, xml, st = [r for r in GC.ROSTER if r[0] == rn][0]
        spec_t, dof_t, rest_t, qn_t, xml_t = E.get_target(xml, E.CACHE_AMASS)
        dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
        rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)
        lo_, hi_, lim_ = _limits(dof_d, 1.0)
        kj_t, _a, _b = V.key_joints(spec_t)
        bodies[rn] = dict(spec=spec_t, g=E.target_geom(spec_t, dof_t, dev,
                                                       sem_encoder=enc),
                          dof=dof_d, rest=rest_d, lo=lo_, hi=hi_, lim=lim_,
                          span=(hi_ - lo_).clamp(min=1e-6),
                          kidx=dict(zip(("root",) + KEYS, kj_t)))

    rows = []
    for si, s in enumerate(seqs):
        gr6 = R6[s["start"]:s["start"] + T]
        hp = V.fk_pos(gr6.numpy(), S)
        pose = torch.cat([gr6, torch.from_numpy(hp).float() / REACH], -1).to(dev)
        fp, fr = model.encode_features(pose, E.geom_batch(sg, T))
        ipS, irS = model.ints_from_features(fp, fr)
        ip, ir = ipS[0], irS[0]
        qa = (torch.cat([ip, ir], 1).float() - 4.0) / 3.5
        codes = torch.cat([ip, ir], 1).long()[None]
        tok, ep = tokens_of(ma, codes.to(dev))
        z = ma.fsq.from_ints(((tok[0][:, None] // W) % 5)[None].float())[None]
        prob = F.softmax(ma.decode(z, ep, T)[0], -1)
        lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
        qb = (prob * lv).sum(-1); qb[0] = qa[0]
        lat = {"A": (model.fsqp.proj_out(qa[:, :128]),
                     model.fsqr.proj_out(qa[:, 128:])),
               "B": (model.fsqp.proj_out(qb[:, :128]),
                     model.fsqr.proj_out(qb[:, 128:]))}
        reach_h = float(np.linalg.norm(
            hp[:, [kidx_h[k] for k in KEYS]] - hp[:, kidx_h["root"]][:, None],
            axis=-1).max())
        hdir = dir_series(hp, kidx_h, kidx_h["root"])
        hamp = amp_series(hp, kidx_h, kidx_h["root"], reach_h)
        for rn, B_ in bodies.items():
            out = {}
            for tag, (zp, zr) in lat.items():
                rot, _ = model.decode(zp, zr, E.geom_batch(B_["g"], T))
                Rg = rot6d_to_matrix(rot).double()
                qa_, _ = fit_angles(Rg, B_["spec"], B_["dof"], rest=B_["rest"],
                                    clamp=True, method="greedy")
                _gR, gpos = MP.fk_from_angles(qa_, B_["spec"], B_["dof"],
                                              rest=B_["rest"], root_R=Rg[:, 0])
                p = gpos.cpu().numpy()
                near = (((qa_ - B_["lo"][None]) / B_["span"][None] < 0.02) |
                        ((B_["hi"][None] - qa_) / B_["span"][None] < 0.02)) \
                    & B_["lim"][None]
                reach_r = float(np.linalg.norm(
                    p[:, [B_["kidx"][k] for k in KEYS]]
                    - p[:, B_["kidx"]["root"]][:, None], axis=-1).max())
                dr = dir_series(p, B_["kidx"], B_["kidx"]["root"])
                am = amp_series(p, B_["kidx"], B_["kidx"]["root"], reach_r)
                # pipeline floor: same numbers with the robot's time axis
                # shuffled. Without it "follow 0.44" has no scale — the control-arm
                # rule (管线地板/随机方向/同尺重测) says build the floor first.
                perm = np.random.default_rng(0).permutation(len(p))
                dsh = dir_series(p[perm], B_["kidx"], B_["kidx"]["root"])
                out[tag] = dict(
                    pos=p,
                    follow_shuf=float(np.mean([
                        np.mean([np.corrcoef(hdir[k][:, a], dsh[k][:, a])[0, 1]
                                 for a in range(3)]) for k in KEYS])),
                    follow=float(np.mean([
                        np.mean([np.corrcoef(hdir[k][:, a], dr[k][:, a])[0, 1]
                                 for a in range(3)]) for k in KEYS])),
                    amp=float(np.mean([am[k].std() / (hamp[k].std() + 1e-9)
                                       for k in KEYS])),
                    jerk=float(np.abs(np.diff(p, 3, axis=0)).mean()),
                    limit=float(near[:, B_["lim"]].float().mean())
                    if B_["lim"].any() else 0.0)
            rows.append({"clip": s["name"], "robot": rn,
                         "followA": out["A"]["follow"], "followB": out["B"]["follow"],
                         "floorA": out["A"]["follow_shuf"],
                         "floorB": out["B"]["follow_shuf"],
                         "ampA": out["A"]["amp"], "ampB": out["B"]["amp"],
                         "jerkA": out["A"]["jerk"], "jerkB": out["B"]["jerk"],
                         "limA": out["A"]["limit"], "limB": out["B"]["limit"],
                         "gap_cm": float(np.median(np.linalg.norm(
                             out["A"]["pos"] - out["B"]["pos"], axis=-1)))})
        if si % 5 == 0:
            print(f"  {si}/{len(seqs)}", flush=True)

    def col(k):
        return np.array([r[k] for r in rows])
    print(f"\n== {len(rows)} 个 (片段 x 机器人) 组合 ==")
    print(f"{'':>22}{'A 直连':>14}{'B 过Equator':>16}{'B 更好的比例':>14}")
    for label, ka, kb, better in (("跟随度(越高越好)", "followA", "followB", "hi"),
                                  ("幅度比(越接近1越好)", "ampA", "ampB", "amp"),
                                  ("jerk(越低越好)", "jerkA", "jerkB", "lo"),
                                  ("限位命中(越低越好)", "limA", "limB", "lo")):
        a, b = col(ka), col(kb)
        if better == "hi":
            frac = float((b > a).mean())
        elif better == "lo":
            frac = float((b < a).mean())
        else:
            frac = float((np.abs(b - 1) < np.abs(a - 1)).mean())
        print(f"{label:>22}{np.median(a):14.3f}{np.median(b):16.3f}"
              f"{frac*100:13.0f}%")
    fl = np.concatenate([col("floorA"), col("floorB")])
    print(f"\n管线地板(打乱机器人时间轴): 跟随度 中位 {np.median(fl):+.3f} "
          f"— 0.44 相对这个地板才有意义")
    g = col("gap_cm")
    print(f"\n两条链在机器人上的差距: 中位 {np.median(g):.2f} cm  "
          f"p90 {np.percentile(g, 90):.2f} cm")
    fa, fb = col("followA"), col("followB")
    print(f"跟随度下降: 中位 {np.median(fa - fb):+.3f}  "
          f"{(fb < fa).mean()*100:.0f}% 的组合里 B 更差")
    json.dump(rows, open(f"{V.RUNS}/CHAIN_DUEL_MULTI_{args.source}_0814.json",
                     "w"), indent=1)
    print(f"-> {V.RUNS}/CHAIN_DUEL_MULTI_{args.source}_0814.json", flush=True)


if __name__ == "__main__":
    main()
