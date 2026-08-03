"""v5: train the topology-agnostic AE with GLOBAL-frame targets.

Hypothesis (from diagnostics): per-rig LOCAL rotation conventions are not
comparable across skeletons (G1 mean 92deg-from-identity vs SOMA 65deg), which
is why a shared net fails to reconstruct local rotation. Predicting per-joint
GLOBAL orientation + root-relative position (both in a common frame) should be
far more learnable across topologies -- WITHOUT any hand-crafted structure.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE_ROOT, SkeletonSpec, rot6d_to_matrix, matrix_to_rot6d,
                    fk_global, geodesic_deg, swing_twist_decompose, swing_twist_compose)
from model import TopoAE, hop_distance
from train_ae import load_skeleton, static_cond


def build_global(local6d, spec, device, chunk=4096):
    """FK all frames -> global_rot6d [N,J,6], global_pos [N,J,3] (root-rel, /reach)."""
    N = local6d.shape[0]
    outs_r, outs_p = [], []
    reach = None
    for i in range(0, N, chunk):
        b = local6d[i:i+chunk].to(device)
        gR, gp = fk_global(b, spec)             # [n,J,3,3],[n,J,3]
        outs_r.append(matrix_to_rot6d(gR).cpu())
        outs_p.append(gp.cpu())
    gr6 = torch.cat(outs_r, 0)
    gp = torch.cat(outs_p, 0)
    reach = gp.norm(dim=-1).max().item()
    return gr6, gp, max(reach, 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{CACHE_ROOT}/cache_big")
    ap.add_argument("--skels", nargs="+", default=["soma77", "g1_30"])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_latent", type=int, default=32)
    ap.add_argument("--use_index", action="store_true")
    ap.add_argument("--use_fsq", action="store_true")
    ap.add_argument("--fsq_stages", type=int, default=1)
    ap.add_argument("--w_twist", type=float, default=1.0)
    ap.add_argument("--target", default="global", choices=["global", "global_st", "glob2loc"])
    ap.add_argument("--pool_mode", default="perceiver", choices=["perceiver", "per_joint"])
    ap.add_argument("--w_pos", type=float, default=5.0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--out", default=f"{CACHE_ROOT}/runs/v5_global")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"; torch.manual_seed(0)

    data = {}
    for name in args.skels:
        spec, local6d = load_skeleton(args.cache, name)
        st = static_cond(spec, device)
        gr6, gp, reach = build_global(local6d, spec, device)
        pose = torch.cat([gr6, gp / reach], -1)         # [N,J,9]
        a0 = torch.nn.functional.normalize(st["rest_off"], dim=-1)  # [J,3] bone axis
        N = pose.shape[0]; perm = torch.randperm(N); nval = max(64, N // 10)
        data[name] = {"spec": spec, "st": st, "pose": pose, "gr6": gr6, "gp": gp, "a0": a0,
                      "local6d": local6d, "reach": reach, "tr": perm[nval:], "val": perm[:nval]}
        print(f"[{name}] J={spec.J} N={N} reach={reach:.1f}cm")

    model = TopoAE(d_model=args.d_model, n_latent=args.n_latent,
                   use_index=args.use_index, target=args.target, use_fsq=args.use_fsq,
                   fsq_stages=args.fsq_stages, pool_mode=args.pool_mode).to(device)
    bits = (model.fsq.bits * args.n_latent) if args.use_fsq else float("inf")
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M target=global "
          f"fsq={args.use_fsq} total_bits/frame={bits:.0f}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    names = args.skels

    def mk(name, idx):
        d = data[name]; B = len(idx)
        pose = d["pose"][idx].to(device)
        st = d["st"]
        return {
            "pose_in": pose,
            "rest_off": st["rest_off"].unsqueeze(0).expand(B, -1, -1),
            "bonelen": st["bonelen"].unsqueeze(0).expand(B, -1),
            "depth": st["depth"].unsqueeze(0).expand(B, -1),
            "hop": st["hop"].unsqueeze(0).expand(B, -1, -1),
            "kpm": None, "target": pose,
            "gt_local": d["local6d"][idx].to(device), "_name": name,
        }

    def losses(pred, tgt, a0, batch=None):
        gt = rot6d_to_matrix(tgt[..., :6])
        if args.target == "glob2loc":
            # encoder saw global; decoder outputs LOCAL rotation. Compare local rot +
            # FK-derived global rotation & position (the real translation targets).
            spec = data[batch["_name"]]["spec"]
            gt_local = rot6d_to_matrix(batch["gt_local"][..., :6])
            pr_local = rot6d_to_matrix(pred[..., :6])
            local_geo = geodesic_deg(pr_local, gt_local).mean()
            gpr, gpp = fk_global(pred[..., :6], spec)
            ggr, ggp = fk_global(batch["gt_local"][..., :6], spec)
            rot = geodesic_deg(gpr, ggr).mean()               # resulting GLOBAL rot err
            reach = data[batch["_name"]]["reach"]
            pos = (gpp - ggp).norm(dim=-1).mean() / reach
            return rot, pos, local_geo
        if args.target == "global_st":
            # pred: [swing_dir(3), twist_cos, twist_sin, pos(3)]
            a0b = a0.unsqueeze(0).expand(gt.shape[0], -1, -1)
            pr = swing_twist_compose(a0b, pred[..., 0:3], pred[..., 3], pred[..., 4])
            pos_pred = pred[..., 5:8]
            # explicit twist supervision: compare twist angle of pred vs gt about a0
            _, cg, sg = swing_twist_decompose(gt, a0b)
            ng = torch.sqrt(cg * cg + sg * sg).clamp(min=1e-6)
            np_ = torch.sqrt(pred[..., 3] ** 2 + pred[..., 4] ** 2).clamp(min=1e-6)
            twist_cos = (pred[..., 3] / np_ * cg / ng + pred[..., 4] / np_ * sg / ng).clamp(-1 + 1e-6, 1 - 1e-6)
            twist_err = torch.rad2deg(torch.arccos(twist_cos)).mean()
        else:
            pr = rot6d_to_matrix(pred[..., :6]); pos_pred = pred[..., 6:9]
            twist_err = torch.tensor(0.0, device=pred.device)
        rot = geodesic_deg(pr, gt).mean()
        pos = (pos_pred - tgt[..., 6:9]).norm(dim=-1).mean()
        return rot, pos, twist_err

    def evaluate():
        model.eval(); out = {}
        with torch.no_grad():
            for name in names:
                d = data[name]; b = mk(name, d["val"])
                pred, _ = model(b)
                rot, pos, tw = losses(pred, b["target"], d["a0"], batch=b)
                lbl = "local_geo_deg" if args.target == "glob2loc" else "twist_err_deg"
                out[name] = {"global_geo_deg": float(rot),
                             lbl: float(tw),
                             "pos_err_pct_reach": float(pos * 100),
                             "pos_err_cm": float(pos * d["reach"])}
        model.train(); return out

    log = []; t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad(); ll = []
        for name in names:
            d = data[name]; tr = d["tr"]
            sel = tr[torch.randint(0, len(tr), (args.bs,))]
            b = mk(name, sel)
            pred, _ = model(b)
            rot, pos, tw = losses(pred, b["target"], d["a0"], batch=b)
            if args.target == "glob2loc":
                loss = (tw / 30.0 + args.w_pos * pos) / len(names)   # tw = local geodesic (primary)
            else:
                loss = (rot / 30.0 + args.w_pos * pos + args.w_twist * tw / 30.0) / len(names)
            loss.backward(); ll.append(f"{name}:R{rot.item():.0f}/T{tw.item():.0f}/P{pos.item()*100:.1f}%")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 100 == 0:
            print(f"step {step+1}/{args.steps} " + " ".join(ll) + f" ({time.time()-t0:.0f}s)")
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev = evaluate(); ev["step"] = step + 1; log.append(ev)
            print("  EVAL", json.dumps(ev))
            json.dump(log, open(f"{args.out}/eval_log.json", "w"), indent=2)
    torch.save(model.state_dict(), f"{args.out}/model.pt")
    json.dump(vars(args), open(f"{args.out}/config.json", "w"), indent=2)
    print("DONE", args.out)


if __name__ == "__main__":
    main()
