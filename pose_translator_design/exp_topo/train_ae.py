"""Train the topology-agnostic SE3 autoencoder across multiple real topologies.

Core feasibility question: can ONE structure-free shared network reconstruct
single-frame per-joint SE3 for genuinely different skeletons (human-78, robot-30)
at once? And (ablation) does a hand-crafted structural prior help?
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from common import (CACHE_ROOT, SkeletonSpec, rot6d_to_matrix, matrix_to_rot6d,
                    fk_global, geodesic_deg)
from model import TopoAE, hop_distance


def load_skeleton(cache, name):
    d = np.load(f"{cache}/{name}_spec.npz", allow_pickle=True)
    spec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], list(d["joint_names"]))
    local6d = torch.load(f"{cache}/{name}_local6d.pt")   # [N,J,6]
    return spec, local6d


def static_cond(spec, device):
    hop = hop_distance(spec.parents)                     # [J,J]
    depth = hop[int(np.where(spec.parents < 0)[0][0])]   # root row -> depth per joint
    return {
        "rest_off": torch.as_tensor(spec.rest_offsets, device=device),
        "bonelen": torch.as_tensor(spec.norm_bonelen, device=device),
        "depth": torch.as_tensor(depth, device=device),
        "hop": torch.as_tensor(hop, device=device),
        "J": spec.J,
        "height": spec.height,
    }


def make_batch(local6d, static, idx, device):
    b = local6d[idx].to(device)                          # [B,J,6]
    B = b.shape[0]
    return {
        "local6d": b,
        "rest_off": static["rest_off"].unsqueeze(0).expand(B, -1, -1),
        "bonelen": static["bonelen"].unsqueeze(0).expand(B, -1),
        "depth": static["depth"].unsqueeze(0).expand(B, -1),
        "hop": static["hop"].unsqueeze(0).expand(B, -1, -1),
        "kpm": None,
    }


def recon_losses(pred6d, gt6d, spec, static):
    pred_R = rot6d_to_matrix(pred6d)
    gt_R = rot6d_to_matrix(gt6d)
    local_geo = geodesic_deg(pred_R, gt_R).mean()
    # FK both
    g_pred_R, g_pred_p = fk_global(pred6d, spec)
    g_gt_R, g_gt_p = fk_global(gt6d, spec)
    # position error normalized by skeleton height (scale-free, comparable across skels)
    pos_err = (g_pred_p - g_gt_p).norm(dim=-1).mean() / static["height"]
    glob_geo = geodesic_deg(g_pred_R, g_gt_R).mean()
    return local_geo, pos_err, glob_geo


def evaluate(model, skels, statics, val_idx, device):
    model.eval()
    out = {}
    with torch.no_grad():
        for name, (spec, local6d) in skels.items():
            idx = val_idx[name]
            batch = make_batch(local6d, statics[name], idx, device)
            pred6d, _ = model(batch)
            lg, pe, gg = recon_losses(pred6d, batch["local6d"], spec, statics[name])
            out[name] = {"local_geo_deg": float(lg), "pos_err_pct_height": float(pe * 100),
                         "global_geo_deg": float(gg)}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{CACHE_ROOT}/cache")
    ap.add_argument("--skels", nargs="+", default=["soma77", "g1_30"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_latent", type=int, default=8)
    ap.add_argument("--structured", action="store_true")
    ap.add_argument("--use_index", action="store_true")
    ap.add_argument("--out", default=f"{CACHE_ROOT}/runs/ae_plain")
    ap.add_argument("--w_pos", type=float, default=1.0)
    ap.add_argument("--w_grot", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.manual_seed(0)

    skels = {}
    statics = {}
    train_idx = {}; val_idx = {}
    for name in args.skels:
        spec, local6d = load_skeleton(args.cache, name)
        skels[name] = (spec, local6d)
        statics[name] = static_cond(spec, device)
        N = local6d.shape[0]
        perm = torch.randperm(N)
        n_val = max(64, N // 10)
        val_idx[name] = perm[:n_val]
        train_idx[name] = perm[n_val:]
        print(f"[{name}] J={spec.J} N={N} train={len(train_idx[name])} val={n_val}")

    model = TopoAE(d_model=args.d_model, n_latent=args.n_latent,
                   structured=args.structured, use_index=args.use_index).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"model params: {nparam/1e6:.2f}M  structured={args.structured}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)

    names = list(skels.keys())
    log = []
    t0 = time.time()
    for step in range(args.steps):
        # Gradient accumulation across ALL skeletons per optimizer step, so every
        # update sees every topology at once (avoids task ping-pong / interference).
        opt.zero_grad()
        logline = []
        for name in names:
            spec, local6d = skels[name]
            ti = train_idx[name]
            sel = ti[torch.randint(0, len(ti), (args.bs,))]
            batch = make_batch(local6d, statics[name], sel, device)
            pred6d, z = model(batch)
            lg, pe, gg = recon_losses(pred6d, batch["local6d"], spec, statics[name])
            loss = (lg / 30.0 + args.w_pos * pe + args.w_grot * gg / 30.0) / len(names)
            loss.backward()
            logline.append(f"{name}:L{lg.item():.1f}/P{pe.item()*100:.1f}%")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 100 == 0:
            print(f"step {step+1}/{args.steps} " + " ".join(logline) + f" ({time.time()-t0:.0f}s)")
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev = evaluate(model, skels, statics, val_idx, device)
            ev["step"] = step + 1
            log.append(ev)
            print("  EVAL", json.dumps(ev))
            with open(f"{args.out}/eval_log.json", "w") as f:
                json.dump(log, f, indent=2)
    torch.save(model.state_dict(), f"{args.out}/model.pt")
    with open(f"{args.out}/config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print("DONE", args.out)


if __name__ == "__main__":
    main()
