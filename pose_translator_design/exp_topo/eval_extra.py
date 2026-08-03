"""Post-training eval: per-frame decode latency (real-time feasibility) +
zero-shot reconstruction on a held-out topology."""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE_ROOT, SkeletonSpec, rot6d_to_matrix, fk_global, geodesic_deg
from model import TopoAE, hop_distance
from train_ae import load_skeleton, static_cond, make_batch, recon_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{CACHE_ROOT}/cache_big")
    ap.add_argument("--run", default=f"{CACHE_ROOT}/runs/plain")
    ap.add_argument("--heldout", default="soma_nohand")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_latent", type=int, default=8)
    ap.add_argument("--structured", action="store_true")
    args = ap.parse_args()
    device = "cuda"

    cfg_path = f"{args.run}/config.json"
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        args.d_model = cfg.get("d_model", args.d_model)
        args.n_latent = cfg.get("n_latent", args.n_latent)
        args.structured = cfg.get("structured", args.structured)

    model = TopoAE(d_model=args.d_model, n_latent=args.n_latent, structured=args.structured).to(device)
    model.load_state_dict(torch.load(f"{args.run}/model.pt", map_location=device))
    model.eval()

    def interpretable(spec, st, batch, pred6d):
        """Report position error in cm and as % of body reach (root->farthest joint),
        a far more meaningful scale than sum-of-all-bonelengths."""
        from common import fk_global
        g_pred_R, g_pred_p = fk_global(pred6d, spec)
        g_gt_R, g_gt_p = fk_global(batch["local6d"], spec)
        per_joint_cm = (g_pred_p - g_gt_p).norm(dim=-1)  # [B,J] in cm (data is cm)
        reach = g_gt_p.norm(dim=-1).max().item()  # farthest posed joint from root
        return {
            "pos_err_cm_mean": float(per_joint_cm.mean()),
            "pos_err_cm_p95": float(per_joint_cm.flatten().quantile(0.95)),
            "pos_err_pct_reach": float(per_joint_cm.mean() / max(reach, 1e-6) * 100),
            "body_reach_cm": float(reach),
        }

    result = {}
    # zero-shot on held-out topology
    spec, local6d = load_skeleton(args.cache, args.heldout)
    st = static_cond(spec, device)
    idx = torch.arange(min(2000, local6d.shape[0]))
    batch = make_batch(local6d, st, idx, device)
    with torch.no_grad():
        pred6d, _ = model(batch)
        lg, pe, gg = recon_losses(pred6d, batch["local6d"], spec, st)
        interp = interpretable(spec, st, batch, pred6d)
    result["heldout_zeroshot"] = {
        "skeleton": args.heldout, "J": spec.J,
        "local_geo_deg": float(lg), "global_geo_deg": float(gg), **interp,
    }

    # seen-topology recon in interpretable units too
    result["seen_recon"] = {}
    for name in args.skels_seen:
        spec, local6d = load_skeleton(args.cache, name)
        st = static_cond(spec, device)
        idx = torch.arange(min(2000, local6d.shape[0]))
        batch = make_batch(local6d, st, idx, device)
        with torch.no_grad():
            pred6d, _ = model(batch)
            lg, pe, gg = recon_losses(pred6d, batch["local6d"], spec, st)
            interp = interpretable(spec, st, batch, pred6d)
        result["seen_recon"][name] = {
            "J": spec.J, "local_geo_deg": float(lg), "global_geo_deg": float(gg), **interp,
        }

    # per-frame decode latency for real-time feasibility (batch=1, encode+decode)
    lat = {}
    for name in ["soma77", "g1_30"]:
        try:
            spec, local6d = load_skeleton(args.cache, name)
        except Exception:
            continue
        st = static_cond(spec, device)
        b1 = make_batch(local6d, st, torch.arange(1), device)
        with torch.no_grad():
            for _ in range(10):  # warmup
                model(b1)
            torch.cuda.synchronize()
            t0 = time.time()
            N = 200
            for _ in range(N):
                model(b1)
            torch.cuda.synchronize()
            ms = (time.time() - t0) / N * 1000
        lat[name] = {"J": spec.J, "encode_decode_ms_bs1": round(ms, 3)}
    result["latency"] = lat
    print(json.dumps(result, indent=2))
    with open(f"{args.run}/eval_extra.json", "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
