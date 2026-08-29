"""Standalone eval harness for DualTranslator runs — the ONE ruler for all E0/E5 ablations.

Rebuilds the model from {run}/config.json + {run}/model.pt, reproduces the exact
train/val split (seed(0) -> load -> randperm, BEFORE model init — this ordering is
load-bearing, see dual_common docstring), recomputes the FIX metrics, and writes
{run}/eval_dual.json. Probes (leakage / partial-observation) hook in via probes.py
once it lands (Phase 3); until then core metrics only.

Usage:
    python eval_dual.py --run /var/tmp/alphamotion/topo_exp/runs/FIX_translate
"""
from __future__ import annotations
import os, sys, json, argparse, itertools
import torch
sys.path.insert(0, os.path.dirname(__file__))
from dual_common import load_paired_data, split_indices, build_model, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing config.json + model.pt")
    ap.add_argument("--cache", default=None, help="override cache dir (default: from run config)")
    ap.add_argument("--out", default=None, help="output json (default: {run}/eval_dual.json)")
    ap.add_argument("--no_probes", action="store_true", help="skip leakage/partial-obs probes")
    args = ap.parse_args()
    device = "cuda"

    cfg = json.load(open(f"{args.run}/config.json"))
    cache = args.cache or cfg["cache"]
    print(f"run={args.run}\ncache={cache}\ncfg={ {k: cfg[k] for k in sorted(cfg)} }")

    torch.manual_seed(0)
    # The run's config decides the text tower: SigLIP is 768-wide, Qwen3 is 1024.
    # Without this, every Qwen-trained run died with
    # "mat1 and mat2 shapes cannot be multiplied (110000x768 and 1024x256)".
    data, names, N = load_paired_data(cache, device,
                                      sem_encoder=cfg.get("sem_encoder", "siglip"))
    val_idx, tr_idx = split_indices(N)          # MUST precede model construction

    model = build_model(cfg, device)
    sd = torch.load(f"{args.run}/model.pt", map_location=device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"WARN state_dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()

    combos = list(itertools.product(names, names))   # eval always reports all directions
    ev = evaluate(model, data, names, combos, val_idx)
    ev["n_val"] = int(len(val_idx))
    ev["state_dict_mismatch"] = bool(missing or unexpected)

    # per-body-part translation error (lloyd 2026-08-03: aggregate error diluted a 74%
    # leg deficit in b16s1 that was obvious in the GIF — legs are 12/30 joints)
    LEG_KW = ("hip", "knee", "ankle", "leg", "foot", "toe")
    ARM_KW = ("shoulder", "elbow", "wrist", "arm", "hand")
    from dual_common import geom as _geom
    with torch.no_grad():
        parts = {}
        for src, tgt in combos:
            if src == tgt:
                continue
            spec = data[tgt]["spec"]
            legm = torch.tensor([any(k in n.lower() for k in LEG_KW) for n in spec.joint_names], device=device)
            armm = torch.tensor([any(k in n.lower() for k in ARM_KW) for n in spec.joint_names], device=device)
            zp, zr = model.encode(data[src]["pose"][val_idx], _geom(data, src, len(val_idx)))
            prot, ppos = model.decode(zp, zr, _geom(data, tgt, len(val_idx)))
            err = (ppos - data[tgt]["pose"][val_idx][..., 6:9]).norm(dim=-1) * data[tgt]["reach"]
            parts[f"{src}->{tgt}"] = {"legs_cm": round(float(err[:, legm].mean()), 2),
                                      "arms_cm": round(float(err[:, armm].mean()), 2),
                                      "other_cm": round(float(err[:, ~(legm | armm)].mean()), 2)}
        ev["parts"] = parts

    if not args.no_probes:
        from probes import leakage_probe, partial_obs_stability
        ev["leakage"] = leakage_probe(model, data, names, tr_idx, val_idx)
        ev["partial_obs"] = partial_obs_stability(model, data, names, val_idx)

    out_path = args.out or f"{args.run}/eval_dual.json"
    json.dump(ev, open(out_path, "w"), indent=2)
    print("EVAL", json.dumps(ev))
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
