"""Encode a temporal paired cache with a FROZEN tokenizer run -> per-frame discrete
codes + continuous pre-round features, for AR-likelihood (E0a lock metric) and E1.

Outputs under {run}/tokens_seq/:
  {emb}_f.pt     float16 [N, 2, K, f]    pre-quantization features (pos, rot streams)
  {emb}_ints.pt  int8    [N, 2, S, K, f] integer FSQ levels (the AR-LM vocabulary)
  meta.json      levels/stages/K + source cache path
  sequences.json copied from the cache (per-file frame ranges)

Usage: python tokenize_cache.py --run .../runs/FIX_translate --cache .../cache_paired_seq
"""
from __future__ import annotations
import os, sys, json, shutil, argparse
import torch
sys.path.insert(0, os.path.dirname(__file__))
from dual_common import load_paired_data, build_model, geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cache", default="/var/tmp/alphamotion/topo_exp/cache_paired_seq")
    ap.add_argument("--bs", type=int, default=2048)
    args = ap.parse_args()
    device = "cuda"

    cfg = json.load(open(f"{args.run}/config.json"))
    data, names, N = load_paired_data(args.cache, device)
    model = build_model(cfg, device)
    missing, unexpected = model.load_state_dict(torch.load(f"{args.run}/model.pt", map_location=device), strict=False)
    if missing or unexpected:
        print(f"WARN state_dict mismatch: missing={missing} unexpected={unexpected}")
    model.eval()

    out_dir = f"{args.run}/tokens_seq"
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        fs, ints = [], []
        for i in range(0, N, args.bs):
            idx = torch.arange(i, min(i + args.bs, N))
            with torch.no_grad():
                fp, fr = model.encode_features(data[name]["pose"][idx], geom(data, name, len(idx)))
                ip, ir = model.ints_from_features(fp, fr)
            fs.append(torch.stack([fp, fr], 1).half().cpu())                 # [B,2,K,f]
            ints.append(torch.stack([ip, ir], 1).permute(2, 1, 0, 3, 4).to(torch.int8).cpu())  # [B,2,S,K,f]
        F = torch.cat(fs, 0); I = torch.cat(ints, 0)
        torch.save(F, f"{out_dir}/{name}_f.pt")
        torch.save(I, f"{out_dir}/{name}_ints.pt")
        print(f"[{name}] f={tuple(F.shape)} ints={tuple(I.shape)}")
    shutil.copy(f"{args.cache}/sequences.json", f"{out_dir}/sequences.json")
    fsq = model.fsqp
    json.dump({"cache": args.cache, "names": names,
               "levels": [int(l) for l in fsq._levels.tolist()],
               "stages": int(getattr(fsq, "n_stages", 1)),
               "n_latent": cfg.get("n_latent", 32),
               "single_codebook": cfg.get("single_codebook", False)},
              open(f"{out_dir}/meta.json", "w"), indent=2)
    print("SAVED", out_dir)


if __name__ == "__main__":
    main()
