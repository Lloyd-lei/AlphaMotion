"""FSQ codebook utilization probe — the ruler for any add-dims/add-levels decision.

    probe_fsq_usage.py --runs PROD_spatial_0808,E19_K128_0805 [--cache <dir>]

Per codebook (pose/root) reports, over temporally-decorrelated frames:
  * per-dim level entropy (ceiling log2(levels)): dead dims (<0.1) should be cut,
    saturated dims (>0.97*ceiling) justify more levels — nothing else does
  * distinct 13-d codes among 200k sampled tokens (lattice is 8^13; what matters
    is how much of it the ENCODER actually reaches)
  * per-slot vocabulary for a spread of the 128 latent slots

lloyd 2026-08-09: codebook-size decisions must cite this probe, not averages.
Output: runs/FSQ_USAGE_<run>.json + stdout table.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                                # noqa: E402
import eval_newbody as E                                             # noqa: E402


@torch.no_grad()
def probe(runname, cache, dev, n_clips=400, stride=10, bs=512):
    model, cfg = E.load_run(f"{V.RUNS}/{runname}", dev)
    enc = cfg.get("sem_encoder", "siglip")
    sq = json.load(open(f"{cache}/sequences.json"))
    rng = np.random.default_rng(0)
    names = [s["name"] for s in rng.choice(sq, min(n_clips, len(sq)), replace=False)]
    src, sspec, _ = E.load_source(cache, names, dev)
    src = src[::stride]
    sdof = np.load(f"{cache}/human_smpl_dof.npz")["dof"]
    sg = E.target_geom(sspec, sdof, dev, sem_encoder=enc)
    ips, irs = [], []
    for i in range(src.shape[0] // bs):
        b = src[i * bs:(i + 1) * bs]
        fp, fr = model.encode_features(b, E.geom_batch(sg, bs))
        ips.append(model.fsqp.ints(fp)[0].cpu())
        irs.append((model.fsqp if model.single_codebook else model.fsqr)
                   .ints(fr)[0].cpu())
    out = {"run": runname, "frames": int(len(ips) * bs),
           "levels": list(model.fsqp.levels.tolist()
                          if torch.is_tensor(model.fsqp.levels)
                          else model.fsqp.levels)}
    for tag, c in (("pose", torch.cat(ips).numpy()), ("root", torch.cat(irs).numpy())):
        F = c.shape[-1]
        flat = c.reshape(-1, F)
        Hd = []
        for d in range(F):
            p = np.bincount(flat[:, d], minlength=8) / len(flat)
            p = p[p > 0]
            Hd.append(float(-(p * np.log2(p)).sum()))
        sub = flat[np.random.default_rng(1).choice(len(flat),
                                                   min(200_000, len(flat)),
                                                   replace=False)]
        codes = len(set(map(tuple, sub)))
        slots = {int(k): len(set(map(tuple, c[:, k]))) for k in range(0, c.shape[1], 16)}
        ceil = float(np.log2(np.asarray(out["levels"])).mean())
        out[tag] = {"per_dim_entropy": [round(h, 3) for h in Hd],
                    "dead_dims": int(sum(h < 0.1 for h in Hd)),
                    "saturated_dims": int(sum(h > 0.97 * ceil for h in Hd)),
                    "distinct_codes_200k": codes, "slot_vocab": slots}
        print(f"[{runname}][{tag}] H/dim min {min(Hd):.2f} med "
              f"{float(np.median(Hd)):.2f} max {max(Hd):.2f} /{ceil:.2f} | "
              f"dead {out[tag]['dead_dims']} sat {out[tag]['saturated_dims']} | "
              f"distinct {codes}", flush=True)
    del model; torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="PROD_spatial_0808")
    ap.add_argument("--cache", default=V.AMASS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    for r in [x for x in args.runs.split(",") if x]:
        out = probe(r, args.cache, args.device)
        p = f"{V.RUNS}/FSQ_USAGE_{r}.json"
        json.dump(out, open(p, "w"), indent=1)
        print("->", p, flush=True)


if __name__ == "__main__":
    main()
