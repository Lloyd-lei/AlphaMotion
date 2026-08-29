"""bones-seed motion-generalization benchmark, multi-run version.

    eval_bones_multi.py --runs PROD_spatial_0808,PROD_data13k_0809,PROD_dim16_13k_0809

0809: with all 13,281 AMASS clips fed to the new arms, AMASS holds no motion-level
zero-shot anymore. bones-seed (soma_body cache) is corpus-level unseen for EVERY
run compared here, so it is the primary ruler of the spatial duel. Reuses
eval_bones_body_e19's self_recon/entry machinery; adds per-run sem_encoder from
config (SigLIP 768 vs Qwen 1024) and tail stats (p95/p99) — the whole point of
this round is the tail.

Output: runs/BONES_<run>.json + stdout table.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE_ROOT                                     # noqa: E402
from dual_common import build_model                               # noqa: E402
import eval_bones_body_e19 as EB                                  # noqa: E402

CACHE = f"{CACHE_ROOT}/cache_bones_body"
RUNS = f"{CACHE_ROOT}/runs"


def tails(v):
    v = np.asarray(v)
    return {k: round(float(f), 2) for k, f in
            (("mean", v.mean()), ("median", np.median(v)),
             ("p90", np.percentile(v, 90)), ("p95", np.percentile(v, 95)),
             ("p99", np.percentile(v, 99)), ("max", v.max()))}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    man = json.load(open(f"{CACHE}/manifest.json"))
    spec_b = EB.load_spec(CACHE, "soma_body")
    g9 = torch.load(f"{CACHE}/soma_body_global9d.pt")
    reach = man["reach_cm"]
    pose = torch.cat([g9[..., :6], g9[..., 6:9] / reach], -1).to(dev)
    seqs = json.load(open(f"{CACHE}/sequences.json"))

    for runname in [r for r in args.runs.split(",") if r]:
        cfg = json.load(open(f"{RUNS}/{runname}/config.json"))
        enc = cfg.get("sem_encoder", "siglip")
        from semantics import build_name_embeddings
        lut = build_name_embeddings(list(spec_b.joint_names), device=dev, encoder=enc)
        data = {"soma_body": EB.entry(CACHE, "soma_body", spec_b, dev, lut)}
        net = build_model(cfg, dev)
        missing, unexpected = net.load_state_dict(
            torch.load(f"{RUNS}/{runname}/model.pt", map_location=dev), strict=False)
        if missing or unexpected:
            print(f"WARN state_dict mismatch {runname}: {missing} {unexpected}")
        net.eval()
        rot, pos, pj, sw = EB.self_recon(net, data, "soma_body", pose, args.bs)
        pos_cm = pos * reach
        fam_rows = {}
        for s in seqs:
            fam_rows.setdefault(s["family"], []).append((s["start"], s["start"] + s["len"]))
        per_fam = {}
        for f, spans in sorted(fam_rows.items()):
            m = torch.zeros(len(rot), dtype=torch.bool)
            for a, b in spans:
                m[a:b] = True
            per_fam[f] = {"rows": int(m.sum()),
                          "rot_deg": round(float(rot[m].mean()), 2),
                          "pos_cm": round(float(pos_cm[m].mean()), 2)}
        names_b = list(spec_b.joint_names)
        order = torch.argsort(pj, descending=True)
        out = {"run": runname, "sem_encoder": enc, "rows": len(rot),
               "rot_deg": tails(rot), "pos_cm": tails(pos_cm),
               "per_family": per_fam,
               "worst_joints": [(names_b[i], round(float(pj[i]), 1))
                                for i in order[:6]]}
        json.dump(out, open(f"{RUNS}/BONES_{runname}.json", "w"), indent=1)
        print(f"[{runname}] rot {out['rot_deg']}  pos {out['pos_cm']}", flush=True)
        print(f"  -> {RUNS}/BONES_{runname}.json", flush=True)
        del net, data
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
