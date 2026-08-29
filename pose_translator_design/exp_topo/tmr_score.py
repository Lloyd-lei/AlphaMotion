"""Stage 2 of the TMR bridge: score exported motion with nvidia/TMR-SOMA-RP-v1.

    tmr_score.py --npz /var/tmp/alphamotion/tmr_bridge/real.npz

Runs in the `kimodo` env (TMR + LLM2Vec-Llama-3-8B live there); stage 1
(tmr_export.py) runs in ours.

Two things are being established here, in order:
  1. does the SMPL22 -> SOMA30 bridge actually work?  If TMR cannot retrieve
     the right caption for REAL corpus motion, the bridge is wrong and nothing
     downstream means anything.
  2. how much does TMR lose when the world translation is removed?  That is
     precisely what our representation throws away (D6), so this number decides
     whether TMR can serve as a relative judge for us at all.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/home/arenalabs/.cache/huggingface")
# kimodo is not pip-installed in its env; it resolves off the repo root
sys.path.insert(0, "/home/arenalabs/Desktop/rot/kimodo")


def retrieval(zt, zm):
    sim = zt @ zm.T
    n = len(zt)
    lab = torch.arange(n, device=sim.device)
    r1 = float((sim.argmax(1) == lab).float().mean())
    r5 = float((sim.topk(min(5, n), 1).indices == lab[:, None]).any(1)
               .float().mean())
    r10 = float((sim.topk(min(10, n), 1).indices == lab[:, None]).any(1)
                .float().mean())
    # median rank of the true match
    order = sim.argsort(1, descending=True)
    rank = (order == lab[:, None]).float().argmax(1) + 1
    return r1, r5, r10, float(rank.median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/var/tmp/alphamotion/tmr_bridge/real.npz")
    ap.add_argument("--model", default="TMR-SOMA-RP-v1")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--tag", default="real")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    from kimodo.model.load_model import load_model
    from kimodo.skeleton import SOMASkeleton30
    m, name = load_model(args.model, device=dev, default_family="TMR",
                         return_resolved_name=True)
    m.eval()
    skel = SOMASkeleton30()
    d = np.load(args.npz, allow_pickle=True)
    caps = [str(c) for c in d["captions"]]
    print(f"[TMR] {name}  clips {len(caps)}", flush=True)

    with torch.no_grad():
        zt = torch.cat([m.encode_raw_text(caps[i:i + args.bs])
                        for i in range(0, len(caps), args.bs)])
        zt = torch.nn.functional.normalize(zt, dim=-1)
        out = {}
        for key, label in (("pos", "带真实世界位移"),
                           ("pos_zero_root", "位移清零(我们的表征)")):
            P = torch.from_numpy(d[key]).float()
            zs = []
            for i in range(0, len(P), args.bs):
                x = P[i:i + args.bs].to(dev)
                ln = torch.full((x.shape[0],), x.shape[1], device=dev,
                                dtype=torch.long)
                zs.append(m.encode_motion(x, original_skeleton=skel,
                                          lengths=ln))
            zm = torch.nn.functional.normalize(torch.cat(zs), dim=-1)
            r1, r5, r10, mr = retrieval(zt, zm)
            out[key] = {"R1": r1, "R5": r5, "R10": r10, "median_rank": mr}
            print(f"  {label:22s} R@1 {r1*100:5.1f}%  R@5 {r5*100:5.1f}%  "
                  f"R@10 {r10*100:5.1f}%  真值中位排名 {mr:.0f}/{len(caps)}",
                  flush=True)
    chance = 1.0 / len(caps)
    print(f"  {'随机基线':22s} R@1 {chance*100:5.1f}%  R@5 "
          f"{5*chance*100:5.1f}%  R@10 {10*chance*100:5.1f}%", flush=True)
    p = f"/var/tmp/alphamotion/tmr_bridge/TMR_{args.tag}.json"
    json.dump(out | {"n": len(caps), "model": name}, open(p, "w"), indent=2)
    print(f"-> {p}", flush=True)


if __name__ == "__main__":
    main()
