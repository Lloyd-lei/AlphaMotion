"""BRIDGE Stage-0 prep: tokenize the full human corpus through Greenwich.

    bridge_tokenize.py [--run PROD_c1_91k_dim20_tail_0810]

Encodes every human_smpl frame of cache_xwbc_amass_full with the frozen spatial
layer and stores the FSQ code INTS (pose codebook, [N, 128, D] int8) plus the
(q,v) anchor states needed for endpoint conditioning. This is the "board
coordinates" file every BRIDGE stage reads; ~10GB, CPU-resident at train time.

Output: /var/tmp/alphamotion/topo_exp/bridge_codes/<run>/
    codes_pose.npy  [N,128,D] int8   FSQ levels 0..7
    codes_root.npy  [N,128,D] int8
    frame_emb.npy   [N,256]  fp16    pooled quantized zp (encoder input shortcut)
    meta.json       {run, cache, N, D, slots}
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                                # noqa: E402
import eval_newbody as E                                             # noqa: E402

FULL = "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full"
OUT_ROOT = "/var/tmp/alphamotion/topo_exp/bridge_codes"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--cache", default=FULL)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    out = f"{OUT_ROOT}/{args.run}"
    os.makedirs(out, exist_ok=True)

    model, cfg = E.load_run(f"{V.RUNS}/{args.run}", dev)
    enc = cfg.get("sem_encoder", "siglip")
    sspec = V.load_spec(args.cache, "human_smpl")
    sdof = np.load(f"{args.cache}/human_smpl_dof.npz")["dof"]
    sg = E.target_geom(sspec, sdof, dev, sem_encoder=enc)
    l6 = torch.load(f"{args.cache}/human_smpl_paired_local6d.pt", mmap=True)
    N = l6.shape[0]
    D = len(model.fsqp.levels) if not torch.is_tensor(model.fsqp.levels) \
        else model.fsqp.levels.numel()
    print(f"[tokenize] {args.run} N={N} D={D}", flush=True)

    # 0810 disk reality: pose codebook only (12GB int8); root codes + frame_emb
    # deferred — the pilot encoder embeds ints directly, root stream joins in v2.
    cp = np.lib.format.open_memmap(f"{out}/codes_pose.npy", mode="w+",
                                   dtype=np.int8, shape=(N, 128, D))
    # v2 (lloyd: 完全体): root codebook too, nibble-packed (levels 0..8 fit 4 bits,
    # two dims per byte -> 6GB instead of 12).
    assert D % 2 == 0
    croot = np.lib.format.open_memmap(f"{out}/codes_root_packed.npy", mode="w+",
                                      dtype=np.uint8, shape=(N, 128, D // 2))
    # build_global in chunks to keep host RAM sane (full pose is 4.7M x 22 x 9)
    from train_global import build_global
    CH = 200_000
    reach = None
    done = 0
    for c0 in range(0, N, CH):
        c1 = min(c0 + CH, N)
        gr6, gp, r = build_global(l6[c0:c1], sspec, dev)
        reach = reach or float(r)                    # pin the first chunk's reach
        pose = torch.cat([gr6, gp / reach], -1)
        for i in range(c0, c1, args.bs):
            j = min(i + args.bs, c1)
            b = pose[i - c0:j - c0].to(dev).float()
            fp, fr = model.encode_features(b, E.geom_batch(sg, b.shape[0]))
            cp[i:j] = model.fsqp.ints(fp)[0].cpu().numpy().astype(np.int8)
            r = (model.fsqp if model.single_codebook else model.fsqr) \
                .ints(fr)[0].cpu().numpy().astype(np.uint8)
            croot[i:j] = r[..., 0::2] | (r[..., 1::2] << 4)
        done = c1
        print(f"  {done}/{N}", flush=True)
    json.dump({"run": args.run, "cache": args.cache, "N": N, "D": D,
               "slots": 128, "reach": reach}, open(f"{out}/meta.json", "w"))
    print(f"DONE {out}", flush=True)


if __name__ == "__main__":
    main()
