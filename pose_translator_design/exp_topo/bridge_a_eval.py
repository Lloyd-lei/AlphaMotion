"""BRIDGE-A acceptance: code accuracy, FK geometry, copy-rate, and the first
A-route (T32) vs B-route (BRIDGE-A) head-to-head on identical windows.

    bridge_a_eval.py

Caveats printed in the verdict (NOT a controlled ablation vs T32):
BRIDGE-A is human-only, no speed token, variable-rate; T32 is 16-body, 60-frame
fixed. Comparison runs on fixed n=60 human val windows, anchor positions, FK.
Root codebook is GT-given for BRIDGE-A rendering (pilot models the pose stream).
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402
import eval_newbody as E                                             # noqa: E402
from bridge_a import BridgeA, CodeData, CODES, FULL                  # noqa: E402
from train_global import build_global                                # noqa: E402
from train_v5 import load_run as load_temporal, to_states            # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402


def tails(v):
    v = np.asarray(v, dtype=np.float64)
    return {k: round(float(f), 2) for k, f in
            (("mean", v.mean()), ("p95", np.percentile(v, 95)),
             ("p99", np.percentile(v, 99)))}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="BRIDGE_A_pilot_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--t32", default="T32_arclen_0809")
    ap.add_argument("--n_win", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    T = 60

    # ---- data & models --------------------------------------------------------
    data = CodeData(args.spatial)
    m = BridgeA(dims=20).to(dev)
    m.load_state_dict(torch.load(f"{V.RUNS}/{args.run}/model.pt", map_location=dev))
    m.eval()
    sp, spcfg = E.load_run(f"{V.RUNS}/{args.spatial}", dev)
    enc = spcfg.get("sem_encoder", "siglip")
    sspec = V.load_spec(FULL, "human_smpl")
    sdof = np.load(f"{FULL}/human_smpl_dof.npz")["dof"]
    sg = E.target_geom(sspec, sdof, dev, sem_encoder=enc)
    l6 = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)
    kidx, knames, _ = V.key_joints(sspec, verbose=False)

    g = np.random.default_rng(5)
    # fixed-n windows so T32 can run the same rows
    starts = []
    pool = [(a, b) for a, b in data.pools["val"] if b - a >= T]
    for _ in range(args.n_win):
        a, b = pool[int(g.integers(len(pool)))]
        starts.append(int(g.integers(a, b - T + 1)))

    acc_all, fk_all, fk_anc, copy_hits, copy_tot = [], [], [], 0, 0
    t32_anc = []
    tok_sets = []
    t32 = load_temporal(f"{V.RUNS}/{args.t32}", dev)[0]
    kn = torch.ones(1, device=dev)

    for s0 in starts:
        rows = torch.arange(s0, s0 + T)
        c = torch.from_numpy(np.ascontiguousarray(
            data.c[s0:s0 + T])).long()[None].to(dev)
        # ---- GT geometry (rigid-bone FK of true global rotations) ----
        gr6, gp, reach = build_global(l6[rows], sspec, dev)
        gt_fk = V.fk_pos(gr6, sspec)                       # [T,J,3] cm
        # GT root codes for rendering (pilot: pose stream only)
        pose = torch.cat([gr6, gp / reach], -1).to(dev).float()
        _, zr_gt = sp.encode(pose, E.geom_batch(sg, T))

        # ---- BRIDGE-A recon ----
        logits, z = m(c)
        pred = logits.argmax(-1)[0]                        # [T,128,20]
        acc_all.append((pred[1:-1] == c[0, 1:-1]).float().mean().item())
        # copy-rate: does the interior prediction match the PREVIOUS frame's codes
        # better than its own frame?
        same_t = (pred[1:-1] == c[0, 1:-1]).float().mean(dim=(1, 2))
        same_prev = (pred[1:-1] == c[0, 0:-2]).float().mean(dim=(1, 2))
        copy_hits += int((same_prev > same_t).sum())
        copy_tot += same_t.numel()
        # endpoints pass through verbatim
        pred[0] = c[0, 0]; pred[-1] = c[0, -1]
        zp_hat = sp.fsqp.from_ints(pred[None].float())     # [T,128,256]
        rot_hat, _ = sp.decode(zp_hat, zr_gt, E.geom_batch(sg, T))
        fk_hat = V.fk_pos(rot_hat.cpu(), sspec)
        err = np.linalg.norm(fk_hat - gt_fk, axis=-1)      # [T,J]
        fk_all.append(err.mean())
        fk_anc.append(err[:, kidx].mean())

        # ---- T32 on the same window ----
        key = torch.cat([torch.as_tensor(gr6[:, kidx], dtype=torch.float32),
                         torch.as_tensor(gp[:, kidx] / reach, dtype=torch.float32)],
                        -1)[None].to(dev)
        e = t32.encode(key, key[:, 0], key[:, -1], kn)
        o = t32.decode(e["path_z"], e["time_z"], key[:, 0], key[:, -1], kn, T)
        s_hat = to_states(o, t32, key[:, 0], key[:, -1], kn).float()[0].cpu().numpy()
        anc_gt = gt_fk[:, kidx]
        anc_hat = s_hat[..., 6:9] * float(reach)
        # T32 anchors are root-relative global pos, same convention as gp; compare
        t32_anc.append(np.linalg.norm(anc_hat - np.asarray(gp[:, kidx]), axis=-1).mean())

    L = ["=" * 92,
         f"BRIDGE-A ACCEPTANCE — {args.run} (board={args.spatial})",
         "NOT a controlled duel vs T32: human-only / no speed token / root codes GT-given.",
         "=" * 92,
         f"windows n=60, count={len(starts)} (val clips, seed 5)",
         f"interior code top-1 acc      : {np.mean(acc_all):.4f}",
         f"copy-rate (pred~prev>own)    : {copy_hits/max(copy_tot,1)*100:.2f}%  "
         f"(frame-AR disease was 93.9%)",
         f"FK all-joint err [cm]        : {tails(fk_all)}",
         f"FK anchor err  [cm]          : {tails(fk_anc)}",
         f"T32 anchor err [cm] (same windows): {tails(t32_anc)}",
         ]
    txt = "\n".join(L)
    print(txt, flush=True)
    open(f"{V.RUNS}/BRIDGE_A_verdict_0810.txt", "w").write(txt + "\n")
    print(f"verdict: {V.RUNS}/BRIDGE_A_verdict_0810.txt", flush=True)


if __name__ == "__main__":
    main()
