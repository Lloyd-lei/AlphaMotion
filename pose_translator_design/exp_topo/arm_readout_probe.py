"""ADJUDICATION PROBE: do frozen Greenwich codes still contain arm detail?

    arm_readout_probe.py

Train a small readout head: frozen C1 codes (per frame, 256x20 ints) ->
root-relative wrist / ankle positions of the paired human GT.  If wrist
readout error is small (few cm), the ENCODER kept the arms and a
decoder-only post-train (lattice frozen, zero downstream cascade) has
headroom.  If wrist readout is large while ankles are small, the arms died
at the encoder and only a full spatial round can revive them.
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402
from bridge_a import CodeData, FULL                                  # noqa: E402
from train_global import build_global                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=200_000)
    ap.add_argument("--n_val", type=int, default=20_000)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)

    data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
    emb = os.environ.get("PROBE_EMB", "human_smpl")
    l6h = torch.load(f"{FULL}/{emb}_paired_local6d.pt", mmap=True)
    sspec = V.load_spec(FULL, emb)
    names = list(sspec.joint_names)
    J = {}
    def find(*kws):
        for i, n in enumerate(names):
            nl = n.lower()
            if all(k in nl for k in kws):
                return i
        raise KeyError(kws)
    J = {"L_Wrist": find("left", "wrist"), "R_Wrist": find("right", "wrist"),
         "L_Elbow": find("left", "elbow"), "R_Elbow": find("right", "elbow"),
         "L_Ankle": find("left", "ankle"), "R_Ankle": find("right", "ankle"),
         "Pelvis": 0} if "human" not in sspec.name else         {w: names.index(w) for w in ("L_Wrist", "R_Wrist", "L_Elbow", "R_Elbow",
                                     "L_Ankle", "R_Ankle", "Pelvis")}
    targets = ["L_Wrist", "R_Wrist", "L_Elbow", "R_Elbow",
               "L_Ankle", "R_Ankle"]

    g = np.random.default_rng(0)

    def pick(split, n):
        pool = data.pools[split]
        rows = []
        for a, b in pool:
            rows.append(np.arange(a, b))
        rows = np.concatenate(rows)
        return np.sort(g.choice(rows, n, replace=False))

    def fetch(idx):
        """codes [n,5120] float + GT joint pos root-relative [n,6,3] cm."""
        cp = torch.from_numpy(np.ascontiguousarray(data.c[idx])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[idx])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        c = torch.cat([cp, r], 1).float()
        x = ((c - 4.0) / 3.5).reshape(len(idx), -1)
        from common import rot6d_to_matrix
        out = []
        for i in range(0, len(idx), 4096):
            sl = torch.as_tensor(idx[i:i + 4096])
            g6, _, _ = build_global(l6h[sl], sspec, dev)
            pos = torch.as_tensor(V.fk_pos(g6, sspec)).float()  # [n,J,3] cm
            pel = pos[:, J["Pelvis"]][:, None]
            R0 = rot6d_to_matrix(g6[:, 0].float())              # root world rot
            rel = pos[:, [J[t] for t in targets]] - pel
            out.append(torch.einsum("nji,nkj->nki", R0, rel))   # root frame
        return x, torch.cat(out).float()

    print("[fetch] train ...", flush=True)
    idx_tr = pick("train", args.n_train)
    x_tr, y_tr = fetch(idx_tr)
    print("[fetch] val ...", flush=True)
    idx_va = pick("val", args.n_val)
    x_va, y_va = fetch(idx_va)

    m = nn.Sequential(nn.Linear(5120, 1024), nn.GELU(),
                      nn.Linear(1024, 512), nn.GELU(),
                      nn.Linear(512, 18)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    x_va_d, y_va_d = x_va.to(dev), y_va.to(dev)
    for step in range(args.steps):
        i = torch.randint(0, len(x_tr), (args.bs,))
        x, y = x_tr[i].to(dev), y_tr[i].to(dev)
        loss = (m(x).reshape(-1, 6, 3) - y).norm(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % 500 == 0:
            with torch.no_grad():
                e = (m(x_va_d).reshape(-1, 6, 3) - y_va_d).norm(dim=-1)
            print(f"step {step} train {loss.item():.2f}cm "
                  f"val {e.mean().item():.2f}cm", flush=True)
    with torch.no_grad():
        e = (m(x_va_d).reshape(-1, 6, 3) - y_va_d).norm(dim=-1)   # [n,6]
    print("\n=== READOUT from frozen C1 codes -> human GT (val, cm) ===")
    for k, t in enumerate(targets):
        v = e[:, k]
        print(f"  {t:8s} mean {v.mean().item():6.2f}  "
              f"p95 {v.quantile(0.95).item():6.2f}")
    arm = e[:, :4].mean().item(); leg = e[:, 4:].mean().item()
    print(f"\n  ARM mean {arm:.2f}cm  vs  LEG mean {leg:.2f}cm  "
          f"(ratio {arm/max(leg,1e-6):.2f})")
    print("  [context] cross-embodiment G1 decode wrist tail was 14-20cm")


if __name__ == "__main__":
    main()
