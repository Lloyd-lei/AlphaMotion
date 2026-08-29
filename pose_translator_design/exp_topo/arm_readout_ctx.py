"""CONTEXT READOUT: does +-2 frames of codes resolve the G1 arm ambiguity?

    arm_readout_ctx.py

Health check refuted jitter (labels smooth, eflip 0).  Remaining suspect:
the GMR arm solve is HISTORY-DEPENDENT — per-frame codes cap at ~30cm p95
because the same pose legally maps to several arm configs.  If a window of
codes recovers the accuracy, the wall is per-frame ambiguity and the cure is
a context-aware decode, not data surgery.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402
from bridge_a import CodeData, FULL                                  # noqa: E402
from train_global import build_global                                # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402

CTX = 2                                                   # +-2 frames
EMB = os.environ.get("PROBE_EMB", "unitree_g1_29dof")


def main():
    dev = "cuda"
    torch.manual_seed(0)
    data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
    l6 = torch.load(f"{FULL}/{EMB}_paired_local6d.pt", mmap=True)
    spec = V.load_spec(FULL, EMB)
    names = [n.lower() for n in spec.joint_names]

    def find(side, kws):
        for kw in kws:
            for i, n in enumerate(names):
                if side in n and kw in n:
                    return i
        raise KeyError((side, kws))
    J = {"L_Wrist": find("left", ("wrist", "hand")),
         "R_Wrist": find("right", ("wrist", "hand")),
         "L_Elbow": find("left", ("elbow",)), "R_Elbow": find("right", ("elbow",)),
         "L_Ankle": find("left", ("ankle",)), "R_Ankle": find("right", ("ankle",))}
    targets = list(J)

    g = np.random.default_rng(0)

    def pick(split, n):
        rows = np.concatenate([np.arange(a + CTX, b - CTX)
                               for a, b in data.pools[split]
                               if b - a > 2 * CTX + 2])
        return np.sort(g.choice(rows, n, replace=False))

    def fetch(idx):
        xs = []
        for off in range(-CTX, CTX + 1):
            cp = torch.from_numpy(np.ascontiguousarray(data.c[idx + off])).long()
            pk = torch.from_numpy(np.ascontiguousarray(data.cr[idx + off])).long()
            r = torch.empty_like(cp)
            r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
            c = torch.cat([cp, r], 1).float()
            xs.append(((c - 4.0) / 3.5).reshape(len(idx), -1).half())
        x = torch.cat(xs, 1)                              # [n, 5*5120] fp16
        out = []
        for i in range(0, len(idx), 4096):
            sl = torch.as_tensor(idx[i:i + 4096])
            g6, _, _ = build_global(l6[sl], spec, dev)
            pos = torch.as_tensor(V.fk_pos(g6, spec)).float()
            pel = pos[:, :1]
            R0 = rot6d_to_matrix(g6[:, 0].float())
            rel = pos[:, [J[t] for t in targets]] - pel
            out.append(torch.einsum("nji,nkj->nki", R0, rel))
        return x, torch.cat(out).float()

    print("[fetch] train ...", flush=True)
    x_tr, y_tr = fetch(pick("train", 80_000))
    print("[fetch] val ...", flush=True)
    x_va, y_va = fetch(pick("val", 10_000))

    m = nn.Sequential(nn.Linear((2 * CTX + 1) * 5120, 1024), nn.GELU(),
                      nn.Linear(1024, 512), nn.GELU(),
                      nn.Linear(512, 18)).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 4000)
    x_va_d, y_va_d = x_va.to(dev).float(), y_va.to(dev)
    for step in range(4000):
        i = torch.randint(0, len(x_tr), (1024,))
        x, y = x_tr[i].to(dev).float(), y_tr[i].to(dev)
        loss = (m(x).reshape(-1, 6, 3) - y).norm(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        if step % 1000 == 0:
            with torch.no_grad():
                e = (m(x_va_d).reshape(-1, 6, 3) - y_va_d).norm(dim=-1)
            print(f"step {step} val {e.mean().item():.2f}cm", flush=True)
    with torch.no_grad():
        e = (m(x_va_d).reshape(-1, 6, 3) - y_va_d).norm(dim=-1)
    print(f"\n=== CONTEXT(+-{CTX}) READOUT codes -> {EMB} (val, cm) ===")
    for k, t in enumerate(targets):
        v = e[:, k]
        print(f"  {t:8s} mean {v.mean().item():6.2f}  p95 {v.quantile(0.95).item():6.2f}")
    print(f"  [single-frame probe was: wrist p95 ~30, ankle p95 ~25]")


if __name__ == "__main__":
    main()
