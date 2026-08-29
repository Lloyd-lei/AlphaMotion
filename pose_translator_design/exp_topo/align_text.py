"""AlphaMotion text layer, step 2: FSQ text codec aligned to Equator tokens.

    align_text.py --out RUNS/TEXT_ALIGN_v1_0810

TextCodec: frozen-Qwen caption vec -> 8 queries -> FSQ 6x5 each -> pooled 256-d.
MotionEnc: window codes -> FROZEN Equator-A3 encoder -> 32 token ids -> small
transformer -> pooled 256-d.  Loss: symmetric InfoNCE (batch-pairing CE).
Greenwich/Equator are READ-ONLY throughout (lloyd's iron rule).

Eval: text<->motion retrieval R@1/R@5 on (a) AMASS val-split clips (clip-level
unseen), (b) bones 20% holdout; plus text-FSQ per-dim usage entropy.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V
from bridge_a import BridgeA, CodeData, load_bridge_a
from model import FSQ

RUNS = V.RUNS
TA = "/var/tmp/alphamotion/topo_exp/text_align"
V5 = 5 ** 6


class TextCodec(nn.Module):
    def __init__(self, d=256, k=8):
        super().__init__()
        self.k = k
        self.inp = nn.Sequential(nn.Linear(1024, k * d), nn.LayerNorm(k * d))
        self.fsq = FSQ(d, (5, 5, 5, 5, 5, 5))
        self.head = nn.Linear(d, 256)

    def forward(self, e):
        h = self.inp(e).reshape(e.shape[0], self.k, -1)
        z, _ = self.fsq(h)
        return F.normalize(self.head(z.mean(1)), dim=-1)

    def codes(self, e):
        h = self.inp(e).reshape(e.shape[0], self.k, -1)
        return self.fsq.ints(self.fsq.features(h))[0]


class MotionEnc(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.emb = nn.Embedding(V5, d)
        self.pos = nn.Parameter(torch.randn(32, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, 8, 4 * d, batch_first=True,
                                        norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(el, 2)
        self.head = nn.Linear(d, 256)

    def forward(self, tok):
        h = self.enc(self.emb(tok) + self.pos[None])
        return F.normalize(self.head(h.mean(1)), dim=-1)


class Pairs:
    def __init__(self, split):
        self.items = [p for p in json.load(open(f"{TA}/pairs_{split}.json"))
                      if p["len"] >= 60]
        self.emb = torch.load(f"{TA}/caption_emb.pt", map_location="cpu")
        self.amass = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
        self.bones = np.load(f"{TA}/bones_codes.npy", mmap_mode="r")

    def window(self, p, g, n=60):
        s = p["start"] + int(g.integers(0, max(1, p["len"] - n + 1)))
        if p["src"] == "amass":
            cp = torch.from_numpy(np.ascontiguousarray(
                self.amass.c[s:s + n])).long()
            pk = torch.from_numpy(np.ascontiguousarray(
                self.amass.cr[s:s + n])).long()
            r = torch.empty_like(cp)
            r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
            return torch.cat([cp, r], 1)
        return torch.from_numpy(np.ascontiguousarray(
            self.bones[s:s + n])).long()

    def batch(self, bs, g, n=60):
        idx = g.choice(len(self.items), bs, replace=False)
        ps = [self.items[i] for i in idx]
        wins = torch.stack([self.window(p, g, n) for p in ps])
        embs = torch.stack([self.emb[p["caption"]].float() for p in ps])
        return wins, embs


@torch.no_grad()
def motion_tokens(ma, wins, dev, masked=False):
    """CONTRACT FIX 0812. This used to apply bridge_a.encode's time-local mask,
    which put the text layer in A3's token space — while B4, the consumer,
    conditions a prior that lives in bridge_b.tokens_of's UNMASKED space. The
    two overlap on 0.15% of tokens, so the alignment was pointing at a
    representation B4 never sees. Default is now the consumer's space."""
    from bridge_b import tokens_of
    tok, _ = tokens_of(ma, wins.to(dev), masked=masked)
    return tok


@torch.no_grad()
def retrieval(tc, me, ma, pairs, dev, n_gallery=400):
    g = np.random.default_rng(1)
    byclip = {}
    for p in pairs.items:
        byclip.setdefault(p["name"], p)
    ps = list(byclip.values())[:n_gallery]
    wins = torch.stack([pairs.window(p, g) for p in ps])
    tok = motion_tokens(ma, wins, dev)
    zm = me(tok)
    zt = tc(torch.stack([pairs.emb[p["caption"]].float() for p in ps]).to(dev))
    sim = zt @ zm.T
    r1 = (sim.argmax(1) == torch.arange(len(ps), device=dev)).float().mean()
    top5 = sim.topk(5, 1).indices
    r5 = (top5 == torch.arange(len(ps), device=dev)[:, None]).any(1).float().mean()
    return float(r1), float(r5), len(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--masked_tokens", action="store_true",
                    help="align to A3's masked contract instead of B4's")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)   # FROZEN
    for p in ma.parameters():
        p.requires_grad_(False)
    tc, me = TextCodec().to(dev), MotionEnc().to(dev)
    scale = nn.Parameter(torch.tensor(2.66, device=dev))
    params = list(tc.parameters()) + list(me.parameters()) + [scale]
    print(f"params {sum(p.numel() for p in params)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    tr = Pairs("train")
    va_a, va_b = Pairs("val_amass"), Pairs("val_bones")
    print(f"train {len(tr.items)} val_a {len(va_a.items)} val_b {len(va_b.items)}",
          flush=True)
    g = np.random.default_rng(0)
    log = []
    t0 = time.time()
    for step in range(args.steps):
        wins, embs = tr.batch(args.bs, g)
        tok = motion_tokens(ma, wins, dev, masked=args.masked_tokens)
        zm = me(tok)
        zt = tc(embs.to(dev))
        sim = zt @ zm.T * scale.exp()
        lab = torch.arange(args.bs, device=dev)
        ce = 0.5 * (F.cross_entropy(sim, lab) + F.cross_entropy(sim.T, lab))
        opt.zero_grad(set_to_none=True)
        ce.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 100 == 0:
            acc = (sim.argmax(1) == lab).float().mean()
            print(f"step {step} ce {ce.item():.4f} batch-acc {acc.item():.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if step and step % 1000 == 0 or step == args.steps - 1:
            tc.eval(); me.eval()
            ra = retrieval(tc, me, ma, va_a, dev)
            rb = retrieval(tc, me, ma, va_b, dev)
            ev = {"step": step,
                  "amass_val_R1": round(ra[0], 3), "amass_val_R5": round(ra[1], 3),
                  "bones_hold_R1": round(rb[0], 3), "bones_hold_R5": round(rb[1], 3),
                  "gallery": [ra[2], rb[2]]}
            log.append(ev)
            print("EVAL", json.dumps(ev), flush=True)
            json.dump(log, open(f"{args.out}/eval_log.json", "w"))
            tc.train(); me.train()
    torch.save({"text": tc.state_dict(), "motion": me.state_dict(),
                "scale": float(scale.detach())}, f"{args.out}/model.pt")
    # text-FSQ usage entropy
    with torch.no_grad():
        embs = torch.stack([tr.emb[p["caption"]].float()
                            for p in tr.items[:4000]]).to(dev)
        codes = tc.codes(embs).reshape(-1, 6).cpu().numpy()
    Hd = []
    for d_ in range(6):
        p_ = np.bincount(codes[:, d_], minlength=5) / len(codes)
        p_ = p_[p_ > 0]
        Hd.append(float(-(p_ * np.log2(p_)).sum()))
    print(f"text-FSQ per-dim entropy: min {min(Hd):.2f} med {np.median(Hd):.2f} "
          f"max {max(Hd):.2f} /2.32", flush=True)
    json.dump(vars(args) | {"arch": "text_align_v1", "k_text": 8,
                            "fsq": "6x5", "entropy": Hd,
                            "token_contract": "masked" if args.masked_tokens
                                              else "legacy_unmasked"},
              open(f"{args.out}/config.json", "w"), indent=2)
    print(f"DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
