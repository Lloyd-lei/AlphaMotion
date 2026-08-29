"""BRIDGE Stage B4 — text-conditioned player, warm-started from FROZEN B3 copy.

    bridge_b4.py --out RUNS/BRIDGE_B4_text_0810

Adds 8 text-FSQ tokens (own Embedding, +1 null id) to the condition slots:
trunk consumes [start, goal, n, text x8, tok x32].  B3's checkpoint is loaded
as init and NEVER touched; Equator title stays with A3+B3 until a duel.

Batch mix vs forgetting: 30% of batches come from the FULL corpus with null
text (B3's original diet); captioned batches drop text per-sample 30% (also the
CFG knob).  Acceptance: on captioned val windows, CE(with text) < CE(null), and
CE(null) not worse than B3's own val CE.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_a import CodeData, load_bridge_a                          # noqa: E402
from bridge_b import BridgeB, tokens_of, V5                           # noqa: E402

RUNS = "/var/tmp/alphamotion/topo_exp/runs"
TA = "/var/tmp/alphamotion/topo_exp/text_align"


class BridgeB4(BridgeB):
    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.pos_id.shape[1]
        self.text_emb = nn.Embedding(V5 + 1, d)         # +1 = <no-text>
        self.text_pos = nn.Parameter(torch.randn(8, d) * 0.02)
        nn.init.zeros_(self.text_emb.weight)            # warm start: text silent

    def forward(self, tok, mask, ep, n_frames, goal_drop=None, text=None):
        """text [B,8] long (0..15624) or None; V5 rows = null slot."""
        B = tok.shape[0]
        if text is None:
            text = torch.full((B, 8), V5, dtype=torch.long, device=tok.device)
        t = torch.where(mask, torch.full_like(tok, V5), tok)
        h = self.tok_emb(t) + self.pos_id[None]
        from bridge_a import fourier_time
        nf = fourier_time((n_frames[:, None] / 120.0).clamp(0, 1))[:, 0]
        cond = torch.stack([self.cond_in(ep[:, 0]),
                            self.cond_in(ep[:, 1]),
                            self.n_in(nf)], 1) + self.cond_id[None]
        if goal_drop is not None:
            cond = cond.clone()
            cond[goal_drop, 1] = 0.0
        tx = self.text_emb(text) + self.text_pos[None]
        h = self.trunk(torch.cat([cond, tx, h], 1))[:, 11:]
        return self.head(h)


class CaptionedWindows:
    """pairs_*.json windows + packed text ids, both corpora."""

    def __init__(self, split, spatial, use_root):
        self.items = [p for p in json.load(open(f"{TA}/pairs_{split}.json"))
                      if p["len"] >= 24]
        self.amass = CodeData(spatial, use_root=use_root)
        self.bones = np.load(f"{TA}/bones_codes.npy", mmap_mode="r")
        tcodes = np.load(f"{TA}/text_codes.npy")
        keys = json.load(open(f"{TA}/text_codes_keys.json"))
        self.tid = {k: tcodes[i] for i, k in enumerate(keys)}

    def window(self, p, g, n):
        s = p["start"] + int(g.integers(0, max(1, p["len"] - n + 1)))
        n = min(n, p["len"])
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

    def batch(self, bs, g):
        n = int(g.integers(24, 121))
        pool = [p for p in self.items if p["len"] >= n]
        ps = [pool[i] for i in g.choice(len(pool), bs, replace=False)]
        c = torch.stack([self.window(p, g, n) for p in ps])
        text = torch.from_numpy(np.stack([self.tid[p["caption"]]
                                          for p in ps])).long()
        return c, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--stage_b_init", default="BRIDGE_B3_prod_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--corpus_frac", type=float, default=0.3)
    ap.add_argument("--text_drop", type=float, default=0.3)
    ap.add_argument("--text_lr", type=float, default=1e-3,
                    help="10x lr for the zero-init text pathway (v2 lesson: "
                         "1e-4 leaves KL(text||null) at 1-3% of dist entropy)")
    ap.add_argument("--cap_goal_drop", type=float, default=0.5,
                    help="goal-drop rate on CAPTIONED batches: text must carry "
                         "the motion identity when the goal is gone (v1 lesson: "
                         "10% left text with no gradient, gain ~= 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    ma, acfg = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)
    for p in ma.parameters():
        p.requires_grad_(False)
    a_root = bool(acfg.get("use_root", False))
    corpus = CodeData(args.spatial, use_root=a_root)
    cap_tr = CaptionedWindows("train", args.spatial, a_root)
    cap_va = CaptionedWindows("val_amass", args.spatial, a_root)
    m = BridgeB4(ep_dim=ma.d).to(dev)
    if args.stage_b_init != "none":
        sd3 = torch.load(f"{RUNS}/{args.stage_b_init}/model.pt", map_location=dev)
        missing, unexpected = m.load_state_dict(sd3, strict=False)
        assert not unexpected, unexpected
        print(f"warm-start from {args.stage_b_init}; new params: {missing}", flush=True)
    else:
        nn.init.normal_(m.text_emb.weight, std=0.02)     # from scratch: text
        print("FROM SCRATCH: text pathway live from step 0", flush=True)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"params {n_par/1e6:.2f}M  captioned train {len(cap_tr.items)}", flush=True)
    text_p = [m.text_emb.weight, m.text_pos]
    text_ids = {id(x) for x in text_p}
    rest_p = [q for q in m.parameters() if id(q) not in text_ids]
    opt = torch.optim.AdamW([{"params": rest_p, "lr": args.lr},
                             {"params": text_p, "lr": args.text_lr}],
                            weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / args.warmup))
    g = np.random.default_rng(args.seed)
    log = []
    t0 = time.time()

    def make_mask(B, gg):
        mode = gg.random()
        r = torch.cos(torch.rand(B, 1, device=dev) * (np.pi / 2))
        if mode < 0.2:
            mask = torch.ones(B, 32, dtype=torch.bool, device=dev)
        elif mode < 0.6:
            mask = torch.rand(B, 32, device=dev) < r
        elif mode < 0.8:
            cut = (32 * (1 - r)).long().clamp(1, 31)
            mask = torch.arange(32, device=dev)[None] >= cut
        else:
            cut = (32 * r).long().clamp(1, 31)
            mask = torch.arange(32, device=dev)[None] < cut
        mask[mask.sum(1) == 0, 0] = True
        return mask

    for step in range(args.steps):
        captioned = g.random() >= args.corpus_frac
        if not captioned:                                # B3's original diet
            c, _n = corpus.sample(args.bs, g, "train")
            text = None
        else:
            c, text = cap_tr.batch(args.bs, g)
            text = text.to(dev)
            drop = torch.rand(args.bs, device=dev) < args.text_drop
            text = torch.where(drop[:, None], torch.full_like(text, V5), text)
        c = c.to(dev)
        tok, ep = tokens_of(ma, c)
        nf = torch.full((args.bs,), float(c.shape[1]), device=dev)
        mask = make_mask(args.bs, g)
        gd_rate = args.cap_goal_drop if captioned else 0.10
        gd = torch.rand(args.bs, device=dev) < gd_rate
        logits = m(tok, mask, ep, nf, goal_drop=gd, text=text)
        ce = F.cross_entropy(logits[mask], tok[mask])
        opt.zero_grad(set_to_none=True)
        ce.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            acc = (logits[mask].argmax(-1) == tok[mask]).float().mean()
            print(f"step {step} ce {ce.item():.4f} acc {acc.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if step and step % args.eval_every == 0 or step == args.steps - 1:
            m.eval()
            with torch.no_grad():
                gv = np.random.default_rng(1)
                ct, cn, cc = [], [], []
                for _ in range(8):
                    c, text = cap_va.batch(args.bs, gv)
                    c, text = c.to(dev), text.to(dev)
                    tok, ep = tokens_of(ma, c)
                    nf = torch.full((args.bs,), float(c.shape[1]), device=dev)
                    mk = torch.ones_like(tok, dtype=torch.bool)
                    lt = m(tok, mk, ep, nf, text=text)
                    ln = m(tok, mk, ep, nf, text=None)
                    ct.append(F.cross_entropy(lt[mk], tok[mk]).item())
                    cn.append(F.cross_entropy(ln[mk], tok[mk]).item())
                    cv, _ = corpus.sample(args.bs, gv, "val")
                    cv = cv.to(dev)
                    tkv, epv = tokens_of(ma, cv)
                    nfv = torch.full((args.bs,), float(cv.shape[1]), device=dev)
                    lv = m(tkv, torch.ones_like(tkv, dtype=torch.bool), epv, nfv,
                           text=None)
                    cc.append(F.cross_entropy(
                        lv[torch.ones_like(tkv, dtype=torch.bool)],
                        tkv.reshape(-1)).item())
                ev = {"step": step,
                      "cap_ce_text": round(float(np.mean(ct)), 4),
                      "cap_ce_null": round(float(np.mean(cn)), 4),
                      "text_gain": round(float(np.mean(cn) - np.mean(ct)), 4),
                      "corpus_ce_null": round(float(np.mean(cc)), 4)}
            log.append(ev)
            print("EVAL", json.dumps(ev), flush=True)
            json.dump(log, open(f"{args.out}/eval_log.json", "w"))
            m.train()
    torch.save(m.state_dict(), f"{args.out}/model.pt")
    json.dump(vars(args) | {"arch": "bridge_b4_text", "params": n_par,
                            "ep_dim": ma.d, "n_text": 8},
              open(f"{args.out}/config.json", "w"), indent=2)
    print(f"DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
