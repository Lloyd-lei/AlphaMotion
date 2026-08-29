"""BRIDGE Stage B5 — text CROSS-ATTENTION player (the composition wrapper).

    bridge_b5.py --out RUNS/BRIDGE_B5_xattn_0814

B4 (four arms, v1..v5) never got text conditioning to work: the best no-goal
text_gain was +0.060 nats, visually indistinguishable.  The diagnosis after
composition_test / rainbow_codes / the atomicity check (97.4% of a hand-grafted
"walk + wave" is already in-vocabulary) is that the vocabulary can express the
composition and the prior does not reject it — text simply had no force:

    B4                                  B5
    pooled last-token Qwen vector       full BPE word sequence  [L,1024]
    -> FSQ 6x5 -> 8 discrete ids        -> linear proj, NO FSQ (no new vocab)
    8 extra INPUT slots                 CROSS-ATTENTION in every trunk layer
    ~70 bits, diluted over 6 layers     every layer can query any word
    masked-CE only                      + composition loss (retrieval critic)

lloyd's call 0814: "没有必要为了 fsq 而 fsq,大语言的 bpe 词表已经很成熟" — the
discrete text branch stays only for retrieval/indexing (TEXT_ALIGN_v3), it is
no longer on the conditioning path.

Warm start: B3's weights load into self-attn/FFN by name; cross-attention is
new and its out_proj is ZERO-init, so B5 at step 0 is bit-identical to B3.
Greenwich (spatial) and Equator A3/B3 checkpoints are READ-ONLY.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge_a import CodeData, load_bridge_a, fourier_time            # noqa: E402
from bridge_b import tokens_of, V5                                    # noqa: E402

RUNS = "/var/tmp/alphamotion/topo_exp/runs"
TA = "/var/tmp/alphamotion/topo_exp/text_align"


class XBlock(nn.Module):
    """pre-norm self-attn -> cross-attn(text) -> FFN.

    Submodule names match nn.TransformerEncoderLayer (self_attn/linear1/
    linear2/norm1/norm2) so B3's trunk loads by name; the cross-attention pair
    (norm_x/cross_attn) is the only new tensor set."""

    def __init__(self, d, n_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.linear1, self.linear2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.norm_x = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        nn.init.zeros_(self.cross_attn.out_proj.weight)   # step 0 == B3 exactly
        nn.init.zeros_(self.cross_attn.out_proj.bias)

    def forward(self, x, mem, mem_pad):
        h = self.norm1(x)
        x = x + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm_x(x)
        x = x + self.cross_attn(h, mem, mem, key_padding_mask=mem_pad,
                                need_weights=False)[0]
        h = self.norm2(x)
        # relu, not gelu: nn.TransformerEncoderLayer's default, which is
        # what B3 was trained with. A silent gelu here would break the
        # warm start without any error message.
        return x + self.linear2(F.relu(self.linear1(h)))


class BridgeB5(nn.Module):
    def __init__(self, d=256, n_tok=32, layers=6, n_heads=8, n_freq=8,
                 ep_dim=None, d_text=1024):
        super().__init__()
        self.n_tok = n_tok
        self.tok_emb = nn.Embedding(V5 + 1, d)
        self.pos_id = nn.Parameter(torch.randn(n_tok, d) * 0.02)
        self.cond_in = nn.Linear(ep_dim or d, d)
        self.cond_id = nn.Parameter(torch.randn(3, d) * 0.02)
        self.n_in = nn.Linear(2 * n_freq, d)
        self.trunk = nn.ModuleDict({"layers": nn.ModuleList(
            [XBlock(d, n_heads) for _ in range(layers)])})
        self.head = nn.Linear(d, V5)
        # text pathway (trained jointly with B — not frozen, lloyd 0814)
        self.text_in = nn.Sequential(nn.Linear(d_text, d), nn.LayerNorm(d))
        self.text_pos = nn.Parameter(torch.randn(32, d) * 0.02)
        self.null_mem = nn.Parameter(torch.randn(1, d) * 0.02)

    def text_memory(self, seq, tlen, drop=None):
        """seq [B,L,1024] fp32, tlen [B] int, drop [B] bool -> mem, pad_mask."""
        B, L = seq.shape[0], seq.shape[1]
        mem = self.text_in(seq) + self.text_pos[None, :L]
        ar = torch.arange(L, device=seq.device)[None]
        pad = ar >= tlen[:, None]                       # True = ignore
        if drop is not None and bool(drop.any()):
            mem = mem.clone(); pad = pad.clone()
            mem[drop, 0] = self.null_mem[0]
            pad[drop] = True
            pad[drop, 0] = False                        # only the null slot
        return mem, pad

    def forward(self, tok, mask, ep, n_frames, seq, tlen,
                goal_drop=None, text_drop=None):
        B = tok.shape[0]
        t = torch.where(mask, torch.full_like(tok, V5), tok)
        h = self.tok_emb(t) + self.pos_id[None]
        nf = fourier_time((n_frames[:, None] / 120.0).clamp(0, 1))[:, 0]
        cond = torch.stack([self.cond_in(ep[:, 0]), self.cond_in(ep[:, 1]),
                            self.n_in(nf)], 1) + self.cond_id[None]
        if goal_drop is not None:
            cond = cond.clone()
            cond[goal_drop, 1] = 0.0
        mem, pad = self.text_memory(seq, tlen, text_drop)
        x = torch.cat([cond, h], 1)
        for blk in self.trunk["layers"]:
            x = blk(x, mem, pad)
        return self.head(x[:, 3:])


class Critic(nn.Module):
    """FROZEN TEXT_ALIGN_v3 retriever used as the composition judge.

    The motion encoder scores a token sequence against the caption embedding.
    Feeding it the model's SOFT distribution (sum_v p_v * emb_v) instead of a
    sampled id keeps the whole thing differentiable, so "the thing you just
    generated must still be recognisable as the prompt" becomes a loss."""

    def __init__(self, run, dev):
        super().__init__()
        from align_text import TextCodec, MotionEnc
        sd = torch.load(f"{RUNS}/{run}/model.pt", map_location=dev)
        self.tc, self.me = TextCodec().to(dev), MotionEnc().to(dev)
        self.tc.load_state_dict(sd["text"]); self.me.load_state_dict(sd["motion"])
        self.scale = float(sd["scale"])
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def motion_soft(self, logits, tok, mask):
        """logits [B,32,V5] -> pooled 256-d, soft at masked slots, true elsewhere."""
        p = logits.softmax(-1)
        e = p @ self.me.emb.weight                      # [B,32,256]
        e = torch.where(mask[..., None], e, self.me.emb(tok))
        h = self.me.enc(e + self.me.pos[None])
        return F.normalize(self.me.head(h.mean(1)), dim=-1)

    def text(self, emb):
        return self.tc(emb)


class CaptionedWindows:
    """pairs_*.json windows + WORD-SEQUENCE text features (caption_seq.pt)."""

    def __init__(self, split, spatial, use_root, seq, keyidx, pool):
        self.items = [p for p in json.load(open(f"{TA}/pairs_{split}.json"))
                      if p["len"] >= 24]
        self.amass = CodeData(spatial, use_root=use_root)
        self.bones = np.load(f"{TA}/bones_codes.npy", mmap_mode="r")
        self.seq, self.keyidx, self.pool = seq, keyidx, pool

    def window(self, p, g, n):
        s = p["start"] + int(g.integers(0, max(1, p["len"] - n + 1)))
        n = min(n, p["len"])
        if p["src"] == "amass":
            cp = torch.from_numpy(np.ascontiguousarray(self.amass.c[s:s + n])).long()
            pk = torch.from_numpy(np.ascontiguousarray(self.amass.cr[s:s + n])).long()
            r = torch.empty_like(cp)
            r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
            return torch.cat([cp, r], 1)
        return torch.from_numpy(np.ascontiguousarray(self.bones[s:s + n])).long()

    def batch(self, bs, g):
        n = int(g.integers(24, 121))
        pool = [p for p in self.items if p["len"] >= n]
        ps = [pool[i] for i in g.choice(len(pool), bs, replace=False)]
        c = torch.stack([self.window(p, g, n) for p in ps])
        ix = torch.tensor([self.keyidx[p["caption"]] for p in ps])
        return c, self.seq["seq"][ix].float(), self.seq["len"][ix].long(), \
            self.pool[ix].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--stage_b_init", default="BRIDGE_B3_prod_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--critic", default="TEXT_ALIGN_v3_fixed_0812")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--text_lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=400)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--corpus_frac", type=float, default=0.25,
                    help="fraction of steps on the FULL corpus w/ null text "
                         "(anti-forgetting; B3's original diet)")
    ap.add_argument("--text_drop", type=float, default=0.2)
    ap.add_argument("--cap_goal_drop", type=float, default=0.6,
                    help="goal-drop on captioned batches: with the goal present "
                         "text is redundant, so it gets no gradient (B4 v1 lesson)")
    ap.add_argument("--w_critic", type=float, default=0.1,
                    help="weight of the composition (retrieval-critic) loss")
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
    seq = torch.load(f"{TA}/caption_seq.pt", map_location="cpu")
    keyidx = {k: i for i, k in enumerate(seq["keys"])}
    pooled = torch.load(f"{TA}/caption_emb.pt", map_location="cpu")
    pool = torch.stack([pooled[k] for k in seq["keys"]])
    corpus = CodeData(args.spatial, use_root=a_root)
    cap_tr = CaptionedWindows("train", args.spatial, a_root, seq, keyidx, pool)
    cap_va = CaptionedWindows("val_amass", args.spatial, a_root, seq, keyidx, pool)

    m = BridgeB5(ep_dim=ma.d).to(dev)
    sd3 = torch.load(f"{RUNS}/{args.stage_b_init}/model.pt", map_location=dev)
    missing, unexpected = m.load_state_dict(sd3, strict=False)
    assert not unexpected, unexpected
    new = sorted({k.split(".")[0] if not k.startswith("trunk") else
                  "trunk." + k.split(".")[3] for k in missing})
    print(f"warm-start <- {args.stage_b_init}; new tensors: {new}", flush=True)
    critic = Critic(args.critic, dev)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"params {n_par/1e6:.2f}M  captioned train {len(cap_tr.items)}", flush=True)

    tp = [m.text_in[0].weight, m.text_in[0].bias, m.text_in[1].weight,
          m.text_in[1].bias, m.text_pos, m.null_mem]
    tp += [q for b in m.trunk["layers"] for q in b.cross_attn.parameters()]
    tp += [q for b in m.trunk["layers"] for q in b.norm_x.parameters()]
    tids = {id(x) for x in tp}
    rest = [q for q in m.parameters() if id(q) not in tids]
    opt = torch.optim.AdamW([{"params": rest, "lr": args.lr},
                             {"params": tp, "lr": args.text_lr}], weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, s / args.warmup))
    g = np.random.default_rng(args.seed)
    log, t0 = [], time.time()

    def maskgit(B):
        r = torch.cos(torch.rand(B, 1, device=dev) * (np.pi / 2))
        mode = g.random()
        if mode < 0.2:
            mk = torch.ones(B, 32, dtype=torch.bool, device=dev)
        elif mode < 0.6:
            mk = torch.rand(B, 32, device=dev) < r
        elif mode < 0.8:
            cut = (32 * (1 - r)).long().clamp(1, 31)
            mk = torch.arange(32, device=dev)[None] >= cut
        else:
            cut = (32 * r).long().clamp(1, 31)
            mk = torch.arange(32, device=dev)[None] < cut
        mk[mk.sum(1) == 0, 0] = True
        return mk

    def null_text(B, L=32):
        return (torch.zeros(B, L, 1024, device=dev),
                torch.ones(B, dtype=torch.long, device=dev),
                torch.ones(B, dtype=torch.bool, device=dev))

    for step in range(args.steps):
        captioned = g.random() >= args.corpus_frac
        if captioned:
            c, s_, l_, pv = cap_tr.batch(args.bs, g)
            s_, l_, pv = s_.to(dev), l_.to(dev), pv.to(dev)
            td = torch.rand(args.bs, device=dev) < args.text_drop
            gdrop = args.cap_goal_drop
        else:
            c, _n = corpus.sample(args.bs, g, "train")
            s_, l_, td = null_text(args.bs)
            pv, gdrop = None, 0.10
        c = c.to(dev)
        tok, ep = tokens_of(ma, c)
        nf = torch.full((args.bs,), float(c.shape[1]), device=dev)
        mask = maskgit(args.bs)
        gd = torch.rand(args.bs, device=dev) < gdrop
        logits = m(tok, mask, ep, nf, s_, l_, goal_drop=gd, text_drop=td)
        ce = F.cross_entropy(logits[mask], tok[mask])
        loss, cri = ce, torch.zeros((), device=dev)
        if captioned and args.w_critic > 0:
            live = ~td
            if int(live.sum()) >= 8:
                zm = critic.motion_soft(logits[live], tok[live], mask[live])
                zt = critic.text(pv[live])
                sim = zt @ zm.T * np.exp(critic.scale)
                lab = torch.arange(int(live.sum()), device=dev)
                cri = 0.5 * (F.cross_entropy(sim, lab) +
                             F.cross_entropy(sim.T, lab))
                loss = ce + args.w_critic * cri
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0:
            print(f"step {step} ce {ce.item():.4f} critic {float(cri):.4f} "
                  f"({'cap' if captioned else 'corpus'}) "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if step and step % args.eval_every == 0 or step == args.steps - 1:
            m.eval()
            with torch.no_grad():
                gv = np.random.default_rng(1)
                with_t, no_t, r5 = [], [], []
                for _ in range(8):
                    c, s_, l_, pv = cap_va.batch(args.bs, gv)
                    c, s_, l_, pv = c.to(dev), s_.to(dev), l_.to(dev), pv.to(dev)
                    tok, ep = tokens_of(ma, c)
                    nf = torch.full((args.bs,), float(c.shape[1]), device=dev)
                    mk = torch.ones_like(tok, dtype=torch.bool)
                    gd = torch.ones(args.bs, dtype=torch.bool, device=dev)  # no goal
                    zt_ = torch.zeros(args.bs, dtype=torch.bool, device=dev)
                    lo = m(tok, mk, ep, nf, s_, l_, goal_drop=gd, text_drop=zt_)
                    with_t.append(F.cross_entropy(lo[mk], tok[mk]).item())
                    lo0 = m(tok, mk, ep, nf, s_, l_, goal_drop=gd,
                            text_drop=~zt_)
                    no_t.append(F.cross_entropy(lo0[mk], tok[mk]).item())
                    zm = critic.motion_soft(lo, tok, mk)
                    sim = critic.text(pv) @ zm.T
                    lab = torch.arange(args.bs, device=dev)[:, None]
                    r5.append((sim.topk(5, 1).indices == lab).any(1)
                              .float().mean().item())
            ev = {"step": step,
                  "ce_text": round(float(np.mean(with_t)), 4),
                  "ce_null": round(float(np.mean(no_t)), 4),
                  "text_gain": round(float(np.mean(no_t) - np.mean(with_t)), 4),
                  "gen_R5": round(float(np.mean(r5)), 3)}
            log.append(ev)
            print("EVAL", json.dumps(ev), flush=True)
            json.dump(log, open(f"{args.out}/eval_log.json", "w"))
            torch.save(m.state_dict(), f"{args.out}/model.pt")
            m.train()
    torch.save(m.state_dict(), f"{args.out}/model.pt")
    json.dump(vars(args) | {"arch": "bridge_b5_xattn", "params": n_par,
                            "text": "qwen word sequence, no FSQ",
                            "token_contract": "legacy_unmasked"},
              open(f"{args.out}/config.json", "w"), indent=2)
    print(f"DONE {args.out}", flush=True)


if __name__ == "__main__":
    main()
