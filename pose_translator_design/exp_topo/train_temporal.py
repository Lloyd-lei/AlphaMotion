"""P1 — train the temporal tokenizer over the frozen E19 frame-code stream.

ARCH_V4 P1 row, all pre-registered choices wired as flags:
  --stream_repr z|codes      P0-b's verdict picks the input arm (both implemented)
  --mode oat                 prefix training, per-sample budget n in {1,2,4,8}
  --mode unordered           control: all 8 registers always visible (matched bits)
  --mode chunk               control: locality-restricted registers (matched bits)
  --aug_time / --aug_strength  time-scale augmentation (P0-a verdict sets strength)
  --w_nce / --nce_embs       InfoNCE across embodiment streams (paired windows)

Loss = cross-entropy on the frame codes' integer FSQ levels (both streams). The
duration seat (last register) is never masked; its codebook is FSQ-learned.

P1c (2026-08-06) adds, all defaulting to the P1 behavior:
  --temporal_arch conv       ARM C1: strided conv keeps the time axis, SNAC multi-scale
  --temporal_arch freq       ARM C2: registers bound to fixed temporal DCT bands
  --w_multires               multi-resolution temporal dynamics loss (audio's
                             multi-resolution STFT loss, ported to motion)
See temporal_common.py's header for the three lines of evidence behind the arms.

Typical GPU run (flags per P0 verdict — see run_p1.sh):
  train_temporal.py --stream_root .../e13_stream --sets E19a1k,E19l24 \
      --embs human_smpl,unitree_h1,booster_t1,fourier_gr3,tienkung \
      --stream_repr z --mode oat --steps 30000 --out .../runs/P1_oat_z_0806
"""
from __future__ import annotations
import os, sys, json, math, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from temporal_common import (TemporalTokenizer, StreamGroup, load_groups, stream_L,  # noqa: E402
                             sample_hidden, budget_hidden, code_metrics, info_nce,
                             BUDGETS, reg_budgets, model_budgets, bit_budget,
                             multires_dyn_loss)

ROOT = "/var/tmp/alphamotion/topo_exp"


def build_temporal(cfg):
    return TemporalTokenizer(
        D=cfg["D"], L=cfg["L"], T=cfg.get("T", 60),
        stream_repr=cfg.get("stream_repr", "z"), d_model=cfg.get("d_model", 384),
        n_heads=cfg.get("n_heads", 8), enc_layers=cfg.get("enc_layers", 5),
        readout_layers=cfg.get("readout_layers", 2), dec_layers=cfg.get("dec_layers", 3),
        n_content=cfg.get("n_content", 8),
        fsq_levels=tuple(int(x) for x in str(cfg.get("fsq_levels", "5,5,5,5,5,5")).split(",")),
        mode=cfg.get("mode", "oat"), code_emb=cfg.get("code_emb", 128),
        dur_bottleneck=bool(cfg.get("dur_bottleneck", False)),
        arch=cfg.get("temporal_arch", "global"), conv_scales=cfg.get("conv_scales", "2,6"),
        conv_down=cfg.get("conv_down", 3), conv_res=cfg.get("conv_res", 2),
        conv_attn=bool(cfg.get("conv_attn", True)),
        freq_mix_layers=cfg.get("freq_mix_layers", 4))


def load_run(run, device):
    """Rebuild a trained temporal tokenizer from {run}/config.json + model.pt."""
    cfg = json.load(open(f"{run}/config.json"))
    model = build_temporal(cfg).to(device)
    model.load_state_dict(torch.load(f"{run}/model.pt", map_location=device))
    model.eval()
    return model, cfg


def fixed_val(groups, n_val, T):
    """Deterministic val windows per group (rng 7, base length T, no augmentation)."""
    out = []
    for gr in groups:
        g = np.random.default_rng(7)
        n = min(n_val, max(32, 8 * len(gr.va_pools)))
        out.append((gr, gr.sample_positions(n, g, split="val")))
    return out


@torch.no_grad()
def evaluate(model, val_sets, device, bs=32, budgets=None):
    budgets = budgets or model_budgets(model)
    model.eval()
    ev = {}
    for gr, pos in val_sets:
        for emb in gr.embs():
            accs = {}
            for n in budgets:
                ms = []
                for i in range(0, len(pos), bs):
                    w = gr.windows(emb, pos[i:i + bs]).to(device)
                    hid = budget_hidden(w.shape[0], model.n_reg, n, device)
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=(device == "cuda")):
                        logits, _ = model(w, hid)
                    ms.append((logits.float().argmax(-1) == w.long()).cpu())
                m = torch.cat(ms)
                accs[f"n{n}"] = round(m.float().mean().item(), 4)
            ev[f"{gr.name}/{emb}"] = accs
    model.train()
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream_root", default=f"{ROOT}/e13_stream")
    ap.add_argument("--sets", required=True, help="comma list; dirs = root/{set}_{emb}")
    ap.add_argument("--embs", required=True, help="comma list of embodiment stream names")
    ap.add_argument("--stream_repr", choices=["z", "codes"], required=True)
    ap.add_argument("--mode", choices=["oat", "unordered", "chunk"], default="oat")
    ap.add_argument("--T", type=int, default=60)          # 2 s @ 30 fps
    ap.add_argument("--n_content", type=int, default=8)   # ordered registers (+1 duration seat)
    ap.add_argument("--fsq_levels", default="5,5,5,5,5,5")
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--enc_layers", type=int, default=5)
    ap.add_argument("--readout_layers", type=int, default=2)
    ap.add_argument("--dec_layers", type=int, default=3)
    ap.add_argument("--code_emb", type=int, default=128)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--aug_time", type=float, default=0.0, help="prob of time-scale aug per sample")
    ap.add_argument("--aug_strength", type=float, default=1.0, help="1.0 = full 0.5x..2x")
    ap.add_argument("--w_motion", type=float, default=0.0,
                    help="motion-weighted CE: code positions whose GT level CHANGES vs "
                         "the previous frame get weight 1+w_motion (renormalized). "
                         "Prices the frame-to-frame deltas a static answer drops.")
    ap.add_argument("--w_vel", type=float, default=0.0,
                    help="aux velocity loss: relative MSE between consecutive-frame "
                         "deltas of softmax-expected dequantized z and GT deltas "
                         "(1.0 = a static prediction's score)")
    ap.add_argument("--w_multires", type=float, default=0.0,
                    help="multi-resolution temporal dynamics loss: relative MSE of the "
                         "first difference at several mean-pool scales (--multires_scales). "
                         "Static prediction = 1.0 at every scale. Motion analogue of "
                         "audio's multi-resolution STFT loss; scale 1 == --w_vel")
    ap.add_argument("--multires_scales", default="1,2,4,8")
    ap.add_argument("--dur_bottleneck", action="store_true",
                    help="duration seat reads ONLY T_in/T; content registers read a "
                         "canonical-length resample (speed invisible to them)")
    # ---------------------------------------------------------------- P1c arch arms
    ap.add_argument("--temporal_arch", choices=["global", "conv", "freq"], default="global",
                    help="global = P1 Perceiver pooling (default, reproduces P1/P1b); "
                         "conv = ARM C1 strided-conv latent TIME steps + SNAC multi-scale; "
                         "freq = ARM C2 registers bound to fixed temporal DCT bands")
    ap.add_argument("--conv_scales", default="2,6",
                    help="C1 only: tokens per SNAC scale, coarse-to-fine, must sum to "
                         "--n_content so the bit budget matches P1. '2,6' = 2 tokens at "
                         "1 Hz + 6 tokens at 3 Hz over a 2 s window")
    ap.add_argument("--conv_down", type=int, default=3,
                    help="C1: stride-2 stages (auto-raised until n_fine*2^down >= T)")
    ap.add_argument("--conv_res", type=int, default=2, help="C1: residual units per stage")
    ap.add_argument("--no_conv_attn", action="store_true",
                    help="C1: skip the self-attention over the latent time steps")
    ap.add_argument("--frame_stride", type=int, default=1,
                    help="keep every N-th frame of the window (same wall-clock span, N x fewer frames to compress). ProMAG/LIT 6.7: a low-compression model on a low frame rate beats a high-compression model on the full rate. 3 = 30fps -> 10fps.")
    ap.add_argument("--freq_mix_layers", type=int, default=4,
                    help="C2: transformer layers mixing the DCT bands on the ANALYSIS "
                         "side only; synthesis stays strictly band-local")
    ap.add_argument("--w_nce", type=float, default=0.0, help="InfoNCE weight across embodiments")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--nce_embs", type=int, default=2, help="embodiments per step incl. base")
    ap.add_argument("--n_val", type=int, default=256)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--in_ram", action="store_true", help="load code arrays into RAM (GPU runs)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    dev = args.device

    groups = load_groups(args.stream_root, args.sets, args.embs, args.T,
                         in_ram=args.in_ram, stride=args.frame_stride)
    meta = groups[0].meta
    D, L, f = meta["D"], stream_L(meta), meta["f"]
    for gr in groups:
        assert gr.meta["D"] == D and stream_L(gr.meta) == L, "stream groups disagree on D/L"
        assert gr.meta["run"] == meta["run"], "streams come from different frozen tokenizers"
    gw = np.array([gr.n_frames for gr in groups], np.float64); gw /= gw.sum()

    cfg = vars(args) | {"D": D, "L": L, "f": f, "frozen_run": meta["run"],
                        "conv_attn": not args.no_conv_attn}
    model = build_temporal(cfg).to(dev)
    off = np.tile(np.asarray(meta["deq_offset"], np.float32), D // f)
    half = np.tile(np.asarray(meta["deq_half"], np.float32), D // f)
    model.set_deq(off, half)
    n_par = sum(p.numel() for p in model.parameters())
    lv = [int(x) for x in args.fsq_levels.split(",")]
    vocab = int(np.prod(lv))
    bits = bit_budget(args.n_content, lv, args.T,
                      scales=getattr(model, "conv_scales", None),
                      stride=args.frame_stride)
    buds = model_budgets(model)
    print(f"arch={args.temporal_arch} mode={args.mode} repr={args.stream_repr} "
          f"D={D} L={L} params={n_par/1e6:.1f}M "
          f"registers={args.n_content}+1(duration) vocab/register={vocab} budgets={buds}")
    if args.temporal_arch == "conv":
        print(f"  conv: scales={model.conv_scales} n_fine={model.n_fine} "
              f"L0={model.conv_L0} down={model.conv_down} attn={model.conv_attn}")
    print("  BIT BUDGET " + json.dumps(bits))
    cfg["bit_budget"] = bits
    cfg["budgets"] = list(buds)
    cfg["n_params"] = n_par
    json.dump(cfg, open(f"{args.out}/config.json", "w"), indent=2)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / args.warmup, 0.5 * (1 + math.cos(math.pi * s / args.steps))))
    val_sets = fixed_val(groups, args.n_val, args.T)
    grng = np.random.default_rng(args.seed + 1)
    mr_scales = tuple(int(x) for x in args.multires_scales.split(",") if x)
    log, t_loop = [], time.time()

    model.train()
    for step in range(args.steps):
        gr = groups[int(grng.choice(len(groups), p=gw))]
        embs = gr.embs()
        n_nce = min(args.nce_embs, len(embs)) if args.w_nce > 0 else 1
        picks = grng.choice(len(embs), n_nce, replace=False)
        # time-scale augmentation: per-STEP window length T_in in [T/2, 2T] — a longer
        # or shorter contiguous span decoded at normalized time (see temporal_common)
        T_in = args.T
        if args.aug_time > 0 and grng.random() < args.aug_time:
            s = math.exp(grng.uniform(-1.0, 1.0) * math.log(2.0) * args.aug_strength)
            T_in = int(np.clip(round(args.T * s), args.T // 2, 2 * args.T))
        pos = gr.sample_positions(args.bs, grng, "train", T_in=T_in)
        hidden = sample_hidden(args.bs, model.n_reg, args.mode, dev, budgets=buds)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            w0 = gr.windows(embs[picks[0]], pos).to(dev)
            zq, q0 = model.encode_regs(w0)
            logits = model.decode_frames(zq, hidden, T_out=T_in)
            lf = logits.float()
            if args.w_motion > 0:
                # motion-weighted CE: upweight (frame, dim) positions whose GT level
                # changed vs the previous frame; renormalize so the scale matches
                # plain mean CE. Frame 0 has no predecessor -> weight 1.
                ce = F.cross_entropy(lf.reshape(-1, L), w0.long().reshape(-1),
                                     reduction="none")
                chg = torch.zeros(w0.shape, dtype=torch.float32, device=dev)
                chg[:, 1:] = (w0[:, 1:] != w0[:, :-1]).float()
                wgt = 1.0 + args.w_motion * chg.reshape(-1)
                loss = (ce * wgt).sum() / wgt.sum()
            else:
                loss = F.cross_entropy(lf.reshape(-1, L), w0.long().reshape(-1))
            vel = torch.zeros((), device=dev)
            mres = torch.zeros((), device=dev)
            if args.w_vel > 0 or args.w_multires > 0:
                # velocity as a first-class target: softmax-expected level -> deq z,
                # relative MSE of consecutive-frame deltas (static prediction -> 1.0).
                # P1b's ONLY validated positive: prefix-1 motion energy
                # walk .73->.86, run .40->.55, jacks .23->.29, stand .50->.79.
                lev = torch.arange(L, device=dev, dtype=torch.float32)
                zh = ((lf.softmax(-1) * lev).sum(-1) - model.deq_off) / model.deq_half
                zg = (w0.float() - model.deq_off) / model.deq_half
                if args.w_vel > 0:
                    dh, dg = zh[:, 1:] - zh[:, :-1], zg[:, 1:] - zg[:, :-1]
                    vel = ((dh - dg) ** 2).mean() / (dg.pow(2).mean() + 1e-4)
                    loss = loss + args.w_vel * vel
                if args.w_multires > 0:
                    # same relative-delta metric at several mean-pool scales, so the
                    # SLOW trajectory is priced too and per-frame jitter cannot fake it
                    mres = multires_dyn_loss(zh, zg, mr_scales)
                    loss = loss + args.w_multires * mres
            nce = torch.zeros((), device=dev)
            for k in picks[1:]:                          # paired windows, other bodies
                wk = gr.windows(embs[k], pos).to(dev)
                _, qk = model.encode_regs(wk)
                nce = nce + info_nce(q0, qk, args.tau)
            if n_nce > 1:
                loss = loss + args.w_nce * nce / (n_nce - 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % 100 == 0:
            sps = 100 / max(time.time() - t_loop, 1e-9) if step else 0.0
            t_loop = time.time()
            print(f"step {step} loss {loss.item():.4f} nce {float(nce.detach()):.3f} "
                  f"vel {float(vel.detach()):.3f} mres {float(mres.detach()):.3f} "
                  f"{sps:.2f} steps/s ({time.time()-t0:.0f}s)", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev = {"step": step + 1, "loss": round(loss.item(), 4),
                  "budget_level_acc": evaluate(model, val_sets, dev, budgets=buds)}
            log.append(ev)
            print("EVAL", json.dumps(ev), flush=True)
            json.dump(log, open(f"{args.out}/eval_log.json", "w"), indent=2)
            torch.save(model.state_dict(), f"{args.out}/model.pt")

    torch.save(model.state_dict(), f"{args.out}/model.pt")
    print("DONE", args.out, f"({(time.time()-t0)/3600:.2f}h)")


if __name__ == "__main__":
    main()
