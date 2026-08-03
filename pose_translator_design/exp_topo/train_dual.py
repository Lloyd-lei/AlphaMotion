"""Any-to-any cross-embodiment translation with DUAL FSQ (separate position/rotation
codes). Trained on paired SOMA<->G1, all 4 direction combos, with train/val split.
Reports rot + pos per direction (cross directions = the real translation metric).

Shared data/eval logic lives in dual_common.py (same code path as eval_dual.py).
RNG order is load-bearing for FIX reproducibility: seed(0) -> load -> split -> model.
"""
from __future__ import annotations
import os, sys, json, time, argparse, itertools
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE_ROOT
from dual_common import load_paired_data, split_indices, build_model, geom, run_dir, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{CACHE_ROOT}/cache_paired")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_latent", type=int, default=32)
    ap.add_argument("--fsq_stages", type=int, default=2)
    ap.add_argument("--fsq_levels", default="8,8,8,5,5,5")   # E0a bit-budget sweep
    ap.add_argument("--single_codebook", action="store_true")  # E0c: one shared FSQ, same total bits
    ap.add_argument("--mask_prob", type=float, default=0.0)  # E0b: per-sample prob of masked encoder input
    ap.add_argument("--mask_mix", default="rand:0.4,ee:0.3,root:0.3")
    ap.add_argument("--w_pos", type=float, default=5.0)
    ap.add_argument("--w_align", type=float, default=0.0)    # L2 same-motion code pull (collapses if strong)
    ap.add_argument("--w_contrast", type=float, default=0.0) # InfoNCE: same-motion attract, others repel (no collapse)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--self_only", action="store_true")   # only reconstruction (soma->soma, g1->g1)
    ap.add_argument("--holdout", default=None)            # E5: embodiment excluded from training, eval zero-shot
    ap.add_argument("--combo_sample", type=int, default=0)  # E5: combos per step (0 = all; bounds N^2 growth)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--out", default=f"{CACHE_ROOT}/runs/D_dual")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"; torch.manual_seed(0)

    data, names, N = load_paired_data(args.cache, device)
    val_idx, tr_idx = split_indices(N)

    model = build_model(vars(args), device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M dual-FSQ bits/frame={model.bits:.0f}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    train_names = [n for n in names if n != args.holdout]
    assert args.holdout is None or args.holdout in names, f"holdout {args.holdout} not in {names}"
    combos = [(n, n) for n in train_names] if args.self_only else list(itertools.product(train_names, train_names))
    eval_combos = list(combos)
    if args.holdout:                                     # zero-shot: self-recon + translation into holdout
        eval_combos += [(args.holdout, args.holdout), (train_names[0], args.holdout)]
    combo_rng = np.random.default_rng(4)                 # own stream: torch RNG order untouched
    print("combos:", combos, "eval extra:", eval_combos[len(combos):])

    mask_mix = None
    if args.mask_prob > 0:
        from masking import parse_mix, sample_joint_mask
        mask_mix = parse_mix(args.mask_mix)

    log = []; t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        sel = tr_idx[torch.randint(0, len(tr_idx), (args.bs,))]
        step_combos = combos
        if args.combo_sample and args.combo_sample < len(combos):
            pick = combo_rng.choice(len(combos), args.combo_sample, replace=False)
            step_combos = [combos[i] for i in pick]
        for src, tgt in step_combos:
            jm = None
            if mask_mix is not None:
                # per-sample: with prob mask_prob the encoder sees a partial pose; loss stays full-body
                jm = sample_joint_mask(data[src]["spec"], args.bs, mask_mix, device=device)
                jm &= (torch.rand(args.bs, 1, device=device) < args.mask_prob)
            rot, pos = run_dir(model, data, src, tgt, sel, joint_mask=jm)
            ((rot / 30.0 + args.w_pos * pos) / len(step_combos)).backward()
        if args.w_align > 0 or args.w_contrast > 0:
            B = len(sel)
            zs = {n: model.encode(data[n]["pose"][sel], geom(data, n, B)) for n in train_names}
            pairs = [(a, b) for i, a in enumerate(train_names) for b in train_names[i + 1:]]
            extra = 0.0
            for a, b in pairs:                       # mean over unordered embodiment pairs
                (zp_s, zr_s), (zp_g, zr_g) = zs[a], zs[b]
                if args.w_align > 0:      # L2 pull (baseline, collapses if strong)
                    extra = extra + args.w_align / len(pairs) * (
                        ((zp_s - zp_g) ** 2).mean() + ((zr_s - zr_g) ** 2).mean())
                if args.w_contrast > 0:   # InfoNCE: same-motion attract, other-motion repel -> no collapse
                    vs = torch.cat([zp_s.reshape(B, -1), zr_s.reshape(B, -1)], -1)
                    vg = torch.cat([zp_g.reshape(B, -1), zr_g.reshape(B, -1)], -1)
                    vs = torch.nn.functional.normalize(vs, dim=-1); vg = torch.nn.functional.normalize(vg, dim=-1)
                    logits = (vs @ vg.t()) / args.tau        # [B,B], diagonal = same motion
                    labels = torch.arange(B, device=vs.device)
                    nce = 0.5 * (torch.nn.functional.cross_entropy(logits, labels) +
                                 torch.nn.functional.cross_entropy(logits.t(), labels))
                    extra = extra + args.w_contrast / len(pairs) * nce
            extra.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev = evaluate(model, data, names, eval_combos, val_idx); ev["step"] = step + 1; log.append(ev)
            print("EVAL", json.dumps(ev)); json.dump(log, open(f"{args.out}/eval_log.json", "w"), indent=2)
    torch.save(model.state_dict(), f"{args.out}/model.pt")
    json.dump(vars(args), open(f"{args.out}/config.json", "w"), indent=2)
    print("DONE", args.out)


if __name__ == "__main__":
    main()
