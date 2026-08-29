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
    # @file argfile support (N3): one token per line, e.g. runs/qwen_recipe.args
    # reproduces E27_qwen_0806's exact flags; later CLI flags override earlier ones.
    ap = argparse.ArgumentParser(fromfile_prefix_chars="@")
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
    ap.add_argument("--visible_boost", type=float, default=0.0)  # upweight visible joints under mask (constraint honoring)
    ap.add_argument("--tail_alpha", type=float, default=0.0)     # 0809 tail cure: per-sample-joint error^alpha reweighting (training only)
    ap.add_argument("--w_pos", type=float, default=5.0)
    ap.add_argument("--w_align", type=float, default=0.0)    # L2 same-motion code pull (collapses if strong)
    ap.add_argument("--w_contrast", type=float, default=0.0) # InfoNCE: same-motion attract, others repel (no collapse)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--self_only", action="store_true")   # only reconstruction (soma->soma, g1->g1)
    ap.add_argument("--holdout", default=None)            # E5: comma-separated embodiments excluded from training; first one gets zero-shot eval
    ap.add_argument("--combo_sample", type=int, default=0)  # E5: combos per step (0 = all; bounds N^2 growth)
    ap.add_argument("--contrast_sample", type=int, default=0)  # InfoNCE embodiments per step (0 = all; bounds memory at large N)
    ap.add_argument("--use_dof", action="store_true")  # feed per-joint rotation axes + limits
    # E27: which frozen text encoder produces the joint-name semantics (same prompt both ways).
    ap.add_argument("--sem_encoder", choices=["siglip", "qwen3"], default="siglip")
    # E25 DOF-masking arms (default = every prior run, bit-for-bit).
    #   --rot_head angles : arm A, decoder emits hinge ANGLES -> FK -> global rotations
    #   --dof_loss        : arm B, free 6D head, loss projected onto the hinge manifold
    ap.add_argument("--rot_head", choices=["free", "angles"], default="free")
    ap.add_argument("--dof_loss", action="store_true")
    ap.add_argument("--judge_mode", choices=["off", "observe", "grad"], default="off")  # E14: sigmoid critics
    # E28: decode encoder-VISIBLE joints as a residual on the observed pose (same-skeleton
    # directions only; cross-body decode is untouched). Default off = every prior run.
    ap.add_argument("--vis_residual", action="store_true")
    ap.add_argument("--w_judge", type=float, default=0.1)
    # N3e (--reenc_consistency W): RE-ENCODE CYCLE CONSISTENCY. The measured failure
    # it exists to fix: encode(decode(z)) explodes even same-body (round-trip error
    # ~8.7 -> ~38.8 deg), which kills any synthetic-data pipeline. With prob
    # --reenc_prob per step: pick one direction, decode, RE-ENCODE the decoded pose
    # (no stop-grad anywhere — the encoder must learn to read decoded input AND the
    # decoder must emit re-encodable poses) and penalize the z-space distance to the
    # original codes with weight W. Runs at HALF batch to bound the double forward.
    # 0 = off = bit-exact prior behavior.
    ap.add_argument("--reenc_consistency", type=float, default=0.0)
    ap.add_argument("--reenc_prob", type=float, default=0.3)
    # N3f (--var_floor W): OUTPUT-VARIANCE FLOOR (VICReg's variance term applied to
    # OUTPUTS): hinge max(0, std_gt - std_decoded) per joint/dim across the batch,
    # std target = the GT batch's own std. Treats amplitude compression (the kick
    # that never lifts). 0 = off = bit-exact prior behavior.
    ap.add_argument("--var_floor", type=float, default=0.0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)  # training randomness only; the val split stays seed-0
    # E24 parameter axis. Everything before it scaled the BOTTLENECK (n_latent, fsq levels)
    # or the data, never the backbone — every run from K=16 to K=128 sat at 17.95-18.01M params.
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--dec_layers", type=int, default=4)
    ap.add_argument("--out", default=f"{CACHE_ROOT}/runs/D_dual")
    ap.add_argument("--device", default="cuda")   # "cpu" = smoke test while the GPU queue runs
    ap.add_argument("--max_rows", type=int, default=0)  # 0 = all; smoke tests only
    # decoder post-train (0811): warm-start a finished run and touch ONLY the decode
    # side. The code lattice (enc/pool/FSQ-in, shared jf) stays byte-frozen, so every
    # downstream consumer (BRIDGE A3/B3, text layer) keeps working unchanged.
    ap.add_argument("--init", default=None, help="path to model.pt to warm-start from")
    ap.add_argument("--train_scope", choices=["all", "decoder"], default="all")
    ap.add_argument("--arm_boost", type=float, default=0.0)
    ap.add_argument("--ee_boost", type=float, default=0.0)
    # 0812 experiments (lloyd): the position head cannot reach the render, so
    #   --fk_loss   supervise the ROTATION head with metric error via FK
    #   --rot_only  spend every code slot on rotation instead of half on a
    #               stream whose output is discarded
    ap.add_argument("--fk_loss", action="store_true")
    ap.add_argument("--rot_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = args.device; torch.manual_seed(0)

    data, names, N = load_paired_data(args.cache, device, max_rows=args.max_rows,
                                      sem_encoder=args.sem_encoder)
    val_idx, tr_idx = split_indices(N)          # consumes global RNG -> must stay at seed 0
    torch.manual_seed(args.seed)                # only init/batching/masking vary across seeds

    cfg = vars(args)
    if args.judge_mode != "off":
        cfg["n_judges"] = len(names)                     # judges cover ALL embodiments incl. holdouts
    emb_idx = {n: i for i, n in enumerate(names)}
    model = build_model(cfg, device)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device))
        print(f"warm-start <- {args.init}")
    train_params = list(model.parameters())
    if args.train_scope == "decoder":
        DEC = ("decp.", "decr.", "selfp.", "selfr.", "hrot", "hpos",
               "fsqp.proj_out", "fsqr.proj_out")
        n_tr = 0
        for nm, prm in model.named_parameters():
            on = nm.startswith(DEC)
            prm.requires_grad_(on)
            n_tr += prm.numel() if on else 0
        train_params = [prm for prm in model.parameters() if prm.requires_grad]
        print(f"train_scope=decoder: {n_tr/1e6:.2f}M trainable, lattice frozen")
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M dual-FSQ bits/frame={model.bits:.0f}")
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    holdouts = [h for h in (args.holdout or "").split(",") if h]
    for h in holdouts:
        assert h in names, f"holdout {h} not in {names}"
    train_names = [n for n in names if n not in holdouts]
    combos = [(n, n) for n in train_names] if args.self_only else list(itertools.product(train_names, train_names))
    if len(combos) <= 20:
        eval_combos = list(combos)
    else:
        # big-N caches: full product eval (169 directions at n=13) costs more than training.
        # Eval = all self-recon + soma-source cross translations; training combos unaffected.
        eval_combos = [(n, n) for n in train_names] + [(train_names[0], t) for t in train_names[1:]]
    if holdouts:                                         # zero-shot: self-recon + translation into first holdout
        eval_combos += [(holdouts[0], holdouts[0]), (train_names[0], holdouts[0])]
    combo_rng = np.random.default_rng(4)                 # own stream: torch RNG order untouched
    reenc_rng = np.random.default_rng(args.seed * 1000 + 77)   # N3e's own stream
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
            ex = {} if args.var_floor > 0 else None                 # N3f
            rot, pos = run_dir(model, data, src, tgt, sel, joint_mask=jm,
                               visible_boost=args.visible_boost, dof_loss=args.dof_loss,
                               extras=ex, tail_alpha=args.tail_alpha,
                               arm_boost=args.arm_boost, ee_boost=args.ee_boost,
                               fk_loss=args.fk_loss)
            term = rot / 30.0 + args.w_pos * pos
            if ex is not None:
                term = term + args.var_floor * ex["var_floor"]
            (term / len(step_combos)).backward()
        if args.reenc_consistency > 0 and float(reenc_rng.random()) < args.reenc_prob:
            # N3e: decode -> re-encode cycle, half batch, one random direction.
            B2 = max(8, args.bs // 2)
            sel2 = sel[:B2]
            a, b = combos[int(reenc_rng.integers(len(combos)))]
            zp_c, zr_c = model.encode(data[a]["pose"][sel2], geom(data, a, B2))
            prot_c, ppos_c = model.decode(zp_c, zr_c, geom(data, b, B2))
            re_in = torch.cat([prot_c, ppos_c], -1)                 # decoded pose [B,J,9]
            zp_r, zr_r = model.encode(re_in, geom(data, b, B2))
            lre = ((zp_r - zp_c) ** 2).mean() + ((zr_r - zr_c) ** 2).mean()
            (args.reenc_consistency * lre).backward()
        if args.w_align > 0 or args.w_contrast > 0:
            B = len(sel)
            cnames = train_names
            if args.contrast_sample and args.contrast_sample < len(train_names):
                pick = combo_rng.choice(len(train_names), args.contrast_sample, replace=False)
                cnames = [train_names[i] for i in pick]
            zs = {n: model.encode(data[n]["pose"][sel], geom(data, n, B)) for n in cnames}
            pairs = [(a, b) for i, a in enumerate(cnames) for b in cnames[i + 1:]]
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
            if args.judge_mode != "off":             # E14 critics: same graph, single backward
                jloss = 0.0
                for n in cnames:
                    zp_j, zr_j = zs[n]
                    if args.judge_mode == "observe": # critics learn, backbone untouched
                        zp_j, zr_j = zp_j.detach(), zr_j.detach()
                    jl = model.judge(zp_j, zr_j)
                    tgt = torch.zeros_like(jl); tgt[:, emb_idx[n]] = 1.0
                    jloss = jloss + torch.nn.functional.binary_cross_entropy_with_logits(jl, tgt)
                extra = extra + args.w_judge * jloss / len(cnames)
            extra.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step(); sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev = evaluate(model, data, names, eval_combos, val_idx); ev["step"] = step + 1; log.append(ev)
            print("EVAL", json.dumps(ev)); json.dump(log, open(f"{args.out}/eval_log.json", "w"), indent=2)
    torch.save(model.state_dict(), f"{args.out}/model.pt")
    json.dump(vars(args), open(f"{args.out}/config.json", "w"), indent=2)
    print("DONE", args.out)


if __name__ == "__main__":
    main()
