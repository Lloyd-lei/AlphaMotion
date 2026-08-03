"""Any-to-any cross-embodiment translation with DUAL FSQ (separate position/rotation
codes). Trained on paired SOMA<->G1, all 4 direction combos, with train/val split.
Reports rot + pos per direction (cross directions = the real translation metric).
"""
from __future__ import annotations
import os, sys, json, time, argparse, itertools
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE_ROOT, SkeletonSpec, rot6d_to_matrix, geodesic_deg, swing_twist_decompose
from model_dual import DualTranslator
from train_ae import static_cond
from train_global import build_global
from train_translate import load_paired


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{CACHE_ROOT}/cache_paired")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_latent", type=int, default=32)
    ap.add_argument("--fsq_stages", type=int, default=2)
    ap.add_argument("--w_pos", type=float, default=5.0)
    ap.add_argument("--w_align", type=float, default=0.0)    # L2 same-motion code pull (collapses if strong)
    ap.add_argument("--w_contrast", type=float, default=0.0) # InfoNCE: same-motion attract, others repel (no collapse)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--self_only", action="store_true")   # only reconstruction (soma->soma, g1->g1)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--out", default=f"{CACHE_ROOT}/runs/D_dual")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"; torch.manual_seed(0)

    paired, g1name = load_paired(args.cache)
    names = ["soma77", g1name]
    data = {}; N = None
    from semantics import build_name_embeddings, embed_matrix
    lut = build_name_embeddings(sum([list(paired[n][0].joint_names) for n in names], []), device=device)
    for name in names:
        spec, local6d = paired[name]
        st = static_cond(spec, device)
        gr6, gp, reach = build_global(local6d, spec, device)
        sem = embed_matrix(spec.joint_names, lut).to(device)     # [J,768] SigLIP joint-name semantics
        data[name] = {"spec": spec, "st": st, "pose": torch.cat([gr6, gp / reach], -1).to(device),
                      "reach": reach, "sem": sem}
        N = local6d.shape[0]; print(f"[{name}] J={spec.J} rows={N} reach={reach:.1f}cm sem={tuple(sem.shape)}")
    perm = torch.randperm(N); nval = max(128, N // 10)
    val_idx, tr_idx = perm[:nval], perm[nval:]

    model = DualTranslator(n_latent=args.n_latent, fsq_stages=args.fsq_stages).to(device)
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M dual-FSQ bits/frame={model.bits:.0f}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    combos = [(n, n) for n in names] if args.self_only else list(itertools.product(names, names))
    print("combos:", combos)
    FINGER_KW = ("Hand", "Thumb", "Index", "Middle", "Ring", "Pinky", "Finger")

    def geom(name, B):
        st = data[name]["st"]
        return dict(rest_off=st["rest_off"][None].expand(B, -1, -1), bonelen=st["bonelen"][None].expand(B, -1),
                    depth=st["depth"][None].expand(B, -1), hop=st["hop"][None].expand(B, -1, -1),
                    sem=data[name]["sem"][None].expand(B, -1, -1))

    def run(src, tgt, idx):
        B = len(idx)
        zp, zr = model.encode(data[src]["pose"][idx], geom(src, B))
        prot, ppos = model.decode(zp, zr, geom(tgt, B))
        gt = data[tgt]["pose"][idx]
        rot = geodesic_deg(rot6d_to_matrix(prot), rot6d_to_matrix(gt[..., :6])).mean()
        pos = (ppos - gt[..., 6:9]).norm(dim=-1).mean()
        return rot, pos

    def swing_twist_report(name):
        d = data[name]; B = len(val_idx)
        zp, zr = model.encode(d["pose"][val_idx], geom(name, B))
        prot, ppos = model.decode(zp, zr, geom(name, B))
        pr = rot6d_to_matrix(prot); gt = rot6d_to_matrix(d["pose"][val_idx][..., :6])
        a0 = torch.nn.functional.normalize(d["st"]["rest_off"], dim=-1)[None].expand(B, -1, -1)
        full = geodesic_deg(pr, gt)                          # [B,J]
        # swing = bone-direction error (position-observable); residual ~ twist
        vp = (pr @ a0.unsqueeze(-1)).squeeze(-1); vg = (gt @ a0.unsqueeze(-1)).squeeze(-1)
        swing = torch.rad2deg(torch.arccos((vp * vg).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)))
        finger = torch.tensor([any(k in nm for k in FINGER_KW) for nm in d["spec"].joint_names], device=pr.device)
        body = ~finger
        return {"full_rot": round(float(full[:, body].mean()), 1),
                "swing": round(float(swing[:, body].mean()), 1),
                "twist_residual": round(float(full[:, body].mean() - swing[:, body].mean()), 1)}

    def evaluate():
        model.eval(); out = {}
        with torch.no_grad():
            for src, tgt in combos:
                rot, pos = run(src, tgt, val_idx)
                out[f"{src[:4]}->{tgt[:4]}"] = {"rot": round(float(rot), 1), "pos_cm": round(float(pos * data[tgt]["reach"]), 2)}
            for name in names:                              # swing/twist breakdown (self-recon)
                out[f"ST_{name[:4]}"] = swing_twist_report(name)
            # code alignment: how close are soma vs g1 codes for the SAME motion?
            zp_s, zr_s = model.encode(data[names[0]]["pose"][val_idx], geom(names[0], len(val_idx)))
            zp_g, zr_g = model.encode(data[names[1]]["pose"][val_idx], geom(names[1], len(val_idx)))
            code_dist = float(((zp_s - zp_g) ** 2).mean() + ((zr_s - zr_g) ** 2).mean())
            code_scale = float((zp_s ** 2).mean() + (zr_s ** 2).mean())
            out["code_align_dist"] = round(code_dist, 4)
            out["code_scale"] = round(code_scale, 4)
        model.train(); return out

    log = []; t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        sel = tr_idx[torch.randint(0, len(tr_idx), (args.bs,))]
        for src, tgt in combos:
            rot, pos = run(src, tgt, sel)
            ((rot / 30.0 + args.w_pos * pos) / len(combos)).backward()
        if args.w_align > 0 or args.w_contrast > 0:
            zp_s, zr_s = model.encode(data[names[0]]["pose"][sel], geom(names[0], len(sel)))
            zp_g, zr_g = model.encode(data[names[1]]["pose"][sel], geom(names[1], len(sel)))
            extra = 0.0
            if args.w_align > 0:      # L2 pull (baseline, collapses if strong)
                extra = extra + args.w_align * (((zp_s - zp_g) ** 2).mean() + ((zr_s - zr_g) ** 2).mean())
            if args.w_contrast > 0:   # InfoNCE: same-motion attract, other-motion repel -> no collapse
                B = len(sel)
                vs = torch.cat([zp_s.reshape(B, -1), zr_s.reshape(B, -1)], -1)
                vg = torch.cat([zp_g.reshape(B, -1), zr_g.reshape(B, -1)], -1)
                vs = torch.nn.functional.normalize(vs, dim=-1); vg = torch.nn.functional.normalize(vg, dim=-1)
                logits = (vs @ vg.t()) / args.tau        # [B,B], diagonal = same motion
                labels = torch.arange(B, device=vs.device)
                nce = 0.5 * (torch.nn.functional.cross_entropy(logits, labels) +
                             torch.nn.functional.cross_entropy(logits.t(), labels))
                extra = extra + args.w_contrast * nce
            extra.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            ev = evaluate(); ev["step"] = step + 1; log.append(ev)
            print("EVAL", json.dumps(ev)); json.dump(log, open(f"{args.out}/eval_log.json", "w"), indent=2)
    torch.save(model.state_dict(), f"{args.out}/model.pt")
    json.dump(vars(args), open(f"{args.out}/config.json", "w"), indent=2)
    print("DONE", args.out)


if __name__ == "__main__":
    main()
