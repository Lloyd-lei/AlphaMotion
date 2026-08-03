"""Any-to-any cross-embodiment translation with a shared FSQ code.

encode(source skeleton pose) -> shared FSQ latent -> decode(target skeleton) -> target pose.
Trained on PAIRED frames (same motion in SOMA & G1) for all 4 direction combos.
This is the REAL cross-embodiment test (vs prior self-reconstruction).
"""
from __future__ import annotations
import os, sys, json, time, argparse, itertools
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE_ROOT, SkeletonSpec, rot6d_to_matrix, matrix_to_rot6d, fk_global, geodesic_deg
from model import TopoAE
from train_ae import static_cond
from train_global import build_global


def load_paired(cache):
    man = json.load(open(f"{cache}/manifest.json"))
    out = {}
    for name in ["soma77", man["g1_name"]]:
        d = np.load(f"{cache}/{name}_spec.npz", allow_pickle=True)
        spec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], list(d["joint_names"]))
        local6d = torch.load(f"{cache}/{name}_paired_local6d.pt")
        out[name] = (spec, local6d)
    return out, man["g1_name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{CACHE_ROOT}/cache_paired")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_latent", type=int, default=32)
    ap.add_argument("--use_fsq", action="store_true")
    ap.add_argument("--fsq_stages", type=int, default=2)
    ap.add_argument("--w_pos", type=float, default=5.0)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--out", default=f"{CACHE_ROOT}/runs/T_translate")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"; torch.manual_seed(0)

    paired, g1name = load_paired(args.cache)
    names = ["soma77", g1name]
    data = {}
    N = None
    for name in names:
        spec, local6d = paired[name]
        st = static_cond(spec, device)
        gr6, gp, reach = build_global(local6d, spec, device)
        pose = torch.cat([gr6, gp / reach], -1)          # [N,J,9]
        data[name] = {"spec": spec, "st": st, "pose": pose, "reach": reach}
        N = pose.shape[0]
        print(f"[{name}] J={spec.J} rows={N} reach={reach:.1f}cm")
    perm = torch.randperm(N); nval = max(128, N // 10)
    val_idx, tr_idx = perm[:nval], perm[nval:]

    model = TopoAE(d_model=args.d_model, n_latent=args.n_latent, use_index=True,
                   target="global", use_fsq=args.use_fsq, fsq_stages=args.fsq_stages).to(device)
    bits = model.fsq.bits * args.n_latent if args.use_fsq else float("inf")
    print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M fsq={args.use_fsq} bits/frame={bits:.0f}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    combos = list(itertools.product(names, names))   # 4 direction combos

    def geom(name, B):
        st = data[name]["st"]
        return dict(rest_off=st["rest_off"][None].expand(B, -1, -1),
                    bonelen=st["bonelen"][None].expand(B, -1),
                    depth=st["depth"][None].expand(B, -1),
                    hop=st["hop"][None].expand(B, -1, -1), kpm=None)

    def run(src, tgt, idx):
        B = len(idx)
        gsrc = geom(src, B); gtgt = geom(tgt, B)
        z = model.encode(data[src]["pose"][idx].to(device), gsrc["rest_off"], gsrc["bonelen"],
                         gsrc["depth"], gsrc["hop"], gsrc["kpm"])
        pred = model.decode(z, gtgt["rest_off"], gtgt["bonelen"], gtgt["depth"], gtgt["hop"], gtgt["kpm"])
        gt = data[tgt]["pose"][idx].to(device)
        rot = geodesic_deg(rot6d_to_matrix(pred[..., :6]), rot6d_to_matrix(gt[..., :6])).mean()
        pos = (pred[..., 6:9] - gt[..., 6:9]).norm(dim=-1).mean()
        return rot, pos

    def evaluate():
        model.eval(); out = {}
        with torch.no_grad():
            for src, tgt in combos:
                rot, pos = run(src, tgt, val_idx)
                out[f"{src[:4]}->{tgt[:4]}"] = {"rot_deg": round(float(rot), 1),
                                                "pos_cm": round(float(pos * data[tgt]["reach"]), 2)}
        model.train(); return out

    log = []; t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        sel = tr_idx[torch.randint(0, len(tr_idx), (args.bs,))]
        for src, tgt in combos:                      # accumulate over all 4 directions
            rot, pos = run(src, tgt, sel)
            ((rot / 30.0 + args.w_pos * pos) / len(combos)).backward()
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
