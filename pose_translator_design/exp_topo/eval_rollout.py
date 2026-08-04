"""E1 duel judge: shared rollout evaluation for the AR-LM (A) and chunked latent
diffusion (B) over the SAME frozen tokenizer, same val sequences, same horizons.

Protocol: context = first 56 frames of a val sequence, roll out 32 frames
autoregressively (A: frame-by-frame greedy codes; B: 4 chunks x 8 frames, DDIM),
decode each predicted frame with the frozen tokenizer decoder (soma77 skeleton),
report pos/rot error vs GT at horizons {1,8,32} + ms/frame.

Usage:
  python eval_rollout.py --arm A --run runs/E1A_v2_0804 --tokens runs/E0b_sparse2_0803/tokens_seq10k --tokenizer runs/E0b_sparse2_0803
  python eval_rollout.py --arm B --run runs/E1B_chunkdiff_0804 ...same...
"""
from __future__ import annotations
import os, sys, json, math, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from common import rot6d_to_matrix, geodesic_deg
from dual_common import load_paired_data, build_model, geom
from train_ar_lm import ARLM, dequant
from train_latent_diffusion import ChunkDenoiser, cosine_alphas, CHUNK

CTX, HOR = 56, 32
HORIZONS = (1, 8, 32)


def decode_frames(tok, data, f_frames, device):
    """f_frames [T,2,K,f] pre-round features -> decoded (rot6d, pos) on soma77."""
    T = f_frames.shape[0]
    fp, fr = f_frames[:, 0], f_frames[:, 1]
    ip = tok.fsqp.ints(fp)
    fsq_r = tok.fsqp if tok.single_codebook else tok.fsqr
    ir = fsq_r.ints(fr)
    with torch.no_grad():
        prot, ppos = tok.decode_from_ints(ip, ir, geom(data, "soma77", T))
    return prot, ppos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--n_seq", type=int, default=64)
    ap.add_argument("--ddim_steps", type=int, default=16)
    args = ap.parse_args()
    device = "cuda"

    # frozen tokenizer; data MUST come from the tokens' source cache (GT rows are
    # indexed against it — loading the tokenizer's train cache here OOB-crashed)
    tcfg = json.load(open(f"{args.tokenizer}/config.json"))
    tmeta = json.load(open(f"{args.tokens}/meta.json"))
    torch.manual_seed(0)
    data, names, _ = load_paired_data(tmeta["cache"], device)
    tok = build_model(tcfg, device)
    tok.load_state_dict(torch.load(f"{args.tokenizer}/model.pt", map_location=device), strict=False)
    tok.eval()

    F = torch.load(f"{args.tokens}/soma77_f.pt").float().to(device)      # [N,2,K,f]
    I = torch.load(f"{args.tokens}/soma77_ints.pt").long().to(device)    # [N,2,S,K,f]
    meta = json.load(open(f"{args.tokens}/meta.json"))
    levels = torch.tensor(meta["levels"], dtype=torch.float32)
    seqs = json.load(open(f"{args.tokens}/sequences.json"))
    rng = np.random.default_rng(0)
    order = rng.permutation(len(seqs))
    nval = max(8, len(seqs) // 10)
    val_seqs = [seqs[i] for i in order[:nval] if seqs[i]["len"] >= CTX + HOR]
    picks = val_seqs[: args.n_seq]
    print(f"val seqs usable {len(val_seqs)}, evaluating {len(picks)}")

    cfg = json.load(open(f"{args.run}/config.json"))
    N, D2, K, f = F.shape
    if args.arm == "A":
        X_all = dequant(I.cpu(), levels).to(device)                       # [N,P] discrete-only input rep
        P = X_all.shape[1]; L = int(I.max().item()) + 1
        model = ARLM(P, P, L, d_model=cfg["d_model"], layers=cfg["layers"],
                     max_w=cfg["window"] + 1).to(device)
        half = (levels - 1) / 2; off = torch.ceil(half)
        S = I.shape[2]
    else:
        D = D2 * K * f
        nrm = torch.load(f"{args.run}/norm.pt")
        mu, sd = nrm["mu"].to(device), nrm["sd"].to(device)
        model = ChunkDenoiser(D, cfg["d_model"], cfg["layers"], n_frames=cfg["window"]).to(device)
        ab = cosine_alphas(cfg["T"]).to(device)
    sd_path = f"{args.run}/model_best.pt"
    model.load_state_dict(torch.load(sd_path if os.path.exists(sd_path) else f"{args.run}/model.pt",
                                     map_location=device))
    model.eval()

    pos_err = {h: [] for h in HORIZONS}; rot_err = {h: [] for h in HORIZONS}
    t_gen = 0.0; n_gen = 0
    reach = data["soma77"]["reach"]
    for s in picks:
        rows = torch.arange(s["start"], s["start"] + CTX + HOR, device=device)
        gt = data["soma77"]["pose"]  # decoded GT source is original pose data
        with torch.no_grad():
            if args.arm == "A":
                win = cfg["window"]
                seq_x = X_all[rows[:CTX]]                                  # [CTX,P]
                pred_f = []
                t0 = time.time()
                cur = seq_x
                ints_ctx = I[rows[:CTX]]                                   # for building f via dequant path
                for t in range(HOR):
                    inp = cur[-win:][None]                                 # [1,W,P]
                    logits = model(inp)[0, -1]                             # [P,L]
                    nxt_int = logits.argmax(-1)                            # [P]
                    nxt = nxt_int.reshape(D2, S, K, f)
                    # ints -> pre-round-equivalent features via from_ints value identity
                    fp = tok.fsqp.from_ints(nxt[0][:, None]) ; fr = (tok.fsqp if tok.single_codebook else tok.fsqr).from_ints(nxt[1][:, None])
                    # store ints for decode later; input rep for next step:
                    x_next = dequant(nxt_int.reshape(1, D2, S, K, f).cpu(), levels).to(device)[0]
                    cur = torch.cat([cur, x_next[None]], 0)
                    pred_f.append(nxt)
                t_gen += time.time() - t0; n_gen += HOR
                ints_pred = torch.stack(pred_f, 0)                         # [HOR,2,S,K,f]
                ip = ints_pred[:, 0].permute(1, 0, 2, 3)
                ir = ints_pred[:, 1].permute(1, 0, 2, 3)
                prot, ppos = None, None
                with torch.no_grad():
                    prot, ppos = tok.decode_from_ints(ip, ir, geom(data, "soma77", HOR))
            else:
                x = ((F[rows].reshape(CTX + HOR, -1) - mu) / sd)
                ctx = x[:CTX]
                out = []
                t0 = time.time()
                for c in range(HOR // CHUNK):
                    ctx_in = ctx[-(cfg["window"] - CHUNK):][None]
                    z = torch.randn(1, CHUNK, ctx.shape[-1], device=device)
                    ts = torch.linspace(cfg["T"], 1, args.ddim_steps, dtype=torch.long, device=device)
                    for i, t in enumerate(ts):
                        a = ab[t]
                        eps = model(ctx_in, z, t[None].float() / cfg["T"])
                        x0 = (z - (1 - a).sqrt() * eps) / a.sqrt()
                        a_prev = ab[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=device)
                        z = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
                    out.append(z[0]); ctx = torch.cat([ctx, z[0]], 0)
                t_gen += time.time() - t0; n_gen += HOR
                pred = (torch.cat(out, 0) * sd + mu).reshape(HOR, D2, K, f)
                prot, ppos = decode_frames(tok, data, pred, device)
            gtw = gt[rows[CTX:CTX + HOR]]
            pe = (ppos - gtw[..., 6:9]).norm(dim=-1).mean(-1) * reach       # [HOR] cm
            re = geodesic_deg(rot6d_to_matrix(prot), rot6d_to_matrix(gtw[..., :6])).mean(-1)
            for h in HORIZONS:
                pos_err[h].append(float(pe[h - 1])); rot_err[h].append(float(re[h - 1]))

    out = {"arm": args.arm, "run": args.run, "n_seq": len(picks),
           "ms_per_frame": round(t_gen / max(n_gen, 1) * 1000, 1)}
    for h in HORIZONS:
        out[f"h{h}"] = {"pos_cm": round(float(np.mean(pos_err[h])), 2),
                        "rot_deg": round(float(np.mean(rot_err[h])), 1)}
    print("ROLLOUT", json.dumps(out))
    json.dump(out, open(f"{args.run}/eval_rollout.json", "w"), indent=2)


if __name__ == "__main__":
    main()
