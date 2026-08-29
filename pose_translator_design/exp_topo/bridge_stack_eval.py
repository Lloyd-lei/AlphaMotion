"""BRIDGE STACK full numeric eval — B samples, A renders, measured properly.

    bridge_stack_eval.py

Sections (all on val windows, seed-fixed; FK = rigid-bone, root-relative, so the
v1 borrowed-root limitation does not contaminate pose metrics):

  1 SAME-ENDPOINT GENERATION (n=60, 48 windows x 4 samples)
      endpoint hit [cm]        must be ~0 (pass-through, verified not assumed)
      anchor FK vs GT [cm]     one-of-many-paths caveat, still informative
      energy ratio vs GT       mean-regression check (T32 punch was 0.917)
      jerk ratio vs GT         over-smoothing check (T32/V9 were ~4x smoother)
      diversity [cm]           mean pairwise anchor-FK distance across 4 samples
                               (0 = mode collapse; GT-vs-GT of different windows
                               shown for scale)
      token NLL / ppl          proper scoring of B on GT tokens (bridge mode)
  2 CROSS-CLIP BRIDGE (walk->kick / run->jacks / dance->walk / punch->run,
                       gap 60, 32 bridges)
      junction snap in/out     |v| jump at both joins over local median |v| —
                               THE number: T32 was 3.79/3.75, jepa 1.94/3.09
      bridge energy ratio      vs surrounding real segments (old arms 0.28-0.31)
      copy-rate inside bridge
  3 SPARSE PIN MODE (give k of 32 GT tokens + endpoints, k=0/8/16/24)
      masked-token acc vs k    does B actually condition on pins?
      anchor FK vs GT vs k     path pulled toward GT as pins increase?

Output: runs/BRIDGE_STACK_verdict_0810.txt
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402
import eval_newbody as E                                             # noqa: E402
from bridge_a import BridgeA, CodeData, CODES, FULL                  # noqa: E402
from bridge_b import BridgeB, V5                                     # noqa: E402
from bridge_demo import sample_tokens                                # noqa: E402
from train_global import build_global                                # noqa: E402

RUNS = V.RUNS


def stats(v):
    v = np.asarray(v, np.float64)
    return f"mean {v.mean():6.2f}  p95 {np.percentile(v,95):6.2f}"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A_pilot_0810")
    ap.add_argument("--stage_b", default="BRIDGE_B_pilot_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, 60

    ma = BridgeA(dims=20).to(dev)
    ma.load_state_dict(torch.load(f"{RUNS}/{args.stage_a}/model.pt",
                                  map_location=dev)); ma.eval()
    mb = BridgeB().to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
                                  map_location=dev)); mb.eval()
    data = CodeData(args.spatial)
    sspec = V.load_spec(FULL, "human_smpl")
    l6 = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)
    kidx, _, _ = V.key_joints(sspec, verbose=False)
    sp, spcfg = E.load_run(f"{RUNS}/{args.spatial}", dev)
    W = 5 ** torch.arange(6, device=dev)
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")

    def decode_tokens(tok, ep_c):
        """B tokens + endpoint frames -> predicted pose codes [T,128,20]."""
        ep = ma.endpoints(ep_c)
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        pred = ma.decode(z, ep, T).argmax(-1)[0]
        pred[0] = ep_c[0, 0]; pred[-1] = ep_c[0, -1]
        return pred

    def fk_of_codes(pred):
        """codes -> HUMAN-side FK positions (root-relative rigid): decode codes
        back on the human skeleton through the frozen spatial layer."""
        sdof = np.load(f"{FULL}/human_smpl_dof.npz")["dof"]
        if not hasattr(fk_of_codes, "sg"):
            fk_of_codes.sg = E.target_geom(sspec, sdof, dev,
                                           sem_encoder=spcfg.get("sem_encoder"))
        zp = sp.fsqp.from_ints(pred[None].float().to(dev))
        # root codes: neutral zeros are wrong; use the GT root of the window at
        # call site via closure attr set by caller
        zr = sp.fsqr.from_ints(fk_of_codes.root[None].float().to(dev))
        rot, _ = sp.decode(zp, zr, E.geom_batch(fk_of_codes.sg, pred.shape[0]))
        return V.fk_pos(rot.cpu(), sspec)

    cr = np.load(f"{CODES}/{args.spatial}/codes_root_packed.npy", mmap_mode="r")

    def root_ints(rows):
        pk = torch.from_numpy(np.ascontiguousarray(cr[rows])).long()
        r = torch.empty(pk.shape[0], 128, 20, dtype=torch.long)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return r

    def vel(p):
        return np.linalg.norm(np.diff(p, axis=0), axis=-1).mean(-1)

    L = ["=" * 96, "BRIDGE STACK verdict — B(sample)->A(render)->FK, human side",
         f"A={args.stage_a}  B={args.stage_b}  board={args.spatial}", "=" * 96]

    # ---------------- section 1 ----------------
    g = np.random.default_rng(5)
    pool = [(a, b) for a, b in data.pools["val"] if b - a >= T]
    ep_hit, anc_err, en_r, jk_r, div, nlls = [], [], [], [], [], []
    for wi in range(48):
        a0, b0 = pool[int(g.integers(len(pool)))]
        s0 = int(g.integers(a0, b0 - T + 1))
        rows = np.arange(s0, s0 + T)
        c = torch.from_numpy(np.ascontiguousarray(data.c[s0:s0 + T])
                             ).long()[None].to(dev)
        fk_of_codes.root = root_ints(rows)
        gr6, _, _ = build_global(l6[torch.as_tensor(rows)], sspec, dev)
        gt_fk = V.fk_pos(gr6, sspec)
        gt_tok, _ = __import__("bridge_b").tokens_of(ma, c)
        # B NLL of GT tokens (bridge mode)
        ep = ma.endpoints(c)
        nf = torch.full((1,), float(T), device=dev)
        lo = mb(gt_tok, torch.ones_like(gt_tok, dtype=torch.bool), ep, nf)
        nlls.append(F.cross_entropy(lo[0], gt_tok[0]).item())
        fks = []
        for sd in range(4):
            tok = sample_tokens(mb, ep, nf, dev, seed=100 * wi + sd)
            pred = decode_tokens(tok, c)
            fk = fk_of_codes(pred)
            fks.append(fk)
        for fk in fks[:2]:
            ep_hit.append(float(np.linalg.norm(
                fk[[0, -1]][:, kidx] - gt_fk[[0, -1]][:, kidx], axis=-1).mean()))
            anc_err.append(float(np.linalg.norm(
                fk[:, kidx] - gt_fk[:, kidx], axis=-1).mean()))
            en_r.append(float(vel(fk[:, kidx]).mean() /
                              (vel(gt_fk[:, kidx]).mean() + 1e-9)))
            j = lambda p: np.linalg.norm(
                p[3:] - 3 * p[2:-1] + 3 * p[1:-2] - p[:-3], axis=-1).mean()
            jk_r.append(float(j(fk[:, kidx]) / (j(gt_fk[:, kidx]) + 1e-9)))
        pw = [float(np.linalg.norm(fks[i][:, kidx] - fks[jx][:, kidx],
                                   axis=-1).mean())
              for i in range(4) for jx in range(i + 1, 4)]
        div.append(float(np.mean(pw)))
    L += ["", "-- 1 SAME-ENDPOINT GENERATION (48 win x 4 samples) --",
          f"endpoint hit [cm]     : {stats(ep_hit)}   (pass-through check)",
          f"anchor FK vs GT [cm]  : {stats(anc_err)}  (one-of-many caveat)",
          f"energy ratio vs GT    : {stats(en_r)}   (mush<1; T32 recon 0.94)",
          f"jerk  ratio vs GT     : {stats(jk_r)}   (oversmooth<1)",
          f"sample diversity [cm] : {stats(div)}   (0=mode collapse)",
          f"B token ppl on GT     : {np.exp(np.mean(nlls)):.1f}  "
          f"(uniform 15625, train-final 96.5)"]

    # ---------------- section 2 ----------------
    PAIRS = [("walk", "kick"), ("run", "jacks"), ("dance", "walk"),
             ("punch", "run")]
    snap_in, snap_out, en_b, copy_b = [], [], [], []
    ga = np.random.default_rng(31)
    for i in range(32):
        fa, fb = PAIRS[i % 4]
        try:
            sa, _ = kd.sample_starts(1, ga, "val", span=T, families=[fa])
            sb, _ = kd.sample_starts(1, ga, "val", span=T, families=[fb])
        except Exception:
            continue
        ra = np.arange(int(sa[0]), int(sa[0]) + T)
        rb = np.arange(int(sb[0]), int(sb[0]) + T)
        ca = torch.from_numpy(np.ascontiguousarray(data.c[ra[0]:ra[0] + T])).long()
        cb = torch.from_numpy(np.ascontiguousarray(data.c[rb[0]:rb[0] + T])).long()
        h = T // 2
        ep_c = torch.stack([ca[h - 1], ca[h], cb[h], cb[h + 1]])[None].to(dev)
        fk_of_codes.root = root_ints(ra)
        ep = ma.endpoints(ep_c)
        nf = torch.full((1,), float(T), device=dev)
        tok = sample_tokens(mb, ep, nf, dev, seed=i)
        pred = decode_tokens(tok, ep_c)
        fk = fk_of_codes(pred)[:, kidx]
        gra, _, _ = build_global(l6[torch.as_tensor(ra)], sspec, dev)
        grb, _, _ = build_global(l6[torch.as_tensor(rb)], sspec, dev)
        pA = V.fk_pos(gra, sspec)[:h, kidx]
        pB = V.fk_pos(grb, sspec)[h:, kidx]
        ref = float(np.median(np.concatenate([vel(pA), vel(pB)])) + 1e-9)
        snap_in.append(float(np.linalg.norm(fk[0] - pA[-1], axis=-1).mean()) / ref)
        snap_out.append(float(np.linalg.norm(pB[0] - fk[-1], axis=-1).mean()) / ref)
        en_b.append(float(vel(fk).mean() /
                          (np.concatenate([vel(pA), vel(pB)]).mean() + 1e-9)))
        same_prev = (pred[1:-1] == pred[0:-2]).float().mean().item()
        copy_b.append(same_prev)
    L += ["", "-- 2 CROSS-CLIP BRIDGE (gap 60 = 2s invented, 32 bridges) --",
          f"junction snap in      : {stats(snap_in)}  (T32 3.79 / jepa 1.94)",
          f"junction snap out     : {stats(snap_out)}  (T32 3.75 / jepa 3.09)",
          f"bridge energy ratio   : {stats(en_b)}   (old arms 0.28-0.31 @2s)",
          f"frame copy inside     : {stats(copy_b)}   (fraction identical codes)"]

    # ---------------- section 3 ----------------
    gs = np.random.default_rng(7)
    L += ["", "-- 3 SPARSE PIN MODE: k GT tokens given --"]
    for k in (0, 8, 16, 24):
        accs, errs = [], []
        for wi in range(24):
            a0, b0 = pool[int(gs.integers(len(pool)))]
            s0 = int(gs.integers(a0, b0 - T + 1))
            rows = np.arange(s0, s0 + T)
            c = torch.from_numpy(np.ascontiguousarray(data.c[s0:s0 + T])
                                 ).long()[None].to(dev)
            fk_of_codes.root = root_ints(rows)
            gt_tok, _ = __import__("bridge_b").tokens_of(ma, c)
            gr6, _, _ = build_global(l6[torch.as_tensor(rows)], sspec, dev)
            gt_fk = V.fk_pos(gr6, sspec)
            keep = torch.zeros(32, dtype=torch.bool)
            keep[torch.from_numpy(gs.choice(32, k, replace=False))] = True if k else False
            tok = gt_tok.clone()
            mask = ~keep[None].to(dev)
            ep = ma.endpoints(c)
            nf = torch.full((1,), float(T), device=dev)
            # one-by-one fill of the masked rest
            tk = tok.clone(); mk = mask.clone()
            gtor = torch.Generator(device=dev).manual_seed(wi)
            for _ in range(int(mk.sum())):
                lo = mb(tk, mk, ep, nf)
                pr = F.softmax(lo[0] / 0.9, -1)
                conf = pr.max(-1).values.masked_fill(~mk[0], -1)
                pos = int(conf.argmax())
                tk[0, pos] = torch.multinomial(pr[pos], 1, generator=gtor)
                mk[0, pos] = False
            accs.append((tk[0][mask[0]] == gt_tok[0][mask[0]]).float().mean().item())
            pred = decode_tokens(tk[0], c)
            fk = fk_of_codes(pred)
            errs.append(float(np.linalg.norm(fk[:, kidx] - gt_fk[:, kidx],
                                             axis=-1).mean()))
        L.append(f"k={k:>2} pins: masked-token acc {np.mean(accs):.3f}  "
                 f"anchor FK vs GT {np.mean(errs):6.2f} cm")

    txt = "\n".join(L)
    print(txt, flush=True)
    open(f"{RUNS}/BRIDGE_STACK_verdict_0810.txt", "w").write(txt + "\n")
    print(f"verdict: {RUNS}/BRIDGE_STACK_verdict_0810.txt", flush=True)


if __name__ == "__main__":
    main()
