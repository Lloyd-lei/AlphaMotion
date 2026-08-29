"""BRIDGE STACK eval v2 — CLEAN RULER (cross-skeleton) + decode-strategy columns.

    bridge_stack_eval2.py --stage_a BRIDGE_A2_full_0810 --stage_b <winner>

F1: v1's ruler decoded human->human with no observation = C1's vis_residual OOD
path; every cm was inflated. v2 decodes to G1 (heldout body, obs=None is the
NORMAL path) and scores against G1's real paired retarget FK — the product ruler.
E3: argmax vs soft-expectation decode, side by side.
Sections mirror v1 so rows are comparable: same-endpoint gen / cross-clip bridge /
sparse pins.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V                                                # noqa: E402
import eval_newbody as E                                             # noqa: E402
from bridge_a import BridgeA, CodeData, load_bridge_a, CODES, FULL, load_bridge_a                  # noqa: E402
from bridge_b import BridgeB, tokens_of                              # noqa: E402
from bridge_demo import sample_tokens                                # noqa: E402
from train_global import build_global                                # noqa: E402

RUNS = V.RUNS
GMR = "/var/tmp/alphamotion/config_harvest/GMR/assets"


def stats(v):
    v = np.asarray(v, np.float64)
    return f"mean {v.mean():6.2f}  p95 {np.percentile(v,95):6.2f}"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A2_full_0810")
    ap.add_argument("--stage_b", required=True)
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, 60

    ma, acfg = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)
    use_root = bool(acfg.get("use_root", False))
    _bcfg = json.load(open(f"{RUNS}/{args.stage_b}/config.json"))
    mb = BridgeB(ep_dim=_bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
                                  map_location=dev)); mb.eval()
    sp, spcfg = E.load_run(f"{RUNS}/{args.spatial}", dev)
    enc = spcfg.get("sem_encoder", "siglip")
    data = CodeData(args.spatial, use_root=use_root)
    l6h = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)
    # ---- G1 clean ruler ----
    G1 = "unitree_g1_29dof"
    spec_g = V.load_spec(FULL, G1)
    gdof = np.load(f"{FULL}/{G1}_dof.npz")["dof"]
    gg = E.target_geom(spec_g, gdof, dev, sem_encoder=enc)
    l6g = torch.load(f"{FULL}/{G1}_paired_local6d.pt", mmap=True)
    names_g = [n.lower() for n in spec_g.joint_names]
    anc_g = [i for i, n in enumerate(names_g)
             if any(k in n for k in ("wrist", "ankle_roll", "head"))]
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    W = 5 ** torch.arange(6, device=dev)
    grid = torch.linspace(-1, 1, 5, device=dev) * 1.0    # fsq 5-level grid values
    # model.py FSQ 5 levels: half=2, values = (round(tanh*2))/2 in {-1,-.5,0,.5,1}

    def sample_win(g):
        pool = [(a, b) for a, b in data.pools["val"] if b - a >= T]
        a0, b0 = pool[int(g.integers(len(pool)))]
        s0 = int(g.integers(a0, b0 - T + 1))
        cp = torch.from_numpy(np.ascontiguousarray(data.c[s0:s0 + T])).long()
        if use_root:
            pk = torch.from_numpy(np.ascontiguousarray(data.cr[s0:s0 + T])).long()
            r = torch.empty_like(cp)
            r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
            cp = torch.cat([cp, r], 1)
        return cp[None].to(dev), np.arange(s0, s0 + T)

    def decode_codes(logits, ep_c, mode):
        """A logits -> (zp, zr) for the spatial layer. argmax or soft-expectation."""
        S = 128
        if mode == "argmax":
            pred = logits.argmax(-1)[0]
            pred[0] = ep_c[0, 0]; pred[-1] = ep_c[0, -1]
            q = (pred.float() - 4.0) / 3.5
        else:
            p = F.softmax(logits[0], -1)
            lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
            q = (p * lv).sum(-1)
            q[0] = (ep_c[0, 0].float() - 4.0) / 3.5
            q[-1] = (ep_c[0, -1].float() - 4.0) / 3.5
        zp = sp.fsqp.proj_out(q[:, :S])
        zr = sp.fsqr.proj_out(q[:, S:]) if use_root else None
        return zp, zr

    def fk_g1(zp, zr):
        rot, _ = sp.decode(zp, zr, E.geom_batch(gg, zp.shape[0]))
        return V.fk_pos(rot.cpu(), spec_g)

    def gen(ep_c, seed, mode):
        ep = ma.endpoints(ep_c)
        nf = torch.full((1,), float(T), device=dev)
        tok = sample_tokens(mb, ep, nf, dev, temp=args.temp, seed=seed)
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        return fk_g1(*decode_codes(ma.decode(z, ep, T), ep_c, mode))

    def vel(p):
        return np.linalg.norm(np.diff(p, axis=0), axis=-1).mean(-1)

    def jerk(p):
        return np.linalg.norm(p[3:] - 3 * p[2:-1] + 3 * p[1:-2] - p[:-3],
                              axis=-1).mean()

    L = ["=" * 96, "BRIDGE STACK v2 — CLEAN RULER: decode->G1(heldout), score vs "
         "real paired retarget FK",
         f"A={args.stage_a}  B={args.stage_b}  temp={args.temp}", "=" * 96]

    # -------- section 1: same-endpoint generation, both decode modes --------
    for mode in ("argmax", "soft"):
        g = np.random.default_rng(5)
        ep_hit, anc, en, jk, div = [], [], [], [], []
        for wi in range(24):
            c, rows = sample_win(g)
            gr6g, _, _ = build_global(l6g[torch.as_tensor(rows)], spec_g, dev)
            gt = V.fk_pos(gr6g, spec_g)
            fks = [gen(c, 100 * wi + sd, mode) for sd in range(3)]
            for fk in fks[:2]:
                ep_hit.append(float(np.linalg.norm(
                    fk[[0, -1]][:, anc_g] - gt[[0, -1]][:, anc_g], axis=-1).mean()))
                anc.append(float(np.linalg.norm(
                    fk[:, anc_g] - gt[:, anc_g], axis=-1).mean()))
                en.append(float(vel(fk[:, anc_g]).mean() /
                                (vel(gt[:, anc_g]).mean() + 1e-9)))
                jk.append(float(jerk(fk[:, anc_g]) / (jerk(gt[:, anc_g]) + 1e-9)))
            pw = [float(np.linalg.norm(fks[i][:, anc_g] - fks[j][:, anc_g],
                                       axis=-1).mean())
                  for i in range(3) for j in range(i + 1, 3)]
            div.append(float(np.mean(pw)))
        L += ["", f"-- 1 SAME-ENDPOINT GEN [{mode}] (24 win x 3 samples) --",
              f"endpoint hit [cm]   : {stats(ep_hit)}  (v1 dirty ruler: 11.90)",
              f"anchor FK vs GT [cm]: {stats(anc)}  (v1: 11.94)",
              f"energy ratio        : {stats(en)}  (v1: 1.83)",
              f"jerk ratio          : {stats(jk)}  (v1: 18.68)",
              f"diversity [cm]      : {stats(div)}  (v1: 0.00)"]

    # -------- section 2: cross-clip bridge, soft decode --------
    PAIRS = [("walk", "kick"), ("run", "jacks"), ("dance", "walk"),
             ("punch", "run")]
    snap_i, snap_o, en_b = [], [], []
    ga = np.random.default_rng(31)
    for i in range(24):
        fa, fb = PAIRS[i % 4]
        try:
            sa, _ = kd.sample_starts(1, ga, "val", span=T, families=[fa])
            sb, _ = kd.sample_starts(1, ga, "val", span=T, families=[fb])
        except Exception:
            continue
        ra = np.arange(int(sa[0]), int(sa[0]) + T)
        rb = np.arange(int(sb[0]), int(sb[0]) + T)
        def codes_at(rr):
            cp = torch.from_numpy(np.ascontiguousarray(
                data.c[rr[0]:rr[0] + T])).long()
            if use_root:
                pk = torch.from_numpy(np.ascontiguousarray(
                    data.cr[rr[0]:rr[0] + T])).long()
                r = torch.empty_like(cp)
                r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
                cp = torch.cat([cp, r], 1)
            return cp
        ca, cb = codes_at(ra), codes_at(rb)
        h = T // 2
        ep_c = torch.stack([ca[h - 1], ca[h], cb[h], cb[h + 1]])[None].to(dev)
        fk = gen(ep_c, i, "soft")[:, anc_g]
        gA, _, _ = build_global(l6g[torch.as_tensor(ra)], spec_g, dev)
        gB, _, _ = build_global(l6g[torch.as_tensor(rb)], spec_g, dev)
        pA = V.fk_pos(gA, spec_g)[:h, anc_g]
        pB = V.fk_pos(gB, spec_g)[h:, anc_g]
        ref = float(np.median(np.concatenate([vel(pA), vel(pB)])) + 1e-9)
        snap_i.append(float(np.linalg.norm(fk[0] - pA[-1], axis=-1).mean()) / ref)
        snap_o.append(float(np.linalg.norm(pB[0] - fk[-1], axis=-1).mean()) / ref)
        en_b.append(float(vel(fk).mean() /
                          (np.concatenate([vel(pA), vel(pB)]).mean() + 1e-9)))
    L += ["", "-- 2 CROSS-CLIP BRIDGE [soft] (gap 60, 24 bridges) --",
          f"junction snap in    : {stats(snap_i)}  (v1: 25.51 | T32 3.79 jepa 1.94)",
          f"junction snap out   : {stats(snap_o)}  (v1: 38.11 | T32 3.75 jepa 3.09)",
          f"bridge energy ratio : {stats(en_b)}  (v1: 1.19, old arms 0.28-0.31)"]

    # -------- section 3: sparse pins --------
    gs = np.random.default_rng(7)
    L += ["", "-- 3 SPARSE PINS [soft] --"]
    for k in (0, 8, 16, 24):
        accs = []
        for wi in range(16):
            c, rows = sample_win(gs)
            gt_tok, _ = tokens_of(ma, c)
            keep = torch.zeros(32, dtype=torch.bool)
            if k:
                keep[torch.from_numpy(gs.choice(32, k, replace=False))] = True
            mask = ~keep[None].to(dev)
            ep = ma.endpoints(c)
            nf = torch.full((1,), float(T), device=dev)
            tk = gt_tok.clone(); mk = mask.clone()
            gtor = torch.Generator(device=dev).manual_seed(wi)
            for _ in range(int(mk.sum())):
                lo = mb(tk, mk, ep, nf)
                pr = F.softmax(lo[0] / args.temp, -1)
                conf = pr.max(-1).values.masked_fill(~mk[0], -1)
                pos = int(conf.argmax())
                tk[0, pos] = torch.multinomial(pr[pos], 1, generator=gtor)
                mk[0, pos] = False
            accs.append((tk[0][mask[0]] == gt_tok[0][mask[0]]).float().mean().item())
        L.append(f"k={k:>2} pins: masked-token acc {np.mean(accs):.3f}"
                 f"  (v1 was flat ~0.08 = pins ignored)")

    txt = "\n".join(L)
    print(txt, flush=True)
    open(f"{RUNS}/BRIDGE_STACK_v2_{args.stage_b}.txt", "w").write(txt + "\n")
    print(f"verdict: {RUNS}/BRIDGE_STACK_v2_{args.stage_b}.txt", flush=True)


if __name__ == "__main__":
    main()
