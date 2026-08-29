"""HOMECOMING + SEMIGROUP: two zero-training studies on frozen Equator.

    homecoming_probe.py

lloyd 0811: "curr -> ... -> curr is a drunkard's-return problem; what does
that unlock?"  Two experiments, no new parameters anywhere:

E-A HOMECOMING.  Bridge from a pose back to ITSELF in n steps.
    return_nll(n)      how costly is a loop of length n from this pose
    path_spread(n)     rendered-FK spread over 8 samples with identical
                       endpoints = how many ways this pose can leave and
                       come back = the pose's FREEDOM.  Low spread on a
                       committed pose (mid-fall, single support) is the
                       "danger map" signal — measured per action family.
    Control: the same starts bridged to a DIFFERENT pose (open loop), so
    spread differences are attributable to the return constraint.

E-B SEMIGROUP.  P(n=60) vs P(n=30) o P(n=30) through the same endpoints.
    If the transition kernel is self-consistent, one-hop and two-hop must
    agree; the gap is a free self-supervised training signal (no GT).
    Reference scale: two independent one-hop samples of the SAME bridge.
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
from bridge_a import CodeData, load_bridge_a                         # noqa: E402
from bridge_b import BridgeB                                         # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402

RUNS = V.RUNS
RM = "/var/tmp/alphamotion/robot_models"
X2 = f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
OUT = "/var/tmp/alphamotion/topo_exp/runs/HOMECOMING_0811.txt"
FAMS = ["walk", "run", "jump", "kick", "punch", "dance", "sit", "jacks"]


def sample_nll(mb, ep, nf, dev, temp=1.0, seed=0):
    """one-by-one confidence sampling; returns (tokens, mean chain NLL)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    tok = torch.zeros(1, 32, dtype=torch.long, device=dev)
    mask = torch.ones(1, 32, dtype=torch.bool, device=dev)
    nll = 0.0
    for _ in range(32):
        logits = mb(tok, mask, ep, nf)
        probs = F.softmax(logits[0] / temp, -1)
        conf = probs.max(-1).values.masked_fill(~mask[0], -1)
        pos = int(conf.argmax())
        pick = torch.multinomial(probs[pos], 1, generator=g)
        nll -= float(F.log_softmax(logits[0, pos], -1)[pick])
        tok[0, pos] = pick
        mask[0, pos] = False
    return tok[0], nll / 32.0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--stage_b", default="BRIDGE_B3_prod_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--n_start", type=int, default=12)
    ap.add_argument("--n_samp", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device

    ma, _ = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)
    bcfg = json.load(open(f"{RUNS}/{args.stage_b}/config.json"))
    mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
                                  map_location=dev)); mb.eval()
    sp, spcfg = E.load_run(f"{RUNS}/{args.spatial}", dev)
    g_t = None
    data = CodeData(args.spatial, use_root=True)
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    W = 5 ** torch.arange(6, device=dev)
    lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
    sspec = V.load_spec(os.environ["V5_AMASS"], "human_smpl")
    sdof = np.load(f"{os.environ['V5_AMASS']}/human_smpl_dof.npz")["dof"]
    g_h = E.target_geom(sspec, sdof, dev, sem_encoder=spcfg.get("sem_encoder"))

    def codes_at(r0, n):
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + n])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + n])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)

    def decode_fk(tok, ep, n):
        """tokens -> human-skeleton joint positions [n,J,3] cm (soft decode)."""
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        p = F.softmax(ma.decode(z, ep, n)[0], -1)
        q = (p * lv).sum(-1)
        zp = sp.fsqp.proj_out(q[:, :128])
        zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_h, q.shape[0]))
        return torch.as_tensor(V.fk_pos(rot.float().cpu(), sspec)).float()

    # ---------- E-A homecoming ----------
    rows, fams = [], []
    gg = np.random.default_rng(0)
    for fam in FAMS:
        try:
            s, _ = kd.sample_starts(args.n_start, gg, "val", span=130,
                                    families=[fam])
        except Exception:
            continue
        for r0 in s.tolist():
            rows.append(int(r0)); fams.append(fam)
    print(f"[E-A] {len(rows)} starts over {len(set(fams))} families", flush=True)

    NS = [20, 45, 90]
    res = {}
    for n in NS:
        for i, (r0, fam) in enumerate(zip(rows, fams)):
            c = codes_at(r0, 130)
            ep_home = ma.endpoints(torch.stack(
                [c[0], c[1], c[0], c[1]])[None].to(dev))          # back to self
            ep_open = ma.endpoints(torch.stack(
                [c[0], c[1], c[120], c[121]])[None].to(dev))      # control
            nf = torch.full((1,), float(n), device=dev)
            for tag, ep in (("home", ep_home), ("open", ep_open)):
                nlls, fks = [], []
                for s_ in range(args.n_samp):
                    tk, nl = sample_nll(mb, ep, nf, dev, seed=i * 100 + s_)
                    nlls.append(nl)
                    fks.append(decode_fk(tk, ep, n))
                fk = torch.stack(fks)                              # [S,n,J,3]
                spread = float((fk - fk.mean(0, keepdim=True)).norm(dim=-1)
                               .mean())
                res.setdefault((fam, n, tag), []).append(
                    (float(np.mean(nlls)), spread))
        print(f"[E-A] n={n} done", flush=True)

    lines = ["E-A HOMECOMING (frozen Equator; 8 samples per start)",
             f"{'family':8s} {'n':>4s} {'home_nll':>9s} {'open_nll':>9s} "
             f"{'home_spread':>12s} {'open_spread':>12s} {'ratio':>7s}"]
    for fam in FAMS:
        for n in NS:
            h = res.get((fam, n, "home")); o = res.get((fam, n, "open"))
            if not h:
                continue
            hn, hs = np.mean([x[0] for x in h]), np.mean([x[1] for x in h])
            on, os_ = np.mean([x[0] for x in o]), np.mean([x[1] for x in o])
            lines.append(f"{fam:8s} {n:4d} {hn:9.3f} {on:9.3f} "
                         f"{hs:12.2f} {os_:12.2f} {hs/max(os_,1e-6):7.2f}")

    # ---------- E-B semigroup ----------
    lines += ["", "E-B SEMIGROUP  P(60) vs P(30)oP(30), same endpoints",
              f"{'metric':28s} {'value_cm':>9s}"]
    one_two, one_one = [], []
    for i, r0 in enumerate(rows[:24]):
        c = codes_at(r0, 130)
        ep = ma.endpoints(torch.stack([c[0], c[1], c[60], c[61]])[None].to(dev))
        nf60 = torch.full((1,), 60.0, device=dev)
        t_a, _ = sample_nll(mb, ep, nf60, dev, seed=i)
        t_b, _ = sample_nll(mb, ep, nf60, dev, seed=1000 + i)
        fk_a, fk_b = decode_fk(t_a, ep, 60), decode_fk(t_b, ep, 60)
        one_one.append(float((fk_a - fk_b).norm(dim=-1).mean()))
        # two-hop: bridge to the midpoint of hop-1's own output, then onward
        mid_q = None
        z = ma.fsq.from_ints(((t_a[:, None] // W) % 5)[None].float())[None]
        p = F.softmax(ma.decode(z, ep, 60)[0], -1)
        q60 = (p * lv).sum(-1)
        mid_c = (q60[29:31] * 3.5 + 4.0).round().clamp(0, 8).long().cpu()
        nf30 = torch.full((1,), 30.0, device=dev)
        ep1 = ma.endpoints(torch.stack([c[0], c[1], mid_c[0], mid_c[1]])[None].to(dev))
        ep2 = ma.endpoints(torch.stack([mid_c[0], mid_c[1], c[60], c[61]])[None].to(dev))
        t1, _ = sample_nll(mb, ep1, nf30, dev, seed=i)
        t2, _ = sample_nll(mb, ep2, nf30, dev, seed=i)
        fk2 = torch.cat([decode_fk(t1, ep1, 30), decode_fk(t2, ep2, 30)], 0)
        one_two.append(float((fk_a - fk2).norm(dim=-1).mean()))
    lines.append(f"{'one-hop vs one-hop (ref)':28s} {np.mean(one_one):9.2f}")
    lines.append(f"{'one-hop vs two-hop (gap)':28s} {np.mean(one_two):9.2f}")
    lines.append(f"{'consistency ratio':28s} "
                 f"{np.mean(one_two)/max(np.mean(one_one),1e-6):9.2f}")
    body = "\n".join(lines)
    print("\n" + body)
    open(OUT, "w").write(body + "\n")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
