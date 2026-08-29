"""EQUATOR native capabilities — three demos that cost zero new training.

    bridge_native_demos.py --demo loop|edit|score

loop   pin start frame == goal frame (with matching in/out velocities lifted from
       the frame's real neighbors) -> a motion that closes on itself. Rendered
       3x consecutively; the seam is exact by pass-through.
edit   take a REAL clip, keep 24 of its 32 time-local tokens, let B resample the
       middle 8 (frames ~22-37) conditioned on the rest -> local motion edit.
       Panels: SOURCE GT | A-recon (original tokens) | edit#1 | edit#2.
score  B's NLL as a motion-naturalness judge: real val windows vs frame-shuffled
       vs hard-cut spliced vs time-reversed. Table -> runs/EQUATOR_NLL_JUDGE.txt.

Any-frame pass-through: pin_frames() overwrites arbitrary rows of the decoded
code grid with known codes (the endpoint clamp generalized).
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import v5_common as V
import eval_newbody as E
import manifold_project as MP
import render_newbody as RN
import greenwich_carnival as GC
from bridge_a import BridgeA, CodeData, load_bridge_a, FULL
from bridge_b import BridgeB, tokens_of
from bridge_knob_sweep import sample_topp
from spatial5 import capsule_tile, pad_band, font
from train_global import build_global
from common import rot6d_to_matrix
from PIL import Image, ImageDraw

RUNS = V.RUNS
RM = "/var/tmp/alphamotion/robot_models"
X2 = f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ", "mp4", "bridge_demo")
TAU = 2.0                                        # knob-sweep winner


def pin_frames(q, pins):
    """generalized pass-through: overwrite decoded grid rows with KNOWN codes.
    pins = [(t, codes[256,20] long)]"""
    for t, c in pins:
        q[t] = (c.float().to(q.device) - 4.0) / 3.5
    return q


class Stack:
    def __init__(self, dev):
        self.dev = dev
        self.ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
        bcfg = json.load(open(f"{RUNS}/BRIDGE_B3_prod_0810/config.json"))
        self.mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
        self.mb.load_state_dict(torch.load(f"{RUNS}/BRIDGE_B3_prod_0810/model.pt",
                                           map_location=dev)); self.mb.eval()
        self.sp, spcfg = E.load_run(f"{RUNS}/PROD_c1_91k_dim20_tail_0810", dev)
        enc = spcfg.get("sem_encoder")
        self.data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
        self.l6 = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)
        self.sspec = V.load_spec(FULL, "human_smpl")
        st = E.get_target(X2, E.CACHE_AMASS)
        self.spec_t, dof_t, rest_t, self.qn_t, self.xml_t = st
        self.g_t = E.target_geom(self.spec_t, dof_t, dev, sem_encoder=enc)
        self.dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
        self.rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)
        self.W = 5 ** torch.arange(6, device=dev)
        self.lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5

    def codes_at(self, r0, n):
        cp = torch.from_numpy(np.ascontiguousarray(self.data.c[r0:r0 + n])).long()
        pk = torch.from_numpy(np.ascontiguousarray(self.data.cr[r0:r0 + n])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)

    def decode_tokens(self, tok, ep, T):
        z = self.ma.fsq.from_ints(((tok[:, None] // self.W) % 5)[None].float())[None]
        logits = self.ma.decode(z, ep, T)
        p = F.softmax(logits[0], -1)
        return (p * self.lv).sum(-1)

    def render_q(self, q):
        zp = self.sp.fsqp.proj_out(q[:, :128])
        zr = self.sp.fsqr.proj_out(q[:, 128:])
        rot, _ = self.sp.decode(zp, zr, E.geom_batch(self.g_t, q.shape[0]))
        Rg = rot6d_to_matrix(rot).double()
        _, _, qj = MP.project(Rg, self.spec_t, self.dof_d, rest=self.rest_d,
                              method="global")
        gR, gp = MP.fk_from_angles(qj, self.spec_t, self.dof_d, rest=self.rest_d,
                                   root_R=Rg[:, 0])
        return GC.render_tile(self.xml_t, self.spec_t, self.qn_t,
                              qj.cpu().numpy(), gR[:, 0].cpu().numpy(),
                              gp.cpu().numpy(), qj.shape[0], status="nodataset")


@torch.no_grad()
def demo_loop(S):
    dev = S.dev
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    g = np.random.default_rng(17)
    sa, _ = kd.sample_starts(1, g, "val", span=64, families=["walk"])
    r0 = int(sa[0]) + 20
    F_, Fp, Fm = S.codes_at(r0, 1)[0], S.codes_at(r0 + 1, 1)[0], \
        S.codes_at(r0 - 1, 1)[0]
    # loop closure: start q=F with outgoing v = F+ - F ; end q=F with incoming
    # v = F - F-  -> velocity continuous across the seam
    ep_c = torch.stack([F_, Fp, Fm, F_])[None].to(dev)
    ep = S.ma.endpoints(ep_c)
    n = 80
    nf = torch.full((1,), float(n), device=dev)
    tok = sample_topp(S.mb, ep, nf, dev, TAU, 1.0, seed=9)
    q = S.decode_tokens(tok, ep, n)
    q = pin_frames(q, [(0, F_), (n - 1, F_)])
    fr = S.render_q(q)
    fr3 = fr[:-1] * 3 + [fr[-1]]                       # play 3x, drop dup seams
    l6r = S.l6[torch.as_tensor(np.arange(r0, r0 + 1))]
    gr6, _, _ = build_global(l6r, S.sspec, dev)
    src = capsule_tile(np.repeat(V.fk_pos(gr6, S.sspec), len(fr3), 0),
                       S.sspec.parents, "PINNED frame (start == goal)")
    frames = []
    per = n - 1
    for t, f_ in enumerate(fr3):
        row = np.concatenate([src[min(t, len(src) - 1)], pad_band(f_)], 1)
        im = Image.fromarray(row).convert("RGB")
        dr = ImageDraw.Draw(im)
        lap = t // per + 1
        dr.text((GC.TILE_W + 8, 6), f"PERFECT LOOP  lap {lap}/3  f{t % per}",
                font=font(15), fill=(140, 255, 200))
        dr.text((8, GC.TILE_H + 24),
                "start pin == goal pin, velocities matched - seam is EXACT "
                "(pass-through)", font=font(11, False), fill=(210, 210, 215))
        frames.append(np.asarray(im))
    p = f"{OUT}/native_loop3x_x2.mp4"
    RN.write_mp4(p, np.stack(frames), 20)
    print("->", p, flush=True)


@torch.no_grad()
def demo_edit(S):
    dev = S.dev
    g = np.random.default_rng(23)
    pool = [(a, b) for a, b in S.data.pools["val"] if b - a >= 60]
    a0, b0 = pool[int(g.integers(len(pool)))]
    r0 = int(g.integers(a0, b0 - 60 + 1))
    c = S.codes_at(r0, 60)[None].to(dev)
    gt_tok, ep = tokens_of(S.ma, c)
    EDIT = list(range(12, 20))                         # tokens 12..19 ~= f22-37
    nf = torch.full((1,), 60.0, device=dev)
    seg = np.round(np.linspace(0, 60, 33)).astype(int)
    f_lo, f_hi = seg[EDIT[0]], seg[EDIT[-1] + 1]

    def resample(seed):
        tk = gt_tok.clone()
        mk = torch.zeros_like(tk, dtype=torch.bool)
        mk[0, EDIT] = True
        gtor = torch.Generator(device=dev).manual_seed(seed)
        for _ in range(len(EDIT)):
            lo = S.mb(tk, mk, ep, nf)
            pr = F.softmax(lo[0] / TAU, -1)
            conf = pr.max(-1).values.masked_fill(~mk[0], -1)
            pos = int(conf.argmax())
            tk[0, pos] = torch.multinomial(pr[pos], 1, generator=gtor)
            mk[0, pos] = False
        return tk[0]

    variants = [("A-recon (original tokens)", gt_tok[0]),
                ("edit #1 (mid 8 tokens resampled)", resample(1)),
                ("edit #2", resample(2))]
    tiles = []
    for _, tk in variants:
        q = S.decode_tokens(tk, ep, 60)
        q = pin_frames(q, [(0, c[0, 0].cpu()), (59, c[0, -1].cpu())])
        tiles.append(S.render_q(q))
    l6r = S.l6[torch.as_tensor(np.arange(r0, r0 + 60))]
    gr6, _, _ = build_global(l6r, S.sspec, dev)
    src = capsule_tile(V.fk_pos(gr6, S.sspec), S.sspec.parents, "SOURCE GT")
    frames = []
    for t in range(60):
        row = [src[min(t, len(src) - 1)]] + \
              [pad_band(ti[min(t, len(ti) - 1)]) for ti in tiles]
        im = Image.fromarray(np.concatenate(row, 1)).convert("RGB")
        dr = ImageDraw.Draw(im)
        inzone = f_lo <= t < f_hi
        for k, (lab, _) in enumerate(variants):
            col = (255, 90, 90) if (inzone and k > 0) else (255, 240, 180)
            dr.text(((k + 1) * GC.TILE_W + 8, 6), lab, font=font(13), fill=col)
        if inzone:
            dr.rectangle([2 * GC.TILE_W, 0, 4 * GC.TILE_W, 6], fill=(255, 90, 90))
        dr.text((8, GC.TILE_H + 24),
                f"time-local tokens 12-19 (frames {f_lo}-{f_hi}) resampled; "
                "rest untouched", font=font(11, False), fill=(210, 210, 215))
        frames.append(np.asarray(im))
    p = f"{OUT}/native_localedit_x2.mp4"
    RN.write_mp4(p, np.stack(frames), 20)
    print("->", p, flush=True)


@torch.no_grad()
def demo_score(S):
    dev = S.dev
    g = np.random.default_rng(11)
    pool = [(a, b) for a, b in S.data.pools["val"] if b - a >= 60]
    nf = torch.full((1,), 60.0, device=dev)

    def nll_of(codes):
        c = codes[None].to(dev)
        tok, ep = tokens_of(S.ma, c)
        lo = S.mb(tok, torch.ones_like(tok, dtype=torch.bool), ep, nf)
        return float(F.cross_entropy(lo[0], tok[0]).item())

    cats = {k: [] for k in ("real", "reversed", "spliced", "shuffled")}
    for i in range(32):
        a0, b0 = pool[int(g.integers(len(pool)))]
        r0 = int(g.integers(a0, b0 - 60 + 1))
        c = S.codes_at(r0, 60)
        cats["real"].append(nll_of(c))
        cats["reversed"].append(nll_of(torch.flip(c, [0])))
        a1, b1 = pool[int(g.integers(len(pool)))]
        r1 = int(g.integers(a1, b1 - 60 + 1))
        c2 = S.codes_at(r1, 60)
        cats["spliced"].append(nll_of(torch.cat([c[:30], c2[30:]], 0)))
        perm = torch.from_numpy(g.permutation(60))
        cats["shuffled"].append(nll_of(c[perm]))
    L = ["=" * 72, "EQUATOR AS JUDGE — B's NLL on token streams (lower = more "
         "natural)", "=" * 72]
    for k, v in cats.items():
        L.append(f"{k:>9}: NLL {np.mean(v):6.3f}  ppl {np.exp(np.mean(v)):8.1f}")
    L.append("expected order: real < reversed < spliced < shuffled")
    txt = "\n".join(L)
    print(txt, flush=True)
    open(f"{RUNS}/EQUATOR_NLL_JUDGE.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", required=True, choices=("loop", "edit", "score"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    S = Stack(a.device)
    {"loop": demo_loop, "edit": demo_edit, "score": demo_score}[a.demo](S)
