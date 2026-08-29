"""HELL MODE — lloyd's ceiling test for bridge attention (0810).

    bridge_hell.py

  [dance, 3s REAL] --5s GAP(150f, n>train-max 120, extrapolated)--
  [punch, REAL but CUT mid-motion] --3s GAP(90f)-- [single target frame: jacks end]

SOURCE panel: GT capsules for real parts, BLANK for gaps, static hold for the
single-frame target. Robot panel: ONE continuous decoded stream on X2.
Honest limits printed on screen: n=150 exceeds the training range (fourier n-cond
clamps at 120); the final pin is a single frame so target velocity = 0.
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
import v5_common as V                                                # noqa: E402
import eval_newbody as E                                             # noqa: E402
import manifold_project as MP                                        # noqa: E402
import render_newbody as RN                                          # noqa: E402
import greenwich_carnival as GC                                      # noqa: E402
from bridge_a import BridgeA, CodeData, load_bridge_a, FULL, load_bridge_a                         # noqa: E402
from bridge_b import BridgeB                                         # noqa: E402
from bridge_demo import sample_tokens                                # noqa: E402
from spatial5 import capsule_tile, pad_band, font                    # noqa: E402
from train_global import build_global                                # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from PIL import Image, ImageDraw                                     # noqa: E402

RUNS = V.RUNS
RM = "/var/tmp/alphamotion/robot_models"
X2 = f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ", "mp4", "bridge_demo")

SEG1, GAP1, SEG2, GAP2, HOLD = 60, 150, 45, 90, 24


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A2_full_0810")
    ap.add_argument("--stage_b", default="BRIDGE_B2_maskgit_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(OUT, exist_ok=True)

    ma, _acfg = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)
    _bcfg = json.load(open(f"{RUNS}/{args.stage_b}/config.json"))
    mb = BridgeB(ep_dim=_bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
                                  map_location=dev)); mb.eval()
    sp, spcfg = E.load_run(f"{RUNS}/{args.spatial}", dev)
    enc = spcfg.get("sem_encoder")
    data = CodeData(args.spatial, use_root=True)
    l6h = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)
    sspec = V.load_spec(FULL, "human_smpl")
    spec_t, dof_t, rest_t, qn_t, xml_t = E.get_target(X2, E.CACHE_AMASS)
    g_t = E.target_geom(spec_t, dof_t, dev, sem_encoder=enc)
    dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
    rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)
    W = 5 ** torch.arange(6, device=dev)
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")

    def codes_at(r0, n):
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + n])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + n])).long()
        r = torch.empty_like(cp)
        r[..., 0::2] = pk & 0x0F; r[..., 1::2] = (pk >> 4) & 0x0F
        return torch.cat([cp, r], 1)

    def bridge_q(f_prev, f_cur, f_tgt, f_tgt2, n, seed):
        """(q,v) both sides -> soft-decoded interior q [n,256,20]."""
        ep_c = torch.stack([f_prev, f_cur, f_tgt, f_tgt2])[None].to(dev)
        ep = ma.endpoints(ep_c)
        nf = torch.full((1,), float(n), device=dev)
        tok = sample_tokens(mb, ep, nf, dev, seed=seed)
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        logits = ma.decode(z, ep, n)
        p = F.softmax(logits[0], -1)
        lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
        q = (p * lv).sum(-1)
        q[0] = (ep_c[0, 1].float() - 4.0) / 3.5     # entry = current frame
        q[-1] = (ep_c[0, 2].float() - 4.0) / 3.5    # exit = target frame
        return q

    # ---- pick three wildly different clips ----
    g = np.random.default_rng(args.seed)
    s1, _ = kd.sample_starts(1, g, "val", span=SEG1 + 2, families=["dance"])
    s2, _ = kd.sample_starts(1, g, "val", span=SEG2 * 2, families=["punch"])
    s3, _ = kd.sample_starts(1, g, "val", span=60, families=["jacks"])
    r1, r2, r3 = int(s1[0]), int(s2[0]), int(s3[0])
    c1 = codes_at(r1, SEG1)
    c2 = codes_at(r2, SEG2 + 1)                     # +1 for velocity at the cut
    c3 = codes_at(r3 + 58, 2)                       # jacks: its LAST two frames
    tgt_frame = c3[1]

    print("[hell] bridging 5s dance->punch ...", flush=True)
    qb1 = bridge_q(c1[-2], c1[-1], c2[0], c2[1], GAP1, args.seed * 10 + 1)
    print("[hell] bridging 3s punch(cut)->single jacks frame ...", flush=True)
    qb2 = bridge_q(c2[SEG2 - 1], c2[SEG2], tgt_frame, tgt_frame, GAP2,
                   args.seed * 10 + 2)              # single-frame pin: v_target=0

    to_q = lambda c: (c.float().to(dev) - 4.0) / 3.5
    q_full = torch.cat([to_q(c1), qb1[1:-1], to_q(c2[:SEG2 + 1]), qb2[1:-1],
                        to_q(tgt_frame[None]).expand(HOLD, -1, -1)], 0)
    print(f"[hell] total {q_full.shape[0]} frames; decoding to X2 ...", flush=True)
    zp = sp.fsqp.proj_out(q_full[:, :128])
    zr = sp.fsqr.proj_out(q_full[:, 128:])
    rot, _ = sp.decode(zp, zr, E.geom_batch(g_t, q_full.shape[0]))
    Rg = rot6d_to_matrix(rot).double()
    _, _, qj = MP.project(Rg, spec_t, dof_d, rest=rest_d, method="global")
    gR, gp = MP.fk_from_angles(qj, spec_t, dof_d, rest=rest_d, root_R=Rg[:, 0])
    fr = GC.render_tile(xml_t, spec_t, qn_t, qj.cpu().numpy(),
                        gR[:, 0].cpu().numpy(), gp.cpu().numpy(),
                        qj.shape[0], status="nodataset")

    # ---- SOURCE panel: GT capsules / blanks / held target frame ----
    g1r, _, _ = build_global(l6h[torch.as_tensor(np.arange(r1, r1 + SEG1))],
                             sspec, dev)
    g2r, _, _ = build_global(l6h[torch.as_tensor(np.arange(r2, r2 + SEG2 + 1))],
                             sspec, dev)
    g3r, _, _ = build_global(l6h[torch.as_tensor(np.arange(r3 + 59, r3 + 60))],
                             sspec, dev)
    cap1 = capsule_tile(V.fk_pos(g1r, sspec), sspec.parents, "SOURCE")
    cap2 = capsule_tile(V.fk_pos(g2r, sspec), sspec.parents, "SOURCE")
    cap3 = capsule_tile(np.repeat(V.fk_pos(g3r, sspec), HOLD, 0), sspec.parents,
                        "SOURCE")
    blank = np.full_like(cap1[0], 18)
    src = (cap1 + [blank] * (GAP1 - 2) + cap2 + [blank] * (GAP2 - 2) + cap3)

    phases = ([("REAL DANCE", (120, 200, 255))] * SEG1
              + [("5s GAP - NO GT (n=150 > train max 120)", (255, 90, 90))] * (GAP1 - 2)
              + [("REAL PUNCH (cut mid-motion!)", (120, 255, 160))] * (SEG2 + 1)
              + [("3s GAP -> SINGLE-FRAME PIN (v=0)", (255, 90, 90))] * (GAP2 - 2)
              + [("TARGET FRAME (jacks end), HELD", (255, 220, 120))] * HOLD)
    frames = []
    for t, f_ in enumerate(fr):
        srcf = src[min(t, len(src) - 1)]
        row = np.concatenate([srcf, pad_band(f_)], 1)
        im = Image.fromarray(row).convert("RGB")
        dr = ImageDraw.Draw(im)
        ph, col = phases[min(t, len(phases) - 1)]
        dr.rectangle([GC.TILE_W, 0, 2 * GC.TILE_W, 8], fill=col)
        dr.text((GC.TILE_W + 8, 12), f"{ph}  f{t}", font=font(14), fill=col)
        frames.append(np.asarray(im))
    p = f"{OUT}/hell_dance5s_punchcut3s_jacks1f_x2.mp4"
    RN.write_mp4(p, np.stack(frames), 20)
    print("->", p, flush=True)


if __name__ == "__main__":
    main()
