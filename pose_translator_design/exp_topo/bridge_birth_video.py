"""THE BIRTH VIDEOS — A2 + B2(maskgit) + Greenwich, composed properly this time.

    bridge_birth_video.py

v1  full sequence: REAL walk (30f, GT codes) -> BRIDGE (60f, generated, red band)
    -> REAL kick (30f, GT codes), decoded to X2 in ONE stream. 6 seconds.
v2  diversity: same endpoints + n=60, three samples side by side on X2, with the
    GT human as SOURCE reference.
Soft-expectation decode (E3 winner: jerk 2.6 vs argmax 6.6). Root stream is A2's
own prediction — no borrowing.
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A2_full_0810")
    ap.add_argument("--stage_b", default="BRIDGE_B2_maskgit_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, 60
    os.makedirs(OUT, exist_ok=True)

    ma, acfg = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)
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
        return torch.cat([cp, r], 1)                       # [n,256,20]

    def gen_soft_q(ep_c, seed):
        ep = ma.endpoints(ep_c)
        nf = torch.full((1,), float(T), device=dev)
        tok = sample_tokens(mb, ep, nf, dev, temp=args.temp, seed=seed)
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        logits = ma.decode(z, ep, T)
        p = F.softmax(logits[0], -1)
        lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
        q = (p * lv).sum(-1)                               # [T,256,20]
        q[0] = (ep_c[0, 0].float() - 4.0) / 3.5
        q[-1] = (ep_c[0, -1].float() - 4.0) / 3.5
        return q

    def render_q(q):
        zp = sp.fsqp.proj_out(q[:, :128])
        zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_t, q.shape[0]))
        Rg = rot6d_to_matrix(rot).double()
        _, _, qj = MP.project(Rg, spec_t, dof_d, rest=rest_d, method="global")
        gR, gp = MP.fk_from_angles(qj, spec_t, dof_d, rest=rest_d, root_R=Rg[:, 0])
        return GC.render_tile(xml_t, spec_t, qn_t, qj.cpu().numpy(),
                              gR[:, 0].cpu().numpy(), gp.cpu().numpy(),
                              qj.shape[0], status="nodataset")

    # ================= video 1: walk -> BRIDGE -> kick, one stream =============
    g = np.random.default_rng(31)
    sa, _ = kd.sample_starts(1, g, "val", span=T, families=["walk"])
    sb, _ = kd.sample_starts(1, g, "val", span=T, families=["kick"])
    ra0, rb0 = int(sa[0]), int(sb[0])
    LA = 30
    ca = codes_at(ra0, T); cb = codes_at(rb0, T)
    # endpoints: last walk frame (with its predecessor) -> first kick frame
    ep_c = torch.stack([ca[LA - 1], ca[LA], cb[LA], cb[LA + 1]])[None].to(dev)
    q_bridge = gen_soft_q(ep_c, seed=7)
    q_walk = (ca[:LA + 1].float().to(dev) - 4.0) / 3.5
    q_kick = (cb[LA:LA + 31].float().to(dev) - 4.0) / 3.5
    q_full = torch.cat([q_walk, q_bridge[1:-1], q_kick], 0)   # ~119 frames
    fr = render_q(q_full)
    n_walk, n_br = LA + 1, T - 2
    # human hard-cut reference
    gra, _, _ = build_global(l6h[torch.as_tensor(np.arange(ra0, ra0 + LA + 1))],
                             sspec, dev)
    grb, _, _ = build_global(l6h[torch.as_tensor(np.arange(rb0 + LA, rb0 + LA + 31))],
                             sspec, dev)
    hp = np.concatenate([V.fk_pos(gra, sspec),
                         np.repeat(V.fk_pos(gra, sspec)[-1:], n_br, 0),
                         V.fk_pos(grb, sspec)], 0)
    src_t = capsule_tile(hp, sspec.parents, "SOURCE: real walk | (frozen: no GT) | real kick")
    frames = []
    for t, f_ in enumerate(fr):
        row = np.concatenate([src_t[min(t, len(src_t) - 1)], pad_band(f_)], 1)
        im = Image.fromarray(row).convert("RGB")
        dr = ImageDraw.Draw(im)
        phase = ("REAL WALK" if t < n_walk else
                 "BRIDGE (invented)" if t < n_walk + n_br else "REAL KICK")
        col = (120, 200, 255) if phase == "REAL WALK" else \
              (255, 90, 90) if phase.startswith("BRIDGE") else (120, 255, 160)
        dr.rectangle([GC.TILE_W, 0, 2 * GC.TILE_W, 8], fill=col)
        dr.text((GC.TILE_W + 8, 12), f"{phase}  f{t}", font=font(15), fill=col)
        frames.append(np.asarray(im))
    p1 = f"{OUT}/birth1_walk_BRIDGE_kick_x2.mp4"
    RN.write_mp4(p1, np.stack(frames), 20)
    print("->", p1, flush=True)

    # ================= video 2: diversity, three samples =======================
    g2 = np.random.default_rng(21)
    pool = [(a, b) for a, b in data.pools["val"] if b - a >= T]
    a0, b0 = pool[int(g2.integers(len(pool)))]
    s0 = int(g2.integers(a0, b0 - T + 1))
    c = codes_at(s0, T)[None].to(dev)
    gr6, _, _ = build_global(l6h[torch.as_tensor(np.arange(s0, s0 + T))], sspec, dev)
    src2 = capsule_tile(V.fk_pos(gr6, sspec), sspec.parents, "SOURCE GT (human)")
    tiles = [src2]
    for sd in (11, 12, 13):
        tiles.append(render_q(gen_soft_q(c, seed=sd)))
    frames = []
    for t in range(T):
        row = np.concatenate(
            [tiles[0][min(t, len(tiles[0]) - 1)]] +
            [pad_band(ti[min(t, len(ti) - 1)]) for ti in tiles[1:]], 1)
        im = Image.fromarray(row).convert("RGB")
        dr = ImageDraw.Draw(im)
        for k, lab in enumerate(["", "sample 1", "sample 2", "sample 3"]):
            if lab:
                dr.text((k * GC.TILE_W + 8, 6), lab, font=font(15),
                        fill=(255, 240, 180))
        dr.text((8, GC.TILE_H + 24),
                "same endpoints + n=60 - three DIFFERENT legal paths",
                font=font(12, False), fill=(210, 210, 215))
        frames.append(np.asarray(im))
    p2 = f"{OUT}/birth2_diversity3_x2.mp4"
    RN.write_mp4(p2, np.stack(frames), 20)
    print("->", p2, flush=True)


if __name__ == "__main__":
    main()
