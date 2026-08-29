"""FIRST end-to-end BRIDGE generation: B samples tokens -> A renders codes ->
Greenwich puts it on a robot. lloyd 0810: "我等不及了".

    bridge_demo.py

Video 1  same-clip endpoints (val), n=60: SOURCE GT | sample#1 | sample#2 on X2.
         Two different samples between the SAME endpoints = the multimodality
         thesis on screen.
Video 2  cross-clip bridge: end of a walk -> start of a kick, n=60, no GT exists.
         SOURCE shows the hard cut; the robot panels show two invented bridges.

Sampling: one-by-one confidence-ordered unmasking (exact chain rule, never
argmax-all): 32 steps, temperature 0.9.
Root stream: v1 Stage-A models pose only; root codes are borrowed from the source
clip (video 1: its own; video 2: clip A's) — noted on screen.
"""
from __future__ import annotations
import os, sys, argparse
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
from bridge_a import BridgeA, CodeData, CODES, FULL                  # noqa: E402
from bridge_b import BridgeB, V5                                     # noqa: E402
from spatial5 import capsule_tile, pad_band, font                    # noqa: E402
from train_global import build_global                                # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from PIL import Image, ImageDraw                                     # noqa: E402

RUNS = V.RUNS
GMR = "/var/tmp/alphamotion/config_harvest/GMR/assets"
RM = "/var/tmp/alphamotion/robot_models"
X2 = f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ", "mp4", "bridge_demo")


@torch.no_grad()
def sample_tokens(mb, ep, n_frames, dev, temp=0.9, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    tok = torch.zeros(1, 32, dtype=torch.long, device=dev)
    mask = torch.ones(1, 32, dtype=torch.bool, device=dev)
    for _ in range(32):
        logits = mb(tok, mask, ep, n_frames)
        probs = F.softmax(logits[0] / temp, -1)
        conf = probs.max(-1).values.masked_fill(~mask[0], -1)
        pos = int(conf.argmax())
        tok[0, pos] = torch.multinomial(probs[pos], 1, generator=g)
        mask[0, pos] = False
    return tok[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A_pilot_0810")
    ap.add_argument("--stage_b", default="BRIDGE_B_pilot_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, args.T
    os.makedirs(OUT, exist_ok=True)

    ma = BridgeA(dims=20).to(dev)
    ma.load_state_dict(torch.load(f"{RUNS}/{args.stage_a}/model.pt",
                                  map_location=dev))
    ma.eval()
    mb = BridgeB().to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
                                  map_location=dev))
    mb.eval()
    sp, spcfg = E.load_run(f"{RUNS}/{args.spatial}", dev)
    enc = spcfg.get("sem_encoder", "siglip")
    sspec = V.load_spec(FULL, "human_smpl")
    sdof = np.load(f"{FULL}/human_smpl_dof.npz")["dof"]
    sg = E.target_geom(sspec, sdof, dev, sem_encoder=enc)
    data = CodeData(args.spatial, use_root=True)
    cr = np.load(f"{CODES}/{args.spatial}/codes_root_packed.npy", mmap_mode="r")
    l6 = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)
    spec_t, dof_t, rest_t, qn_t, xml_t = E.get_target(X2, E.CACHE_AMASS)
    g_t = E.target_geom(spec_t, dof_t, dev, sem_encoder=enc)
    dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
    rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)
    W = 5 ** torch.arange(6, device=dev)

    def unpack_root(rows):
        pk = torch.from_numpy(np.ascontiguousarray(cr[rows])).long()
        r = torch.empty(pk.shape[0], 128, 20, dtype=torch.long)
        r[..., 0::2] = pk & 0x0F
        r[..., 1::2] = (pk >> 4) & 0x0F
        return r

    def render_codes(pred_pose, root_rows):
        """code ints [T,128,20] + root rows -> X2 stream frames."""
        zp = sp.fsqp.from_ints(pred_pose[None].float().to(dev))
        zr = sp.fsqr.from_ints(unpack_root(root_rows)[None].float().to(dev))
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_t, pred_pose.shape[0]))
        Rg = rot6d_to_matrix(rot).double()
        _, _, q = MP.project(Rg, spec_t, dof_d, rest=rest_d, method="global")
        gR, gp = MP.fk_from_angles(q, spec_t, dof_d, rest=rest_d, root_R=Rg[:, 0])
        return GC.render_tile(xml_t, spec_t, qn_t, q.cpu().numpy(),
                              gR[:, 0].cpu().numpy(), gp.cpu().numpy(),
                              q.shape[0], status="nodataset")

    def gen(ep_c, root_rows, seed):
        ep = ma.endpoints(ep_c)
        nf = torch.full((1,), float(T), device=dev)
        tok = sample_tokens(mb, ep, nf, dev, seed=seed)
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        logits = ma.decode(z, ep, T)
        pred = logits.argmax(-1)[0]
        pred[0] = ep_c[0, 0]; pred[-1] = ep_c[0, -1]      # endpoint pass-through
        return render_codes(pred.cpu(), root_rows)

    def compose(path, tiles, labels, header):
        Tm = max(len(t) for t in tiles)
        frames = []
        for t in range(Tm):
            row = np.concatenate(
                [ti[min(t, len(ti) - 1)] if ti[0].shape[0] > GC.TILE_H
                 else pad_band(ti[min(t, len(ti) - 1)]) for ti in tiles], 1)
            im = Image.fromarray(row).convert("RGB")
            dr = ImageDraw.Draw(im)
            for k, lab in enumerate(labels):
                dr.text((k * GC.TILE_W + 8, 4), lab, font=font(14),
                        fill=(255, 240, 180))
            dr.text((8, GC.TILE_H + 24), header, font=font(12, False),
                    fill=(200, 200, 205))
            frames.append(np.asarray(im))
        RN.write_mp4(path, np.stack(frames), 20)
        print("->", path, flush=True)

    # ---------------- video 1: same-clip endpoints, two samples ----------------
    g = np.random.default_rng(21)
    pool = [(a, b) for a, b in data.pools["val"] if b - a >= T]
    a0, b0 = pool[int(g.integers(len(pool)))]
    s0 = int(g.integers(a0, b0 - T + 1))
    rows = np.arange(s0, s0 + T)
    c = torch.from_numpy(np.ascontiguousarray(data.c[s0:s0 + T])).long()[None].to(dev)
    gr6, _, _ = build_global(l6[torch.as_tensor(rows)], sspec, dev)
    src_t = capsule_tile(V.fk_pos(gr6, sspec), sspec.parents, "SOURCE GT (human)")
    t1 = gen(c, rows, seed=1)
    t2 = gen(c, rows, seed=2)
    compose(f"{OUT}/gen1_same_endpoints_x2.mp4", [src_t, t1, t2],
            ["", "sample #1 (B->A->Greenwich)", "sample #2 (same endpoints)"],
            "same endpoints + n=60, two sampled paths | X2 zero-shot body")

    # ---------------- video 2: cross-clip bridge, no GT ------------------------
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    ga = np.random.default_rng(31)
    sa, _ = kd.sample_starts(1, ga, "val", span=T, families=["walk"])
    sb, _ = kd.sample_starts(1, ga, "val", span=T, families=["kick"])
    ra = np.arange(int(sa[0]), int(sa[0]) + T)
    rb = np.arange(int(sb[0]), int(sb[0]) + T)
    ca = torch.from_numpy(np.ascontiguousarray(data.c[ra[0]:ra[0] + T])).long()
    cb = torch.from_numpy(np.ascontiguousarray(data.c[rb[0]:rb[0] + T])).long()
    ep_c = torch.stack([ca[T // 2 - 1], ca[T // 2], cb[T // 2], cb[T // 2 + 1]]
                       )[None].to(dev)                     # (q,v) frames both sides
    # hard-cut reference on the human
    gra, _, _ = build_global(l6[torch.as_tensor(ra)], sspec, dev)
    grb, _, _ = build_global(l6[torch.as_tensor(rb)], sspec, dev)
    cut = np.concatenate([V.fk_pos(gra, sspec)[T // 2 - T // 4:T // 2],
                          V.fk_pos(grb, sspec)[T // 2:T // 2 + T // 4]], 0)
    src_t = capsule_tile(cut, sspec.parents, "SOURCE hard cut walk->kick")
    b1 = gen(ep_c, ra, seed=3)                            # root borrowed from walk
    b2 = gen(ep_c, ra, seed=4)
    compose(f"{OUT}/gen2_bridge_walk_kick_x2.mp4", [src_t, b1, b2],
            ["", "bridge sample #1", "bridge sample #2"],
            "cross-clip bridge (walk end -> kick start), NO GT exists; root "
            "borrowed from walk (v1 limitation)")


if __name__ == "__main__":
    main()
