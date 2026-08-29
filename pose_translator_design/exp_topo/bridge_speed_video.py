"""EQUATOR speed control — the n-knob replaces the old speed-token transplant.

    bridge_speed_video.py

Same (start, goal) pair, THREE frame budgets n = 30 / 60 / 120: the model plans
the same journey at three urgencies. Panels play at the same fps, so the n=30
panel finishes in 1.5s and holds; n=120 takes 6s. SOURCE = GT human (its own
natural pacing, 60f).
"""
from __future__ import annotations
import os, sys, json
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
from bridge_b import BridgeB
from bridge_demo import sample_tokens
from spatial5 import capsule_tile, pad_band, font
from train_global import build_global
from common import rot6d_to_matrix
from PIL import Image, ImageDraw

RUNS = V.RUNS
RM = "/var/tmp/alphamotion/robot_models"
X2 = f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ", "mp4", "bridge_demo")
NS = (30, 60, 120)


@torch.no_grad()
def main():
    dev = "cuda"
    ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
    bcfg = json.load(open(f"{RUNS}/BRIDGE_B3_prod_0810/config.json"))
    mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/BRIDGE_B3_prod_0810/model.pt",
                                  map_location=dev)); mb.eval()
    sp, spcfg = E.load_run(f"{RUNS}/PROD_c1_91k_dim20_tail_0810", dev)
    enc = spcfg.get("sem_encoder")
    data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
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

    g = np.random.default_rng(41)
    sa, _ = kd.sample_starts(1, g, "val", span=64, families=["kick"])
    r0 = int(sa[0])
    c = codes_at(r0, 62)[None].to(dev)
    ep_c = torch.stack([c[0, 0], c[0, 1], c[0, -2], c[0, -1]])[None].to(dev)
    ep = ma.endpoints(ep_c)

    def gen_n(n, seed=3):
        nf = torch.full((1,), float(n), device=dev)
        tok = sample_tokens(mb, ep, nf, dev, seed=seed)
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        logits = ma.decode(z, ep, n)
        p = F.softmax(logits[0], -1)
        lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
        q = (p * lv).sum(-1)
        q[0] = (ep_c[0, 1].float() - 4.0) / 3.5
        q[-1] = (ep_c[0, 2].float() - 4.0) / 3.5
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

    tiles = []
    for n in NS:
        print(f"[speed] n={n}", flush=True)
        tiles.append(render_q(gen_n(n)))
    gr6, _, _ = build_global(l6h[torch.as_tensor(np.arange(r0, r0 + 62))],
                             sspec, dev)
    src = capsule_tile(V.fk_pos(gr6, sspec), sspec.parents,
                       "SOURCE GT (natural pace, 62f)")
    Tm = 126
    frames = []
    for t in range(Tm):
        row = [src[min(t, len(src) - 1)]]
        for n, ti in zip(NS, tiles):
            row.append(pad_band(ti[min(t, len(ti) - 1)]))
        im = Image.fromarray(np.concatenate(row, 1)).convert("RGB")
        dr = ImageDraw.Draw(im)
        for k, n in enumerate(NS):
            done = " DONE" if t >= n else ""
            dr.text(((k + 1) * GC.TILE_W + 8, 6),
                    f"n={n} ({n/20:.1f}s){done}", font=font(15),
                    fill=(255, 240, 180) if not done else (140, 255, 140))
        dr.text((8, GC.TILE_H + 24),
                "SAME endpoints, three frame budgets - speed control IS the n knob",
                font=font(12, False), fill=(210, 210, 215))
        frames.append(np.asarray(im))
    p = f"{OUT}/equator_speed_n30_60_120_x2.mp4"
    RN.write_mp4(p, np.stack(frames), 20)
    print("->", p, flush=True)


if __name__ == "__main__":
    main()
