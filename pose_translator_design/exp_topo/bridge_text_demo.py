"""TEXT -> MOTION demo: same start pose, goal DROPPED, only the prompt differs.

    bridge_text_demo.py --cfg 3.0

Pipeline: prompt --frozen Qwen--> TextCodec(8 FSQ ids) --> B4 one-by-one
sampling with classifier-free guidance (text vs null logits) --> A3 soft
decode --> Greenwich C1 --> X2 render.  No GT exists for generation; the
on-screen banner says so.  All towers frozen; only B4 (a B3 copy) knows text.
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
from bridge_a import CodeData, load_bridge_a, FULL                   # noqa: E402
from bridge_b4 import BridgeB4                                       # noqa: E402
from bridge_b import V5                                              # noqa: E402
from align_text import TextCodec                                     # noqa: E402
from semantics import _encode_qwen3                                  # noqa: E402
from spatial5 import pad_band, font                                  # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from PIL import Image, ImageDraw                                     # noqa: E402

RUNS = V.RUNS
TA = "/var/tmp/alphamotion/topo_exp/text_align"
RM = "/var/tmp/alphamotion/robot_models"
X2 = f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ", "mp4", "bridge_demo")

PROMPTS = ["a person walks forward",
           "a person punches with their right arm",
           "a person waves with their hand",
           "a person does jumping jacks"]


def sample_cfg(mb, ep, nf, text, dev, w=3.0, temp=0.9, seed=0):
    """one-by-one confidence-ordered sampling, goal dropped, CFG on text."""
    g = torch.Generator(device=dev).manual_seed(seed)
    gd = torch.ones(1, dtype=torch.bool, device=dev)
    tok = torch.zeros(1, 32, dtype=torch.long, device=dev)
    mask = torch.ones(1, 32, dtype=torch.bool, device=dev)
    for _ in range(32):
        lt = mb(tok, mask, ep, nf, goal_drop=gd, text=text)
        ln = mb(tok, mask, ep, nf, goal_drop=gd, text=None)
        logits = ln + w * (lt - ln)
        probs = F.softmax(logits[0] / temp, -1)
        conf = probs.max(-1).values.masked_fill(~mask[0], -1)
        pos = int(conf.argmax())
        tok[0, pos] = torch.multinomial(probs[pos], 1, generator=g)
        mask[0, pos] = False
    return tok[0]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--stage_b", default="BRIDGE_B4v2_gdrop_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--text_run", default="TEXT_ALIGN_v2_0810")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--T", type=int, default=90)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, args.T
    os.makedirs(OUT, exist_ok=True)

    ma, _ = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)
    mb = BridgeB4(ep_dim=ma.d).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
                                  map_location=dev)); mb.eval()
    tc = TextCodec().to(dev)
    tc.load_state_dict(torch.load(f"{RUNS}/{args.text_run}/model.pt",
                                  map_location=dev)["text"]); tc.eval()
    sp, spcfg = E.load_run(f"{RUNS}/{args.spatial}", dev)
    enc = spcfg.get("sem_encoder")
    data = CodeData(args.spatial, use_root=True)
    spec_t, dof_t, rest_t, qn_t, xml_t = E.get_target(X2, E.CACHE_AMASS)
    g_t = E.target_geom(spec_t, dof_t, dev, sem_encoder=enc)
    dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
    rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)
    W = 5 ** torch.arange(6, device=dev)

    # prompts -> 8 text FSQ ids each
    pe = _encode_qwen3(PROMPTS, dev, batch=8).float().to(dev)
    ints = tc.codes(pe)                                   # [P,8,6]
    tids = (ints * W[None, None, :]).sum(-1).long()       # [P,8]

    # one shared neutral start: first frames of a val walk clip
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    g = np.random.default_rng(args.seed)
    s0, _ = kd.sample_starts(1, g, "val", span=4, families=["walk"])
    r0 = int(s0[0])
    cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + 2])).long()
    pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + 2])).long()
    rr = torch.empty_like(cp)
    rr[..., 0::2] = pk & 0x0F; rr[..., 1::2] = (pk >> 4) & 0x0F
    c01 = torch.cat([cp, rr], 1)                          # [2,256,20]
    ep_c = torch.stack([c01[0], c01[1], c01[1], c01[1]])[None].to(dev)
    ep = ma.endpoints(ep_c)
    nf = torch.full((1,), float(T), device=dev)

    def decode_q(tok):
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        logits = ma.decode(z, ep, T)
        p = F.softmax(logits[0], -1)
        lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
        q = (p * lv).sum(-1)
        q[0] = (ep_c[0, 1].float() - 4.0) / 3.5           # pin entry only
        return q                                          # exit free: no goal

    def render_q(q):
        zp = sp.fsqp.proj_out(q[:, :128])
        zr = sp.fsqr.proj_out(q[:, 128:])
        rot, _ = sp.decode(zp, zr, E.geom_batch(g_t, q.shape[0]))
        Rg = rot6d_to_matrix(rot).double()
        _, _, qj = MP.project(Rg, spec_t, dof_d, rest=rest_d, method="global")
        gR, gp = MP.fk_from_angles(qj, spec_t, dof_d, rest=rest_d,
                                   root_R=Rg[:, 0])
        return GC.render_tile(xml_t, spec_t, qn_t, qj.cpu().numpy(),
                              gR[:, 0].cpu().numpy(), gp.cpu().numpy(),
                              qj.shape[0], status="nodataset")

    tiles = []
    for i, pr in enumerate(PROMPTS):
        print(f"[gen] {pr!r} cfg={args.cfg}", flush=True)
        tok = sample_cfg(mb, ep, nf, tids[i:i + 1], dev, w=args.cfg,
                         temp=0.9, seed=args.seed * 100 + i)
        tiles.append(render_q(decode_q(tok)))

    frames = []
    for t in range(T):
        row = np.concatenate([pad_band(ti[min(t, len(ti) - 1)])
                              for ti in tiles], 1)
        im = Image.fromarray(row).convert("RGB")
        dr = ImageDraw.Draw(im)
        for k, pr in enumerate(PROMPTS):
            dr.text((k * GC.TILE_W + 8, 6), f'"{pr}"', font=font(13),
                    fill=(255, 240, 180))
        dr.text((8, GC.TILE_H + 24),
                f"same start pose | goal DROPPED | n={T} | CFG w={args.cfg} | "
                "pure generation: no GT exists", font=font(12, False),
                fill=(210, 210, 215))
        frames.append(np.asarray(im))
    p = f"{OUT}/text2motion_4prompts_cfg{args.cfg:g}_x2.mp4"
    RN.write_mp4(p, np.stack(frames), 20)
    print("->", p, flush=True)


if __name__ == "__main__":
    main()
