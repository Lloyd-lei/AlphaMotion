"""TEXT -> GOAL(retrieval) -> BRIDGE demo: validated parts only, no new training.

    bridge_text_goal_demo.py

prompt --frozen Qwen + TextCodec--> shared space --retrieve--> best-matching
train window --> its mid frames become the GOAL --> Equator B3 (unmodified!)
bridges from a shared neutral start --> A3 soft decode --> Greenwich C1 --> X2.

Text steers by CHOOSING the destination; the bridge does the walking.  Every
component here has already passed its own acceptance.  Matched caption is
printed on each tile so the retrieval step is inspectable.
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
from bridge_a import CodeData, load_bridge_a                         # noqa: E402
from bridge_b import BridgeB                                         # noqa: E402
from bridge_demo import sample_tokens                                # noqa: E402
from align_text import TextCodec, MotionEnc, motion_tokens           # noqa: E402
from bridge_b4 import CaptionedWindows                               # noqa: E402
from semantics import _encode_qwen3                                  # noqa: E402
from spatial5 import pad_band, font                                  # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from PIL import Image, ImageDraw                                     # noqa: E402

RUNS = V.RUNS
RM = "/var/tmp/alphamotion/robot_models"
X2 = f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ", "mp4", "bridge_demo")

PROMPTS = ["a person walks forward",
           "a person punches with their right arm",
           "a person waves with their hand",
           "a person does jumping jacks"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage_a", default="BRIDGE_A3_prod_0810")
    ap.add_argument("--stage_b", default="BRIDGE_B3_prod_0810")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--text_run", default="TEXT_ALIGN_v2_0810")
    ap.add_argument("--T", type=int, default=90)
    ap.add_argument("--n_gallery", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, args.T
    os.makedirs(OUT, exist_ok=True)

    ma, _ = load_bridge_a(f"{RUNS}/{args.stage_a}", dev)
    bcfg = json.load(open(f"{RUNS}/{args.stage_b}/config.json"))
    mb = BridgeB(ep_dim=bcfg.get("ep_dim")).to(dev)
    mb.load_state_dict(torch.load(f"{RUNS}/{args.stage_b}/model.pt",
                                  map_location=dev)); mb.eval()
    ck = torch.load(f"{RUNS}/{args.text_run}/model.pt", map_location=dev)
    tc = TextCodec().to(dev); tc.load_state_dict(ck["text"]); tc.eval()
    me = MotionEnc().to(dev); me.load_state_dict(ck["motion"]); me.eval()
    sp, spcfg = E.load_run(f"{RUNS}/{args.spatial}", dev)
    enc = spcfg.get("sem_encoder")
    data = CodeData(args.spatial, use_root=True)
    spec_t, dof_t, rest_t, qn_t, xml_t = E.get_target(X2, E.CACHE_AMASS)
    g_t = E.target_geom(spec_t, dof_t, dev, sem_encoder=enc)
    dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
    rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)
    W = 5 ** torch.arange(6, device=dev)

    # ---- gallery: one 60f window per captioned TRAIN clip ----
    cap = CaptionedWindows("train", args.spatial, True)
    g = np.random.default_rng(args.seed + 3)
    byclip = {}
    for p in cap.items:
        if p["len"] >= 62:
            byclip.setdefault(p["name"], p)
    ps = list(byclip.values())
    if len(ps) > args.n_gallery:
        ps = [ps[i] for i in g.choice(len(ps), args.n_gallery, replace=False)]
    print(f"[gallery] {len(ps)} clips", flush=True)
    wins, zg = [], []
    for i in range(0, len(ps), 128):
        w = torch.stack([cap.window(p, g, 60) for p in ps[i:i + 128]])
        wins.append(w)
        zg.append(me(motion_tokens(ma, w, dev)))
    wins = torch.cat(wins); zg = torch.cat(zg)

    pe = _encode_qwen3(PROMPTS, dev, batch=8).float().to(dev)
    zt = tc(pe)
    top = (zt @ zg.T).argmax(1).tolist()

    # ---- shared neutral start ----
    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu")
    s0, _ = kd.sample_starts(1, np.random.default_rng(args.seed), "val",
                             span=4, families=["walk"])
    r0 = int(s0[0])
    cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + 2])).long()
    pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + 2])).long()
    rr = torch.empty_like(cp)
    rr[..., 0::2] = pk & 0x0F; rr[..., 1::2] = (pk >> 4) & 0x0F
    c01 = torch.cat([cp, rr], 1)

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

    tiles, hits = [], []
    for i, pr in enumerate(PROMPTS):
        p = ps[top[i]]
        hits.append(p["caption"])
        print(f"[match] {pr!r} -> [{p['src']}] {p['caption'][:70]!r}", flush=True)
        w = wins[top[i]]                                  # [60,256,20]
        # goal = mid-action moment of the matched window (frames 30,31)
        ep_c = torch.stack([c01[0], c01[1], w[30], w[31]])[None].to(dev)
        ep = ma.endpoints(ep_c)
        nf = torch.full((1,), float(T), device=dev)
        tok = sample_tokens(mb, ep, nf, dev, temp=0.9,
                            seed=args.seed * 100 + i)
        z = ma.fsq.from_ints(((tok[:, None] // W) % 5)[None].float())[None]
        logits = ma.decode(z, ep, T)
        prob = F.softmax(logits[0], -1)
        lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
        q = (prob * lv).sum(-1)
        q[0] = (ep_c[0, 1].float() - 4.0) / 3.5
        q[-1] = (ep_c[0, 2].float() - 4.0) / 3.5
        # after reaching the goal, play the matched window's continuation
        cont = (w[31:].float().to(dev) - 4.0) / 3.5
        tiles.append(render_q(torch.cat([q, cont], 0)))

    L = max(len(t) for t in tiles)
    frames = []
    for t in range(L):
        row = np.concatenate([pad_band(ti[min(t, len(ti) - 1)])
                              for ti in tiles], 1)
        im = Image.fromarray(row).convert("RGB")
        dr = ImageDraw.Draw(im)
        for k, (pr, hc) in enumerate(zip(PROMPTS, hits)):
            dr.text((k * GC.TILE_W + 8, 6), f'"{pr}"', font=font(13),
                    fill=(255, 240, 180))
            dr.text((k * GC.TILE_W + 8, 26), f"match: {hc[:44]}",
                    font=font(10, False), fill=(160, 200, 255))
            ph = "BRIDGE to goal" if t < T else "matched motion plays"
            dr.text((k * GC.TILE_W + 8, 44), ph, font=font(10, False),
                    fill=(255, 120, 120) if t < T else (120, 255, 160))
        dr.text((8, GC.TILE_H + 24),
                "text picks the GOAL by retrieval (aligned FSQ space) | "
                "Equator B3 bridges from the same neutral start | no GT",
                font=font(12, False), fill=(210, 210, 215))
        frames.append(np.asarray(im))
    p = f"{OUT}/text2goal_retrieval_4prompts_x2.mp4"
    RN.write_mp4(p, np.stack(frames), 20)
    print("->", p, flush=True)


if __name__ == "__main__":
    main()
