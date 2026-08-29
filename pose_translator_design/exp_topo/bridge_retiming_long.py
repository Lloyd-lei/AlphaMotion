"""EQUATOR long-clip self-supervised re-timing: chunked token re-decode.

    bridge_retiming_long.py --src amass|bones

Mechanism: split a LONG clip into ~90f chunks; A's encoder extracts each chunk's
own 32 tokens (self-supervised, no labels); decode the SAME tokens at T_out =
scale * n. Endpoint pass-through pins every chunk boundary to GT codes, so the
re-timed chunks concatenate with exact seams.

amass: ACCAD Male2 MartialArtsExtended, first 300f (10s), Equator clip-level val.
bones: dance chaines 240f - CORPUS zero-shot for Equator (never saw bones codes).
Scales 0.5x (fast) / 1x (recon) / 2x (slow; chunk T_out ~180 > train max 120 =
duration extrapolation, printed on screen).
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
import eval_bones_body_e19 as EB
from bridge_a import BridgeA, CodeData, load_bridge_a, FULL
from spatial5 import capsule_tile, pad_band, font
from train_global import build_global
from common import rot6d_to_matrix, CACHE_ROOT
from semantics import build_name_embeddings
from PIL import Image, ImageDraw

RUNS = V.RUNS
RM = "/var/tmp/alphamotion/robot_models"
X2 = f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
BONES = f"{CACHE_ROOT}/cache_bones_body"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ", "mp4", "bridge_demo")
SCALES = (0.5, 1.0, 2.0)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="amass", choices=("amass", "bones"))
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--chunk", type=int, default=90)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(OUT, exist_ok=True)

    ma, _ = load_bridge_a(f"{RUNS}/BRIDGE_A3_prod_0810", dev)
    sp, spcfg = E.load_run(f"{RUNS}/PROD_c1_91k_dim20_tail_0810", dev)
    enc = spcfg.get("sem_encoder")
    spec_t, dof_t, rest_t, qn_t, xml_t = E.get_target(X2, E.CACHE_AMASS)
    g_t = E.target_geom(spec_t, dof_t, dev, sem_encoder=enc)
    dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
    rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)

    # ---------------- get the code stream + human reference positions ----------
    if args.src == "amass":
        data = CodeData("PROD_c1_91k_dim20_tail_0810", use_root=True)
        seqs = json.load(open(f"{FULL}/sequences.json"))
        s = next(x for x in seqs
                 if "MartialArtsExtended" in x["name"] and x["len"] >= 500)
        r0, n_all = s["start"], min(args.frames, s["len"])
        cp = torch.from_numpy(np.ascontiguousarray(data.c[r0:r0 + n_all])).long()
        pk = torch.from_numpy(np.ascontiguousarray(data.cr[r0:r0 + n_all])).long()
        rr = torch.empty_like(cp)
        rr[..., 0::2] = pk & 0x0F; rr[..., 1::2] = (pk >> 4) & 0x0F
        codes = torch.cat([cp, rr], 1)
        sspec = V.load_spec(FULL, "human_smpl")
        l6 = torch.load(f"{FULL}/human_smpl_paired_local6d.pt", mmap=True)
        gr6, _, _ = build_global(l6[torch.as_tensor(np.arange(r0, r0 + n_all))],
                                 sspec, dev)
        hp = V.fk_pos(gr6, sspec)
        parents = sspec.parents
        tag, zshot = s["name"][:40], "Equator clip-level val"
    else:
        seqs = json.load(open(f"{BONES}/sequences.json"))
        s = max((x for x in seqs if x.get("family") == "dance"),
                key=lambda x: x["len"])
        man = json.load(open(f"{BONES}/manifest.json"))
        spec_b = EB.load_spec(BONES, "soma_body")
        g9 = torch.load(f"{BONES}/soma_body_global9d.pt")[
            s["start"]:s["start"] + min(args.frames, s["len"])]
        n_all = g9.shape[0]
        pose = torch.cat([g9[..., :6], g9[..., 6:9] / man["reach_cm"]], -1)
        lut = build_name_embeddings(list(spec_b.joint_names), device=dev,
                                    encoder=enc)
        entry = EB.entry(BONES, "soma_body", spec_b, dev, lut)
        entry["reach"] = man["reach_cm"]
        from dual_common import geom
        cps, crs = [], []
        for i in range(0, n_all, 512):
            b = pose[i:i + 512].to(dev).float()
            fp, fr = sp.encode_features(b, geom({"soma_body": entry},
                                                "soma_body", b.shape[0]))
            cps.append(sp.fsqp.ints(fp)[0].cpu())
            crs.append(sp.fsqr.ints(fr)[0].cpu())
        codes = torch.cat([torch.cat(cps), torch.cat(crs)], 1).long()
        hp = np.asarray(g9[..., 6:9])
        parents = spec_b.parents
        tag, zshot = s["name"][:40], "CORPUS ZERO-SHOT (bones-seed)"

    # ---------------- chunked re-timing ---------------------------------------
    K = args.chunk
    bounds = list(range(0, n_all - 1, K - 1)) + [n_all - 1]
    lv = (torch.arange(9, device=dev).float() - 4.0) / 3.5
    streams = {}
    for sc in SCALES:
        parts = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            ch = codes[a:b + 1][None].to(dev)
            n = ch.shape[1]
            T_out = max(8, int(round(sc * n)))
            z = ma.encode(ch)
            ep = ma.endpoints(ch)
            logits = ma.decode(z, ep, T_out)
            p = F.softmax(logits[0], -1)
            q = (p * lv).sum(-1)
            q[0] = (ch[0, 0].float() - 4.0) / 3.5
            q[-1] = (ch[0, -1].float() - 4.0) / 3.5
            parts.append(q if not parts else q[1:])       # share boundary frame
        streams[sc] = torch.cat(parts, 0)
        print(f"[retime] scale {sc}: {n_all} -> {streams[sc].shape[0]} frames",
              flush=True)

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

    tiles = [render_q(streams[sc]) for sc in SCALES]
    src_t = capsule_tile(hp, parents, f"SOURCE natural pace ({n_all}f)")
    Tm = max(len(t) for t in tiles)
    frames = []
    for t in range(Tm):
        row = [src_t[min(t, len(src_t) - 1)]]
        for sc, ti in zip(SCALES, tiles):
            row.append(pad_band(ti[min(t, len(ti) - 1)]))
        im = Image.fromarray(np.concatenate(row, 1)).convert("RGB")
        dr = ImageDraw.Draw(im)
        for k, (sc, ti) in enumerate(zip(SCALES, tiles)):
            done = " DONE" if t >= len(ti) else ""
            warn = " [extrap>120]" if sc == 2.0 else ""
            dr.text(((k + 1) * GC.TILE_W + 8, 6),
                    f"{sc}x ({len(ti)}f){warn}{done}", font=font(14),
                    fill=(140, 255, 140) if done else (255, 240, 180))
        dr.text((8, GC.TILE_H + 24),
                f"{tag} | chunked self-retiming | {zshot}",
                font=font(11, False), fill=(210, 210, 215))
        frames.append(np.asarray(im))
    p = f"{OUT}/equator_retime_{args.src}_{n_all}f_x2.mp4"
    RN.write_mp4(p, np.stack(frames), 20)
    print("->", p, flush=True)


if __name__ == "__main__":
    main()
