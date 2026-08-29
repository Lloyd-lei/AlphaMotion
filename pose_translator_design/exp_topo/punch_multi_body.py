"""BACKFIST exam, any-body version: every embodiment in the full cache has a REAL
paired retarget, so the exam runs on trained bodies too.

    punch_multi_body.py --body adam|gr3|g1 --frozen <run dir>

Panels: GT retarget (real, from cache) | spatial full-body | spatial 6-anchor.
(The temporal panels of punch_g1_rollout are dropped here — the question tonight is
purely whether the SPATIAL layer learned the motion on bodies it trained on.)
Prints wrist mean/max cm vs the real retarget, then renders the mesh video.

lloyd 2026-08-09: "backfist 试一试adam还有福利耶gr3 这个动作只有g1有原始数据的gt吗"
— no: all 18 are paired; G1 is only special for being held out.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
import v5_common as V                                                # noqa: E402
import manifold_project as MP                                        # noqa: E402
import eval_newbody as E                                             # noqa: E402
import v7_bridge_mesh as BM                                          # noqa: E402
import exam_c_mesh as XM                                             # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from speed_token_rollout import EMB, FAMS                            # noqa: E402
import prep_xwbc_cache as PX                                         # noqa: E402

BODY = {"g1": ("unitree_g1_29dof", "unitree_g1 (HELD OUT)"),
        "adam": ("pnd_adam_lite", "pnd_adam_lite (TRAINED body)"),
        "gr3": ("fourier_gr3", "fourier_gr3 (TRAINED body)")}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="adam", choices=sorted(BODY))
    ap.add_argument("--frozen", default=None,
                    help="spatial run dir; default = V5_FROZEN/E19 chain")
    ap.add_argument("--fam", default="punch")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, V.T_WIN
    emb, label = BODY[args.body]

    if args.frozen:
        os.environ["V5_FROZEN"] = args.frozen
        V.FROZEN = args.frozen
    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    g = np.random.default_rng(23)                    # same clips as every demo
    win = {}
    for fam in FAMS:
        st, cid = kd.sample_starts(1, g, "val", span=T, families=[fam])
        win[fam] = (int(st[0]), kd.clips[int(cid[0])]["name"])
    s0, cname = win[args.fam]
    rows = torch.arange(s0, s0 + T)
    print(f"[{args.fam}] {cname} on {emb}", flush=True)

    xml = PX.resolve_xml(emb)
    spec, dof, rest, qnames, xml2 = E.get_target(xml, E.CACHE_AMASS)
    dof_d = torch.as_tensor(dof, device=dev, dtype=torch.float64)
    rest_d = torch.as_tensor(rest, device=dev, dtype=torch.float64)

    # ---- GT: the cache's real paired retarget on THIS body ----
    l6 = torch.load(f"{V.AMASS}/{emb}_paired_local6d.pt", map_location="cpu")[rows]
    gr6, _, _ = V.build_global(l6, spec, dev)
    Rg = rot6d_to_matrix(gr6.to(dev)).double()
    qgt, _ = MP.fit_angles(Rg, spec, dof_d, rest=rest_d, clamp=False, method="greedy")
    gRt, gtp = MP.fk_from_angles(qgt, spec, dof_d, rest=rest_d, root_R=Rg[:, 0])
    gt_stream = (qgt.cpu().numpy(), gRt[:, 0].cpu().numpy(), gtp.cpu().numpy())
    gtp_np = gtp.cpu().numpy()

    tok, tcfg = V.load_frozen(dev, run=args.frozen) if args.frozen \
        else V.load_frozen(dev)
    enc = tcfg.get("sem_encoder", "siglip") if isinstance(tcfg, dict) else V.frozen_sem_encoder()
    sq = {x["name"]: x for x in json.load(open(f"{V.AMASS}/sequences.json"))}
    off = s0 - sq[cname]["start"]
    src, sspec, _ = E.load_source(V.AMASS, [cname], dev)
    src = src[off:off + T]
    sdof = np.load(f"{V.AMASS}/{EMB}_dof.npz")["dof"]
    sg = E.target_geom(sspec, sdof, dev, sem_encoder=enc)
    gg = E.target_geom(spec, dof, dev, sem_encoder=enc)
    B = src.shape[0]

    def project_stream(Rp):
        r6, _, q = MP.project(Rp, spec, dof_d, rest=rest_d, method="global")
        gR, gp = MP.fk_from_angles(q, spec, dof_d, rest=rest_d, root_R=Rp[:, 0])
        return (q.cpu().numpy(), gR[:, 0].cpu().numpy(), gp.cpu().numpy())

    # full-body spatial
    zp, zr = tok.encode(src, E.geom_batch(sg, B))
    rot, _ = tok.decode(zp, zr, E.geom_batch(gg, B))
    full_stream = project_stream(rot6d_to_matrix(rot).double())
    # 6-anchor spatial
    names_s = [n.lower() for n in sspec.joint_names]
    kidx = kd.kidx[EMB]
    x = torch.zeros_like(src)
    ki = torch.as_tensor(kidx, dtype=torch.long)
    x[:, ki] = src[:, ki]
    m = torch.ones(B, sspec.J, dtype=torch.bool, device=dev)
    m[:, ki] = False
    zp2, zr2 = tok.encode(x, E.geom_batch(sg, B), joint_mask=m)
    rot2, _ = tok.decode(zp2, zr2, E.geom_batch(gg, B))
    anchor_stream = project_stream(rot6d_to_matrix(rot2).double())

    names = [n.lower() for n in spec.joint_names]
    wl = [i for i, n in enumerate(names) if "wrist" in n or "hand" in n]
    for tag, (q_, r_, gp_) in (("full", full_stream), ("6-anchor", anchor_stream)):
        d = np.linalg.norm(gp_[:, wl] - gtp_np[:, wl], axis=-1)
        print(f"  [{tag}] wrist mean {d.mean():.2f}cm max {d.max():.2f}cm", flush=True)

    # ---- render through MeshCtx machinery (precomputed streams) ----
    C = XM.MeshCtx.__new__(XM.MeshCtx)          # skip heavy __init__
    C.dev = dev
    C.tg = {args.body: dict(label=label, xml=xml2, spec=spec, qnames=qnames)}
    variants = {"GT retarget (real)": gt_stream,
                "spatial full-body": full_stream,
                "spatial 6-anchor": anchor_stream}

    def hdr(t, TT):
        return (f"BACKFIST exam | {emb} | frozen={os.path.basename(V.FROZEN)} "
                f"| f{t}/{TT - 1}", [("", (245, 245, 245))] * len(variants))

    p = XM.MeshCtx.render(C, args.body, variants, f"punch_{args.body}", hdr,
                          "all panels vs the dataset's own paired retarget")
    print("  ->", p, flush=True)


if __name__ == "__main__":
    main()
