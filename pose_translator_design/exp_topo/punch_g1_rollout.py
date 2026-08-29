"""The backfist exam: G1 (held out) with a REAL paired ground-truth retarget.

    punch_g1_rollout.py

lloyd, 2026-08-09: the punch clip (ACCAD E11 backfist_left) is AMASS, and
cache_xwbc_amass carries a real unitree_g1_29dof paired retarget for it. G1 was
held out of spatial training, so every panel except GT is zero-shot embodiment.
lloyd's hypothesis: the spatial layer cannot learn this motion. Five panels:

  GT retarget       the cache's real G1 motion (from qpos, exactly realizable)
  spatial full      all 22 human joints -> frozen spatial encode -> G1 -> project
  spatial 6-anchor  only the 6 key joints visible (the temporal pipeline's diet)
  MSBA              V9_arclen temporal encode/decode -> 6-anchor path -> G1
  JEPA              V9_jepa   temporal encode/decode -> 6-anchor path -> G1

Reading: GT vs "spatial full" isolates what the SPATIAL layer loses on this motion;
"spatial full" vs "spatial 6-anchor" isolates the anchor bottleneck; the last two
add the temporal codec on top. Output: R4_VIZ/mp4/exam_c_mesh/punch_gt_g1.mp4
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
import v5_common as V                                                # noqa: E402
import exam_c_mesh as XM                                             # noqa: E402
import manifold_project as MP                                        # noqa: E402
from train_v5 import load_run, to_states                             # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from speed_token_rollout import EMB, FAMS, full_gt_stream            # noqa: E402

G1 = "unitree_g1_29dof"


@torch.no_grad()
def gt_retarget_stream(C, rows, dev):
    """The cache's REAL paired G1 motion for these rows -> (q, rootR, gp)."""
    t_ = C.tg["g1"]
    l6 = torch.load(f"{V.AMASS}/{G1}_paired_local6d.pt", map_location="cpu")
    l6 = l6[torch.as_tensor(rows, dtype=torch.long)]
    gr6, _, _ = V.build_global(l6, t_["spec"], dev)
    Rg = rot6d_to_matrix(gr6.to(dev)).double()
    q, _ = MP.fit_angles(Rg, t_["spec"], t_["dof"], rest=t_["rest"],
                         clamp=False, method="greedy")
    gR, gp = MP.fk_from_angles(q, t_["spec"], t_["dof"], rest=t_["rest"],
                               root_R=Rg[:, 0])
    return q.cpu().numpy(), gR[:, 0].cpu().numpy(), gp.cpu().numpy()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", default="punch")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, V.T_WIN

    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    C = XM.MeshCtx(dev)
    C.kd = kd
    g = np.random.default_rng(23)                       # SAME clips as the demo
    kn = torch.ones(1, device=dev)
    win = {}
    for fam in FAMS:                                    # replay rng to keep clips
        st, cid = kd.sample_starts(1, g, "val", span=T, families=[fam])
        win[fam] = (int(st[0]), kd.clips[int(cid[0])]["name"])
    s0, cname = win[args.fam]
    rows = list(range(s0, s0 + T))
    w = kd.window(EMB, np.array([s0]), T, T).to(dev).float()
    cur, tgt = w[:, 0], w[:, -1]
    print(f"[{args.fam}] {cname}", flush=True)

    variants = {
        "GT retarget (real)": gt_retarget_stream(C, rows, dev),
        "spatial full-body": full_gt_stream(C, "g1", rows, dev),
        "spatial 6-anchor": w[0].cpu(),
    }
    for label, run in (("MSBA", "V9_arclen_pt_0808"), ("JEPA", "V9_jepa_pt_0808")):
        m = load_run(f"{V.RUNS}/{run}", dev)[0]
        e = m.encode(w, cur, tgt, kn)
        o = m.decode(e["path_z"], e["time_z"], cur, tgt, kn, T)
        variants[label] = to_states(o, m, cur, tgt, kn).float()[0].cpu()
        del m; torch.cuda.empty_cache()

    def hdr(t, TT):
        return (f"BACKFIST exam | {args.fam} | AMASS clip w/ REAL G1 retarget "
                f"| f{t}/{TT - 1}", [("", (245, 245, 245))] * len(variants))

    p = C.render("g1", variants, f"punch_gt", hdr,
                 "G1 held out of training; GT panel is the dataset's paired retarget",
                 size=(2000, 780), dist=4.4)
    print("  ->", p, flush=True)


if __name__ == "__main__":
    main()
