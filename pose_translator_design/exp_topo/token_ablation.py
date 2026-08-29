"""WHAT DID THE TEMPORAL LAYER ADD? — one video, five columns, no polish.

    token_ablation.py --fam kick

lloyd, 2026-08-08: "我们加了时序轴 内容token和速度控制token后 相对于纯spatial
有了哪些新功能?" and "整个模型能只输入内容token或者只输入速度token吗?"

Five columns, all decoded onto the same robot, all from the SAME two endpoint frames:

  1 GT                 the held-out truth
  2 SPATIAL ONLY       what the world looked like BEFORE the temporal layer existed:
                       you have the frozen spatial layer and two keyframes, so all you
                       can do is interpolate the key joints and complete each frame
                       independently. No notion of time, no motion prior.
                       (This is literally model.baseline() -> spatial completion.)
  3 BOTH TOKENS        the full temporal layer: 8 content + 1 speed
  4 CONTENT ONLY       speed token zeroed
  5 SPEED ONLY         content tokens zeroed

Measured before rendering (kick, 16 windows, V8): both 8.41cm/E1.01, content-only
14.73/E0.32, speed-only 19.06/E0.77, neither 14.93/E0.32. So content-alone is worth
0.2 cm over nothing — in V8 the content tokens are written in ARC-LENGTH coordinates
and the decoder cannot place them on the clock without the speed token. The two are
multiplicative, not additive: content says WHICH PATH, speed says WHEN ALONG IT, and
neither on its own is a motion.

No cherry-picking: the clip is drawn from the val split with a fixed seed and used
as-is, whatever it looks like.
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
from train_v5 import load_run, to_states                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="V8_arclen_0807")
    ap.add_argument("--fam", default="kick")
    ap.add_argument("--targets", default="x2,adam")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, V.T_WIN

    kd = V.KeyData(V.AMASS, [EMB := "human_smpl"], device="cpu")
    g = np.random.default_rng(args.seed)
    st, cids = kd.sample_starts(1, g, "val", span=T, families=[args.fam])
    print(f"clip: {kd.clips[int(cids[0])]['name']}  ({args.fam})", flush=True)
    w = kd.window(EMB, st, T, T).to(dev).float()
    kn = torch.ones(1, device=dev)
    m = load_run(f"{V.RUNS}/{args.run}", dev)[0]

    with torch.no_grad():
        e = m.encode(w, w[:, 0], w[:, -1], kn)
        z0p, z0t = torch.zeros_like(e["path_z"]), torch.zeros_like(e["time_z"])

        def dec(pz, tz):
            return to_states(m.decode(pz, tz, w[:, 0], w[:, -1], kn, T),
                             m, w[:, 0], w[:, -1], kn).float()[0].cpu()
        seqs = {
            "GT": w[0].cpu(),
            "SPATIAL ONLY (no temporal layer)": m.baseline(
                w[:, 0], w[:, -1], kn, T).float()[0].cpu(),
            "BOTH tokens": dec(e["path_z"], e["time_z"]),
            "CONTENT token only": dec(e["path_z"], z0t),
            "SPEED token only": dec(z0p, e["time_z"]),
        }

    prog = {}
    for k, s_ in seqs.items():
        c = V.arclen_curve(s_[None].to(dev).float())[0].cpu().numpy()
        prog[k] = c / max(c[-1], 1e-9)

    def hdr(t, TT):
        nts = [(f"path {prog[k][min(t, len(prog[k]) - 1)] * 100:5.1f}%",
                (245, 245, 245)) for k in seqs]
        return (f"what the temporal layer adds | {args.fam} | frame {t}/{TT-1}", nts)

    C = XM.MeshCtx(dev)
    for tk in args.targets.split(","):
        if tk not in C.tg:
            continue
        try:
            print("  ->", C.render(tk, seqs, f"tokenablate_{args.fam}", hdr,
                                   "all five start from the SAME two endpoint frames"),
                  flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"WARN {tk}: {ex}", flush=True)


if __name__ == "__main__":
    main()
