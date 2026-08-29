"""SPEED DEMO — one long clip, two arms, three speeds, on X2 and PND Adam.

    speed_demo.py --fam dance --secs 4

TWO DIFFERENT WAYS TO MAKE IT SLOW, and the whole point is that they are different:

  (1) CLOCK  (both arms)   decode the SAME tokens at a different T_out. The decoder's
      per-frame queries are fourier_time(linspace(0,1,T_out)) — a NORMALIZED clock —
      so asking for 120 frames instead of 60 simply samples the same normalized
      trajectory twice as finely. Nothing about the tokens changes. This is why it
      works on both arms and why the path barely drifts (measured 0.53-0.75 cm).

  (2) SPEED-TOKEN TRANSPLANT  (only arms that HAVE a speed token: jepa and V8)
      keep the content tokens of clip A, replace the SPEED token with one encoded
      from a re-timed copy. The speed token's only input is `curve_summary` =
      [8-point normalized cumulative arc-length curve ; log(total path) ,
       log2(T/60) , mean speed]. A re-timing bends that curve without moving the
      path, so the transplant changes the PACING PROFILE (where the motion lingers
      and where it rushes) rather than the total duration.
      maskfsq structurally cannot do this — it has no speed token.

So (1) is "play the same recording slower" and (2) is "perform it with different
timing". A demo that only shows (1) is not showing the speed token at all.

Output: R4_VIZ/mp4/speed_demo/*.mp4 — mesh, X2 (in no dataset) + PND Adam (trained).
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
import v7_bridge_mesh as BM                                          # noqa: E402
import exam_c_mesh as XM                                             # noqa: E402
from train_v5 import load_run, to_states                             # noqa: E402

OUT = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
       'R4_VIZ/mp4/speed_demo')
EMB = "human_smpl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fam", default="dance")
    ap.add_argument("--secs", type=float, default=4.0, help="source clip length")
    ap.add_argument("--targets", default="x2,adam")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=71)
    args = ap.parse_args()
    dev = args.device
    os.makedirs(OUT, exist_ok=True)
    T = V.T_WIN                                   # the model's canonical window
    span = int(round(args.secs * V.FPS))          # a LONGER source than one window

    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    g = np.random.default_rng(args.seed)
    st, cids = kd.sample_starts(1, g, "val", span=span, families=[args.fam])
    print(f"clip: {kd.clips[int(cids[0])]['name']}  family={kd.family[int(cids[0])]}  "
          f"span={span} frames ({args.secs}s)", flush=True)
    # read the long span at the model's window length (same path, model's own rate)
    w = kd.window(EMB, st, T, span).to(dev).float()
    kn = torch.ones(1, device=dev)

    models = {}
    for a in ("V7_jepa_0807", "V8_arclen_0807"):
        rd = f"{V.RUNS}/{a}"
        if os.path.exists(f"{rd}/model.pt"):
            models[a.replace("_0807", "")] = load_run(rd, dev)[0]

    # -------- the two mechanisms, as TWO videos (5 panels each; 7 in one row is
    #          unreadable and overruns the palette) ------------------------------
    clock, trans = {"GT (1x)": w[0].cpu()}, {}
    with torch.no_grad():
        for nm, m in models.items():
            e = m.encode(w, w[:, 0], w[:, -1], kn)
            for T_out, lab in ((30, "0.5x"), (120, "2x")):
                o = m.decode(e["path_z"], e["time_z"], w[:, 0], w[:, -1], kn, T_out)
                clock[f"{nm} {lab}"] = to_states(
                    o, m, w[:, 0], w[:, -1], kn).float()[0].cpu()
            o = m.decode(e["path_z"], e["time_z"], w[:, 0], w[:, -1], kn, T)
            trans[f"{nm} ORIGINAL rhythm"] = to_states(
                o, m, w[:, 0], w[:, -1], kn).float()[0].cpu()
            if e["time_z"].shape[1]:
                fid = torch.as_tensor(V.warp_fidx(T, T, np.random.default_rng(5),
                                                  strength=1.0),
                                      device=dev, dtype=w.dtype)
                lo = fid.long().clamp(max=T - 2)
                u = (fid - lo)[None, :, None, None]
                warped = V.orthonormalize6d((1 - u) * w[:, lo] + u * w[:, lo + 1])
                ew = m.encode(warped, w[:, 0], w[:, -1], kn)
                o = m.decode(e["path_z"], ew["time_z"], w[:, 0], w[:, -1], kn, T)
                trans[f"{nm} SWAPPED rhythm"] = to_states(
                    o, m, w[:, 0], w[:, -1], kn).float()[0].cpu()

    def pad(d):
        L = max(s.shape[0] for s in d.values())
        lens = {k: s.shape[0] for k, s in d.items()}
        return {k: (torch.cat([s, s[-1:].expand(L - s.shape[0], -1, -1)])
                    if s.shape[0] < L else s) for k, s in d.items()}, lens
    clock, clen = pad(clock)
    trans, tlen = pad(trans)

    def progress(d):
        """每个面板的归一化累计弧长 -> 同一帧下"走到路径几成"。节奏不同 = 这个数不同。"""
        out = {}
        for k, s_ in d.items():
            c = V.arclen_curve(s_[None].to(dev).float())[0].cpu().numpy()
            out[k] = c / max(c[-1], 1e-9)
        return out

    def mk_hdr(lens, tag, prog=None):
        def hdr(t, TT):
            nts = []
            for k in lens:
                if prog is not None:
                    i = min(t, len(prog[k]) - 1)
                    nts.append((f"path {prog[k][i]*100:5.1f}%",
                                (150, 155, 162) if t >= lens[k] else (245, 245, 245)))
                else:
                    nts.append(("DONE - holding" if t >= lens[k]
                                else f"{min(t,lens[k])}/{lens[k]}",
                                (150, 155, 162) if t >= lens[k] else (240, 240, 240)))
            return (f"{tag}: {args.fam} ({args.secs}s source)  |  frame {t}", nts)
        return hdr

    C = XM.MeshCtx(dev)
    for tk in args.targets.split(","):
        if tk not in C.tg:
            print(f"[skip] {tk} not in roster", flush=True); continue
        tprog = progress(trans)
        for d, lens, tag, name, pr in (
                (clock, clen, "CLOCK (same tokens, new T_out)",
                 f"speedA_clock_{args.fam}", None),
                (trans, tlen, "SPEED-TOKEN SWAP: same content tokens, different "
                 "speed token", f"speedB_transplant_{args.fam}", tprog)):
            try:
                print("  ->", C.render(tk, d, name, mk_hdr(lens, tag, pr),
                                       "path% = how far along the motion each one is"),
                      flush=True)
            except Exception as ex:
                import traceback; traceback.print_exc()
                print(f"WARN {tk} {name}: {ex}", flush=True)


if __name__ == "__main__":
    main()
