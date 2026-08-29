"""SPEED-TOKEN rollout on X2 — speed varied ONLY through the speed token.

    speed_token_rollout.py

lloyd, 2026-08-09: "如果有速度token 记得用速度token去变速 别再给我傻不拉几的
加速/减速了" — so NO T_out playback tricks anywhere in this file. Every panel is
decoded at the SAME 60 frames; the only thing that differs is the SPEED TOKEN.

Mechanism per motion:
  content tokens  : encoded once from the original clip, reused in every panel
  speed token     : re-encoded from a deliberately re-timed copy of the SAME clip
      original    : the clip's own pacing
      FRONT-loaded: ease-out warp  s(t)=1-(1-t)^2  -> burst first, glide after
      BACK-loaded : ease-in  warp  s(t)=t^2        -> wind up, burst at the end
  duration        : identical (60 frames) in all panels, by construction

Model: V9_arclen_pt_0808 — the pure-timing speed token (amplitude scalars stripped;
timing-shape probe R^2 0.965, amplitude 0.17). Honesty note carried from the
transplant exam: swapping the speed token still drags the PATH somewhat (measured
12-17 cm shape drift on dance/walk); the path% readout under each panel makes both
the intended pacing change AND any unintended path change visible. Judge by eye.

Output: R4_VIZ/mp4/speed_token/*.mp4  (X2, zero-shot body, mesh, projected)
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

EMB = "human_smpl"
FAMS = ("walk", "kick", "punch", "jacks")


@torch.no_grad()
def full_gt_stream(C, tkey, rows, dev):
    """GT the honest way: ALL 22 human joints -> spatial encode (no mask) -> decode
    on the target -> feasibility projection. Same chain as robot_stream except the
    encoder sees the full body instead of 6 anchors."""
    t_ = C.tg[tkey]
    x, _ = V.full_pose_of(V.AMASS, EMB, rows, dev, reach=C.kd.reach[EMB])
    zp, zr = C.tok.encode(x.to(dev).float(), V.geom_for(C.src, x.shape[0]))
    rot, _ = C.tok.decode(zp, zr, V.geom_for(t_["ent"], x.shape[0]))
    Rg = rot6d_to_matrix(rot).double()
    r6, _, q = MP.project(Rg, t_["spec"], t_["dof"], rest=t_["rest"], method="global")
    gR, gp = MP.fk_from_angles(q, t_["spec"], t_["dof"], rest=t_["rest"],
                               root_R=Rg[:, 0])
    return q.cpu().numpy(), gR[:, 0].cpu().numpy(), gp.cpu().numpy()


def retime(w, s_of_t):
    """Same path, new clock: sample the clip at fractional times s(t)*(T-1)."""
    T = w.shape[1]
    tau = torch.linspace(0, 1, T, device=w.device, dtype=w.dtype)
    fid = s_of_t(tau).clamp(0, 1) * (T - 1)
    lo = fid.long().clamp(max=T - 2)
    u = (fid - lo)[None, :, None, None]
    return V.orthonormalize6d((1 - u) * w[:, lo] + u * w[:, lo + 1])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="V9_arclen_pt_0808")
    ap.add_argument("--target", default="x2")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, V.T_WIN
    arm = "msba" if "arclen" in args.run else ("jepa" if "jepa" in args.run else args.run)

    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    m = load_run(f"{V.RUNS}/{args.run}", dev)[0]
    assert m.fsq_time is not None, "this demo requires an arm WITH a speed token"
    C = XM.MeshCtx(dev)
    g = np.random.default_rng(args.seed)
    kn = torch.ones(1, device=dev)

    for fam in FAMS:
        try:
            st, cid = kd.sample_starts(1, g, "val", span=T, families=[fam])
            w = kd.window(EMB, st, T, T).to(dev).float()
            print(f"[{fam}] clip {kd.clips[int(cid[0])]['name']}", flush=True)
        except Exception as ex:
            print(f"[{fam}] sample failed: {ex}", flush=True); continue
        cur, tgt = w[:, 0], w[:, -1]
        e = m.encode(w, cur, tgt, kn)                       # content + own timing

        def dec(time_z):
            o = m.decode(e["path_z"], time_z, cur, tgt, kn, T)
            return to_states(o, m, cur, tgt, kn).float()[0].cpu()

        s0 = int(st[0])
        gt = full_gt_stream(C, args.target, list(range(s0, s0 + T)), dev)
        variants = {"GT (full-body spatial)": gt, "own pacing": dec(e["time_z"])}
        for name, s_of_t in (("FRONT-loaded", lambda t: 1 - (1 - t) ** 2),
                             ("BACK-loaded", lambda t: t ** 2)):
            wv = retime(w, s_of_t)
            ev = m.encode(wv, cur, tgt, kn)                 # ONLY the speed token used
            variants[name] = dec(ev["time_z"])

        prog = {}
        for k, s_ in variants.items():
            if isinstance(s_, tuple):                 # GT tuple: pacing = the clip's
                s_ = w[0].cpu()
            c = V.arclen_curve(s_[None].to(dev).float())[0].cpu().numpy()
            prog[k] = c / max(c[-1], 1e-9)

        def hdr(t, TT):
            nts = [(f"path {prog[k][min(t, len(prog[k]) - 1)] * 100:5.1f}%",
                    (245, 245, 245)) for k in variants]
            return (f"SPEED TOKEN swap | {arm.upper()} ({args.run}) | {fam} | "
                    f"same content tokens, same 60 frames | f{t}/{TT - 1}", nts)

        try:
            p = C.render(args.target, variants, f"speedtok_{arm}_{fam}", hdr,
                         "pacing changed ONLY via the speed token — no playback tricks")
            print("  ->", p, flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"WARN render {fam}: {ex}", flush=True)


if __name__ == "__main__":
    main()
