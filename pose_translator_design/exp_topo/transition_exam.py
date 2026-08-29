"""TRANSITION exam: splice two families at frame 30, seam hidden — who crosses better?

    transition_exam.py --runs T32_arclen_0809,T32_jepa_0809

lloyd 0810: "为什么我肉眼感觉jepa比msba好呢? transition哪个更好?" — the verdict
table scores position error and energy, neither of which measures what the EYE sees:
jitter and seam pops. A time-local code (MSBA) can snap at its group boundaries; a
global code (jepa) is built to be smooth. This exam scores BOTH families of metrics:

  accuracy   key pos err [cm] (whole window / seam±5)
  smoothness jerk RMS (3rd difference of anchor positions, cm/frame^3)
             seam accel spike = max |a| in seam±3 / median |a| elsewhere
             boundary jitter (MSBA only): |a| at its 32 group boundaries vs between
  physics    GT's own numbers as the reference row (the eye calibrates on GT)

Also renders a 3-panel mesh video per pair (GT splice | MSBA | JEPA) on X2 with a
live per-frame |accel| readout under each panel, so the numbers can be checked by eye.

Output: runs/TRANSITION_verdict_0810.txt + R4_VIZ/mp4/exam_c_mesh/transit_*.mp4
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

EMB = "human_smpl"
PAIRS = [("walk", "kick"), ("run", "jacks"), ("dance", "walk"), ("punch", "run")]
SEAM = 30


def pos_of(s):
    """[T,nk,9] key states -> anchor positions [T,nk,3] (reach units)."""
    return s[..., 6:9]


def metrics(s, model_T=V.T_WIN):
    p = pos_of(s)                                        # [T,nk,3]
    v = p[1:] - p[:-1]
    a = v[1:] - v[:-1]
    j = a[1:] - a[:-1]
    amag = a.norm(dim=-1).mean(-1)                       # [T-2] mean over anchors
    seam_lo, seam_hi = SEAM - 3 - 1, SEAM + 3 - 1        # accel index offset
    seam = amag[max(seam_lo, 0):seam_hi]
    off = torch.cat([amag[:max(seam_lo, 0)], amag[seam_hi:]])
    return {"jerk_rms": float(j.norm(dim=-1).mean(-1).pow(2).mean().sqrt()),
            "seam_spike": float(seam.max() / (off.median() + 1e-9)),
            "amag": amag}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="T32_arclen_0809,T32_jepa_0809")
    ap.add_argument("--labels", default="MSBA,JEPA")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--target", default="x2")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev, T = args.device, V.T_WIN
    runs = args.runs.split(","); labels = args.labels.split(",")

    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    models = {lab: load_run(f"{V.RUNS}/{r}", dev)[0] for lab, r in zip(labels, runs)}
    C = XM.MeshCtx(dev)
    g = np.random.default_rng(args.seed)
    kn = torch.ones(1, device=dev)
    L = ["=" * 100,
         "TRANSITION EXAM — splice at f30, seam hidden. accuracy AND smoothness.",
         "jerk = 3rd-diff RMS (lower=smoother). seam spike = max|a| at seam / median|a| off-seam.",
         "GT row calibrates the eye: a real transition has its own nonzero spike.",
         "=" * 100]
    agg = {lab: {"key": [], "jerk": [], "spike": [], "gjit": []} for lab in labels}
    agg["GT"] = {"key": [], "jerk": [], "spike": [], "gjit": []}

    for pi, (fa, fb) in enumerate(PAIRS):
        errs = {lab: [] for lab in labels}
        first = None
        for i in range(args.n):
            try:
                sa, _ = kd.sample_starts(1, g, "val", span=T, families=[fa])
                sb, _ = kd.sample_starts(1, g, "val", span=T, families=[fb])
            except Exception:
                break
            wa = kd.window(EMB, sa, T, T); wb = kd.window(EMB, sb, T, T)
            w = torch.cat([wa[:, :SEAM], wb[:, SEAM:]], 1).to(dev).float()
            cur, tgt = w[:, 0], w[:, -1]
            gm = metrics(w[0].cpu())
            agg["GT"]["jerk"].append(gm["jerk_rms"])
            agg["GT"]["spike"].append(gm["seam_spike"])
            outs = {}
            for lab in labels:
                m = models[lab]
                e = m.encode(w, cur, tgt, kn)
                o = m.decode(e["path_z"], e["time_z"], cur, tgt, kn, T)
                s = to_states(o, m, cur, tgt, kn).float()[0].cpu()
                outs[lab] = s
                mm = metrics(s)
                key = float((pos_of(s) - pos_of(w[0].cpu())).norm(dim=-1).mean()
                            * kd.reach[EMB])
                agg[lab]["key"].append(key)
                agg[lab]["jerk"].append(mm["jerk_rms"])
                agg[lab]["spike"].append(mm["seam_spike"])
                # MSBA group-boundary jitter: |a| at group bounds vs between
                if lab == labels[0] and hasattr(m, "n_groups"):
                    from train_v7 import group_bounds
                    gb = list(group_bounds(T, m.n_groups))[1:-1]
                    amag = mm["amag"]
                    at = amag[[b - 1 for b in gb if 0 < b - 1 < len(amag)]]
                    agg[lab]["gjit"].append(float(at.mean() / (amag.median() + 1e-9)))
            if first is None:
                first = (w, outs, gm)
        # ---- video for the first splice of this pair ----
        w, outs, gm = first
        amags = {"GT splice": metrics(w[0].cpu())["amag"]}
        variants = {"GT splice": w[0].cpu()}
        for lab in labels:
            variants[f"{lab}"] = outs[lab]
            amags[lab] = metrics(outs[lab])["amag"]

        def hdr(t, TT, amags=amags, fa=fa, fb=fb):
            nts = []
            for k in variants:
                am = amags[k]
                v = float(am[min(max(t - 2, 0), len(am) - 1)])
                nts.append((f"|accel| {v * kd.reach[EMB]:5.2f}"
                            + ("  <SEAM>" if abs(t - SEAM) <= 2 else ""),
                            (255, 120, 120) if abs(t - SEAM) <= 2 else (245, 245, 245)))
            return (f"TRANSITION {fa}->{fb} @f{SEAM} (seam hidden) | f{t}/{TT - 1}", nts)

        p = C.render(args.target, variants, f"transit_{fa}_{fb}", hdr,
                     "3rd panel question: which arm crosses the seam more naturally")
        print("  ->", p, flush=True)

    L.append(f"{'metric':<28}" + "".join(f"{lab:>14}" for lab in ["GT"] + labels))
    rows = [("key pos err [cm]", "key"), ("jerk RMS", "jerk"),
            ("seam accel spike", "spike"), ("group-boundary jitter", "gjit")]
    for name, k in rows:
        vals = []
        for lab in ["GT"] + labels:
            a = agg[lab][k]
            vals.append(f"{np.mean(a):>14.3f}" if a else f"{'—':>14}")
        L.append(f"{name:<28}" + "".join(vals))
    txt = "\n".join(L)
    print(txt, flush=True)
    open(f"{V.RUNS}/TRANSITION_verdict_0810.txt", "w").write(txt + "\n")
    print(f"verdict: {V.RUNS}/TRANSITION_verdict_0810.txt", flush=True)


if __name__ == "__main__":
    main()
