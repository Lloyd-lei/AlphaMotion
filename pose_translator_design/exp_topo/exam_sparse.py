"""SPARSE-CELL ROLLOUT — can the model reconstruct a whole clip from a handful of
cells anywhere in the (joint x time) grid?  (lloyd, 2026-08-08)

    exam_sparse.py --runs V8_arclen_0807,V7_maskfsq_0807

Two constraint SHAPES that m4 did not cover. m4 pinned ONE joint's DENSE track;
these are the two that matter for a real controller:

  A. TRACK    5 of the 6 key joints get their FULL reference trajectory; the 6th is
              unconstrained and must be invented. Sweep WHICH joint is free.
              -> "I have a reference motion for most of the body, fill in the rest."
  B. SPARSE   k RANDOM frames give the WHOLE skeleton (all 6 key joints), every other
              frame is empty. k = 1, 2, 3, 5, 10.
              -> "I have two keyframes somewhere in the clip. Roll out the rest."
              k=1 is the hard case: a single interior pose, no endpoints at all.

MECHANISM (same as m4 v2 — the constraint enters through the ENCODER, never as a
paste-over): build a sketch = the model's straight-line prior everywhere, overwritten
with the commanded cells, then linearly interpolated between commanded frames on the
constrained joints. Encode the sketch -> the tokens carry the constraint -> decode.
Commanded cells are re-imposed on the output as a hard guarantee.

HONEST FRAMING, printed with the numbers:
  * The endpoints of the window are NOT automatically given in mode B (unless a drawn
    frame happens to be 0 or T-1). V7/V8 were trained with cond_drop=False, i.e. they
    have NEVER seen a missing endpoint, so mode B is OUT OF DISTRIBUTION and its
    numbers are a measurement of the gap, not of the ceiling.
  * Baselines on every row: `interp` = linear interpolation through the given cells
    alone (no model), and `straight` = the endpoint line. A model row only means
    something if it beats interp.

Output: runs/SPARSE_verdict.txt + mesh videos (X2 zero-shot + G1 held out).
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
import exam_c_mesh as XM                                             # noqa: E402
from train_v5 import load_run, to_states                             # noqa: E402

EMB = "human_smpl"
FAMS = ("kick", "jacks", "walk")
SPARSE_K = [1, 2, 3, 5, 10]


def lerp_track(cells, fr, T, dev):
    """cells [B,k,nk,9] at frames fr -> [B,T,nk,9] piecewise-linear in time.
    Outside [fr[0], fr[-1]] the nearest commanded frame is held."""
    tt = torch.as_tensor(fr, device=dev, dtype=torch.float32)
    q = torch.arange(T, device=dev, dtype=torch.float32)
    idx = torch.searchsorted(tt, q).clamp(1, max(len(fr) - 1, 1))
    if len(fr) == 1:
        return cells[:, 0:1].expand(-1, T, -1, -1).contiguous()
    t0, t1 = tt[idx - 1], tt[idx]
    u = ((q - t0) / (t1 - t0).clamp_min(1e-6)).clamp(0, 1)[None, :, None, None]
    return V.orthonormalize6d((1 - u) * cells[:, (idx - 1).cpu()] + u * cells[:, idx.cpu()])


@torch.no_grad()
def run_case(model, w, cells_j, fr, dev, T):
    """cells_j: list of key-joint indices that are constrained at frames `fr`."""
    B = w.shape[0]
    kn = torch.ones(B, device=dev)
    sketch = model.baseline(w[:, 0], w[:, -1], kn, T).float().clone()
    track = lerp_track(w[:, fr][:, :, cells_j], fr, T, dev)
    sketch[:, :, cells_j] = track
    e = model.encode(sketch, sketch[:, 0], sketch[:, -1], kn)
    out = to_states(model.decode(e["path_z"], e["time_z"],
                                 sketch[:, 0], sketch[:, -1], kn, T),
                    model, sketch[:, 0], sketch[:, -1], kn).float()
    o = out.clone()
    fr_t = torch.as_tensor(np.asarray(fr), device=dev, dtype=torch.long)
    for jj in cells_j:                      # hard guarantee on commanded cells only
        o[:, fr_t, jj] = w[:, fr_t, jj]
    return o, sketch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="V8_arclen_0807,V7_maskfsq_0807")
    ap.add_argument("--targets", default="x2,g1")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_render", action="store_true")
    args = ap.parse_args()
    dev, T = args.device, V.T_WIN
    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    knames, reach = kd.knames[EMB], kd.reach[EMB]
    models = {a.replace("_0807", ""): load_run(f"{V.RUNS}/{a}", dev)[0]
              for a in args.runs.split(",") if a}
    C = None if args.no_render else XM.MeshCtx(dev)
    rows, vids = {}, []

    def cm(x, y):
        return float((x[..., 6:9] - y[..., 6:9]).norm(dim=-1).mean() * reach)

    for fam in FAMS:
        g = np.random.default_rng(4096 + len(fam))
        st, _ = kd.sample_starts(args.n, g, "val", span=T, families=[fam])
        w = kd.window(EMB, st, T, T).to(dev).float()
        kn = torch.ones(w.shape[0], device=dev)
        m0 = next(iter(models.values()))
        line = m0.baseline(w[:, 0], w[:, -1], kn, T).float()

        # ---------- MODE A: 5 joints tracked, 1 free -------------------------
        allf = np.arange(T)
        for free_j in range(len(knames)):
            given = [i for i in range(len(knames)) if i != free_j]
            base_i = lerp_track(w[:, allf][:, :, [free_j]], allf, T, dev)  # = GT, ref
            for nm, m in models.items():
                o, _ = run_case(m, w, given, allf, dev, T)
                rows[f"A trackers | {fam} | free={knames[free_j]} | {nm}"] = {
                    "free_joint_cm": round(cm(o[:, :, [free_j]], w[:, :, [free_j]]), 2),
                    "straight_baseline_cm": round(cm(line[:, :, [free_j]],
                                                     w[:, :, [free_j]]), 2)}

        # ---------- MODE B: k random full-skeleton keyframes ------------------
        alljoints = list(range(len(knames)))
        for k in SPARSE_K:
            fr = np.sort(g.choice(T, size=k, replace=False))
            interp_only = lerp_track(w[:, fr], fr, T, dev)
            r = {"cells": f"{k} frames x 6 joints",
                 "frames": fr.tolist(),
                 "interp_only_cm": round(cm(interp_only, w), 2),
                 "straight_line_cm": round(cm(line, w), 2),
                 "endpoints_given": bool(0 in fr and T - 1 in fr)}
            for nm, m in models.items():
                o, _ = run_case(m, w, alljoints, fr, dev, T)
                r[f"{nm}_cm"] = round(cm(o, w), 2)
                r[f"{nm}_energy"] = round(float(o[..., 6:9].std(1).mean()
                                                / w[..., 6:9].std(1).mean()), 3)
            rows[f"B sparse | {fam} | k={k}"] = r
            print(f"  B {fam:<6} k={k:<3} interp {r['interp_only_cm']:5.2f}  " +
                  "  ".join(f"{nm} {r[nm+'_cm']:5.2f}(E{r[nm+'_energy']:.2f})"
                            for nm in models), flush=True)

            if C is not None and k in (1, 3, 10) and fam == "kick":
                nm0 = list(models)[0]
                o, sk = run_case(models[nm0], w, alljoints, fr, dev, T)
                for tk in args.targets.split(","):
                    try:
                        vids.append(C.render(
                            tk, {"GT (never shown)": w[0].cpu(),
                                 f"interp only ({k} kf)": interp_only[0].cpu(),
                                 f"{nm0} ({k} kf)": o[0].cpu()},
                            f"sparse_{fam}_k{k}",
                            XM.plain_hdr(f"sparse rollout: {k} keyframe(s), whole body"),
                            f"frames {fr.tolist()}"))
                    except Exception as ex:
                        print(f"WARN render {fam} k={k} {tk}: {ex}", flush=True)

    L = ["=" * 100,
         "SPARSE-CELL ROLLOUT — 5 tracked joints (A) / k random keyframes (B)",
         "=" * 100,
         "MODE B IS OUT OF DISTRIBUTION: V7/V8 trained cond_drop=False and have never",
         "seen a window without both endpoints. `interp_only` is the no-model control;",
         "a model row is only meaningful if it beats it.",
         "-" * 100]
    for k in sorted(rows):
        L.append(f"{k:<52} {json.dumps(rows[k])}")
    txt = "\n".join(L)
    print("\n" + txt, flush=True)
    open(f"{V.RUNS}/SPARSE_verdict.txt", "w").write(txt + "\n")
    json.dump({"rows": rows, "videos": vids},
              open(f"{V.RUNS}/SPARSE_verdict.json", "w"), indent=1)
    print(f"\nverdict: {V.RUNS}/SPARSE_verdict.txt", flush=True)


if __name__ == "__main__":
    main()
