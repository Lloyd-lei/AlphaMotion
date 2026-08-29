"""EXAM C — the generative-capability exam for the temporal layer (lloyd's spec,
2026-08-07), six input regimes, visual gate FIRST, numbers second.

    exam_c.py --modes m1,m2,m3,m5,m6 --runs V7_jepa_0807,V7_maskfsq_0807

The six regimes, each a different CONTRACT between caller and model:
  m1  full motion in -> 125 bits -> motion out          (the codec contract)
  m2  ONLY endpoint states, zero tokens -> model fills  (the prior contract)
      m2b control: endpoints stripped to ROTATION ONLY (positions zeroed)
  m3  ONLY ONE endpoint + tokens -> forward / backward  (the extrapolation contract)
  m4  one joint's whole trajectory pinned -> rest fills (the constraint contract)
      [implemented via temporal prior + project_constrained, separate runner]
  m5  K sparse keyframes -> tokens -> WHICH dataset motion is this?
                                                        (the vocabulary contract)
  m6  same path tokens, different clock -> speed control (the timing contract)

HONESTY BOX — two regimes are OUT OF DISTRIBUTION for V7 by construction:
  * V7 trained with cond_drop = False (train_v7.py:203): the decoder has never once
    seen a missing or degraded endpoint. m2b (rot-only) and m3 (single endpoint) are
    therefore expected to degrade; the point of running them anyway is to MEASURE the
    gap so the production retrain knows what dropout to add. Their rows say [OOD].
  * m2's "model fills" = decode with path_z zeroed — the established tokens_zeroed
    diagnostic, i.e. what the decoder's learned prior invents beyond the straight
    line, not a trained regime either.

Every mode emits stick GIFs (R4_VIZ/mp4/exam_c/) for the visual gate, and a metrics
block in EXAMC_verdict.txt. No number in this file is a GT-frame-error unless the
mode's contract is literally reconstruction (m1).
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
import v5_common as V                                             # noqa: E402
from train_v5 import load_run, to_states                          # noqa: E402
import v7_render as R7                                            # noqa: E402

OUT = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
       'R4_VIZ/mp4/exam_c')
EMB = "human_smpl"
FAMS = ("walk", "run", "jacks", "kick", "dance")


def fam_of(name):
    n = name.lower()
    for f in ("walk", "run", "jump", "jacks", "kick", "punch", "dance", "fight",
              "sprint", "sidestep"):
        if f in n:
            return f
    return "other"


class Ctx:
    def __init__(self, runs, dev):
        self.dev = dev
        self.kd = V.KeyData(V.AMASS, [EMB], device="cpu")
        self.tok, _ = V.load_frozen(dev)
        self.ent = V.body_entry(V.AMASS, EMB, dev, self.kd.reach[EMB])
        self.spec = self.kd.spec[EMB]
        self.kidx = self.kd.kidx[EMB]
        self.parents = np.asarray(self.spec.parents)
        self.models = {}
        for a in runs:
            rd = f"{V.RUNS}/{a}"
            if os.path.exists(f"{rd}/model.pt"):
                self.models[a.replace("_0807", "")] = load_run(rd, dev)[0]

    def fk(self, key):
        """[T,nk,9] -> [T,J,3] cm through the frozen spatial completion."""
        return R7.to_full_body(self.tok, self.ent, key.cpu(), self.kidx,
                               self.dev, self.spec)

    def win(self, fam, T, g, n=1):
        st, _ = self.kd.sample_starts(n, g, "val", span=T, families=[fam])
        return self.kd.window(EMB, st, T, T)

    def animate(self, pans, titles, name, seam=None, note=""):
        p = R7.animate(pans, titles, f"{OUT}/{name}.gif", self.parents,
                       seam=seam, note=note)
        print(f"  -> {p}", flush=True)
        return p


def energy(p):
    return float(np.asarray(p).std(0).mean())


def path_shape(p, n=60):
    """[T,J,3] -> [n,J,3] resampled by NORMALIZED cumulative arc length. Two motions
    tracing the same path at different speeds map to (nearly) the same array — the
    whole point of the m6 metric."""
    p = np.asarray(p, np.float64)
    d = np.linalg.norm(np.diff(p, axis=0), axis=-1).mean(-1)
    c = np.concatenate([[0], np.cumsum(d)])
    c = c / max(c[-1], 1e-9)
    ti = np.interp(np.linspace(0, 1, n), c, np.arange(len(c)))
    lo = np.clip(ti.astype(int), 0, len(p) - 2)
    w = (ti - lo)[:, None, None]
    return p[lo] * (1 - w) + p[lo + 1] * w


# =========================================================================== m1
@torch.no_grad()
def m1_codec(C, rows, paths, g, T=60):
    """Full motion -> 125 bits -> motion. The only mode where GT error IS the metric."""
    for fam in FAMS:
        w = C.win(fam, T, g).to(C.dev).float()
        cur, tgt, kn = w[:, 0], w[:, -1], torch.ones(1, device=C.dev)
        pans, ttl = [C.fk(w[0])], ["GT"]
        for nm, m in C.models.items():
            p = to_states(m(w, cur, tgt, kn)[0], m, cur, tgt, kn).float()
            pe, _ = V.key_se3_err(p, w, C.kd.reach[EMB])
            pans.append(C.fk(p[0].cpu())); ttl.append(nm)
            rows[f"m1 {fam} | {nm}"] = {
                "key_pos_cm": round(pe, 2),
                "energy_vs_gt": round(energy(pans[-1]) / energy(pans[0]), 3)}
        paths.append(C.animate(R7.deyaw(pans, _R0(w)), ttl, f"m1_{fam}",
                     note=f"m1 codec: full {fam} in, 125 bits, motion out"))


def _R0(w):
    from common import rot6d_to_matrix
    return rot6d_to_matrix(w[0, 0, 0, :6].double().cpu())[None].numpy()[0]


# =========================================================================== m2
@torch.no_grad()
def m2_prior(C, rows, paths, g, T=60):
    """Endpoints only, ZERO tokens: what does the decoder's prior invent beyond the
    straight line?  m2b: same, endpoints stripped to rotation only [OOD]."""
    for fam in ("kick", "jacks"):
        w = C.win(fam, T, g).to(C.dev).float()
        cur, tgt, kn = w[:, 0], w[:, -1], torch.ones(1, device=C.dev)
        pans, ttl = [C.fk(w[0])], ["GT (never shown)"]
        m0 = next(iter(C.models.values()))
        line = m0.baseline(cur, tgt, kn, T)
        pans.append(C.fk(line[0].cpu())); ttl.append("straight line")
        for tag, (c2, t2) in {"": (cur, tgt),
                              "b rot-only": (_rot_only(cur), _rot_only(tgt))}.items():
            for nm, m in C.models.items():
                e = m.encode(w, c2, t2, kn)
                z0 = torch.zeros_like(e["path_z"])
                o = m.decode(z0, e["time_z"][:, :0] if e["time_z"].shape[1] == 0
                             else torch.zeros_like(e["time_z"]), c2, t2, kn, T)
                p = to_states(o, m, c2, t2, kn).float()
                fkp = C.fk(p[0].cpu())
                key = f"m2{tag} {fam} | {nm}"
                rows[key] = {"energy_vs_gt": round(energy(fkp) / energy(pans[0]), 3),
                             "ood": bool(tag)}
                if not tag:
                    pans.append(fkp); ttl.append(f"{nm} prior-fill")
        paths.append(C.animate(R7.deyaw(pans, _R0(w)), ttl, f"m2_{fam}",
                     note=f"m2: model given ONLY frame 0 + frame {T-1}, ZERO tokens. "
                          f"GT panel shown for reference only."))


def _rot_only(x):
    y = x.clone(); y[..., 6:9] = 0.0
    return y


# =========================================================================== m3
@torch.no_grad()
def m3_extrap(C, rows, paths, g, T=60):
    """ONE endpoint + self tokens -> forward (from frame 0) / backward (from frame
    T-1). [OOD]: V7 never trained a missing endpoint; both endpoints are set to the
    same frame, so the architectural prior collapses to 'hold' and ALL motion must
    come from the tokens."""
    for fam in ("run", "kick"):
        w = C.win(fam, T, g).to(C.dev).float()
        kn = torch.ones(1, device=C.dev)
        pans, ttl = [C.fk(w[0])], ["GT"]
        for direction in ("fwd", "bwd"):
            a = w[:, 0] if direction == "fwd" else w[:, -1]
            for nm, m in C.models.items():
                e = m.encode(w, a, a, kn)          # tokens DO see the full motion
                o = m.decode(e["path_z"], e["time_z"], a, a, kn, T)
                p = to_states(o, m, a, a, kn).float()
                pe, _ = V.key_se3_err(p, w, C.kd.reach[EMB])
                fkp = C.fk(p[0].cpu())
                rows[f"m3 {fam} {direction} | {nm}"] = {
                    "key_pos_cm": round(pe, 2),
                    "energy_vs_gt": round(energy(fkp) / energy(pans[0]), 3),
                    "ood": True}
                if direction == "fwd":
                    pans.append(fkp); ttl.append(f"{nm} fwd")
        paths.append(C.animate(R7.deyaw(pans, _R0(w)), ttl, f"m3_{fam}",
                     note=f"m3 [OOD]: ONLY frame 0 given (tgt=cur), tokens from the "
                          f"full clip. All motion must come from the 125 bits."))


# =========================================================================== m5
@torch.no_grad()
def m5_vocab(C, rows, paths, g, T=60, n_bank=400, K=4):
    """K sparse keyframes -> piecewise-linear sketch -> tokens -> retrieve the most
    similar dataset motion BY TOKEN Hamming distance. If the retrieved clip is the
    right family, the codebook is a semantic vocabulary, not a hash."""
    # BALANCED token bank: equal windows per family. A uniform draw over val is 77%
    # "other" (memory: AMASS family skew), so top-1 family would be decided by base
    # rates, not by the tokens. Family labels come from kd.family — the same labeler
    # sample_starts filters by — not from ad-hoc name matching.
    gb = np.random.default_rng(7)
    per = max(20, n_bank // (len(FAMS) + 1))
    st_all, fam_all = [], []
    for f in list(FAMS) + [None]:
        try:
            st, cids = C.kd.sample_starts(per, gb, "val", span=T,
                                          families=[f] if f else None)
        except Exception:
            continue
        st_all.append(st)
        fam_all += [C.kd.family[int(c)] for c in cids]
    st = np.concatenate(st_all)
    n_bank = len(st)
    bank_w = C.kd.window(EMB, st, T, T)
    names = fam_all                                   # retrieval reports the FAMILY
    banks = {}
    for nm, m in C.models.items():
        ints = []
        for i in range(0, n_bank, 128):
            wb = bank_w[i:i + 128].to(C.dev).float()
            e = m.encode(wb, wb[:, 0], wb[:, -1], torch.ones(wb.shape[0], device=C.dev))
            pi, _ = m.token_ints(e)
            ints.append(pi.reshape(pi.shape[0], -1).cpu())
        banks[nm] = torch.cat(ints)
    kf_idx = np.round(np.linspace(0, T - 1, K)).astype(int)
    for fam in ("kick", "jacks", "walk"):
        w = C.win(fam, T, g).to(C.dev).float()
        cur0, tgt0 = w[:, 0], w[:, -1]
        kn = torch.ones(1, device=C.dev)
        # piecewise-linear sketch through the K keyframes
        m0 = next(iter(C.models.values()))
        segs = []
        for i in range(K - 1):
            L = kf_idx[i + 1] - kf_idx[i] + 1
            segs.append(m0.baseline(w[:, kf_idx[i]], w[:, kf_idx[i + 1]],
                                    kn, L)[:, :-1] if i < K - 2 else
                        m0.baseline(w[:, kf_idx[i]], w[:, kf_idx[i + 1]], kn, L))
        sketch = torch.cat(segs, 1)
        pans = [C.fk(w[0]), C.fk(sketch[0].cpu())]
        ttl = ["GT", f"sketch ({K} keyframes)"]
        for nm, m in C.models.items():
            e = m.encode(sketch, sketch[:, 0], sketch[:, -1], kn)
            o = m.decode(e["path_z"], e["time_z"], sketch[:, 0], sketch[:, -1], kn, T)
            p = to_states(o, m, sketch[:, 0], sketch[:, -1], kn).float()
            pans.append(C.fk(p[0].cpu())); ttl.append(f"{nm} fill")
            def retrieve(ints):
                q2 = ints.reshape(1, -1).cpu()
                dist = (banks[nm] != q2).float().mean(1)
                top = torch.topk(-dist, 5).indices.tolist()
                return [names[t] for t in top], round(float(dist[top[0]]), 3)
            pi, _ = m.token_ints(e)
            fams_top, d1 = retrieve(pi)
            # CONTROL: the GT window's own tokens. If even THESE do not retrieve the
            # right family, token-space retrieval itself is broken and the sketch row
            # above says nothing about sketches.
            eg = m.encode(w, cur0, tgt0, kn)
            pg, _ = m.token_ints(eg)
            fams_gt, dg = retrieve(pg)
            rows[f"m5 {fam} | {nm}"] = {
                "sketch_top5_fam": fams_top, "sketch_hit": fam in fams_top[:3],
                "GTctrl_top5_fam": fams_gt, "GTctrl_hit": fam in fams_gt[:3],
                "hamming_sketch/GT": [d1, dg]}
        paths.append(C.animate(R7.deyaw(pans, _R0(w)), ttl, f"m5_{fam}",
                     note=f"m5: {K} keyframes -> sketch -> tokens -> fill. Token also "
                          f"used to retrieve the nearest dataset motion (see verdict)."))


# =========================================================================== m6
@torch.no_grad()
def m6_speed(C, rows, paths, g, T=60):
    """Same path tokens, different clock. Two knobs:
      T_out sweep (both arms): decode the same tokens at 0.5x/1x/2x duration.
      time-code swap (jepa only): path tokens from clip A + TIME tokens re-encoded
        from a slowed copy of A. If path/time are separated, the output traces A's
        path at the donor's timing.
    Verdict metric = a RESPONSE CURVE, not a GT error: realized mean speed must fall
    as 1/duration (monotone), while path SHAPE drift across variants stays flat."""
    for fam in ("kick", "run"):
        w = C.win(fam, T, g).to(C.dev).float()
        cur, tgt, kn = w[:, 0], w[:, -1], torch.ones(1, device=C.dev)
        for nm, m in C.models.items():
            e = m.encode(w, cur, tgt, kn)
            pans, ttl, shapes, speeds = [C.fk(w[0])], ["GT (1x)"], [], []
            for T_out, lab in ((30, "0.5x"), (60, "1x"), (120, "2x")):
                o = m.decode(e["path_z"], e["time_z"], cur, tgt, kn, T_out)
                p = to_states(o, m, cur, tgt, kn).float()
                fkp = C.fk(p[0].cpu())
                shapes.append(path_shape(fkp))
                d = np.linalg.norm(np.diff(fkp, axis=0), axis=-1).mean()
                speeds.append(d * 30)                      # cm/s at 30 fps
                hold = np.concatenate([fkp, np.repeat(fkp[-1:], 120 - T_out, 0)]) \
                    if T_out < 120 else fkp
                pans.append(hold); ttl.append(f"{nm} {lab}")
            drift = float(np.mean([np.linalg.norm(shapes[i] - shapes[1], axis=-1).mean()
                                   for i in (0, 2)]))
            rows[f"m6 {fam} Tsweep | {nm}"] = {
                "speed_cm_s @0.5x/1x/2x": [round(s, 1) for s in speeds],
                "speed_monotone": bool(speeds[0] > speeds[1] > speeds[2]),
                "path_shape_drift_cm": round(drift, 2)}
            paths.append(C.animate(R7.deyaw(pans, _R0(w)), ttl, f"m6_{fam}_{nm}",
                         note=f"m6: SAME tokens decoded at 0.5x/1x/2x duration; fast "
                              f"variants freeze when done. Same path, different clock?"))
        # jepa-only: time-code transplant
        if "V7_jepa" in C.models:
            m = C.models["V7_jepa"]
            # SAME path, NON-UNIFORMLY re-timed (a 2x-then-decimate roundtrips to the
            # identical frames — the first version of this measured drift 0.000 for
            # exactly that reason). warp_fidx bends the clock inside the same span.
            fid = torch.as_tensor(
                V.warp_fidx(T, T, np.random.default_rng(5), strength=1.0),
                device=w.device, dtype=w.dtype)
            lo = fid.long().clamp(max=T - 2); u = (fid - lo)[None, :, None, None]
            warped = V.orthonormalize6d((1 - u) * w[:, lo] + u * w[:, lo + 1])
            ef = m.encode(w, cur, tgt, kn)
            es = m.encode(warped, cur, tgt, kn)                # same path, bent clock
            o = m.decode(ef["path_z"], es["time_z"], cur, tgt, kn, T)
            p = to_states(o, m, cur, tgt, kn).float()
            fkp = C.fk(p[0].cpu())
            base = C.fk(to_states(m.decode(ef["path_z"], ef["time_z"], cur, tgt, kn, T),
                                  m, cur, tgt, kn).float()[0].cpu())
            rows[f"m6 {fam} time-swap | V7_jepa"] = {
                "shape_drift_vs_own_cm": round(float(np.linalg.norm(
                    path_shape(fkp) - path_shape(base), axis=-1).mean()), 2),
                "speed_ratio": round(float(
                    np.linalg.norm(np.diff(fkp, axis=0), axis=-1).mean() /
                    max(np.linalg.norm(np.diff(base, axis=0), axis=-1).mean(), 1e-9)), 3)}


# ========================================================================= main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="V7_jepa_0807,V7_maskfsq_0807")
    ap.add_argument("--modes", default="m1,m2,m3,m5,m6")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=2027)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    C = Ctx([r for r in args.runs.split(",") if r], args.device)
    rows, paths = {}, []
    g = np.random.default_rng(args.seed)
    fns = {"m1": m1_codec, "m2": m2_prior, "m3": m3_extrap,
           "m5": m5_vocab, "m6": m6_speed}
    for mo in args.modes.split(","):
        print(f"\n===== {mo} =====", flush=True)
        try:
            fns[mo](C, rows, paths, g)
        except Exception:
            import traceback; traceback.print_exc()
    L = ["=" * 100, "EXAM C — generative-capability exam (visual gate FIRST; these "
         "numbers are the second gate)", "=" * 100]
    for k in sorted(rows):
        L.append(f"{k:<44} {json.dumps(rows[k], ensure_ascii=False)}")
    txt = "\n".join(L)
    print("\n" + txt, flush=True)
    open(f"{V.RUNS}/EXAMC_verdict.txt", "w").write(txt + "\n")
    json.dump({"rows": rows, "gifs": paths},
              open(f"{V.RUNS}/EXAMC_verdict.json", "w"), indent=1)
    print(f"\nverdict: {V.RUNS}/EXAMC_verdict.txt", flush=True)


if __name__ == "__main__":
    main()
