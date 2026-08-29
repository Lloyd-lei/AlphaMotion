"""V0-a — THE CORE-ASSUMPTION FALSIFICATION. No training, no learned temporal layer.

ARCH_V5 stakes everything on one sentence: "其余关节由冻结的空间层补全,实测误差
0.8cm". That 0.80 cm was measured on a SINGLE FRAME with 5 SE3 constraints (31 other
joints -> a 36-joint body, i.e. G1). A temporal layer is not a single frame. This
script asks the only question that matters before anything is built:

    given ONLY the 6 key joints' SE3 trajectories over a 2 s window, can the FROZEN
    spatial layer (E19_K128_0805) reconstruct the full-body MOTION — and does it
    jitter?

Method (nothing learned, nothing fitted):
  GT window -> take the 6 key joints (root, head, both wrists, both ankles)
            -> mask every other joint, PER FRAME, into E19's masked-encode path
            -> decode on the target skeleton -> manifold_project (joint limits)
            -> compare against GT, per frame and across time.

THREE conditions, because "0.8 cm" and "what V5 needs" are not the same measurement:
  self_g1     G1's own 6 key joints -> G1 full body.  This is the setting the 0.80 cm
              reference number was measured in (31 other joints), so it is the row
              that either reproduces it or does not.
  self_human  human's 6 key joints -> human full body. Isolates the completion from
              any retargeting.
  cross_h2g1  human's 6 key joints -> G1 full body. This is what V5 actually does at
              deploy time: one temporal token stream, decoded onto another body.

and for every condition a FULL-OBSERVATION FLOOR (mask NOTHING, all joints visible)
— without it "8 cm" cannot be attributed: it might be the masking, or it might just
be what frozen E19 costs on this cache. Only floor-vs-masked prices the assumption.

TWO position metrics, because they answer different questions:
  pos_dec_cm  the decoder's own position stream vs GT positions. This is the metric
              the 0.80 cm reference used, so it is the comparable one.
  pos_fk_cm   positions FK'd from the decoded ROTATIONS with rigid bones. This is
              what actually gets rendered (p1_fkfix: rendering the raw position
              stream gave rubber bones and cost a day), so it is the honest one.

SMOOTHNESS is the point of the probe as much as accuracy: the completion is
INDEPENDENT per frame and has no temporal prior whatsoever, so jitter is the
EXPECTED failure mode and is priced explicitly — step median/p95/max, acceleration
(the sharp one: smooth motion has small acceleration even when fast), and the
fraction of steps that reverse direction (white jitter -> 0.5).

  python v5_probe_keyjoints.py            (defaults, ~5-10 min on GPU)
  python v5_probe_keyjoints.py --tiny     (CPU smoke)
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v5_common as V                                        # noqa: E402
import manifold_project as MP                                # noqa: E402
from common import rot6d_to_matrix, geodesic_deg             # noqa: E402

OUT_JSON = f"{V.RUNS}/V5_v0a_keyjoints.json"
HUM, G1 = "human_smpl", "unitree_g1_29dof"
# (tag, source body whose key joints are given, target body to complete)
CONDITIONS = [("self_g1", G1, G1), ("self_human", HUM, HUM), ("cross_h2g1", HUM, G1)]


# --------------------------------------------------------------------------- helpers
def project_limits(rot6d, spec, cache, emb, soft_margin=1.0, lm_iters=25):
    """manifold_project onto the target's mechanical manifold. -> (rot6d, stats)."""
    dof, rest = MP.load_dof(cache, emb, dtype=torch.float64)
    Rg = rot6d_to_matrix(torch.as_tensor(np.asarray(rot6d), dtype=torch.float64))
    q_raw, _ = MP.fit_angles(Rg, spec, dof, rest=rest, clamp=False, method="greedy")
    lo, hi, lim = MP._limits(dof, soft_margin)
    A, act = MP._slot_axes(dof)
    m = (lim & act)[None]
    nlim = max(int(m.sum()) * q_raw.shape[0], 1)
    over = (q_raw - hi[None]).clamp_min(0) + (lo[None] - q_raw).clamp_min(0)
    pre = 100.0 * float(((over > 1e-9) & m).sum()) / nlim
    worst = float(torch.rad2deg(over[m.expand_as(over)].max())) if int(m.sum()) else 0.0
    r6, _, q = MP.project(Rg, spec, dof, soft_margin=soft_margin, rest=rest, clamp=True,
                          method="global", lm_iters=lm_iters)
    over2 = (q - hi[None]).clamp_min(0) + (lo[None] - q).clamp_min(0)
    post = 100.0 * float(((over2 > 1e-9) & m).sum()) / nlim
    moved = float(geodesic_deg(rot6d_to_matrix(r6), Rg).mean())
    return r6.float(), {"pre_viol_pct": round(pre, 3), "pre_worst_deg": round(worst, 1),
                        "post_viol_pct": round(post, 5), "moved_deg": round(moved, 2)}


@torch.no_grad()
def complete(tok, src_ent, tgt_ent, pose_src, kidx, dev, masked=True, bs=128):
    """pose_src [n,J_s,9] -> (rot6d [n,J_t,6], pos [n,J_t,3] reach units).
    masked=True hides every joint except kidx (the V5 contract); masked=False is the
    FULL-OBSERVATION FLOOR — frozen E19 with nothing hidden."""
    J = src_ent["spec"].J
    ki = torch.as_tensor(kidx, dtype=torch.long)
    rr, pp = [], []
    for i in range(0, pose_src.shape[0], bs):
        x = pose_src[i:i + bs].to(dev).float()
        b = x.shape[0]
        m = None
        if masked:
            m = torch.ones(b, J, dtype=torch.bool, device=dev); m[:, ki] = False
        zp, zr = tok.encode(x, V.geom_for(src_ent, b), joint_mask=m)
        rot, pos = tok.decode(zp, zr, V.geom_for(tgt_ent, b))
        rr.append(rot.cpu()); pp.append(pos.cpu())
    return torch.cat(rr), torch.cat(pp)


def pick_windows(kd, families, n_clips, n_win, T):
    """-> [(family, clip_id, start)] with real family coverage, val clips preferred."""
    out = []
    for fam in families:
        cand = [c for c in range(len(kd.clips))
                if kd.family[c] == fam and kd.clips[c]["len"] >= T]
        cand.sort(key=lambda c: (0 if c in kd.val_clips else 1, -kd.clips[c]["len"]))
        for c in cand[:n_clips]:
            s, L = kd.clips[c]["start"], kd.clips[c]["len"]
            for st in np.linspace(s, s + L - T, n_win).astype(np.int64):
                out.append((fam, c, int(st)))
    return out


def measure(rot, pos, gt, spec, kj, reach, project=None):
    """One decoded window -> the full metric block."""
    stats = {}
    if project is not None:
        rot, stats = project
    p_dec = pos.numpy() * reach
    p_fk = V.fk_pos(rot, spec)
    g_dec = gt[..., 6:9].numpy() * reach
    g_fk = V.fk_pos(gt[..., :6], spec)
    other = [j for j in range(spec.J) if j not in kj]
    e_dec = np.linalg.norm(p_dec - g_dec, axis=-1)
    e_fk = np.linalg.norm(p_fk - g_fk, axis=-1)
    rerr = geodesic_deg(rot6d_to_matrix(torch.as_tensor(np.asarray(rot))),
                        rot6d_to_matrix(gt[..., :6])).numpy()
    e = {"pos_dec_cm": float(e_dec[:, other].mean()),
         "pos_fk_cm": float(e_fk[:, other].mean()),
         "pos_fk_p95_cm": float(np.quantile(e_fk[:, other], .95)),
         "rot_deg": float(rerr[:, other].mean()),
         "key_pos_dec_cm": float(e_dec[:, kj].mean()),
         "key_rot_deg": float(rerr[:, kj].mean()),
         "energy_pred_cm": V.motion_energy(p_fk), "energy_gt_cm": V.motion_energy(g_fk),
         "jit_pred": V.jitter_stats(p_fk), "jit_gt": V.jitter_stats(g_fk)}
    e["energy_ratio"] = e["energy_pred_cm"] / max(e["energy_gt_cm"], 1e-6)
    for k in ("step_med_cm", "step_p95_cm", "step_max_cm", "accel_mean_cm",
              "accel_p95_cm"):
        e[f"ratio_{k}"] = e["jit_pred"][k] / max(e["jit_gt"][k], 1e-6)
    e["rev_frac_pred"] = e["jit_pred"]["rev_frac"]
    e["rev_frac_gt"] = e["jit_gt"]["rev_frac"]
    e.update({f"proj_{k}": v for k, v in stats.items()})
    return e


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=V.AMASS)
    ap.add_argument("--frozen", default=V.FROZEN)
    ap.add_argument("--T", type=int, default=V.T_WIN)
    ap.add_argument("--families", default=",".join(V.V5_FAMILIES))
    ap.add_argument("--n_clips", type=int, default=3)
    ap.add_argument("--n_win", type=int, default=2)
    ap.add_argument("--no_project", action="store_true")
    ap.add_argument("--lm_iters", type=int, default=25)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke: 2 families, 1 window")
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    if args.tiny:
        args.families, args.n_clips, args.n_win = "walk,punch", 1, 1
        args.T, args.device, args.lm_iters = 20, "cpu", 4
    t0, dev = time.time(), args.device
    fams = [f for f in args.families.split(",") if f]

    print("=== V5 V0-a  key-joint trajectory -> full-body motion (NO TRAINING) ===")
    print(f"frozen spatial layer: {args.frozen}")
    kd = V.KeyData(args.cache, [HUM, G1], device=dev)
    tok, fcfg = V.load_frozen(dev, args.frozen)
    ent = {b: V.body_entry(args.cache, b, dev, kd.reach[b]) for b in (HUM, G1)}
    held = [x for x in str(fcfg.get("holdout", "")).split(",") if x]
    print(f"key joints ({len(kd.kidx[HUM])}/{kd.spec[HUM].J} human, "
          f"{len(kd.kidx[G1])}/{kd.spec[G1].J} G1)")
    print(f"  human: {kd.knames[HUM]}")
    print(f"  G1   : {kd.knames[G1]}")
    print(f"E19 trained on {os.path.basename(fcfg.get('cache',''))}; this probe runs on "
          f"{os.path.basename(args.cache)}; E19 holdout bodies = {held}"
          f"{'  (G1 IS HELD OUT — zero-shot)' if G1 in held else ''}")
    print(f"reference: 5-constraint single frame, 31 other joints, G1 vs GT = "
          f"{V.REF['constraint5_pos_cm_g1']} cm", flush=True)

    wins = pick_windows(kd, fams, args.n_clips, args.n_win, args.T)
    print(f"windows: {len(wins)}  " +
          str({f: sum(1 for w in wins if w[0] == f) for f in fams}), flush=True)
    assert wins, "no windows matched the requested families"

    per = []
    for wi, (fam, cid, st) in enumerate(wins):
        rows = np.arange(st, st + args.T)
        rec = {"family": fam, "clip": kd.clips[cid]["name"][:70], "start": int(st),
               "val_clip": bool(cid in kd.val_clips)}
        gt = {b: V.full_pose_of(args.cache, b, rows, dev, kd.reach[b])[0]
              for b in (HUM, G1)}
        for tag, sb, tb in CONDITIONS:
            spec, kj, reach = kd.spec[tb], kd.kidx[tb], kd.reach[tb]
            for mode in ("masked", "floor"):
                rot, pos = complete(tok, ent[sb], ent[tb], gt[sb], kd.kidx[sb], dev,
                                    masked=(mode == "masked"))
                pr = None
                if not args.no_project:
                    pr = project_limits(rot, spec, args.cache, tb,
                                        lm_iters=args.lm_iters)
                rec[f"{tag}/{mode}"] = measure(rot, pos, gt[tb], spec, kj, reach, pr)
        per.append(rec)
        if wi % 4 == 0 or wi + 1 == len(wins):
            c = rec["cross_h2g1/masked"]
            print(f"  [{wi+1}/{len(wins)}] {fam:6s} cross_h2g1 dec "
                  f"{c['pos_dec_cm']:5.2f}cm fk {c['pos_fk_cm']:5.2f}cm energy "
                  f"{c['energy_ratio']:.2f} accelx {c['ratio_accel_mean_cm']:.2f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ aggregate
    KEYS = ["pos_dec_cm", "pos_fk_cm", "pos_fk_p95_cm", "rot_deg", "key_pos_dec_cm",
            "key_rot_deg", "energy_gt_cm", "energy_pred_cm", "energy_ratio",
            "ratio_step_med_cm", "ratio_step_p95_cm", "ratio_accel_mean_cm",
            "ratio_accel_p95_cm", "rev_frac_pred", "rev_frac_gt",
            "proj_pre_viol_pct", "proj_post_viol_pct", "proj_moved_deg"]
    cols = [f"{t}/{m}" for t, _, _ in CONDITIONS for m in ("masked", "floor")]

    def agg(sel, col, k):
        v = [r[col][k] for r in sel if k in r[col] and r[col][k] is not None]
        return round(float(np.mean(v)), 4) if v else None

    res = {"probe": "V0-a key-joint trajectory -> full body (frozen spatial layer)",
           "frozen": args.frozen, "frozen_train_cache": fcfg.get("cache"),
           "frozen_holdout": held, "cache": args.cache, "T": args.T,
           "key_joints": {HUM: kd.knames[HUM], G1: kd.knames[G1]},
           "conditions": {t: f"{s} key joints -> {g} full body" for t, s, g in CONDITIONS},
           "n_windows": len(wins), "reference": V.REF,
           "note": "pos_dec_cm = decoder position stream (the metric the 0.80 cm "
                   "reference used). pos_fk_cm = FK from decoded ROTATIONS, rigid "
                   "bones — what actually gets rendered. 'floor' = the SAME frozen "
                   "encoder with NOTHING masked, so masked-minus-floor is the price "
                   "of key-joint-only conditioning. The completion is per frame and "
                   "has no temporal prior, so jitter is the expected failure mode.",
           "by_family": {}, "per_window": per}
    for fam in fams + ["ALL"]:
        sel = per if fam == "ALL" else [r for r in per if r["family"] == fam]
        if not sel:
            continue
        res["by_family"][fam] = {"n": len(sel)} | {
            c: {k: agg(sel, c, k) for k in KEYS} for c in cols}

    # ------------------------------------------------------------------- tables
    A = res["by_family"]["ALL"]
    print("\n" + "=" * 104)
    print("V0-a TABLE 1 — accuracy, ALL families. 'floor' = same encoder, nothing "
          "masked (E19's own cost).")
    print(f"{'condition':>22s} | {'pos_dec cm':>10s} {'pos_fk cm':>9s} {'rot deg':>7s} "
          f"| {'key dec cm':>10s} {'key rot':>7s} | {'limits pre%':>11s}")
    for t, _, _ in CONDITIONS:
        for m in ("masked", "floor"):
            b = A[f"{t}/{m}"]
            print(f"{t + '/' + m:>22s} | {b['pos_dec_cm']:10.2f} {b['pos_fk_cm']:9.2f} "
                  f"{b['rot_deg']:7.1f} | {b['key_pos_dec_cm']:10.2f} "
                  f"{b['key_rot_deg']:7.1f} | "
                  f"{(b['proj_pre_viol_pct'] if b['proj_pre_viol_pct'] is not None else float('nan')):11.3f}")
    print("=" * 104)

    print("\nV0-a TABLE 2 — MOTION over time, masked condition, per family "
          "(energy = FK-position temporal std ratio vs GT)")
    print(f"{'family':7s} {'n':>3s} | {'E self_g1':>9s} {'E self_hum':>10s} "
          f"{'E cross':>8s} | {'accel x':>8s} {'step p95 x':>10s} {'rev pred/gt':>12s} "
          f"| {'cross dec cm':>12s}")
    for fam in fams + ["ALL"]:
        if fam not in res["by_family"]:
            continue
        b = res["by_family"][fam]
        g, h, c = b["self_g1/masked"], b["self_human/masked"], b["cross_h2g1/masked"]
        print(f"{fam:7s} {b['n']:3d} | {g['energy_ratio']:9.2f} "
              f"{h['energy_ratio']:10.2f} {c['energy_ratio']:8.2f} | "
              f"{c['ratio_accel_mean_cm']:8.2f} {c['ratio_step_p95_cm']:10.2f} "
              f"{c['rev_frac_pred']:5.2f}/{c['rev_frac_gt']:.2f} | "
              f"{c['pos_dec_cm']:12.2f}")
    print("=" * 104)

    # ------------------------------------------------------------------- verdict
    ref = V.REF["constraint5_pos_cm_g1"]
    sg, sgf = A["self_g1/masked"], A["self_g1/floor"]
    ch = A["cross_h2g1/masked"]
    hm = A["self_human/masked"]
    repro = sg["pos_dec_cm"] <= 2.0 * ref                 # reproduces the 0.80 cm claim
    mask_cost = sg["pos_dec_cm"] - sgf["pos_dec_cm"]      # what masking actually costs
    ener_ok = min(sg["energy_ratio"], hm["energy_ratio"], ch["energy_ratio"]) >= 0.80
    jit_bad = (max(sg["ratio_accel_mean_cm"], ch["ratio_accel_mean_cm"]) > 1.5
               or ch["rev_frac_pred"] > ch["rev_frac_gt"] + 0.10)
    res["verdict"] = {
        "reproduces_0p80cm_reference": bool(repro),
        "self_g1_pos_dec_cm": sg["pos_dec_cm"], "self_g1_floor_pos_dec_cm": sgf["pos_dec_cm"],
        "masking_cost_cm": round(mask_cost, 3),
        "motion_energy_ok": bool(ener_ok), "jitters": bool(jit_bad),
        "bar_pos_cm": round(2 * ref, 2), "bar_energy_ratio": 0.80}

    print("\nVERDICT (plain words)")
    print(f"  1. same-body completion on G1, the reference setting: "
          f"{sg['pos_dec_cm']:.2f} cm on the other joints "
          f"(reference {ref:.2f} cm; E19's own floor with NOTHING masked is "
          f"{sgf['pos_dec_cm']:.2f} cm, so key-joint-only conditioning itself costs "
          f"{mask_cost:+.2f} cm) -> {'REPRODUCES' if repro else 'DOES NOT REPRODUCE'}")
    print(f"  2. what V5 actually needs (human key joints -> G1 body): "
          f"{ch['pos_dec_cm']:.2f} cm decoder-stream / {ch['pos_fk_cm']:.2f} cm "
          f"FK-from-rotation")
    print(f"  3. whole-body MOTION retained over the window: self_g1 "
          f"{sg['energy_ratio']*100:.0f}%, self_human {hm['energy_ratio']*100:.0f}%, "
          f"cross {ch['energy_ratio']*100:.0f}% of GT motion energy -> "
          f"{'PRESERVED' if ener_ok else 'NOT PRESERVED'}")
    print(f"  4. temporal smoothness: acceleration {ch['ratio_accel_mean_cm']:.2f}x GT, "
          f"step p95 {ch['ratio_step_p95_cm']:.2f}x GT, direction reversals "
          f"{ch['rev_frac_pred']:.2f} vs GT {ch['rev_frac_gt']:.2f} -> "
          f"{'JITTERS' if jit_bad else 'smooth enough'}")
    print(f"  SO: key-joint-only conditioning "
          f"{'DOES' if (repro and ener_ok) else 'DOES NOT'} preserve whole-body "
          f"motion, and it {'DOES' if jit_bad else 'does NOT'} jitter.")

    msg = []
    if not repro:
        msg.append(f"CORE ASSUMPTION NOT REPRODUCED: 6 key joints -> G1 body gives "
                   f"{sg['pos_dec_cm']:.2f} cm on the other joints, not "
                   f"{ref:.2f} cm. E19's unmasked floor on this cache is "
                   f"{sgf['pos_dec_cm']:.2f} cm, so "
                   + ("the MASKING is what costs it."
                      if mask_cost > 0.5 * sg['pos_dec_cm'] else
                      "most of the gap is E19's own cost on this data, NOT the masking."))
    if not ener_ok:
        msg.append(f"CORE ASSUMPTION FALSIFIED: whole-body MOTION is not preserved "
                   f"(energy self_g1 {sg['energy_ratio']:.2f}, self_human "
                   f"{hm['energy_ratio']:.2f}, cross {ch['energy_ratio']:.2f}). "
                   f"ARCH_V5 §2 '降 60 倍是安全的' does not survive the temporal test.")
    if jit_bad:
        msg.append(f"EXPECTED AND CONFIRMED: the per-frame completion JITTERS "
                   f"(accel {ch['ratio_accel_mean_cm']:.2f}x GT, direction reversals "
                   f"{ch['rev_frac_pred']:.2f} vs GT {ch['rev_frac_gt']:.2f}). The "
                   f"spatial layer supplies NO temporal prior; V5 must add one on "
                   f"top of the completion (or filter it) before anything renders.")
    if msg:
        V.loud(msg)
    else:
        V.loud(["V0-a: the core assumption HOLDS on all three axes "
                "(position, motion energy, smoothness)."], mark="===")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=1, default=float)
    print(f"SAVED {args.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
