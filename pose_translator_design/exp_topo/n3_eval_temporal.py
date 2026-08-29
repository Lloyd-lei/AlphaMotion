"""N3 TEMPORAL VERDICT — V5c_0806 (control) vs N3a_regAR / N3b_xbody / N3c_root.

One table per the N3 queue contract (-> runs/N3_temporal_verdict.txt + .json):
  (1) the 2c warp exam  — same-span warped-readout path-token invariance
      (warp_exam.py recipe verbatim: seed-13 windows, warps seeded 1000..1005;
      oracle = centered arclen descriptor, 0.961; gate >= 0.90)
  (2) content share     — stat8_ladder's shuffled-token control (gate >= 0.6),
      plus key cm (gate <= 7.5) and the full OAT prefix ladder
  (3) motion energy per family (walk/run/jacks/dance/kick/punch), with
      root_vel-aware extras for arms trained with the side channel (vel-channel
      speed-magnitude ratio + per-window mean-forward-speed correlation)
  (4) kNN body leakage  — eval_v5 stat7 (P1 reference 0.141)
  (5) GT-FREE foot slip — decoded G1 trajectories through the frozen spatial
      layer -> manifold projection -> MuJoCo qpos -> physeval, same LAFAN clips
      and root-world convention as runs/PHYSEVAL_report.md, so numbers line up
      with the 0806 baseline table (V5c slip p50 there: 0.751 m/s).

Measurement guards:
  * every invariance/content metric is measured on the HUMAN stream — an
    xbody-trained model decodes with its default (index-0 = human) body token,
    so N3b's numbers are directly comparable to V5c's.
  * arms trained with --root_vel get their OWN KeyData (7-channel); pal_talos is
    excluded from that leakage probe (no root_world side channel), so its chance
    level is 1/5 not 1/6 and is printed as such.
Every section is independently guarded; a broken stat never costs the others.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
NIGHT2_LIB = "/var/tmp/alphamotion/topo_exp/night2_lib"
if os.path.isdir(NIGHT2_LIB):
    sys.path.append(NIGHT2_LIB)

import v5_common as V                                         # noqa: E402
from train_v5 import load_run, to_states                      # noqa: E402
from eval_v5 import (tokens_of, cos_rows, stat7_leakage,      # noqa: E402
                     stat8_ladder, R7_BODIES)

RUNS = V.RUNS
ARMS = ["V5c_0806", "N3a_regAR_0807", "N3b_xbody_0807", "N3c_root_0807"]
EMB = "human_smpl"
G1 = "unitree_g1_29dof"
PHYS_CLIPS = ["walk1_subject1", "walk3_subject1", "run1_subject2", "dance1_subject1"]
GATES = {"warp_2c": 0.90, "content_share": 0.60, "key_cm": 7.5}


# ------------------------------------------------------------------ 2c warp exam
def arclen_descriptor(w, M=60):
    """Oracle descriptor (warp_exam.py verbatim): positions arc-length-resampled
    to M points, per-window mean removed. 6-channel windows only."""
    x = w[..., 6:9].double().numpy()
    n, Tn, K, C = x.shape
    flat = x.reshape(n, Tn, -1)
    seg = np.linalg.norm(np.diff(flat, axis=1), axis=-1)
    s = np.concatenate([np.zeros((n, 1)), np.cumsum(seg, axis=1)], 1)
    out = np.empty((n, M, K * C))
    for i in range(n):
        ti = (np.linspace(0, Tn - 1, M) if s[i, -1] < 1e-9 else
              np.interp(np.linspace(0, s[i, -1], M), s[i], np.arange(Tn)))
        i0 = np.floor(ti).astype(int); i1 = np.minimum(i0 + 1, Tn - 1)
        a = (ti - i0)[:, None]
        out[i] = (1 - a) * flat[i, i0] + a * flat[i, i1]
    d = torch.from_numpy(out).float()
    return d - d.mean(dim=1, keepdim=True)


def warp_windows(kd, n, T, span, k_warps=6, seed=13):
    g = np.random.default_rng(seed)                     # stat2_speed's exact windows
    st, _ = kd.sample_starts(n, g, "val", span=2 * span + 8)
    base = kd.window(EMB, st, T, span)
    warps = []
    for k in range(k_warps):
        rng = np.random.default_rng(1000 + k)
        warps.append(kd.window_fidx(EMB, st, V.warp_fidx(span, T, rng), span))
    perm = torch.from_numpy(np.random.default_rng(99).permutation(n))
    return base, warps, perm


def exam_2c(model, kd, dev, n, T, span):
    base, warps, perm = warp_windows(kd, n, T, span)
    p0 = tokens_of(model, base, dev)[0]
    cs = [float(cos_rows(p0, tokens_of(model, w, dev)[0]).mean()) for w in warps]
    return {"control_unrelated": round(float(cos_rows(p0, p0[perm]).mean()), 4),
            "warp_frac_mean": round(float(np.mean(cs)), 4),
            "warp_frac_per": [round(c, 4) for c in cs]}


# ------------------------------------------------------------------ family energy
@torch.no_grad()
def family_energy(model, kd, dev, T, span, n=64, seed=41):
    g = np.random.default_rng(seed)
    out = {}
    root_vel = kd.key[EMB].shape[1] == V.NK_VEL
    for fam in V.V5_FAMILIES:
        try:
            st, _ = kd.sample_starts(n, g, "val", span=span, families=[fam])
        except AssertionError:
            continue
        w = kd.window(EMB, st, T, span).to(dev).float()
        cur, tgt = w[:, 0], w[:, -1]
        kn = torch.ones(w.shape[0], device=dev)
        o, _ = model(w, cur, tgt, kn)
        p = to_states(o, model, cur, tgt, kn).float()
        pe, re = V.key_se3_err(p, w, kd.reach[EMB])
        row = {"energy_ratio": round(float(p.std(1).mean() / w.std(1).mean()), 3),
               "key_pos_cm": round(pe, 2), "key_rot_deg": round(re, 1)}
        if root_vel:
            # root-vel-aware extras: does the decoded side channel carry the
            # LOCOMOTION the articulated channels cannot see?
            pv, gv = p[..., -1, 6:9], w[..., -1, 6:9]        # leg-lengths/s
            row["vel_mag_ratio"] = round(
                float(pv.norm(dim=-1).mean() / gv.norm(dim=-1).mean().clamp_min(1e-6)), 3)
            a = pv[..., 2].mean(1).cpu().numpy()             # mean forward speed
            b = gv[..., 2].mean(1).cpu().numpy()
            row["fwd_speed_corr"] = (round(float(np.corrcoef(a, b)[0, 1]), 3)
                                     if a.std() > 1e-6 and b.std() > 1e-6 else 0.0)
        out[fam] = row
    return out


# ------------------------------------------------------------------ foot slip
@torch.no_grad()
def recon_key_traj(model, key, dev):
    """[n,nk,9] GT key stream -> full-prefix window-by-window reconstruction."""
    Tw = model.T
    segs = []
    for i in range(0, key.shape[0], Tw):
        w = key[i:i + Tw]
        if w.shape[0] < 4:
            segs.append(w.cpu()); continue
        ww = w[None].to(dev).float()
        cur, tgt = ww[:, 0], ww[:, -1]
        kn = torch.ones(1, device=dev)
        o, _ = model(ww, cur, tgt, kn)
        segs.append(to_states(o, model, cur, tgt, kn)[0].float().cpu())
    return torch.cat(segs, 0)[:key.shape[0]]


def foot_slip(models, dev, frames=240, start_sec=2.0, tiny=False):
    """models: {tag: (model, cfg)}. -> {tag: {clip: headline}} + GT_retarget row.
    Same clips / GT-root-world / projection settings as runs/PHYSEVAL_report.md."""
    import physeval_run as PR                    # staged validated pipeline
    import physeval as PE
    from common import fk_global, matrix_to_rot6d
    LAFAN = V.LAFAN
    seqs = {s["name"]: s for s in json.load(open(f"{LAFAN}/sequences.json"))}
    clips = PHYS_CLIPS[:1] if tiny else PHYS_CLIPS
    body = PR.body_of(G1, LAFAN)
    tok, _ = V.load_frozen(dev)
    hent = V.body_entry(LAFAN, EMB, dev)
    gent = V.body_entry(LAFAN, G1, dev)
    kds = {}
    for tag, (m, cfg) in models.items():
        rv = bool(cfg.get("root_vel", False))
        if rv not in kds:
            kds[rv] = V.KeyData(LAFAN, [EMB], device="cpu", root_vel=rv,
                                verbose=False)
    g1spec, _ = PR.spec_of(LAFAN, G1)
    cfgP = PE.PhysEvalConfig(fps=30.0)
    out = {t: {} for t in list(models) + ["GT_retarget"]}
    for clip in clips:
        s = seqs[clip]
        o0 = int(start_sec * 30)
        n = min(frames if not tiny else 62, s["len"] - o0)
        idx = s["start"] + o0 + np.arange(n)
        rw = PR.root_world_muj(LAFAN, G1, idx)
        # GT retarget row: the floor every decode is judged against
        gl6 = torch.load(f"{LAFAN}/{G1}_paired_local6d.pt")[idx]
        gR, _ = fk_global(gl6.to(dev), g1spec)
        gt_rot = matrix_to_rot6d(gR).float().cpu().numpy()
        qpos, _, _ = PR.to_qpos(body, gt_rot, rw)
        out["GT_retarget"][clip] = PE.evaluate(body["model"], qpos, cfgP,
                                               body["floor_gid"],
                                               label=f"{clip}/GT")["headline"]
        for tag, (m, cfg) in models.items():
            rv = bool(cfg.get("root_vel", False))
            kd = kds[rv]
            key = kd.key[EMB][torch.from_numpy(idx)]
            ktraj = recon_key_traj(m, key, dev)
            rot, _ = V.complete_from_key(tok, hent, gent, ktraj,
                                         kd.kidx[EMB], dev)
            qpos, _, _ = PR.to_qpos(body, np.asarray(rot), rw)
            out[tag][clip] = PE.evaluate(body["model"], qpos, cfgP,
                                         body["floor_gid"],
                                         label=f"{clip}/{tag}")["headline"]
            print(f"  [phys] {clip} {tag} slip50="
                  f"{out[tag][clip]['slip_p50_ms']:.3f}", flush=True)
    return out


def med(d, k):
    v = [c[k] for c in d.values() if k in c]
    return float(np.median(v)) if v else float("nan")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=",".join(ARMS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--n_ladder", type=int, default=256)
    ap.add_argument("--n_knn", type=int, default=250)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--no_phys", action="store_true")
    ap.add_argument("--out", default=f"{RUNS}/N3_temporal_verdict")
    args = ap.parse_args()
    if args.tiny:
        args.n, args.n_ladder, args.n_knn = 48, 32, 24
    dev, t0 = args.device, time.time()

    arms = []
    for a in [r for r in args.runs.split(",") if r]:
        rd = a if os.path.isabs(a) else f"{RUNS}/{a}"
        if os.path.exists(f"{rd}/model.pt"):
            arms.append((a, rd))
        else:
            print(f"[skip] {a}: no model.pt", flush=True)
    if not arms:
        raise SystemExit("no finished arm found")

    # KeyData flavors: 6-channel always; 7-channel only if some arm used --root_vel
    cfgs = {a: json.load(open(f"{rd}/config.json")) for a, rd in arms}
    bodies6 = list(dict.fromkeys([EMB] + R7_BODIES))
    kd = {False: V.KeyData(V.AMASS, bodies6, device="cpu")}
    if any(c.get("root_vel") for c in cfgs.values()):
        b7 = [b for b in bodies6 if
              os.path.exists(f"{V.AMASS}/{b}_root_world.pt")]
        kd[True] = V.KeyData(V.AMASS, b7, device="cpu", root_vel=True)
        print(f"[kd7] bodies={b7} (pal_talos has no root_world)", flush=True)

    tok = ent = None
    try:
        tok, _ = V.load_frozen(dev)
        ent = V.body_entry(V.AMASS, EMB, dev, kd[False].reach[EMB])
    except Exception as e:
        print(f"WARN frozen spatial layer unavailable: {e}", flush=True)

    # oracle row for the 2c exam (data property, 6-channel descriptor)
    T0, S0 = V.T_WIN, V.T_WIN
    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "arms": {}, "gates": GATES}
    try:
        base, warps, perm = warp_windows(kd[False], args.n, T0, S0)
        dA = arclen_descriptor(base)
        res["oracle_2c"] = {
            "control": round(float(cos_rows(dA, dA[perm]).mean()), 4),
            "warp": round(float(np.mean([float(cos_rows(dA, arclen_descriptor(w)).mean())
                                         for w in warps])), 4)}
        print("oracle_2c", json.dumps(res["oracle_2c"]), flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        res["oracle_2c"] = {"error": str(e)}

    models = {}
    for a, rd in arms:
        print(f"\n===== {a}", flush=True)
        model, cfg = load_run(rd, dev)
        models[a] = (model, cfg)
        k = kd[bool(cfg.get("root_vel", False))]
        T, span = cfg.get("T", V.T_WIN), cfg.get("span", V.T_WIN)
        out = {"config_steps": cfg.get("steps"), "root_vel": bool(cfg.get("root_vel")),
               "reg_ar": bool(cfg.get("reg_ar")),
               "xbody_loss": float(cfg.get("xbody_loss", 0) or 0)}
        jobs = [
            ("warp_2c", lambda: exam_2c(model, k, dev, args.n, T, span)),
            ("ladder", lambda: stat8_ladder(model, k, EMB, dev, args.n_ladder,
                                            T, span, tok, ent)),
            ("family_energy", lambda: family_energy(model, k, dev, T, span,
                                                    n=16 if args.tiny else 64)),
            ("leakage", lambda: stat7_leakage(model, k, dev, args.n_knn, T, span)),
        ]
        for name, fn in jobs:
            try:
                out[name] = fn()
                print(f"  {name}: ok", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                out[name] = {"error": f"{type(e).__name__}: {e}"}
        res["arms"][a] = out

    if not args.no_phys:
        try:
            res["foot_slip"] = foot_slip(models, dev, tiny=args.tiny)
        except Exception as e:
            import traceback; traceback.print_exc()
            res["foot_slip"] = {"error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------- the table
    names = [a for a, _ in arms]
    W = max([len(n) for n in names] + [12]) + 2
    L = []
    A = L.append
    A("=" * 100)
    A("N3 TEMPORAL VERDICT — register-AR / xbody / root_vel vs V5c (control), "
      + time.strftime("%Y-%m-%d %H:%M"))
    A("controls: V5c recipe (warp aug linear + Riemann both + cdrop .15/.05 + "
      "time-detach + no pair-NCE)")
    A("=" * 100)

    def row(label, fn, fmt="{:.3f}"):
        vals = []
        for n_ in names:
            try:
                v = fn(res["arms"][n_])
                vals.append(fmt.format(v) if v is not None else "-")
            except Exception:
                vals.append("-")
        A(f"{label:42s}" + "".join(f"{v:>{W}s}" for v in vals))

    A(f"{'metric':42s}" + "".join(f"{n_:>{W}s}" for n_ in names))
    o = res.get("oracle_2c", {})
    A(f"-- (1) 2c warp exam [gate >= {GATES['warp_2c']}; oracle "
      f"{o.get('warp', '?')} / control {o.get('control', '?')}] --")
    row("  warp invariance (frac)", lambda r: r["warp_2c"]["warp_frac_mean"])
    row("  control (unrelated)", lambda r: r["warp_2c"]["control_unrelated"])
    A(f"-- (2) content / error [gates: share >= {GATES['content_share']}, "
      f"key cm <= {GATES['key_cm']}] --")
    row("  content share", lambda r: r["ladder"]["_token_content"]["content_share"])
    row("  registers buy [cm]", lambda r: r["ladder"]["_token_content"]["registers_buy_cm"])
    row("  key pos prefix8 [cm]", lambda r: r["ladder"]["prefix8"]["key_pos_cm"], "{:.2f}")
    row("  key rot prefix8 [deg]", lambda r: r["ladder"]["prefix8"]["key_rot_deg"], "{:.1f}")
    row("  energy ratio prefix8", lambda r: r["ladder"]["prefix8"]["energy_ratio"])
    for b in (1, 2, 4):
        row(f"  prefix{b} key pos [cm]", lambda r, b=b: r["ladder"][f"prefix{b}"]["key_pos_cm"], "{:.2f}")
    row("  tokens shuffled [cm]", lambda r: r["ladder"]["tokens_shuffled"]["key_pos_cm"], "{:.2f}")
    row("  tokens zeroed [cm]", lambda r: r["ladder"]["tokens_zeroed"]["key_pos_cm"], "{:.2f}")
    row("  straight line [cm]", lambda r: r["ladder"]["straight_line"]["key_pos_cm"], "{:.2f}")
    A("-- (3) motion energy per family (energy ratio; root_vel arms add "
      "vel-mag ratio / fwd-speed corr) --")
    for fam in V.V5_FAMILIES:
        row(f"  {fam}", lambda r, f=fam: r["family_energy"][f]["energy_ratio"])
    for fam in V.V5_FAMILIES:
        row(f"  {fam} vel-mag ratio", lambda r, f=fam: r["family_energy"][f].get("vel_mag_ratio"))
    for fam in V.V5_FAMILIES:
        row(f"  {fam} fwd-speed corr", lambda r, f=fam: r["family_energy"][f].get("fwd_speed_corr"))
    A("-- (4) kNN body leakage [P1 ref 0.141; root_vel arms: 5 bodies, chance 0.20] --")
    row("  knn body acc", lambda r: r["leakage"]["knn_body_acc"])
    row("  chance", lambda r: r["leakage"]["chance"])
    fs = res.get("foot_slip", {})
    if fs and "error" not in fs:
        A("-- (5) GT-free physeval, decoded G1, median over clips "
          f"({','.join(PHYS_CLIPS)}) --")
        A("   [reference floor GT_retarget: slip50="
          f"{med(fs.get('GT_retarget', {}), 'slip_p50_ms'):.3f}  "
          "V5c@PHYSEVAL_0806 was 0.751]")
        for lab, k_ in (("foot slip p50 [m/s]", "slip_p50_ms"),
                        ("foot slip p95 [m/s]", "slip_p95_ms"),
                        ("torque over limit [%]", "torque_over_pct"),
                        ("worst torque ratio [x]", "torque_worst_ratio"),
                        ("jerk p95 [rad/s^3]", "jerk_p95")):
            vals = []
            for n_ in names:
                try:
                    vals.append(f"{med(fs[n_], k_):.3f}")
                except Exception:
                    vals.append("-")
            A(f"  {lab:40s}" + "".join(f"{v:>{W}s}" for v in vals))
    A("")
    A("VERDICT vs gates (2c >= 0.90, content >= 0.6, key <= 7.5 cm):")
    for n_ in names:
        r = res["arms"][n_]
        try:
            w2 = r["warp_2c"]["warp_frac_mean"]
            cs = r["ladder"]["_token_content"]["content_share"]
            kc = r["ladder"]["prefix8"]["key_pos_cm"]
            A(f"  {n_:16s} 2c {w2:.3f} {'PASS' if w2 >= GATES['warp_2c'] else 'FAIL'} | "
              f"content {cs:.3f} {'PASS' if (cs or 0) >= GATES['content_share'] else 'FAIL'} | "
              f"key {kc:.2f} {'PASS' if kc <= GATES['key_cm'] else 'FAIL'}")
        except Exception:
            A(f"  {n_:16s} (incomplete)")
    A("=" * 100)
    txt = "\n".join(L)
    print("\n" + txt, flush=True)
    open(f"{args.out}.txt" if not args.out.endswith(".txt") else args.out, "w"
         ).write(txt + "\n")
    json.dump(res, open(f"{args.out.removesuffix('.txt')}.json", "w"), indent=1,
              default=float)
    print(f"\nSAVED {args.out}(.txt/.json)  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
