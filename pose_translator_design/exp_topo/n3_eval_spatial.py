"""N3 SPATIAL VERDICT — E27_qwen_0806 (control) vs N3e_reenc / N3f_varfloor.

One table (-> runs/N3_spatial_verdict.txt + .json):
  (i)   zero-shot human->G1 per-part errors + POST-PROJECTION rot (eval_e25
        harness, the night2 (iii) section verbatim) — G1 is holdout for all
        three arms, so this is behavior-level zero-shot evidence
  (ii)  ROUND-TRIP error, the metric N3e exists to fix: gen1 = decode(encode(x)),
        gen2 = decode(encode(gen1)) — same-body on human (train) and G1
        (holdout), plus the via-robot cycle human -> unitree_h1 -> human.
        The 0806 measurement: same-body gen1 ~8.7 deg explodes to ~38.8 deg at
        gen2; the gate is gen2/gen1 blow-up, not the absolute number.
  (iii) motion energy / amplitude per family on held-out AMASS clips including
        KICK (N3f's target — does the kick lift now?): per-clip temporal std of
        decoded positions vs GT, and ankle-lift (max-median ankle height) ratio.
  (iv)  GT-free physeval on decoded G1 trajectories (foot slip / torque), same
        LAFAN clips + root-world convention as runs/PHYSEVAL_report.md
        (E27_qwen there: slip p50 0.162 m/s).
  (v)   the standard night2 error decomposition (full-vis / continuous /
        constraint-mode wrist error).
Sections independently guarded. All three arms are qwen3-semantics models.
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

from common import rot6d_to_matrix, geodesic_deg                  # noqa: E402
from dual_common import load_paired_data, split_indices, build_model, geom  # noqa: E402
import v5_common as V                                             # noqa: E402

ROOT = "/var/tmp/alphamotion/topo_exp"
RUNS = f"{ROOT}/runs"
LAFAN = f"{ROOT}/cache_xwbc_lafan"
AMASS = f"{ROOT}/cache_xwbc_amass"
HUMAN, G1, H1 = "human_smpl", "unitree_g1_29dof", "unitree_h1"
ARMS = ["E27_qwen_0806", "N3e_reenc_0807", "N3f_varfloor_0807"]
PHYS_CLIPS = ["walk1_subject1", "walk3_subject1", "run1_subject2", "dance1_subject1"]
FAMS = ["walk", "run", "dance", "kick", "punch", "jacks"]


# ------------------------------------------------------------------ round trip
@torch.no_grad()
def enc_dec(model, data, src, tgt, x, chunk=256):
    R, P = [], []
    for i in range(0, x.shape[0], chunk):
        xs = x[i:i + chunk]
        zp, zr = model.encode(xs, geom(data, src, len(xs)))
        r, p = model.decode(zp, zr, geom(data, tgt, len(xs)))
        R.append(r); P.append(p)
    return torch.cat(R), torch.cat(P)


@torch.no_grad()
def roundtrip(model, data, idx):
    """gen1 = decode(encode(x)); gen2 = decode(encode(gen1)). Errors vs the GT x.
    Same-body on human (train) and G1 (holdout); via-robot human->h1->human."""
    out = {}
    for emb in (HUMAN, G1):
        x = data[emb]["pose"][idx]
        reach = data[emb]["reach"]
        r1, p1 = enc_dec(model, data, emb, emb, x)
        r2, p2 = enc_dec(model, data, emb, emb, torch.cat([r1, p1], -1))
        Rt = rot6d_to_matrix(x[..., :6])
        e1 = float(geodesic_deg(rot6d_to_matrix(r1), Rt).mean())
        e2 = float(geodesic_deg(rot6d_to_matrix(r2), Rt).mean())
        out[emb] = {"gen1_rot_deg": round(e1, 2), "gen2_rot_deg": round(e2, 2),
                    "blowup_x": round(e2 / max(e1, 1e-6), 2),
                    "gen1_pos_cm": round(float((p1 - x[..., 6:9]).norm(dim=-1).mean() * reach), 2),
                    "gen2_pos_cm": round(float((p2 - x[..., 6:9]).norm(dim=-1).mean() * reach), 2)}
    # via-robot cycle: encode human -> decode h1 -> re-encode (as h1) -> decode human
    x = data[HUMAN]["pose"][idx]
    reach = data[HUMAN]["reach"]
    rh, ph = enc_dec(model, data, HUMAN, H1, x)
    r2, p2 = enc_dec(model, data, H1, HUMAN, torch.cat([rh, ph], -1))
    Rt = rot6d_to_matrix(x[..., :6])
    d1, _ = enc_dec(model, data, HUMAN, HUMAN, x)          # direct, for reference
    out["via_h1"] = {
        "cycle_rot_deg": round(float(geodesic_deg(rot6d_to_matrix(r2), Rt).mean()), 2),
        "cycle_pos_cm": round(float((p2 - x[..., 6:9]).norm(dim=-1).mean() * reach), 2),
        "direct_rot_deg": round(float(geodesic_deg(rot6d_to_matrix(d1), Rt).mean()), 2)}
    return out


# ------------------------------------------------------------------ family amp
def pick_family_clips(n_per=3, min_len=90):
    seqs = json.load(open(f"{AMASS}/sequences.json"))
    out = {f: [] for f in FAMS}
    for s in seqs:
        f = V.family_of(s["name"])
        if f in out and len(out[f]) < n_per and s["len"] >= min_len:
            out[f].append(s)
    return out


@torch.no_grad()
def family_amplitude(model, dev, clips, n_frames=90, chunk=128):
    """Held-out AMASS clips -> decode human->human and human->G1; per family:
    energy ratio (temporal std of positions, decoded/GT) and ankle-lift ratio
    (max - median ankle height over the clip, decoded/GT) — the kick-lift meter."""
    from train_global import build_global
    import physeval_run as PR
    from semantics import build_name_embeddings
    hspec = V.load_spec(AMASS, HUMAN)
    gspec = V.load_spec(AMASS, G1)
    names = list(hspec.joint_names) + list(gspec.joint_names)
    lut = build_name_embeddings(names, device=dev, encoder="qwen3")
    hz = np.load(f"{AMASS}/{HUMAN}_dof.npz", allow_pickle=True)
    gz = np.load(f"{AMASS}/{G1}_dof.npz", allow_pickle=True)
    hE = PR.entry_of(hspec, hz["dof"], dev, lut, hz["rest"])
    gE = PR.entry_of(gspec, gz["dof"], dev, lut, gz["rest"])
    hreach = float(np.load(f"{V.KEYCACHE}/cache_xwbc_amass_{HUMAN}.npz")["reach"])
    greach = float(np.load(f"{V.KEYCACHE}/cache_xwbc_amass_{G1}.npz")["reach"])
    hl6 = torch.load(f"{AMASS}/{HUMAN}_paired_local6d.pt")
    gl6 = torch.load(f"{AMASS}/{G1}_paired_local6d.pt")
    h_ank = [V.key_joints(hspec)[0][i] for i in (4, 5)]     # l_ankle, r_ankle
    g_ank = [V.key_joints(gspec)[0][i] for i in (4, 5)]
    data2 = {HUMAN: hE, G1: gE}

    def lift(pos, ank):
        y = pos[:, ank, 1]                                   # [T,2]
        return float((y.max(0).values - y.median(0).values).mean())

    out = {}
    for fam, cl in clips.items():
        if not cl:
            continue
        rows_en, rows_lift_h, rows_lift_g = [], [], []
        for s in cl:
            idx = torch.arange(s["start"], s["start"] + min(s["len"], n_frames))
            gr6, gp, _ = build_global(hl6[idx], hspec, dev)
            hx = torch.cat([gr6, gp / hreach], -1).float().to(dev)
            gg6, ggp, _ = build_global(gl6[idx], gspec, dev)
            gx = torch.cat([gg6, ggp / greach], -1).float().to(dev)
            hE["pose"] = hx
            _, ph = enc_dec(model, data2, HUMAN, HUMAN, hx, chunk)
            _, pg = enc_dec(model, data2, HUMAN, G1, hx, chunk)
            en_h = float(ph.std(0).mean() / hx[..., 6:9].std(0).mean().clamp_min(1e-6))
            en_g = float(pg.std(0).mean() / gx[..., 6:9].std(0).mean().clamp_min(1e-6))
            rows_en.append((en_h, en_g))
            gt_lh = lift(hx[..., 6:9], h_ank)
            gt_lg = lift(gx[..., 6:9], g_ank)
            rows_lift_h.append(lift(ph, h_ank) / max(gt_lh, 1e-6))
            rows_lift_g.append(lift(pg, g_ank) / max(gt_lg, 1e-6))
        out[fam] = {"n_clips": len(cl),
                    "energy_ratio_self": round(float(np.mean([a for a, _ in rows_en])), 3),
                    "energy_ratio_g1": round(float(np.mean([b for _, b in rows_en])), 3),
                    "ankle_lift_ratio_self": round(float(np.mean(rows_lift_h)), 3),
                    "ankle_lift_ratio_g1": round(float(np.mean(rows_lift_g)), 3)}
    return out


# ------------------------------------------------------------------ physeval
def foot_slip(models, data, dev, frames=240, start_sec=2.0, tiny=False):
    import physeval_run as PR
    import physeval as PE
    from common import fk_global, matrix_to_rot6d
    seqs = {s["name"]: s for s in json.load(open(f"{LAFAN}/sequences.json"))}
    clips = PHYS_CLIPS[:1] if tiny else PHYS_CLIPS
    body = PR.body_of(G1, LAFAN)
    g1spec, _ = PR.spec_of(LAFAN, G1)
    cfgP = PE.PhysEvalConfig(fps=30.0)
    out = {t: {} for t in list(models) + ["GT_retarget"]}
    for clip in clips:
        s = seqs[clip]
        o0 = int(start_sec * 30)
        n = min(frames if not tiny else 62, s["len"] - o0)
        idx = s["start"] + o0 + np.arange(n)
        if int(idx.max()) >= data[HUMAN]["pose"].shape[0]:
            print(f"  [phys] skip {clip}: rows beyond loaded data "
                  f"(max_rows smoke?)", flush=True)
            continue
        rw = PR.root_world_muj(LAFAN, G1, idx)
        gl6 = torch.load(f"{LAFAN}/{G1}_paired_local6d.pt")[idx]
        with torch.no_grad():
            gR, _ = fk_global(gl6.to(dev), g1spec)
        gt_rot = matrix_to_rot6d(gR).float().cpu().numpy()
        qpos, _, _ = PR.to_qpos(body, gt_rot, rw)
        out["GT_retarget"][clip] = PE.evaluate(body["model"], qpos, cfgP,
                                               body["floor_gid"],
                                               label=f"{clip}/GT")["headline"]
        x = data[HUMAN]["pose"][torch.from_numpy(idx)]
        for tag, model in models.items():
            r, _ = enc_dec(model, data, HUMAN, G1, x, chunk=64)
            qpos, _, _ = PR.to_qpos(body, r.float().cpu().numpy(), rw)
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
    ap.add_argument("--runs", nargs="*", default=ARMS)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_decomp", type=int, default=512)
    ap.add_argument("--n_eval", type=int, default=4096)
    ap.add_argument("--n_proj", type=int, default=128)
    ap.add_argument("--n_rt", type=int, default=512)
    ap.add_argument("--max_rows", type=int, default=0)     # smoke only (changes split)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--no_phys", action="store_true")
    ap.add_argument("--out", default=f"{RUNS}/N3_spatial_verdict")
    args = ap.parse_args()
    if args.tiny:
        args.n_decomp, args.n_eval, args.n_proj, args.n_rt = 32, 64, 8, 32
        args.max_rows = args.max_rows or 2000
    dev, t0 = args.device, time.time()

    torch.manual_seed(0)                       # split contract: seed -> load -> perm
    data, names, N = load_paired_data(LAFAN, dev, max_rows=args.max_rows,
                                      sem_encoder="qwen3")
    val_idx, _ = split_indices(N)
    idx_d = val_idx[:args.n_decomp]
    idx_e = val_idx[:args.n_eval]
    idx_r = val_idx[:args.n_rt]

    import night2_eval as N2
    import eval_e25 as E25
    from manifold_project import load_dof
    masks = {k: v.cpu() for k, v in E25.part_masks(data[G1]["spec"], dev).items()}
    dofG1, restG1 = load_dof(LAFAN, G1, device="cpu")
    fam_clips = pick_family_clips(n_per=1 if args.tiny else 3)

    res = {"generated": time.strftime("%Y-%m-%d %H:%M"), "cache": LAFAN,
           "n_decomp": len(idx_d), "n_eval": len(idx_e), "runs": {}}
    models = {}
    for run in args.runs:
        rd = run if os.path.isabs(run) else f"{RUNS}/{run}"
        if not (os.path.exists(f"{rd}/config.json") and os.path.exists(f"{rd}/model.pt")):
            print(f"[skip] {run}: not finished", flush=True)
            continue
        cfg = json.load(open(f"{rd}/config.json"))
        model = build_model(cfg, dev)
        model.load_state_dict(torch.load(f"{rd}/model.pt", map_location=dev), strict=True)
        model.eval()
        models[run] = model
        print(f"\n===== {run}", flush=True)
        r = {"steps": cfg.get("steps"),
             "reenc": float(cfg.get("reenc_consistency", 0) or 0),
             "var_floor": float(cfg.get("var_floor", 0) or 0)}
        jobs = [
            ("decomposition", lambda: N2.decomposition(model, data, idx_d)),
            ("roundtrip", lambda: roundtrip(model, data, idx_r)),
            ("family_amplitude", lambda: family_amplitude(model, dev, fam_clips,
                                                          n_frames=60 if args.tiny else 90)),
        ]
        for name, fn in jobs:
            try:
                r[name] = fn()
                print(f"  {name}: ok", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                r[name] = {"error": f"{type(e).__name__}: {e}"}
        try:
            prot, ppos = E25.decode_all(model, data, HUMAN, G1, idx_e)
            gt = data[G1]["pose"][idx_e].cpu()
            tab, _ = E25.err_table(prot, ppos, gt, data[G1]["reach"], masks)
            tab["projected_rot_deg"] = E25.projected_error(
                prot, gt, data[G1]["spec"], dofG1, restG1, masks, args.n_proj)
            r["zeroshot"] = tab
            print("  zeroshot: ok", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            r["zeroshot"] = {"error": f"{type(e).__name__}: {e}"}
        res["runs"][run] = r

    if not args.no_phys and models:
        try:
            res["foot_slip"] = foot_slip(models, data, dev, tiny=args.tiny)
        except Exception as e:
            import traceback; traceback.print_exc()
            res["foot_slip"] = {"error": f"{type(e).__name__}: {e}"}

    # ------------------------------------------------------------- the table
    runs = list(res["runs"])
    W = max([len(r_) for r_ in runs] + [12]) + 2
    L = []
    A = L.append
    A("=" * 100)
    A("N3 SPATIAL VERDICT — reenc-consistency / var-floor vs E27_qwen (control), "
      + time.strftime("%Y-%m-%d %H:%M"))
    A("noise band: same-recipe seed spread ~2.2 deg pose error (R9); "
      "deltas below that are not results")
    A("=" * 100)
    A(f"{'metric':46s}" + "".join(f"{r_:>{W}s}" for r_ in runs))

    def row(label, fn, fmt="{:.2f}"):
        vals = []
        for r_ in runs:
            try:
                v = fn(res["runs"][r_])
                vals.append(fmt.format(v) if v is not None else "-")
            except Exception:
                vals.append("-")
        A(f"{label:46s}" + "".join(f"{v:>{W}s}" for v in vals))

    A("-- (ii) ROUND TRIP (the N3e metric): gen1 = dec(enc(x)), gen2 = dec(enc(gen1)) --")
    for emb in (HUMAN, G1):
        row(f"  {emb} gen1 rot [deg]", lambda r, e=emb: r["roundtrip"][e]["gen1_rot_deg"])
        row(f"  {emb} gen2 rot [deg]", lambda r, e=emb: r["roundtrip"][e]["gen2_rot_deg"])
        row(f"  {emb} blow-up [x]", lambda r, e=emb: r["roundtrip"][e]["blowup_x"])
    row("  via-h1 cycle rot [deg]", lambda r: r["roundtrip"]["via_h1"]["cycle_rot_deg"])
    row("  (direct h->h rot [deg])", lambda r: r["roundtrip"]["via_h1"]["direct_rot_deg"])
    A("-- (i) zero-shot human->G1 (holdout), per part --")
    for part in ["all", "legs", "arms", "wrists"]:
        row(f"  {part} rot [deg]", lambda r, p=part: r["zeroshot"][p]["rot_deg"])
        row(f"  {part} pos [cm]", lambda r, p=part: r["zeroshot"][p]["pos_cm"])
    row("  POST-PROJECTION rot [deg]", lambda r: r["zeroshot"]["projected_rot_deg"]["all"])
    A("-- (iii) amplitude per family, held-out AMASS (decoded/GT; 1.0 = full amplitude) --")
    for fam in FAMS:
        row(f"  {fam} energy self", lambda r, f=fam: r["family_amplitude"][f]["energy_ratio_self"], "{:.3f}")
    for fam in FAMS:
        row(f"  {fam} energy ->G1", lambda r, f=fam: r["family_amplitude"][f]["energy_ratio_g1"], "{:.3f}")
    row("  KICK ankle-lift self", lambda r: r["family_amplitude"]["kick"]["ankle_lift_ratio_self"], "{:.3f}")
    row("  KICK ankle-lift ->G1", lambda r: r["family_amplitude"]["kick"]["ankle_lift_ratio_g1"], "{:.3f}")
    A("-- (v) night2 decomposition (G1 wrist, val512) --")
    row("  full-vis wrist [cm]", lambda r: r["decomposition"]["fullvis_wrist_pos_cm"])
    row("  continuous-z wrist [cm]", lambda r: r["decomposition"]["continuous_wrist_pos_cm"])
    row("  constraint-mode wrist [cm]", lambda r: r["decomposition"]["constraint_wrist_pos_cm"])
    fs = res.get("foot_slip", {})
    if fs and "error" not in fs:
        A("-- (iv) GT-free physeval, human->G1 decode, median over clips "
          f"({','.join(PHYS_CLIPS)}) --")
        A(f"   [GT_retarget floor: slip50={med(fs.get('GT_retarget', {}), 'slip_p50_ms'):.3f}; "
          "E27_qwen@PHYSEVAL_0806: 0.162]")
        for lab, k_ in (("foot slip p50 [m/s]", "slip_p50_ms"),
                        ("foot slip p95 [m/s]", "slip_p95_ms"),
                        ("torque over limit [%]", "torque_over_pct"),
                        ("worst torque ratio [x]", "torque_worst_ratio"),
                        ("jerk p95 [rad/s^3]", "jerk_p95")):
            vals = []
            for r_ in runs:
                try:
                    vals.append(f"{med(fs[r_], k_):.3f}")
                except Exception:
                    vals.append("-")
            A(f"  {lab:44s}" + "".join(f"{v:>{W}s}" for v in vals))
    A("")
    A("READ: N3e wins if the blow-up column collapses toward 1x without zero-shot "
      "regressing > 0.5 cm; N3f wins if kick ankle-lift moves toward 1.0 without "
      "the same regression.")
    A("=" * 100)
    txt = "\n".join(L)
    print("\n" + txt, flush=True)
    open(f"{args.out}.txt", "w").write(txt + "\n")
    json.dump(res, open(f"{args.out}.json", "w"), indent=1, default=float)
    print(f"\nSAVED {args.out}.txt/.json  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
