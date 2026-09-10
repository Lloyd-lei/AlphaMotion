"""Height-sensitivity decisive experiment (lloyd 0831) — decode-side only, no retarget.

Question: is Greenwich's "legs too straight" on G1 caused by decoder HEIGHT/PROPORTION
sensitivity (short legs -> straighter decode), or by URDF/projection limits?

Three parts (see HEIGHT_SENSITIVITY_REPORT.md):
  1. Projection-limits comparison, STOCK files only (lloyd's order: no synthetic
     limit widening anywhere). Same ground-window decode, projected twice:
       - limits from GMR's g1_mocap_29dof.xml (as shipped)
       - limits from the official-lineage g1_mjlab/g1_29dof.xml (as shipped)
     The pre-projection DECODED rotations are the unclamped reference.
  2. 18-embodiment flexion regression: decode the SAME ground/crawl window
     (aiming1_subject4 @11023) + the deepest-crouch moment of the walk/dyn windows
     onto every cache embodiment (manifest minus human); regress decoded knee/hip
     flexion vs spec height and leg/height ratio.
  3. Synthetic-height G1 (causal): unitree_g1_29dof rest_offsets uniformly scaled
     x{0.9, 1.0, 1.1, 1.2} — IN-MEMORY analysis-only skeleton variants, no files
     written, no URDF/XML touched. Topology, joints, dof, semantics identical;
     only scale (and therefore reach, recomputed per the training convention)
     varies. If flexion tracks scale monotonically, height sensitivity is causal.

Decode path = v2_judge's verbatim (production Greenwich checkpoint
PT_c1dec_reenc_0812, holdout direction human_smpl -> robots, fixed windows
v2_windows.json, training-convention reach). Flexion = geodesic angle of the
DECODED rotations relative to the rest pose:
  knee  = geodesic( R_parent(knee)^T R_knee , rest_knee )
  hip   = geodesic( R_root^T R_parent(knee) , prod(rest along root->thigh chain) )
(thigh-vs-pelvis composition, robust to how many sub-joints a hip block has).

Reach for the scaled specs: FK positions are LINEAR in rest_offsets
(gp_j = sum R_parent @ off), so build_global's full-data max scales exactly by s;
reach_s = s * reach_g1 IS the training-convention value, verified numerically below.

Usage: python height_sweep.py [--device cuda]
Writes height_sweep_results.json next to this file. GPU-light (270 decode rows).
"""
from __future__ import annotations
import os, sys, json, re, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import SkeletonSpec, fk_global, rot6d_to_matrix, geodesic_deg   # noqa: E402
from train_ae import static_cond                                            # noqa: E402
from dual_common import build_model, geom                                   # noqa: E402
from manifold_project import fit_angles, fk_from_angles, load_dof, _limits  # noqa: E402
from dof_head import DofKin                                                 # noqa: E402
import v2_judge as VJ                                                       # noqa: E402

CKPT = "/var/tmp/alphamotion/topo_exp/runs/PT_c1dec_reenc_0812"     # production Greenwich
LAFAN = VJ.LAFAN
SRC = "human_smpl"
G1 = "unitree_g1_29dof"
T = VJ.T
PICKS = {"walk": 7, "ground": 17, "dyn": 15}        # same indices as pilot_decode_viz
XML_GMR = "/var/tmp/alphamotion/config_harvest/GMR/assets/unitree_g1/g1_mocap_29dof.xml"
XML_UFO = "/home/arenalabs/reference_code/ufo/humanoidverse/data/robots/g1_mjlab/g1_29dof.xml"
SCALES = [0.9, 1.0, 1.1, 1.2]
CROUCH_HALF = 7                                     # crouch moment = 15 frames
# knee joint per embodiment via its dof.npz slot_joints (MJCF hinge names);
# covers knee/calf/shank naming + kuavo leg_[lr]4 + talos leg_left/right_4.
KNEE_PAT = re.compile(r"knee|calf|shank|leg_l4|leg_r4|leg_left_4|leg_right_4")


def fast_reach(cache, name, spec, device):
    """Same value as v2_judge.cache_reach (chunked max of full-data FK |gp|, the
    training build_global convention), bigger chunks; written to the SAME json so
    v2_judge / load_emb reuse it."""
    rc = json.load(open(VJ.REACH_FILE)) if os.path.exists(VJ.REACH_FILE) else {}
    key = f"{cache}::{name}"
    if key in rc:
        return rc[key]
    l6 = torch.load(f"{cache}/{name}_paired_local6d.pt", mmap=True)
    mx = 0.0
    with torch.no_grad():
        for i in range(0, l6.shape[0], 131072):
            _, gp = fk_global(l6[i:i + 131072].to(device).float(), spec)
            mx = max(mx, float(gp.norm(dim=-1).max()))
    rc = json.load(open(VJ.REACH_FILE)) if os.path.exists(VJ.REACH_FILE) else {}
    rc[key] = max(mx, 1e-6)
    json.dump(rc, open(VJ.REACH_FILE, "w"), indent=1)
    print(f"[reach] {name} @ {os.path.basename(cache)}: {rc[key]:.2f}cm", flush=True)
    return rc[key]


def knee_slots(dofz, spec=None):
    sj = [str(s) for s in dofz["slot_joints"]]
    ks = [k for k, s in enumerate(sj) if s and KNEE_PAT.search(s.lower())]
    if not ks and spec is not None:      # human_smpl: slot_joints are 'ball(x,y,z)';
        ks = [k for k, n in enumerate(spec.joint_names)   # fall back to link names
              if KNEE_PAT.search(str(n).lower())]
    return ks, sj


def leg_geometry(spec, ks):
    """thigh + shank length (cm) from the rest_offsets chain hip->knee->ankle."""
    off = np.asarray(spec.rest_offsets)
    L = []
    for k in ks:
        thigh = float(np.linalg.norm(off[k]))                       # parent(hip yaw)->knee
        kids = [j for j, p in enumerate(spec.parents) if p == k]
        assert kids, f"{spec.name}: knee joint {k} has no child"
        shank = max(float(np.linalg.norm(off[j])) for j in kids)    # knee->ankle
        L.append(thigh + shank)
    return float(np.mean(L))


def flexion(rot6, spec, dofz, rest):
    """Decoded GLOBAL rot6d -> per-frame knee / hip flexion (deg, geodesic vs rest).
    Returns (knee [B,K], hip [B,K]) over the K knee joints (normally 2)."""
    R = rot6d_to_matrix(rot6) if rot6.shape[-1] == 6 else rot6
    parents = np.asarray(spec.parents)
    ks, _ = knee_slots(dofz, spec)
    assert len(ks) >= 2, f"{spec.name}: found {len(ks)} knee slots"
    dev, dt = R.device, R.dtype
    rest = rest.to(dev, dt)
    knees, hips = [], []
    for k in ks:
        p = int(parents[k])
        Rl = R[:, p].transpose(-1, -2) @ R[:, k]
        knees.append(geodesic_deg(Rl, rest[k][None].expand_as(Rl)))
        chain = []
        j = p
        while parents[j] >= 0:
            chain.append(j)
            j = int(parents[j])
        root = j
        chain.reverse()
        P = torch.eye(3, device=dev, dtype=dt)
        for c in chain:
            P = P @ rest[c]
        Rrel = R[:, root].transpose(-1, -2) @ R[:, p]
        hips.append(geodesic_deg(Rrel, P[None].expand_as(Rrel)))
    return torch.stack(knees, -1), torch.stack(hips, -1)


def dof_from_xml(xml, dofz, tag):
    """Cache dof tensor with (limited, half_range, center) replaced per hinge joint
    by the STOCK xml's jnt_range — files swapped whole, nothing hand-edited."""
    import mujoco as mj
    m = mj.MjModel.from_xml_path(xml)
    dof = torch.as_tensor(np.asarray(dofz["dof"]), dtype=torch.float32).clone()
    sj = [str(s) for s in dofz["slot_joints"]]
    n_hit = 0
    for k, names in enumerate(sj):
        for s, jn in enumerate([x for x in names.split(",") if x]):
            jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, jn)
            assert jid >= 0, f"{tag}: joint {jn} missing in {xml}"
            lo, hi = m.jnt_range[jid]
            lim = bool(m.jnt_limited[jid])
            dof[k, s * 6 + 3] = 1.0 if lim else 0.0
            dof[k, s * 6 + 4] = (hi - lo) / 2 / np.pi if lim else 0.0
            dof[k, s * 6 + 5] = (hi + lo) / 2 / np.pi if lim else 0.0
            n_hit += 1
    print(f"[dof:{tag}] {n_hit} hinge slots re-limited from {os.path.basename(xml)}")
    return dof


def stats(x):
    x = x.reshape(-1).float()
    return {"mean": round(float(x.mean()), 2), "p95": round(float(x.quantile(.95)), 2),
            "max": round(float(x.max()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    torch.manual_seed(0)
    res = {"ckpt": CKPT, "windows": {}}

    cfg = json.load(open(f"{CKPT}/config.json"))
    caches = [c for c in str(cfg["cache"]).split(",") if c]
    assert "cache_xwbc_lafan" in os.path.basename(caches[0].rstrip("/"))
    man = json.load(open(f"{LAFAN}/manifest.json"))
    robots = [n for n in man["embodiments"] if n != SRC]

    # ---- windows: ground (whole) + deepest-crouch 15 frames of walk/dyn ----
    win = json.load(open(VJ.WIN_FILE))
    picks = {tag: win["windows"][i] for tag, i in PICKS.items()}
    rows_per = {tag: torch.arange(w["start"], w["start"] + T) for tag, w in picks.items()}
    rows = torch.cat(list(rows_per.values()))
    seg = {tag: slice(i * T, (i + 1) * T) for i, tag in enumerate(picks)}
    rw = torch.load(f"{LAFAN}/{SRC}_root_world.pt", mmap=True)[rows].float()   # cm, Y-up
    hy = rw[:, 1]
    cand = torch.full_like(hy, 1e9)
    for tag in ("walk", "dyn"):
        cand[seg[tag]] = hy[seg[tag]]
    c0 = int(cand.argmin())
    lo_, hi_ = max(c0 - CROUCH_HALF, 0), min(c0 + CROUCH_HALF + 1, len(rows))
    crouch_sl = slice(lo_, hi_)
    crouch_tag = [t for t in ("walk", "dyn") if seg[t].start <= c0 < seg[t].stop][0]
    for tag, w in picks.items():
        print(f"[win] {tag}: {w['clip']} @ {w['start']}")
    print(f"[crouch] from '{crouch_tag}' window, frame {c0 - seg[crouch_tag].start}, "
          f"human root height {float(hy[c0]):.1f}cm (ground window: "
          f"{float(hy[seg['ground']].min()):.1f}-{float(hy[seg['ground']].max()):.1f}cm)")
    res["windows"] = {t: picks[t] | {} for t in picks}
    res["crouch"] = {"window": crouch_tag, "n_frames": hi_ - lo_,
                     "root_height_cm": round(float(hy[c0]), 1)}

    # ---- reach (training convention, both caches), then data via judge loader ----
    lafan_specs = {}
    for n in [SRC] + robots:
        d = np.load(f"{LAFAN}/{n}_spec.npz", allow_pickle=True)
        lafan_specs[n] = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                                      list(d["joint_names"]))
        for c in caches:
            fast_reach(c, n, lafan_specs[n], dev)

    from semantics import build_name_embeddings
    names_all = sum([list(lafan_specs[n].joint_names) for n in [SRC] + robots], [])
    lut = build_name_embeddings(names_all, device=dev,
                                encoder=cfg.get("sem_encoder", "siglip"))
    data = {n: VJ.load_emb(n, rows, caches, cfg, lut, dev) for n in [SRC] + robots}

    model = build_model(cfg, dev)
    model.load_state_dict(torch.load(f"{CKPT}/model.pt", map_location=dev))
    model.eval()

    B = len(rows)
    with torch.no_grad():
        zp, zr = model.encode(data[SRC]["pose"], geom(data, SRC, B))

    def decode_to(entry):
        d2 = {"_t": entry}
        with torch.no_grad():
            prot, _ = model.decode(zp, zr, geom({"_t": entry}, "_t", B))
        return prot

    # =========================== Part 2: 18-embodiment sweep ==================
    print("\n===== Part 2: per-embodiment decoded flexion vs height =====")
    rows2 = []
    for n in robots:
        e = data[n]
        dofz = np.load(f"{LAFAN}/{n}_dof.npz", allow_pickle=True)
        _, rest = load_dof(LAFAN, n, device=dev)
        prot = decode_to(e)
        kf, hf = flexion(prot, e["spec"], dofz, rest)                # [B,2]
        gkf, ghf = flexion(e["pose"][..., :6], e["spec"], dofz, rest)
        ks, _ = knee_slots(dofz)
        r = {"name": n, "height": round(float(e["spec"].height), 1),
             "leg_cm": round(leg_geometry(e["spec"], ks), 1)}
        r["leg_ratio"] = round(r["leg_cm"] / r["height"], 4)
        for tag, sl in [("ground", seg["ground"]), ("crouch", crouch_sl)]:
            r[f"knee_{tag}"] = stats(kf[sl])
            r[f"hip_{tag}"] = stats(hf[sl])
            r[f"gt_knee_{tag}"] = stats(gkf[sl])
        rows2.append(r)
    # human source reference (same measure on the GT human window)
    dofz_h = np.load(f"{LAFAN}/{SRC}_dof.npz", allow_pickle=True)
    _, rest_h = load_dof(LAFAN, SRC, device=dev)
    hkf, hhf = flexion(data[SRC]["pose"][..., :6], data[SRC]["spec"], dofz_h, rest_h)
    res["human_src"] = {t: {"knee": stats(hkf[sl]), "hip": stats(hhf[sl])}
                        for t, sl in [("ground", seg["ground"]), ("crouch", crouch_sl)]}
    print(f"[human src] ground knee {res['human_src']['ground']['knee']} "
          f"crouch knee {res['human_src']['crouch']['knee']}")

    rows2.sort(key=lambda r: r["height"])
    hdr = (f"{'embodiment':18s} {'height':>7s} {'leg':>6s} {'ratio':>6s} "
           f"{'kneeG.mean':>10s} {'kneeG.max':>9s} {'hipG.mean':>9s} "
           f"{'kneeC.mean':>10s} {'gtKneeG':>8s} {'gtKneeC':>8s}")
    print(hdr); print("-" * len(hdr))
    for r in rows2:
        print(f"{r['name']:18s} {r['height']:7.1f} {r['leg_cm']:6.1f} {r['leg_ratio']:6.3f} "
              f"{r['knee_ground']['mean']:10.2f} {r['knee_ground']['max']:9.2f} "
              f"{r['hip_ground']['mean']:9.2f} {r['knee_crouch']['mean']:10.2f} "
              f"{r['gt_knee_ground']['mean']:8.2f} {r['gt_knee_crouch']['mean']:8.2f}")

    def corr(xs, ys):
        return round(float(np.corrcoef(np.asarray(xs), np.asarray(ys))[0, 1]), 3)

    h = [r["height"] for r in rows2]; lr = [r["leg_ratio"] for r in rows2]
    res["part2"] = {"table": rows2, "corr": {}}
    for tag in ("ground", "crouch"):
        km = [r[f"knee_{tag}"]["mean"] for r in rows2]
        hm = [r[f"hip_{tag}"]["mean"] for r in rows2]
        dg = [r[f"knee_{tag}"]["mean"] - r[f"gt_knee_{tag}"]["mean"] for r in rows2]
        res["part2"]["corr"][tag] = {
            "knee_vs_height": corr(h, km), "knee_vs_legratio": corr(lr, km),
            "hip_vs_height": corr(h, hm),
            "kneeMinusGT_vs_height": corr(h, dg)}
    print("[corr]", json.dumps(res["part2"]["corr"]))

    # =========================== Part 1: stock-limits projection ==============
    print("\n===== Part 1: G1 ground window, stock GMR vs stock UFO limits =====")
    e1 = data[G1]
    dofz1 = np.load(f"{LAFAN}/{G1}_dof.npz", allow_pickle=True)
    dof_cache, rest1 = load_dof(LAFAN, G1, device=dev)
    prot_g1 = decode_to(e1)[seg["ground"]]
    kf0, hf0 = flexion(prot_g1, e1["spec"], dofz1, rest1)
    res["part1"] = {"decoded_preproj": {"knee": stats(kf0), "hip": stats(hf0)}}
    print(f"decoded (pre-projection, unclamped reference): knee {stats(kf0)} hip {stats(hf0)}")
    variants = {"gmr_stock": dof_from_xml(XML_GMR, dofz1, "gmr"),
                "ufo_stock": dof_from_xml(XML_UFO, dofz1, "ufo")}
    # lineage check: which stock file does the cache dof equal?
    for tag, dv in variants.items():
        d = float((dv.to(dev) - dof_cache).abs().max())
        print(f"[lineage] cache dof vs {tag}: max abs diff {d:.4f} "
              f"({'IDENTICAL' if d < 1e-4 else 'differs'})")
        res["part1"][f"cache_vs_{tag}_maxdiff"] = round(d, 5)
    sj1 = [str(s) for s in dofz1["slot_joints"]]
    ks1, _ = knee_slots(dofz1)
    hp1 = [k for k, s in enumerate(sj1) if "hip_pitch" in s]
    qs = {}
    for tag, dv in variants.items():
        q, _ = fit_angles(prot_g1, e1["spec"], dv.to(dev), rest=rest1,
                          clamp=True, method="global")
        qs[tag] = q
        with torch.no_grad():
            gR, _ = fk_from_angles(q, e1["spec"], dv.to(dev), rest=rest1,
                                   root_R=rot6d_to_matrix(prot_g1)[:, 0])
        kfp, hfp = flexion(gR, e1["spec"], dofz1, rest1)
        gd = geodesic_deg(gR, rot6d_to_matrix(prot_g1))
        kq = torch.rad2deg(q[:, ks1, 0]); hq = torch.rad2deg(q[:, hp1, 0])
        res["part1"][tag] = {"knee_flex": stats(kfp), "hip_flex": stats(hfp),
                             "knee_q_deg": stats(kq),
                             "hip_pitch_q_deg": {"mean": round(float(hq.mean()), 2),
                                                 "min": round(float(hq.min()), 2),
                                                 "max": round(float(hq.max()), 2)},
                             "fk_gap_deg_mean": round(float(gd.mean()), 2)}
        print(f"{tag:10s}: knee_flex {stats(kfp)}  hip_flex {stats(hfp)}  "
              f"knee_q {stats(kq)}  hip_pitch_q [{float(hq.min()):.1f},{float(hq.max()):.1f}] "
              f"fk_gap {float(gd.mean()):.2f}deg")
    dq = torch.rad2deg((qs["gmr_stock"] - qs["ufo_stock"]).abs())
    res["part1"]["max_q_diff_deg"] = round(float(dq.max()), 3)
    res["part1"]["mean_q_diff_deg"] = round(float(dq.mean()), 4)
    print(f"GMR-vs-UFO projected q diff: max {float(dq.max()):.3f} deg, "
          f"mean {float(dq.mean()):.4f} deg")

    # =========================== Part 3: synthetic-height G1 ==================
    print("\n===== Part 3: G1 rest_offsets x scale (in-memory variants) =====")
    reach_g1 = e1["reach"]
    spec1 = e1["spec"]
    # numeric check of reach linearity on the window rows
    l6w = torch.load(f"{LAFAN}/{G1}_paired_local6d.pt", mmap=True)[rows].to(dev).float()
    res["part3"] = {"reach_g1": round(reach_g1, 2), "scales": {}}
    tab3 = []
    for s in SCALES:
        spec_s = SkeletonSpec(f"g1_s{s}", spec1.parents, spec1.rest_offsets * s,
                              list(spec1.joint_names))
        with torch.no_grad():
            _, gps = fk_global(l6w, spec_s)
            _, gp1 = fk_global(l6w, spec1)
            lin_err = float((gps - s * gp1).abs().max())
        entry = {"spec": spec_s, "st": static_cond(spec_s, dev), "sem": e1["sem"],
                 "reach": s * reach_g1, "dof": e1["dof"],
                 "kin": DofKin(spec_s, dof_cache, rest1, device=dev)}
        prot = decode_to(entry)
        kf, hf = flexion(prot, spec_s, dofz1, rest1)
        row = {"scale": s, "height": round(spec_s.height, 1),
               "reach": round(s * reach_g1, 2), "fk_linearity_err_cm": round(lin_err, 6)}
        for tag, sl in [("ground", seg["ground"]), ("crouch", crouch_sl)]:
            row[f"knee_{tag}"] = stats(kf[sl]); row[f"hip_{tag}"] = stats(hf[sl])
        if s == 1.0:  # must equal the real G1 decode bit-for-bit (same inputs)
            row["equals_real_g1"] = bool(torch.allclose(prot, decode_to(e1), atol=1e-5))
        tab3.append(row)
        print(f"scale {s:.1f}: height {row['height']:7.1f} reach {row['reach']:6.1f} "
              f"kneeG {row['knee_ground']['mean']:6.2f} kneeC {row['knee_crouch']['mean']:6.2f} "
              f"hipG {row['hip_ground']['mean']:6.2f} hipC {row['hip_crouch']['mean']:6.2f}"
              + (f"  [==real_g1: {row.get('equals_real_g1')}]" if s == 1.0 else "")
              + f"  linERR {lin_err:.2e}cm")
    res["part3"]["scales"] = tab3
    km = [r["knee_ground"]["mean"] for r in tab3]
    kc = [r["knee_crouch"]["mean"] for r in tab3]
    mono_g = all(km[i] <= km[i + 1] for i in range(len(km) - 1))
    mono_c = all(kc[i] <= kc[i + 1] for i in range(len(kc) - 1))
    res["part3"]["monotone_knee_ground"] = mono_g
    res["part3"]["monotone_knee_crouch"] = mono_c
    print(f"knee flexion monotone in scale? ground={mono_g} crouch={mono_c}")

    out = os.path.join(HERE, "height_sweep_results.json")
    json.dump(res, open(out, "w"), indent=1)
    print("\nWROTE", out)


if __name__ == "__main__":
    main()
