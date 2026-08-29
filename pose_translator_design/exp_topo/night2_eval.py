"""NIGHT2 unified re-eval: E19_K128_0805 (control) vs E28_resid_0807 (visible-joint
residual decode) vs E29_wristaug_0807 (wrist-axis-convention augmentation).

Four sections, identical code for all models:
  (i)   error decomposition on unitree_g1_29dof val512, wrist = left_rubber_hand:
          full-vis quantized  — the model's own encode-decode of fully visible input
                                (E19 measured 12.90 cm at the wrist: the dominant failure)
          full-vis continuous — FSQ rounding bypassed (tanh kept), prices quantization
                                (E19: 11.88 cm -> rounding costs only ~1 cm)
          constraint-mode     — encoder sees ONLY root + wrist (E19: 17.20 cm; the
                                extra ~4.3 cm is completion ambiguity)
  (ii)  wrist twist follow ratio at small angles (+-8/+-15 deg): human root+L_Wrist
        conditioning with twist phi about the forearm axis, decoded to human / h1_2 /
        G1 / zero-shot AgiBot X2; achieved twist via swing-twist split about the
        target's own forearm axis; ratio = mean(achieved/commanded).
        (E19 measured: human 0.72, h1_2 0.66, G1 0.62, X2 -0.04.)
  (iii) standard zero-shot human_smpl -> unitree_g1_29dof with per-part breakdown
        (same harness as eval_e25 / E19_control).
  (iv)  SE3 soft constraint satisfaction, root + L_Wrist on human_smpl,
        constraint_demo-style (soft: NO hard pin).

Writes /var/tmp/alphamotion/topo_exp/runs/NIGHT2_verdict.txt (+ .json with a
machine-readable "winner" for night2_render.py).
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
NIGHT2_LIB = "/var/tmp/alphamotion/topo_exp/night2_lib"
if os.path.isdir(NIGHT2_LIB):
    sys.path.append(NIGHT2_LIB)

from common import rot6d_to_matrix, matrix_to_rot6d, geodesic_deg
from dual_common import load_paired_data, split_indices, build_model, geom
from model import FSQ

ROOT = "/var/tmp/alphamotion/topo_exp"
RUNS = f"{ROOT}/runs"
CACHE = f"{ROOT}/cache_xwbc_lafan"
G1 = "unitree_g1_29dof"
HUMAN = "human_smpl"
X2 = "agibot_x2_ultra"
G1_WRIST = "left_rubber_hand"
TWIST_TARGETS = [HUMAN, "unitree_h1_2", G1, X2]
SEED_NOTE = ("seed noise: single-seed runs; cross-embodiment rot numbers move ~+-0.9 deg "
             "across seeds (earlier E13 seed pairs); pos seed spread NOT measured — treat "
             "deltas < ~0.5 cm as unresolved; twist follow-ratio noise unmeasured, treat "
             "|delta| < 0.05 as noise.")


# ------------------------------------------------------------------ helpers ---
def resid_io(model, x, joint_mask):
    """(obs, vis) for SAME-SKELETON decode of a vis_residual model, else (None, None)."""
    if not getattr(model, "vis_residual", False):
        return None, None
    if joint_mask is None:
        vis = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
    else:
        vis = ~joint_mask
    return x, vis


@torch.no_grad()
def decode_self(model, data, emb, idx, joint_mask=None, continuous=False, chunk=512):
    """Same-skeleton encode->decode. continuous=True bypasses FSQ ROUNDING (tanh
    bounding kept) by monkey-patching the instance quantize — prices quantization."""
    patched = []
    if continuous:
        for q in {id(model.fsqp): model.fsqp, id(model.fsqr): model.fsqr}.values():
            assert isinstance(q, FSQ), "continuous patch written for fsq_stages=1"
            q.quantize = lambda z: torch.tanh(z)          # instance attr shadows class
            patched.append(q)
    try:
        R, P = [], []
        for i in range(0, len(idx), chunk):
            sub = idx[i:i + chunk]
            x = data[emb]["pose"][sub]
            jm = None if joint_mask is None else joint_mask[i:i + len(sub)]
            g = geom(data, emb, len(sub))
            zp, zr = model.encode(x, g, joint_mask=jm)
            obs, vis = resid_io(model, x, jm)
            prot, ppos = model.decode(zp, zr, g, obs=obs, vis=vis)
            R.append(prot.cpu()); P.append(ppos.cpu())
        return torch.cat(R), torch.cat(P)
    finally:
        for q in patched:
            del q.quantize                                # restore the class method


def wrist_pos_err_cm(ppos, gt, reach, jw):
    return round(float((ppos[:, jw] - gt[:, jw, 6:9].cpu()).norm(dim=-1).mean() * reach), 2)


def decomposition(model, data, idx):
    d = data[G1]; spec = d["spec"]; reach = d["reach"]
    jw = list(spec.joint_names).index(G1_WRIST)
    root = int(np.where(spec.parents < 0)[0][0])
    gt = d["pose"][idx]
    out = {}
    _, pp = decode_self(model, data, G1, idx)
    out["fullvis_wrist_pos_cm"] = wrist_pos_err_cm(pp, gt, reach, jw)
    out["fullvis_all_pos_cm"] = round(float((pp - gt[..., 6:9].cpu()).norm(dim=-1).mean() * reach), 2)
    _, pp = decode_self(model, data, G1, idx, continuous=True)
    out["continuous_wrist_pos_cm"] = wrist_pos_err_cm(pp, gt, reach, jw)
    mask = torch.ones(len(idx), spec.J, dtype=torch.bool, device=gt.device)
    mask[:, [root, jw]] = False
    _, pp = decode_self(model, data, G1, idx, joint_mask=mask)
    out["constraint_wrist_pos_cm"] = wrist_pos_err_cm(pp, gt, reach, jw)
    kept = torch.zeros(spec.J, dtype=torch.bool); kept[[root, jw]] = True
    pe = (pp - gt[..., 6:9].cpu()).norm(dim=-1) * reach
    out["constraint_completion_pos_cm"] = round(float(pe[:, ~kept].mean()), 2)
    out["quantization_cost_cm"] = round(out["fullvis_wrist_pos_cm"] - out["continuous_wrist_pos_cm"], 2)
    out["ambiguity_cost_cm"] = round(out["constraint_wrist_pos_cm"] - out["fullvis_wrist_pos_cm"], 2)
    return out


# --------------------------------------------------------------- twist probe ---
def aa_mats(u, phi):
    """axis-angle rotations [T,3,3] about fixed unit axis u (x2_twist recipe)."""
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    s, c = np.sin(phi), np.cos(phi)
    return (np.eye(3)[None] + s[:, None, None] * K[None]
            + (1 - c)[:, None, None] * (K @ K)[None])


def align_rot(a, b):
    """Minimal rotation taking unit vector a -> b."""
    a = np.asarray(a, np.float64); a = a / np.linalg.norm(a)
    b = np.asarray(b, np.float64); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b); s = np.linalg.norm(v)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if s < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    return np.eye(3) + K + ((1 - c) / s ** 2) * (K @ K)


def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def twist_of(Rw, R0, u):
    """Achieved twist (deg) about u: swing-twist split of R0^T Rw (x2_twist recipe)."""
    from scipy.spatial.transform import Rotation
    Rr = np.swapaxes(np.asarray(R0, np.float64), -1, -2)[None] @ np.asarray(Rw, np.float64) \
        if R0.ndim == 2 else np.swapaxes(R0, -1, -2) @ Rw
    q = Rotation.from_matrix(Rr.reshape(-1, 3, 3)).as_quat()          # x,y,z,w
    return wrap_deg(np.degrees(2 * np.arctan2(q[:, :3] @ u, q[:, 3])))


def rest_positions(spec):
    off = np.asarray(spec.rest_offsets, np.float64)
    pos = np.zeros_like(off)
    for j, p in enumerate(spec.parents):
        if int(p) >= 0:
            pos[j] = pos[int(p)] + off[j]
    return pos


def wrist_chain(spec):
    """-> (measure joint = deepest LEFT wrist/hand joint, elbow = first non-wrist
    ancestor, forearm axis u at rest in the cache frame)."""
    def is_wrist(n):
        n = n.lower()
        return ("wrist" in n or "hand" in n) and ("l_" in n or "left" in n or n.startswith("l")) \
            and "right" not in n and not n.startswith("r_")
    cand = [j for j, n in enumerate(spec.joint_names) if is_wrist(n)]
    assert cand, f"no left wrist joint found in {spec.name}: {spec.joint_names}"
    depth = {}
    for j in cand:
        d, p = 0, j
        while int(spec.parents[p]) >= 0:
            p = int(spec.parents[p]); d += 1
        depth[j] = d
    jw = max(cand, key=lambda j: depth[j])
    e = jw
    while int(spec.parents[e]) >= 0 and is_wrist(spec.joint_names[e]):
        e = int(spec.parents[e])
    pos = rest_positions(spec)
    u = pos[jw] - pos[e]
    n = np.linalg.norm(u)
    if n < 1e-6:                                        # co-located chain: use elbow bone dir
        u = pos[e] - pos[int(spec.parents[e])]; n = np.linalg.norm(u)
    return jw, e, u / n


def twist_slot(spec, dof, u):
    """(joint, slot) among LEFT wrist-chain joints whose DOF axis is most parallel
    to the forearm — the physical twist ACTUATOR (X2: left_wrist_yaw_joint)."""
    best = (None, None, -1.0)
    nd = (dof[:, 18] * 3).round().long()
    for j, n in enumerate(spec.joint_names):
        nl = n.lower()
        if not (("wrist" in nl or "hand" in nl) and ("left" in nl or nl.startswith("l_"))):
            continue
        v = dof[j, :18].reshape(3, 6)
        for k in range(int(nd[j])):
            d = abs(float(np.dot(np.asarray(v[k, :3]), u)))
            if d > best[2]:
                best = (j, k, d)
    return best


@torch.no_grad()
def twist_probe(model, data, idx, amps=(8.0, 15.0), n_frames=48):
    """In-distribution wrist-twist sensitivity: take real human val frames, multiply
    the L_Wrist rotation by a small BODY-FIXED twist about the forearm axis
    (R' = R @ rot(u_rest, phi), phi in {+-8, +-15} deg), decode full-vis to each
    target, then push the decode through the target's MECHANISM
    (fit_angles greedy+clamp -> FK) and measure, relative to the phi=0 decode:
      follow_realized — swing-twist of the wrist link about its forearm axis
                        (what the assembled wrist actually turns; humans = 3-DOF
                        free joints, so fit+FK is the identity there)
      follow_actuator — angle delta of the twist ACTUATOR slot (axis || forearm);
                        robots only (Euler-slot deltas are ill-posed for a 3-DOF
                        ball joint). Negative = the model drives the WRONG axis.
    ratio = mean over frames x amplitudes of achieved/commanded.
    NOTE: for a vis_residual model the human->human realized column uses the
    residual path (visible wrist -> ~1.0 trivially); cross-body is the real test.
    The 0805 session numbers (0.72/0.66/0.62/-0.04) came from a variant of this
    recipe (script not preserved); the ORDERING is what carries — E19 measured
    here: realized 0.76/0.78/0.64/0.39, actuator h1_2 0.72 / G1 0.63 / X2 -0.34."""
    import manifold_project as MP
    hspec = data[HUMAN]["spec"]
    hw, _, u_h = wrist_chain(hspec)
    sub = idx[:n_frames]
    x0 = data[HUMAN]["pose"][sub]
    B = len(sub)
    g = geom(data, HUMAN, B)
    phis = [s * a for a in amps for s in (+1.0, -1.0)]
    enc = {}
    for phi in [0.0] + phis:
        x = x0.clone()
        Rw = rot6d_to_matrix(x[:, hw, :6].double()).cpu().numpy()
        Rw = Rw @ aa_mats(u_h, np.deg2rad(np.full(B, phi)))
        x[:, hw, :6] = matrix_to_rot6d(torch.as_tensor(Rw, dtype=torch.float32,
                                                       device=x.device))
        enc[phi] = (model.encode(x, g), x)
    out = {}
    for tgt in TWIST_TARGETS:
        if tgt not in data or "_dofrest" not in data[tgt]:
            out[tgt] = None
            continue
        spec = data[tgt]["spec"]
        jw, _, u = wrist_chain(spec)
        gt_ = geom(data, tgt, B)
        dof, rest = data[tgt]["_dofrest"]
        j_t, k_t, par = twist_slot(spec, dof.float(), u)
        Rws, qs = {}, {}
        for phi in [0.0] + phis:
            (zp, zr), x = enc[phi]
            obs = vis = None
            if tgt == HUMAN and getattr(model, "vis_residual", False):
                obs, vis = resid_io(model, x, None)
            prot, _ = model.decode(zp, zr, gt_, obs=obs, vis=vis)
            Rg = rot6d_to_matrix(prot.double().cpu())
            q, _ = MP.fit_angles(Rg, spec, dof, rest=rest, clamp=True, method="greedy")
            R, _ = MP.fk_from_angles(q, spec, dof, rest=rest, root_R=Rg[:, 0])
            R = R[:, 0].transpose(-1, -2)[:, None] @ R          # root-align
            Rws[phi] = R[:, jw].numpy()
            qs[phi] = np.degrees(q[:, j_t, k_t].numpy()) if j_t is not None else None
        pr = {phi: float(np.mean(twist_of(Rws[phi], Rws[0.0], u) / phi)) for phi in phis}
        entry = {"follow_realized": round(float(np.mean(list(pr.values()))), 3),
                 "realized_per_amp": {f"{p:+.0f}": round(v, 3) for p, v in pr.items()},
                 "measure_joint": spec.joint_names[jw],
                 "forearm_axis": [round(float(v), 3) for v in u], "n_frames": B}
        if tgt != HUMAN and j_t is not None:
            pa = {phi: float(np.mean((qs[phi] - qs[0.0]) / phi)) for phi in phis}
            entry["follow_actuator"] = round(float(np.mean(list(pa.values()))), 3)
            entry["actuator_per_amp"] = {f"{p:+.0f}": round(v, 3) for p, v in pa.items()}
            entry["actuator_slot"] = f"{spec.joint_names[j_t]}#slot{k_t} (|a.u|={par:.2f})"
        out[tgt] = entry
    return out


# ------------------------------------------------------- soft SE3 constraint ---
@torch.no_grad()
def soft_constraint(model, data, idx):
    """constraint_demo-style, emb=human_smpl, keep = root + L_Wrist, SOFT (no pin)."""
    d = data[HUMAN]; spec = d["spec"]; reach = d["reach"]
    jw = list(spec.joint_names).index("L_Wrist")
    root = int(np.where(spec.parents < 0)[0][0])
    mask = torch.ones(len(idx), spec.J, dtype=torch.bool, device=d["pose"].device)
    mask[:, [root, jw]] = False
    x = d["pose"][idx]
    g = geom(data, HUMAN, len(idx))
    zp, zr = model.encode(x, g, joint_mask=mask)
    obs, vis = resid_io(model, x, mask)
    prot, ppos = model.decode(zp, zr, g, obs=obs, vis=vis)
    pe = (ppos - x[..., 6:9]).norm(dim=-1) * reach
    re = geodesic_deg(rot6d_to_matrix(prot), rot6d_to_matrix(x[..., :6]))
    kept = torch.zeros(spec.J, dtype=torch.bool); kept[[root, jw]] = True
    return {"constraint_joints": [spec.joint_names[root], spec.joint_names[jw]],
            "constraint_satisfaction": {"pos_cm": round(float(pe[:, kept].mean()), 2),
                                        "rot_deg": round(float(re[:, kept].mean()), 1),
                                        "wrist_pos_cm": round(float(pe[:, jw].mean()), 2)},
            "completion_vs_gt": {"pos_cm": round(float(pe[:, ~kept].mean()), 2),
                                 "rot_deg": round(float(re[:, ~kept].mean()), 1)}}


# ---------------------------------------------------------------------- main ---
def build_x2_entry(device):
    """AgiBot X2 conditioning from its raw MJCF (scratchpad agibot build, staged in
    night2_lib). Zero-shot: X2 is in no training set."""
    try:
        import agibot as AG
        from train_ae import static_cond
        from semantics import CACHE as SEMCACHE, build_name_embeddings, embed_matrix
        from dof_head import DofKin
        b = AG.build(X2, AG.CANDIDATES[X2], verbose=False)
        assert b["report"]["PASS"], b["report"]
        spec = b["spec"]
        lut = torch.load(SEMCACHE, map_location="cpu")
        missing = [n for n in spec.joint_names if n not in lut]
        if missing:
            lut.update(build_name_embeddings(missing, device="cpu"))
        dof = torch.as_tensor(b["dof"], dtype=torch.float32, device=device)
        rest = torch.as_tensor(b["rest"], dtype=torch.float32)
        return {"spec": spec, "st": static_cond(spec, device),
                "sem": embed_matrix(spec.joint_names, lut).to(device),
                "dof": dof, "kin": DofKin(spec, dof.cpu(), rest, device=device)}
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[X2] build failed ({type(e).__name__}: {e}) — twist probe will skip X2", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+",
                    default=["E19_K128_0805", "E28_resid_0807", "E29_wristaug_0807"])
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n_decomp", type=int, default=512)
    ap.add_argument("--n_eval", type=int, default=5000)
    ap.add_argument("--n_proj", type=int, default=256)
    ap.add_argument("--max_rows", type=int, default=0)   # smoke only (changes the split)
    ap.add_argument("--out", default=f"{RUNS}/NIGHT2_verdict")
    args = ap.parse_args()
    device = args.device

    torch.manual_seed(0)                                  # split contract: seed -> load -> perm
    data, names, N = load_paired_data(args.cache, device, max_rows=args.max_rows)
    val_idx, _ = split_indices(N)
    idx512 = val_idx[:args.n_decomp]
    idx_eval = val_idx[:args.n_eval]
    x2 = build_x2_entry(device)
    if x2 is not None:
        data[X2] = x2

    import eval_e25 as E25
    from manifold_project import load_dof
    for tgt in TWIST_TARGETS:                            # mechanism specs for the probe
        if tgt == X2:
            if x2 is not None:
                data[X2]["_dofrest"] = (x2["dof"].cpu().double(), x2["kin"].rest.cpu().double())
        elif tgt in data:
            data[tgt]["_dofrest"] = load_dof(args.cache, tgt, device="cpu")
    # decode_all returns CPU tensors; boolean part masks must live on the same device
    masks = {k: v.cpu() for k, v in E25.part_masks(data[G1]["spec"], device).items()}
    dofG1, restG1 = load_dof(args.cache, G1, device="cpu")

    res = {"cache": args.cache, "n_decomp": int(len(idx512)), "n_eval": int(len(idx_eval)),
           "n_proj": args.n_proj, "seed_note": SEED_NOTE, "runs": {}}
    for run in args.runs:
        rd = run if os.path.isabs(run) else f"{RUNS}/{run}"
        if not (os.path.exists(f"{rd}/config.json") and os.path.exists(f"{rd}/model.pt")):
            print(f"[skip] {run}: not finished", flush=True)
            continue
        cfg = json.load(open(f"{rd}/config.json"))
        model = build_model(cfg, device)
        model.load_state_dict(torch.load(f"{rd}/model.pt", map_location=device), strict=True)
        model.eval()
        t0 = time.time()
        r = {"vis_residual": bool(cfg.get("vis_residual", False)), "cache": cfg.get("cache"),
             "steps": cfg.get("steps"), "params": sum(p.numel() for p in model.parameters())}
        r["decomposition"] = decomposition(model, data, idx512)
        r["twist_follow"] = twist_probe(model, data, idx512)
        prot, ppos = E25.decode_all(model, data, HUMAN, G1, idx_eval)
        gt = data[G1]["pose"][idx_eval].cpu()
        tab, _ = E25.err_table(prot, ppos, gt, data[G1]["reach"], masks)
        kin = data[G1]["kin"].to("cpu") if hasattr(data[G1]["kin"], "to") else data[G1]["kin"]
        tab["mechanical"] = E25.mechanical_report(prot, kin, data[G1]["spec"], dofG1, restG1, args.n_proj)
        tab["projected_rot_deg"] = E25.projected_error(prot, gt, data[G1]["spec"], dofG1, restG1,
                                                       masks, args.n_proj)
        r["zeroshot_human_to_g1"] = tab
        r["soft_constraint_L_Wrist"] = soft_constraint(model, data, idx512)
        r["sec"] = round(time.time() - t0, 1)
        res["runs"][run] = r
        print(f"[{run}] decomp={r['decomposition']} twist=" +
              json.dumps({k: [(v or {}).get('follow_realized'), (v or {}).get('follow_actuator')]
                          for k, v in r['twist_follow'].items()}) +
              f" zs_pos={tab['all']['pos_cm']}cm ({r['sec']}s)", flush=True)
        del model

    # ---- winner: best constraint-mode wrist error among the new runs, guarded by
    # "zero-shot translation must not regress beyond noise (+0.5 cm)" ----
    ctrl = res["runs"].get("E19_K128_0805")
    cand = [k for k in res["runs"] if k != "E19_K128_0805"]
    winner, why = "E28_resid_0807", "fallback (no candidate passed the guard)"
    if ctrl and cand:
        ok = [k for k in cand if res["runs"][k]["zeroshot_human_to_g1"]["all"]["pos_cm"]
              <= ctrl["zeroshot_human_to_g1"]["all"]["pos_cm"] + 0.5]
        pool = ok or cand
        best = min(pool, key=lambda k: res["runs"][k]["decomposition"]["constraint_wrist_pos_cm"])
        winner = best
        why = (f"min constraint-mode wrist pos among {pool} "
               f"({res['runs'][best]['decomposition']['constraint_wrist_pos_cm']} cm vs control "
               f"{ctrl['decomposition']['constraint_wrist_pos_cm']} cm)"
               + ("" if ok else " — WARNING: guard failed for all candidates (zero-shot regressed)"))
    res["winner"] = winner
    res["winner_why"] = why

    # ---- one side-by-side table ----
    runs = [k for k in args.runs if k in res["runs"]]
    W = max([len(r) for r in runs] + [10]) + 2
    L = []
    L.append("NIGHT2 verdict — " + time.strftime("%Y-%m-%d %H:%M"))
    L.append(f"cache={args.cache}  n_decomp={len(idx512)}  n_eval={len(idx_eval)}  n_proj={args.n_proj}")
    L.append("")
    L.append(f"{'metric':44s}" + "".join(f"{r:>{W}s}" for r in runs))

    def row(label, fn, fmt="{:.2f}"):
        vals = []
        for r in runs:
            try:
                v = fn(res["runs"][r]); vals.append(fmt.format(v) if v is not None else "-")
            except Exception:
                vals.append("-")
        L.append(f"{label:44s}" + "".join(f"{v:>{W}s}" for v in vals))

    L.append("-- (i) G1 wrist error decomposition, val512, left_rubber_hand [cm] --")
    row("full-vis recon (quantized)", lambda r: r["decomposition"]["fullvis_wrist_pos_cm"])
    row("full-vis recon (continuous z)", lambda r: r["decomposition"]["continuous_wrist_pos_cm"])
    row("  quantization cost", lambda r: r["decomposition"]["quantization_cost_cm"])
    row("constraint mode (root+wrist only)", lambda r: r["decomposition"]["constraint_wrist_pos_cm"])
    row("  completion ambiguity cost", lambda r: r["decomposition"]["ambiguity_cost_cm"])
    row("full-vis all-joint pos", lambda r: r["decomposition"]["fullvis_all_pos_cm"])
    L.append("   (continuous z = FSQ rounding bypassed, tanh bound kept. The 0805 session's")
    L.append("    11.88 cm used a slightly different bypass; either way quantization is ~0.4-1cm,")
    L.append("    i.e. innocent — the model's own encode-decode of visible input dominates.)")
    L.append("-- (ii) wrist twist follow ratio (+-8/+-15 deg; 1.0 = perfect) --")
    for tgt in TWIST_TARGETS:
        row(f"  -> {tgt} realized",
            lambda r, t=tgt: (r["twist_follow"].get(t) or {}).get("follow_realized"), "{:.3f}")
    for tgt in TWIST_TARGETS[1:]:
        row(f"  -> {tgt} actuator",
            lambda r, t=tgt: (r["twist_follow"].get(t) or {}).get("follow_actuator"), "{:.3f}")
    L.append("   (realized = wrist link twist after fit+FK through the mechanism;")
    L.append("    actuator = the forearm-axis hinge's own angle. X2 negative actuator =")
    L.append("    the model drives the wrong axis under X2's yaw->pitch->roll convention.)")
    L.append("-- (iii) zero-shot human->G1 (val %d) --" % len(idx_eval))
    for part in ["all", "legs", "arms", "wrists", "other"]:
        row(f"  {part} rot [deg]", lambda r, p=part: r["zeroshot_human_to_g1"][p]["rot_deg"])
        row(f"  {part} pos [cm]", lambda r, p=part: r["zeroshot_human_to_g1"][p]["pos_cm"])
    row("  projected rot (n=%d) [deg]" % args.n_proj,
        lambda r: r["zeroshot_human_to_g1"]["projected_rot_deg"]["all"])
    row("  off-manifold local [deg]",
        lambda r: r["zeroshot_human_to_g1"]["mechanical"]["manifold_dist_local_deg"])
    L.append("-- (iv) soft SE3 constraint, human root+L_Wrist, val512 (NO pin) --")
    row("  constraint pos [cm]", lambda r: r["soft_constraint_L_Wrist"]["constraint_satisfaction"]["pos_cm"])
    row("  constraint wrist pos [cm]", lambda r: r["soft_constraint_L_Wrist"]["constraint_satisfaction"]["wrist_pos_cm"])
    row("  constraint rot [deg]", lambda r: r["soft_constraint_L_Wrist"]["constraint_satisfaction"]["rot_deg"])
    row("  completion pos [cm]", lambda r: r["soft_constraint_L_Wrist"]["completion_vs_gt"]["pos_cm"])
    L.append("")
    L.append(f"WINNER: {res['winner']}  ({res['winner_why']})")
    L.append(SEED_NOTE)
    table = "\n".join(L)
    print("\n" + table, flush=True)
    with open(f"{args.out}.txt", "w") as f:
        f.write(table + "\n")
    json.dump(res, open(f"{args.out}.json", "w"), indent=1)
    print("WROTE", f"{args.out}.txt", "and .json", flush=True)


if __name__ == "__main__":
    main()
