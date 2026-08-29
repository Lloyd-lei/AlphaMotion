"""Synthetic embodiments: bodies whose JOINT FRAMES are perturbed (tilted /
non-orthogonal / tighter limit boxes) and whose motion is REGENERATED so it is
exactly what that body can do.

Why not just augment the axis features. The axes in `{emb}_dof.npz` are a physical
fact about the robot that the cached rotations already obey: `{emb}_paired_local6d.pt`
holds rotations lying on the manifold {rot(a0,q0) rot(a1,q1) rot(a2,q2) R_rest}.
Rotating the axes and keeping the rotations tells the model a lie -- the descriptor
would no longer describe the motion. So instead we build a NEW body and give it its
OWN motion:

    variant descriptor  D' = perturb(D)               (axes, extra slots, limit box)
    variant motion      R'(t) = argmin_{R in M(D')} |R - R(t)|   (manifold_project,
                                                       method="global", box-clamped)

R'(t) is on the variant's manifold BY CONSTRUCTION, frame-for-frame aligned with the
source (and therefore with every other embodiment in the cache), so combo / InfoNCE
sampling keeps working unchanged.

What the perturbations are for (measured gap, see `--stats`):
  * every one of the 18 cached bodies has max |dot| between two axes of the SAME
    merged frame = 0.000. Apptronik Apollo has 0.259. Non-orthogonal frames are
    literally absent from training -> op `skew`.
  * Apollo has 3 three-DOF merged frames (both hips + torso); only kuavo_s45 has
    any (2, orthogonal) -> op `promote`.
  * Apollo has 34% of its axes more than 5 deg off a canonical +-X/Y/Z direction
    (worst cached body: unitree_h2, 26%; ten bodies are at 0%) -> op `tilt`.
  * Apollo's median half-range is 49.5 deg, the tightest in the set (next: 63.9,
    widest 180) with 47% of boxes off-center by >20 deg -> op `tighten`.

Geometry is NOT touched: parents, rest offsets and joint names are copied from the
source, so a variant is "the same robot with different hinges", the cleanest possible
isolation of the mechanical descriptor.

Output: a NEW cache dir = hard links to the source cache's 18 bodies + the synthetic
ones, plus manifest.json (all bodies) and synth_manifest.json (every perturbation,
its RNG seed, and the validation numbers).

Usage:
  python make_synth_embodiments.py --stats                      # gap table only
  python make_synth_embodiments.py --dry                        # descriptors + checks
  python make_synth_embodiments.py --out /var/tmp/.../cache_xwbc_lafan_synth
"""
from __future__ import annotations
import os, sys, re, json, time, shutil, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SkeletonSpec, rot6d_to_matrix, matrix_to_rot6d, geodesic_deg, fk_global
from train_global import build_global
import manifold_project as MP

SRC_CACHE = "/var/tmp/alphamotion/topo_exp/cache_xwbc_lafan"
OUT_CACHE = "/var/tmp/alphamotion/topo_exp/cache_xwbc_lafan_synth"

LEG = r"(hip|knee|ankle|thigh|shank|calf|leg|foot)"
ARM = r"(shoulder|elbow|wrist|arm|hand)"

# ---------------------------------------------------------------------------
# The synthetic bodies. 6 bodies from 6 DIFFERENT source robots, so no single
# vendor's quirk can explain a result. Sources exclude the three holdouts
# (unitree_g1_29dof / unitree_g1_23dof / pal_talos) — a synthetic derived from a
# held-out body would leak it into training.
# ---------------------------------------------------------------------------
RECIPES = {
    # 1. non-orthogonal 3-DOF triads: kuavo already HAS two 3-DOF arm frames and
    #    five 2-DOF frames; skewing them is the purest "Apollo triad" analogue.
    "synth_kuavo_skew": dict(src="kuavo_s45", seed=101, ops=[
        dict(op="skew",  sel=r".", min_dof=2, deg=[8, 22]),
        dict(op="tilt",  sel=r"(leg|zarm)", deg=[10, 25]),
    ]),
    # 2. Apollo's actual hip: a co-located 3-DOF gimbal, tilted and non-orthogonal.
    #    unitree_h1 is the most rigidly canonical body we have (all axes exactly
    #    +-X/Y/Z, hip triple |dot| = 0.0000), so it is the hardest case to fix.
    "synth_h1_hipball": dict(src="unitree_h1", seed=102, ops=[
        dict(op="promote", sel=r"(hip_yaw|shoulder_pitch)", to=3, half=[35, 70], ctr=[-25, 25]),
        dict(op="skew",  sel=r"(hip_yaw|shoulder_pitch)", min_dof=2, deg=[10, 25]),
        dict(op="tilt",  sel=LEG + r"|" + ARM, deg=[15, 40]),
    ]),
    # 3. tilt-only, strong: does the decoder track a rotated frame at all?
    "synth_r1_tilt": dict(src="unitree_r1", seed=103, ops=[
        dict(op="tilt",  sel=r".", deg=[20, 45]),
    ]),
    # 4. tight + strongly OFF-CENTER boxes, mild tilt. This is Apollo's other
    #    outlier axis and, per the control measurement, the one that actually costs
    #    pose error: Apollo's infeasibility sits on its 1-DOF (22.8 deg) and 2-DOF
    #    (32.1 deg) joints, not on its 3-DOF triads (0.15 deg). The shift range is
    #    +-35 deg so the variant reproduces Apollo's "47% of boxes off-center by
    #    more than 20 deg"; +-20 produced ~0% of those.
    "synth_t1_tight": dict(src="booster_t1", seed=104, ops=[
        dict(op="tighten", sel=r".", scale=[0.4, 0.65], shift=[-35, 35]),
        dict(op="tilt",    sel=r".", deg=[5, 15]),
    ]),
    # 5. 2-DOF frames made non-orthogonal + moderate tilt on a big tree.
    "synth_gr3_skew": dict(src="fourier_gr3", seed=105, ops=[
        dict(op="skew", sel=r".", min_dof=2, deg=[10, 25]),
        dict(op="tilt", sel=ARM + r"|" + LEG, deg=[10, 30]),
    ]),
    # 6. everything at once, on a body whose limits are the widest in the set
    #    (median half-range 180 deg) so tightening actually bites.
    "synth_hi_mixed": dict(src="hightorque_hi", seed=106, ops=[
        dict(op="promote", sel=r"(hip_pitch|shoulder_pitch)", to=3, half=[30, 60], ctr=[-20, 20]),
        dict(op="skew",    sel=r".", min_dof=2, deg=[8, 20]),
        dict(op="tilt",    sel=r".", deg=[10, 30]),
        dict(op="tighten", sel=r".", scale=[0.3, 0.55], shift=[-30, 30]),
    ]),
}


# ---------------------------------------------------------------------------
# NIGHT2 / E29 wrist-axis augmentation (cache_xwbc_lafan_wrist). Measured failure it
# attacks: the wrist TWIST channel does not transfer to unseen wrist-axis conventions
# (twist follow ratio: human 0.72 / h1_2 0.66 / G1 0.62 / zero-shot X2 -0.04; X2's
# wrist chain order is yaw->pitch->roll where A3/our trained bodies are
# roll->pitch->yaw). E22's lesson: the objective must ask for the rule — here the
# rule IS the wrist axis mapping, so the variants permute/flip ONLY the wrist DOF
# and re-solve the motion on the new manifold; everything else is byte-identical.
# Sources are TRAINED bodies (holdouts g1_29/g1_23/talos untouched); only h1_2 and
# gr3 have multi-joint wrist chains, so they carry the four variants.
# ---------------------------------------------------------------------------
WRIST_RECIPES = {
    # h1_2 chain is roll(Z)->pitch(X)->yaw(Y); X2's convention is yaw->pitch->roll.
    "wrist_h1_2_ypr": dict(src="unitree_h1_2", seed=201, ops=[
        dict(op="chainperm", chains=[
            dict(joints=["left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link"], perm=[2, 1, 0]),
            dict(joints=["right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link"], perm=[2, 1, 0]),
        ]),
    ]),
    # a third convention on the same chain: pitch->yaw->roll.
    "wrist_h1_2_pyr": dict(src="unitree_h1_2", seed=202, ops=[
        dict(op="chainperm", chains=[
            dict(joints=["left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link"], perm=[1, 2, 0]),
            dict(joints=["right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link"], perm=[1, 2, 0]),
        ]),
    ]),
    # gr3: flip the wrist ROLL axis sign (slot 1 of *_hand_pitch_link, the Z axis)
    # and the yaw axis sign — same frame, opposite-handed convention.
    "wrist_gr3_flip": dict(src="fourier_gr3", seed=203, ops=[
        dict(op="flipslot", sel=r"hand_pitch_link", slot=1),
        dict(op="flipslot", sel=r"hand_yaw_link", slot=0),
    ]),
    # gr3: swap the composition order of the 2-DOF wrist frame (roll-then-pitch).
    "wrist_gr3_swap": dict(src="fourier_gr3", seed=204, ops=[
        dict(op="slotperm", sel=r"hand_pitch_link", perm=[1, 0]),
    ]),
}



# ------------------------------------------------------------ descriptor ops ---

def unpack(D):
    """[J,19] -> axes [J,3,3], limited [J,3], half [J,3] rad, ctr [J,3] rad, nd [J]."""
    J = D.shape[0]
    v = D[:, :18].reshape(J, 3, 6).copy()
    nd = np.round(D[:, 18] * 3).astype(int)
    return v[:, :, :3], v[:, :, 3], v[:, :, 4] * np.pi, v[:, :, 5] * np.pi, nd


def pack(axes, limited, half, ctr, nd):
    J = axes.shape[0]
    v = np.zeros((J, 3, 6), np.float32)
    v[:, :, :3] = axes
    v[:, :, 3] = limited
    v[:, :, 4] = half / np.pi
    v[:, :, 5] = ctr / np.pi
    for j in range(J):                       # absent slots are EXACTLY zero
        v[j, nd[j]:] = 0.0
    return np.concatenate([v.reshape(J, 18), (nd / 3.0)[:, None]], 1).astype(np.float32)


def rand_rot(rng, deg_lo, deg_hi):
    """Random rotation with angle drawn uniformly in [deg_lo, deg_hi]."""
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    th = np.deg2rad(rng.uniform(deg_lo, deg_hi))
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def select(names, nd, sel, min_dof=1, exact_dof=None):
    pat = re.compile(sel, re.I)
    out = []
    for j, n in enumerate(names):
        if nd[j] < min_dof or not pat.search(n):
            continue
        if exact_dof is not None and nd[j] != exact_dof:
            continue
        out.append(j)
    return out


def apply_ops(D, names, ops, seed):
    """-> (D' [J,19], per-op record). Deterministic in `seed`."""
    rng = np.random.default_rng(seed)
    axes, limited, half, ctr, nd = unpack(D)
    rec = []
    for spec_op in ops:
        op = spec_op["op"]
        if op == "chainperm":
            # NIGHT2 / E29: permute which joint of a (wrist) chain carries which DOF.
            # chains = [{joints: [names...], perm: [...]}]: joint i takes the FULL dof
            # rows (axes+limits+slot count) of joints[perm[i]]. Geometry/names stay —
            # exactly the "same links, different axis convention" gap that breaks
            # zero-shot twist on X2 (its joint NAMED wrist_roll is a perpendicular axis).
            for ch in spec_op["chains"]:
                idxs = [names.index(n) for n in ch["joints"]]
                perm = list(ch["perm"])
                assert sorted(perm) == list(range(len(idxs))), perm
                src = [(axes[j].copy(), limited[j].copy(), half[j].copy(),
                        ctr[j].copy(), int(nd[j])) for j in idxs]
                for i, j in enumerate(idxs):
                    a, l, h_, c, n = src[perm[i]]
                    axes[j], limited[j], half[j], ctr[j], nd[j] = a, l, h_, c, n
                rec.append(dict(op=op, joints=ch["joints"], perm=perm))
            continue
        if op == "slotperm":
            # permute the COMPOSITION ORDER of a multi-DOF frame's slots:
            # rot(a0,q0)rot(a1,q1) -> rot(a1,q0')rot(a0,q1') is a different manifold.
            perm = list(spec_op["perm"])
            js = [j for j in select(names, nd, spec_op["sel"], min_dof=1)
                  if nd[j] == len(perm)]
            for j in js:
                axes[j, :nd[j]] = axes[j, perm]
                limited[j, :nd[j]] = limited[j, perm]
                half[j, :nd[j]] = half[j, perm]
                ctr[j, :nd[j]] = ctr[j, perm]
            rec.append(dict(op=op, joints=[names[j] for j in js], perm=perm))
            continue
        if op == "flipslot":
            # flip one DOF axis' SIGN (rot(-a, q) == rot(a, -q), so the limit box
            # center mirrors; the half-range is unchanged).
            k = int(spec_op["slot"])
            js = [j for j in select(names, nd, spec_op["sel"], min_dof=1) if nd[j] > k]
            for j in js:
                axes[j, k] = -axes[j, k]
                ctr[j, k] = -ctr[j, k]
            rec.append(dict(op=op, joints=[names[j] for j in js], slot=k))
            continue
        if op == "promote":
            js = select(names, nd, spec_op["sel"], min_dof=1, exact_dof=1)
            for j in js:
                a0 = axes[j, 0] / np.linalg.norm(axes[j, 0])
                # complete a0 to an orthonormal frame, then take the extra slots from it
                tmp = np.eye(3)[np.argmin(np.abs(a0))]
                a1 = np.cross(a0, tmp); a1 /= np.linalg.norm(a1)
                a2 = np.cross(a0, a1)
                new = [a0, a1, a2][:spec_op["to"]]
                for k, a in enumerate(new):
                    axes[j, k] = a
                    if k >= nd[j]:
                        limited[j, k] = 1.0
                        half[j, k] = np.deg2rad(rng.uniform(*spec_op["half"]))
                        ctr[j, k] = np.deg2rad(rng.uniform(*spec_op["ctr"]))
                nd[j] = spec_op["to"]
            rec.append(dict(op=op, joints=[names[j] for j in js], to=spec_op["to"]))
        elif op == "tilt":                       # rigid rotation of a WHOLE frame
            js = select(names, nd, spec_op["sel"], spec_op.get("min_dof", 1))
            ang = {}
            for j in js:
                P = rand_rot(rng, *spec_op["deg"])
                axes[j, :nd[j]] = axes[j, :nd[j]] @ P.T
                ang[names[j]] = round(float(np.rad2deg(np.arccos(
                    np.clip((np.trace(P) - 1) / 2, -1, 1)))), 1)
            rec.append(dict(op=op, deg_per_joint=ang))
        elif op == "skew":                       # each slot rotated INDEPENDENTLY
            js = select(names, nd, spec_op["sel"], spec_op.get("min_dof", 2))
            got = {}
            for j in js:
                for k in range(nd[j]):
                    axes[j, k] = rand_rot(rng, *spec_op["deg"]) @ axes[j, k]
                v = axes[j, :nd[j]]
                v = v / np.linalg.norm(v, axis=-1, keepdims=True)
                G = np.abs(v @ v.T); np.fill_diagonal(G, 0)
                got[names[j]] = round(float(G.max()), 3)
            rec.append(dict(op=op, max_abs_dot_per_joint=got))
        elif op == "tighten":
            js = select(names, nd, spec_op["sel"], spec_op.get("min_dof", 1))
            n = 0
            for j in js:
                for k in range(nd[j]):
                    if limited[j, k] < 0.5:      # unlimited -> give it a real box
                        limited[j, k] = 1.0
                        half[j, k] = np.deg2rad(rng.uniform(60, 110))
                        ctr[j, k] = 0.0
                    half[j, k] *= rng.uniform(*spec_op["scale"])
                    ctr[j, k] += np.deg2rad(rng.uniform(*spec_op["shift"]))
                    n += 1
            rec.append(dict(op=op, n_slots=n, scale=spec_op["scale"], shift=spec_op["shift"]))
        else:
            raise ValueError(op)
        nrm = np.linalg.norm(axes, axis=-1, keepdims=True)
        axes = np.where(nrm > 1e-9, axes / np.maximum(nrm, 1e-9), 0.0)
    return pack(axes, limited, half, ctr, nd), rec


def descriptor_stats(D):
    axes, limited, half, ctr, nd = unpack(D)
    J = D.shape[0]
    act = np.zeros((J, 3), bool)
    for j in range(J):
        act[j, :nd[j]] = True
    a = axes[act]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    tilt = np.degrees(np.arccos(np.abs(a).max(-1).clip(0, 1)))
    non = [0.0]
    for j in range(J):
        if nd[j] >= 2:
            v = axes[j, :nd[j]]
            v = v / np.linalg.norm(v, axis=-1, keepdims=True)
            G = np.abs(v @ v.T); np.fill_diagonal(G, 0)
            non.append(float(G.max()))
    lm = limited[act] > 0.5
    return dict(J=J, nDOF=int(act.sum()), n3dof=int((nd == 3).sum()), n2dof=int((nd == 2).sum()),
                pct_limited=round(100 * float(lm.mean()), 1),
                tilt_mean=round(float(tilt.mean()), 1), tilt_max=round(float(tilt.max()), 1),
                tilt_gt5_pct=round(100 * float((tilt > 5).mean()), 1),
                nonorth_max=round(max(non), 3),
                half_med_deg=round(float(np.degrees(np.median(half[act][lm]))), 1) if lm.any() else 0.0,
                absctr_med_deg=round(float(np.degrees(np.median(np.abs(ctr[act][lm])))), 1) if lm.any() else 0.0,
                absctr_gt20_pct=round(100 * float(
                    (np.abs(np.degrees(ctr[act][lm])) > 20).mean()), 1) if lm.any() else 0.0)


# ---------------------------------------------------------------- generation ---

def load_body(cache, name):
    d = np.load(f"{cache}/{name}_spec.npz", allow_pickle=True)
    spec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                        [str(x) for x in d["joint_names"]])
    z = np.load(f"{cache}/{name}_dof.npz", allow_pickle=True)
    return spec, z, d


def part_of(name):
    n = name.lower()
    if re.search(LEG, n):
        return "leg"
    if re.search(ARM, n):
        return "arm"
    return "torso"


def per_part(err_j, names):
    p = np.array([part_of(n) for n in names])
    out = {k: round(float(err_j[:, np.where(p == k)[0]].mean()), 2)
           for k in ["leg", "arm", "torso"] if (p == k).any()}
    out["all"] = round(float(err_j.mean()), 2)
    return out


def project_motion(local6d, spec, D, rest, device, batch=4096, lm_iters=30, probe=None):
    """Source local6d -> variant local6d (on the variant's manifold) + diagnostics."""
    N = local6d.shape[0]
    rows = np.arange(N) if probe is None else \
        np.linspace(0, N - 1, min(probe, N)).astype(int)
    dof_d = torch.as_tensor(D, device=device, dtype=torch.float64)
    rest_d = torch.as_tensor(rest, device=device, dtype=torch.float64)
    par = np.asarray(spec.parents)
    out = torch.empty(len(rows), spec.J, 6, dtype=torch.float32)
    geo_sum = torch.zeros(spec.J, dtype=torch.float64)
    fk_err = 0.0
    viol = 0.0
    n_done = 0
    for i in range(0, len(rows), batch):
        idx = rows[i:i + batch]
        b = local6d[idx].to(device)
        gR, _ = fk_global(b, spec)                                  # [n,J,3,3]
        Rg = gR.double()
        r6, _, q = MP.project(Rg, spec, dof_d, rest=rest_d, method="global",
                              lm_iters=lm_iters)
        Rp = rot6d_to_matrix(r6)
        geo_sum += geodesic_deg(Rg, Rp).sum(0).cpu()
        # variant local rotations (what gets cached)
        pidx = torch.as_tensor(np.maximum(par, 0), device=device)
        L = Rp[:, pidx].transpose(-1, -2) @ Rp
        L[:, par < 0] = Rp[:, par < 0]
        out[i:i + len(idx)] = matrix_to_rot6d(L).float().cpu()
        # FK round trip of the stored local6d
        gR2, _ = fk_global(matrix_to_rot6d(L).float(), spec)
        fk_err = max(fk_err, float(geodesic_deg(gR2.double(), Rp).max()))
        # residual limit violation of the emitted motion
        v = MP._limits(dof_d, 1.0)
        lo, hi, lim = v
        over = torch.maximum(q - hi[None], lo[None] - q).clamp_min(0)
        A, active = MP._slot_axes(dof_d)
        m = lim & active
        if m.any():
            viol = max(viol, float(torch.rad2deg(over[:, m].max())))
        n_done += len(idx)
    return out, dict(recognizability_deg=per_part((geo_sum / n_done)[None].numpy(),
                                                  spec.joint_names),
                     fk_roundtrip_max_deg=round(fk_err, 6),
                     residual_limit_viol_deg=round(viol, 4), frames=n_done)


def upright(local6d, spec, device, n=512):
    """Cached convention: the skeleton stands along +Y. Returns (top joint, mean Y,
    max |other axis|) — Y must dominate."""
    idx = np.linspace(0, local6d.shape[0] - 1, min(n, local6d.shape[0])).astype(int)
    _, gp = fk_global(local6d[idx].to(device), spec)
    m = gp.mean(0).cpu().numpy()                                     # [J,3]
    top = int(np.argmax(m[:, 1]))
    return dict(top_joint=spec.joint_names[top], mean_pos=[round(float(x), 1) for x in m[top]],
                ok=bool(m[top, 1] > 2 * np.abs(m[top, [0, 2]]).max()))


def link_source(src, out):
    os.makedirs(out, exist_ok=True)
    for f in sorted(os.listdir(src)):
        s, d = f"{src}/{f}", f"{out}/{f}"
        if os.path.exists(d) or os.path.islink(d):
            continue
        if f == "manifest.json":
            continue
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_CACHE)
    ap.add_argument("--out", default=OUT_CACHE)
    ap.add_argument("--set", default="default", choices=["default", "wrist"],
                    help="recipe set: 'wrist' = NIGHT2/E29 wrist-axis-convention variants")
    ap.add_argument("--only", default=None, help="comma list of synthetic names")
    ap.add_argument("--probe", type=int, default=0, help="project only this many frames (dev)")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lm_iters", type=int, default=30)
    ap.add_argument("--recog_gate_deg", type=float, default=25.0,
                    help="drop a variant whose projected motion drifts more than this")
    ap.add_argument("--dry", action="store_true", help="descriptors + stats, no motion")
    ap.add_argument("--stats", action="store_true", help="print the source-vs-Apollo gap table")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.stats:
        man = json.load(open(f"{args.src}/manifest.json"))["embodiments"]
        print(f"{'body':24s} " + " ".join(f"{k:>14s}" for k in
              ["J", "nDOF", "n3dof", "tilt_gt5_pct", "nonorth_max", "half_med_deg"]))
        for e in man:
            _, z, _ = load_body(args.src, e)
            s = descriptor_stats(z["dof"])
            print(f"{e:24s} " + " ".join(f"{str(s[k]):>14s}" for k in
                  ["J", "nDOF", "n3dof", "tilt_gt5_pct", "nonorth_max", "half_med_deg"]))
        return

    recipes = WRIST_RECIPES if args.set == "wrist" else RECIPES
    names = list(recipes) if not args.only else [x for x in args.only.split(",") if x]
    man_src = json.load(open(f"{args.src}/manifest.json"))
    report = {"src_cache": args.src, "out_cache": args.out, "rows": man_src["rows"],
              "recipes": {}, "dropped": {}}
    if not args.dry:
        link_source(args.src, args.out)

    kept = []
    for nm in names:
        rc = recipes[nm]
        t0 = time.time()
        spec, z, raw = load_body(args.src, rc["src"])
        D0 = np.asarray(z["dof"], np.float32)
        rest = np.asarray(z["rest"], np.float32)
        D1, rec = apply_ops(D0, spec.joint_names, rc["ops"], rc["seed"])
        s0, s1 = descriptor_stats(D0), descriptor_stats(D1)
        entry = {"src": rc["src"], "seed": rc["seed"], "ops": rc["ops"], "applied": rec,
                 "stats_src": s0, "stats_synth": s1}
        print(f"\n=== {nm}  <- {rc['src']}")
        print(f"    src   {json.dumps(s0)}")
        print(f"    synth {json.dumps(s1)}")
        if args.dry:
            report["recipes"][nm] = entry
            continue
        local6d = torch.load(f"{args.src}/{rc['src']}_paired_local6d.pt")
        new6d, diag = project_motion(local6d, spec, D1, rest, device, batch=args.batch,
                                     lm_iters=args.lm_iters,
                                     probe=args.probe or None)
        entry.update(diag)
        entry["upright"] = upright(new6d, spec, device)
        entry["sec"] = round(time.time() - t0, 1)
        print(f"    motion {json.dumps(diag)}  upright={entry['upright']}  {entry['sec']}s")
        drift = diag["recognizability_deg"]["all"]
        if drift > args.recog_gate_deg or not entry["upright"]["ok"]:
            entry["DROPPED"] = f"drift {drift} deg (gate {args.recog_gate_deg}) / upright"
            report["dropped"][nm] = entry
            print(f"    !! DROPPED: {entry['DROPPED']}")
            continue
        if not args.probe:
            assert new6d.shape[0] == local6d.shape[0], (new6d.shape, local6d.shape)
            torch.save(new6d, f"{args.out}/{nm}_paired_local6d.pt")
            np.savez(f"{args.out}/{nm}_spec.npz", name=nm, parents=spec.parents,
                     rest_offsets=spec.rest_offsets, norm_bonelen=spec.norm_bonelen,
                     joint_names=np.array(spec.joint_names, dtype=object),
                     height=spec.height)
            np.savez(f"{args.out}/{nm}_dof.npz", dof=D1, rest=rest,
                     rest6d=np.ascontiguousarray(
                         rest[:, :, :2].transpose(0, 2, 1)).reshape(-1, 6),
                     joint_names=np.array(spec.joint_names, dtype=object),
                     slot_joints=np.array([f"synth<-{rc['src']}"] * spec.J, dtype=object))
        report["recipes"][nm] = entry
        kept.append(nm)

    if not args.dry and not args.probe:
        man = dict(man_src)
        man["embodiments"] = list(man_src["embodiments"]) + kept
        man["synthetic"] = kept
        json.dump(man, open(f"{args.out}/manifest.json", "w"), indent=2)
        # row alignment across EVERY body in the new cache
        rows = {}
        for e in man["embodiments"]:
            rows[e] = int(torch.load(f"{args.out}/{e}_paired_local6d.pt",
                                     map_location="meta").shape[0])
        report["row_alignment"] = {"all_equal": len(set(rows.values())) == 1, "rows": rows}
        report["kept"] = kept
        json.dump(report, open(f"{args.out}/synth_manifest.json", "w"), indent=2)
        print("\nrow alignment:", report["row_alignment"]["all_equal"], set(rows.values()))
        print("WROTE", args.out, "kept:", kept)
    else:
        print(json.dumps(report, indent=2)[:4000])


if __name__ == "__main__":
    main()
