"""Zero-shot decode onto a robot that has NO cached motion, judged by its own
mechanics: joint-limit violations BEFORE projection, how far the feasibility
projection has to move the pose, and (where ground truth exists) per-part geodesic
error against the cache's own retarget.

Why this file: the tokenizer never sees the target's motion, only its descriptor
(rest offsets + joint names + `{emb}_dof.npz` axes/limits). So a body whose hinge
triad is TILTED and NON-ORTHOGONAL (Apptronik Apollo: hip_ie = [-0.26,0.84,-0.48],
ie.fe = -0.26) is a pure out-of-distribution test of whether the decoder actually
reads the axis features or just pattern-matches the +-X/Y/Z triads it trained on.

Targets can be
  * an MJCF path  (built here with prep_gmr_cache.build_merge/build_spec +
    prep_dof_features.mjcf_dof — the SAME code that built the cache, so the
    descriptor is byte-comparable), or
  * an embodiment name already in a cache (then GT is available and per-part
    geodesic error is reported too).

Usage:
  python eval_newbody.py --run RUN --targets apollo,pal_talos,unitree_g1_29dof \
      --clips MoSh__50020__salsa_stageii --out out.json
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SkeletonSpec, rot6d_to_matrix, matrix_to_rot6d, geodesic_deg
from dual_common import build_model
from train_ae import static_cond
from train_global import build_global
import manifold_project as MP

MENAGERIE = "/var/tmp/alphamotion/robot_models/menagerie"
CACHE_AMASS = "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass"
CACHE_LAFAN = "/var/tmp/alphamotion/topo_exp/cache_xwbc_lafan"

# extra bodies that exist only as an MJCF (no motion anywhere)
MJCF_BODIES = {
    "apollo": f"{MENAGERIE}/apptronik_apollo/apptronik_apollo.xml",
    "op3": f"{MENAGERIE}/robotis_op3/op3.xml",
}

PART_KW = {
    "leg": ("hip", "knee", "ankle", "thigh", "shin", "calf", "foot", "toe", "leg"),
    "arm": ("shoulder", "elbow", "wrist", "hand", "arm", "clav", "scap"),
    "torso": ("waist", "torso", "spine", "pelvis", "chest", "trunk", "neck", "head", "base"),
}


# ------------------------------------------------------------------ targets ---

def build_from_mjcf(xml, name):
    """MJCF -> (spec, dof [J,19], rest [J,3,3], slot_joint_names [J][<=3], xml).
    Uses the exact cache-build path, so the descriptor matches a cached robot's.
    The slot->MJCF-joint-name map is what lets a renderer write the projected angles
    back into the robot's own qpos."""
    import prep_gmr_cache as PG
    import prep_dof_features as PD
    import mujoco
    model = mujoco.MjModel.from_xml_path(xml)
    keep, frame = PG.build_merge(model)
    spec = PG.build_spec(model, name, keep)
    D, snames, model2, keep2, frame2, rest_raw, axes_raw, nonhinge, Rst = PD.mjcf_dof(
        name, xml, {"parents": spec.parents, "joint_names": spec.joint_names})
    qn = [[x for x in s.split(",") if x] for s in snames]
    return spec, D.astype(np.float32), Rst.astype(np.float32), qn, xml


def build_from_cache(cache, name):
    d = np.load(f"{cache}/{name}_spec.npz", allow_pickle=True)
    spec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                        [str(x) for x in d["joint_names"]])
    z = np.load(f"{cache}/{name}_dof.npz", allow_pickle=True)
    return spec, z["dof"].astype(np.float32), z["rest"].astype(np.float32), None, None


def get_target(spec_name, cache):
    if spec_name in MJCF_BODIES:
        return build_from_mjcf(MJCF_BODIES[spec_name], spec_name)
    if spec_name.endswith(".xml"):
        return build_from_mjcf(spec_name, os.path.basename(spec_name)[:-4])
    return build_from_cache(cache, spec_name)


# --------------------------------------------------------------- model side ---

_DEFAULT_SEM = "siglip"       # main() overwrites from the run's config (0809 fix)


def target_geom(spec, dof, device, lut=None, sem_encoder=None):
    sem_encoder = sem_encoder or _DEFAULT_SEM
    """sem_encoder MUST match the run's text tower: SigLIP is 768-wide, Qwen3 1024.
    Rendering a Qwen run with SigLIP semantics died with
    "mat1 and mat2 shapes cannot be multiplied (3300x768 and 1024x256)", and a
    side-by-side of a SigLIP run and a Qwen run needs a DIFFERENT geom per run."""
    from semantics import build_name_embeddings, embed_matrix
    st = static_cond(spec, device)
    lut = lut or build_name_embeddings(list(spec.joint_names), device=device,
                                       encoder=sem_encoder)
    missing = [n for n in spec.joint_names if n not in lut]
    if missing:
        lut = build_name_embeddings(list(lut.keys()) + list(spec.joint_names),
                                    device=device, encoder=sem_encoder)
    sem = embed_matrix(spec.joint_names, lut).to(device)
    return {"st": st, "sem": sem, "dof": torch.as_tensor(dof, device=device)}


def geom_batch(g, B):
    st = g["st"]
    return dict(rest_off=st["rest_off"][None].expand(B, -1, -1),
                bonelen=st["bonelen"][None].expand(B, -1),
                depth=st["depth"][None].expand(B, -1),
                hop=st["hop"][None].expand(B, -1, -1),
                sem=g["sem"][None].expand(B, -1, -1),
                dof=g["dof"][None].expand(B, -1, -1))


def load_source(cache, clips, device, name="human_smpl", max_frames=None):
    """-> (pose [T,J,9] on device, spec, clip slices)."""
    d = np.load(f"{cache}/{name}_spec.npz", allow_pickle=True)
    spec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                        [str(x) for x in d["joint_names"]])
    seqs = {s["name"]: s for s in json.load(open(f"{cache}/sequences.json"))}
    local6d = torch.load(f"{cache}/{name}_paired_local6d.pt")
    rows, slices, at = [], {}, 0
    for c in clips:
        s = seqs[c]
        n = s["len"] if max_frames is None else min(s["len"], max_frames)
        rows.append(torch.arange(s["start"], s["start"] + n))
        slices[c] = (at, at + n); at += n
    rows = torch.cat(rows)
    gr6, gp, reach = build_global(local6d[rows], spec, device)
    pose = torch.cat([gr6, gp / reach], -1).to(device)
    return pose, spec, slices


# ------------------------------------------------------------------ metrics ---

def limit_report(q, dof, device):
    """q [T,J,3] hinge angles (UNCLAMPED fit) vs the descriptor's limit box."""
    dof_t = torch.as_tensor(dof, device=device)
    v = dof_t[:, :18].reshape(-1, 3, 6)
    limited = v[:, :, 3] > 0.5
    n_dof = (dof_t[:, 18] * 3).round().long()
    slot = torch.arange(3, device=device)[None, :]
    axis_n = v[:, :, :3].norm(dim=-1)
    active = (axis_n > 1e-6) & (slot < n_dof[:, None])
    m = limited & active                                       # [J,3]
    half = v[:, :, 4] * np.pi
    ctr = v[:, :, 5] * np.pi
    lo, hi = ctr - half, ctr + half
    over = torch.maximum(q - hi[None], lo[None] - q).clamp_min(0.0)   # [T,J,3]
    over = torch.where(m[None], over, torch.zeros_like(over))
    n_lim = int(m.sum())
    T = q.shape[0]
    if n_lim == 0:
        return dict(n_limited_dof=0, viol_dof_pct=0.0, viol_frame_pct=0.0,
                    worst_overshoot_deg=0.0), over, m
    bad = over > np.deg2rad(0.5)
    return dict(n_limited_dof=n_lim,
                viol_dof_pct=round(100.0 * float(bad.sum()) / (n_lim * T), 3),
                viol_frame_pct=round(100.0 * float(bad.flatten(1).any(-1).float().mean()), 2),
                worst_overshoot_deg=round(float(torch.rad2deg(over.max())), 2)), over, m


def infeasibility(Rg, spec, dof_d, rest_d, device):
    """TRUE infeasibility, free of the Euler-chart artifact that inflates
    `pre_viol_*` on 3-DOF frames.

    `pre_viol_*` fits angles WITHOUT the box and asks whether they land inside it.
    On a 3-DOF frame the angle triple is a chart of SO(3), not a unique label: two
    branches describe the SAME rotation, and the unclamped fitter is free to return
    the out-of-box one. Measured on Apollo, the hips report 47-53% "violation" with
    a 150 deg overshoot yet their CLAMPED fit reproduces the predicted rotation to
    0.1-0.25 deg — the pose was always reachable.

    This measures the thing that actually matters: the residual between the predicted
    LOCAL rotation and the nearest one the joint can reach inside its box.
    """
    q, _ = MP.fit_angles(Rg, spec, dof_d, rest=rest_d, clamp=True, method="greedy")
    par = torch.as_tensor(np.asarray(spec.parents), device=device)
    Rl = Rg[:, par.clamp_min(0)].transpose(-1, -2) @ Rg
    Rl = torch.where((par < 0)[None, :, None, None], Rg, Rl)
    A, _ = MP._slot_axes(dof_d)
    Ax = A[None].expand(q.shape[0], -1, -1, -1)
    L = (MP._rot_axis_angle(Ax[:, :, 0], q[:, :, 0])
         @ MP._rot_axis_angle(Ax[:, :, 1], q[:, :, 1])
         @ MP._rot_axis_angle(Ax[:, :, 2], q[:, :, 2]) @ rest_d[None])
    err = geodesic_deg(L, Rl)                                       # [T,J]
    nd = (dof_d[:, 18] * 3).round().long().cpu().numpy()
    out = {"all": round(float(err[:, nd > 0].mean()), 2)}
    for k in (1, 2, 3):
        if (nd == k).any():
            out[f"{k}dof"] = round(float(err[:, np.where(nd == k)[0]].mean()), 2)
    return out


def part_of(name):
    n = name.lower()
    for p, kws in PART_KW.items():
        if any(k in n for k in kws):
            return p
    return "other"


def per_part(err_j, spec):
    """err_j [T,J] -> {part: mean deg}."""
    out = {}
    parts = np.array([part_of(n) for n in spec.joint_names])
    for p in ["leg", "arm", "torso", "other"]:
        idx = np.where(parts == p)[0]
        if len(idx):
            out[p] = round(float(err_j[:, idx].mean()), 2)
    out["all"] = round(float(err_j.mean()), 2)
    return out


@torch.no_grad()
def eval_target(model, src_pose, src_geom, tgt_name, tgt, device, gt_pose=None,
                chunk=512, lm_iters=25):
    spec, dof, rest, _, _ = tgt
    g = target_geom(spec, dof, device)
    T = src_pose.shape[0]
    preds = []
    for i in range(0, T, chunk):
        b = src_pose[i:i + chunk]
        zp, zr = model.encode(b, geom_batch(src_geom, b.shape[0]))
        prot, ppos = model.decode(zp, zr, geom_batch(g, b.shape[0]))
        preds.append(prot)
    prot = torch.cat(preds, 0)                                     # [T,J,6]
    Rg = rot6d_to_matrix(prot).double()
    dof_d = torch.as_tensor(dof, device=device, dtype=torch.float64)
    rest_d = torch.as_tensor(rest, device=device, dtype=torch.float64)

    t0 = time.time()
    q_raw, _ = MP.fit_angles(Rg, spec, dof_d, rest=rest_d, clamp=False, method="greedy")
    pre, over, mask = limit_report(q_raw, dof, device)
    infeas = infeasibility(Rg, spec, dof_d, rest_d, device)
    r6, pos, q_p = MP.project(Rg, spec, dof_d, rest=rest_d, method="global",
                              lm_iters=lm_iters)
    post, _, _ = limit_report(q_p, dof, device)
    Rp = rot6d_to_matrix(r6)
    moved = geodesic_deg(Rg, Rp)                                   # [T,J]
    out = {"J": spec.J, "frames": T,
           "pre_viol_dof_pct": pre["viol_dof_pct"],
           "pre_viol_frame_pct": pre["viol_frame_pct"],
           "pre_worst_overshoot_deg": pre["worst_overshoot_deg"],
           "n_limited_dof": pre["n_limited_dof"],
           "post_viol_dof_pct": post["viol_dof_pct"],
           "post_worst_overshoot_deg": post["worst_overshoot_deg"],
           "projection_moved_mean_deg": round(float(moved.mean()), 2),
           "projection_moved_per_part": per_part(moved, spec),
           "infeasibility_deg": infeas,
           "sec": round(time.time() - t0, 1)}
    if gt_pose is not None:
        gt_R = rot6d_to_matrix(gt_pose[..., :6])
        out["gt_geo_pre"] = per_part(geodesic_deg(Rg.float(), gt_R), spec)
        out["gt_geo_post"] = per_part(geodesic_deg(Rp.float(), gt_R), spec)
    return out, prot.float().cpu(), r6.float().cpu(), q_p.float().cpu()


@torch.no_grad()
def descriptor_sensitivity(model, src_pose, src_geom, spec, dof, device):
    """How much does the decode MOVE when a channel of the dof descriptor is
    destroyed? A channel the decoder ignores gives ~0.

    `axes_canonicalized` is the load-bearing one: it snaps every axis to the nearest
    +-X/Y/Z, i.e. it deletes exactly the tilt and the non-orthogonality that make
    Apollo out-of-distribution, while leaving the axis pattern and the limits intact.
    On the control model that costs 1.15 deg of decode on Apollo, against a 25 deg
    feasibility-projection distance — the decoder is effectively blind to the tilt.
    """
    J = dof.shape[0]
    v = dof[:, :18].reshape(J, 3, 6).copy()
    outs = {}
    c = v.copy()
    a = c[:, :, :3]
    snap = np.zeros_like(a)
    k = np.abs(a).argmax(-1)
    for j in range(J):
        for s in range(3):
            if np.linalg.norm(a[j, s]) > 1e-6:
                snap[j, s, k[j, s]] = np.sign(a[j, s, k[j, s]])
    c[:, :, :3] = snap
    outs["axes_canonicalized"] = c
    z = v.copy(); z[:, :, :3] = 0.0
    outs["axes_zeroed"] = z
    w = v.copy(); w[:, :, 4] = 1.0; w[:, :, 5] = 0.0
    outs["limits_widened"] = w
    n = v.copy(); n[:, :, 3:] = 0.0
    outs["limits_dropped"] = n

    def dec(dd):
        g = target_geom(spec, dd.astype(np.float32), device)
        zp, zr = model.encode(src_pose, geom_batch(src_geom, src_pose.shape[0]))
        p, _ = model.decode(zp, zr, geom_batch(g, src_pose.shape[0]))
        return rot6d_to_matrix(p)
    base = dec(dof)
    return {k2: round(float(geodesic_deg(base, dec(
        np.concatenate([vv.reshape(J, 18), dof[:, 18:]], 1))).mean()), 2)
        for k2, vv in outs.items()}


def load_run(run, device):
    cfg = json.load(open(f"{run}/config.json"))
    model = build_model(cfg, device)
    sd = torch.load(f"{run}/model.pt", map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--targets", default="apollo,pal_talos,unitree_g1_29dof,unitree_g1_23dof")
    ap.add_argument("--clips", default="MoSh__50020__salsa_stageii,"
                                       "ACCAD__Female1Walking_c3d__B16_-_walk_turn_change_direction_stageii")
    ap.add_argument("--src_cache", default=CACHE_AMASS)
    ap.add_argument("--gt_cache", default=CACHE_AMASS)
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--lm_iters", type=int, default=25)
    ap.add_argument("--save_pred", default=None)
    ap.add_argument("--sensitivity", action="store_true",
                    help="also report how far the decode moves when a dof channel is destroyed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)

    model, cfg = load_run(args.run, device)
    global _DEFAULT_SEM
    _DEFAULT_SEM = cfg.get("sem_encoder", "siglip")
    clips = [c for c in args.clips.split(",") if c]
    res = {"run": args.run, "cfg_cache": cfg["cache"], "clips": clips}
    saved = {}
    for clip in clips:
        src_pose, src_spec, _ = load_source(args.src_cache, [clip], device,
                                            max_frames=args.max_frames)
        sg = target_geom(src_spec, np.load(f"{args.src_cache}/human_smpl_dof.npz")["dof"],
                         device)
        res[clip] = {}
        for name in args.targets.split(","):
            name = name.strip()
            if not name:
                continue
            tgt = get_target(name, args.gt_cache)
            gt = None
            if name not in MJCF_BODIES and not name.endswith(".xml"):
                gp_, spec_, sl_ = load_source(args.gt_cache, [clip], device, name=name,
                                              max_frames=args.max_frames)
                gt = gp_
            r, prot, r6, q = eval_target(model, src_pose, sg, name, tgt, device,
                                         gt_pose=gt, lm_iters=args.lm_iters)
            if args.sensitivity:
                r["descriptor_sensitivity_deg"] = descriptor_sensitivity(
                    model, src_pose, sg, tgt[0], tgt[1], device)
            res[clip][name] = r
            print(f"[{clip[:34]:34s}] {name:20s} {json.dumps(r)}", flush=True)
            if args.save_pred:
                saved[f"{clip}|{name}"] = {"raw6d": prot, "proj6d": r6, "q": q}
    json.dump(res, open(args.out, "w"), indent=2)
    if args.save_pred:
        torch.save(saved, args.save_pred)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
