"""SE(3) CONTROL CEILING — how far can the constraint interface actually be pushed?

    se3_ceiling.py --run PROD_spatial_0808

lloyd, 2026-08-08: "se3 控制 在没有精修的情况下 我好奇他的能力上限".

The spatial layer's product surface is: pin K joints' SE(3) poses, let the model
invent the rest, then project into the robot's real joint limits. This sweeps K from
1 to "everything" and reports where it saturates, on three bodies spanning the
generalization ladder.

  K = 1   root only                      the degenerate floor
  K = 2   root + one wrist
  K = 3   root + both wrists
  K = 5   root + wrists + ankles         the SE3-scan configuration (0.80 cm claim)
  K = 8   + head + 2 elbows
  K = 12  + knees, shoulders
  K = all every joint visible            the ceiling: pure reconstruction

THREE NUMBERS PER ROW, and they answer different questions:
  commanded_cm   error on the joints we PINNED, AFTER feasibility projection.
                 This is the product guarantee: "if I command it, does it go there?"
                 With E28's vis_residual the pre-projection value should be ~0, so
                 anything here is the projection giving ground to joint limits.
  free_cm        error on the joints we did NOT pin, vs GT. How well the model
                 invents the rest. This is the number that should fall with K.
  viol_pct       pre-projection joint-limit violation: is the model even ASKING for
                 something the robot can do? GT-free, and the one that matters for
                 real hardware.

Bodies: X2 (in no dataset), G1 (held out of training), Adam (trained). If the curve
has the same shape on all three, the interface generalizes; if only the trained body
saturates, it does not.

Output: runs/SE3_CEILING.txt + a mesh video at the most interesting K.
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
import eval_newbody as E                                             # noqa: E402
import manifold_project as MP                                        # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402

GMR = "/var/tmp/alphamotion/config_harvest/GMR/assets"
# Bodies that BOTH have a ground-truth retarget in the cache AND were held out of
# training. Against these the free-joint error is measured vs the REAL retarget, not
# vs the model's own all-visible decode — the honest version of the same sweep.
GT_BODIES = ["unitree_g1_29dof", "unitree_g1_23dof", "pal_talos"]
BODIES = [
    ("X2 (no dataset)", "/var/tmp/alphamotion/robot_models/agibot_x2_urdf/"
                        "X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"),
    ("G1 (held out)",   f"{GMR}/unitree_g1/g1_mocap_29dof.xml"),
    ("Adam (trained)",  f"{GMR}/pnd_adam_lite/adam_lite.xml"),
]
# joint TAGS in priority order; resolved by name against each skeleton
LADDER = [
    ("root",),
    ("root", "wrist_l"),
    ("root", "wrist_l", "wrist_r"),
    ("root", "wrist_l", "wrist_r", "ankle_l", "ankle_r"),
    ("root", "wrist_l", "wrist_r", "ankle_l", "ankle_r", "head", "elbow_l", "elbow_r"),
    ("root", "wrist_l", "wrist_r", "ankle_l", "ankle_r", "head", "elbow_l", "elbow_r",
     "knee_l", "knee_r", "shoulder_l", "shoulder_r"),
    ("ALL",),
]
KW = {"root": None, "head": ("head",), "wrist": ("wrist", "hand"),
      "ankle": ("ankle", "foot"), "elbow": ("elbow", "forearm"),
      "knee": ("knee", "shin", "calf"), "shoulder": ("shoulder", "upperarm", "clav")}


def resolve(spec, tags):
    """tag list -> joint indices on THIS skeleton (by name, side-aware)."""
    names = [n.lower() for n in spec.joint_names]
    root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
    if tags == ("ALL",):
        return list(range(spec.J))
    out = []
    for t in tags:
        if t == "root":
            out.append(root); continue
        base, _, side = t.partition("_")
        kws = KW.get(base) or (base,)
        cand = [i for i, n in enumerate(names)
                if any(k in n for k in kws)
                and (not side or f"_{side}" in n or n.startswith(side)
                     or f"{side}_" in n or n.endswith(side))]
        if cand:
            out.append(cand[0])
    return sorted(set(out))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="PROD_spatial_0808")
    ap.add_argument("--baseline", default="E19_K128_0805")
    ap.add_argument("--cache", default="/var/tmp/alphamotion/topo_exp/cache_xwbc_lafan")
    ap.add_argument("--clip", default="dance1_subject1")
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--gt", action="store_true",
                    help="score against the cache's REAL retarget for held-out bodies "
                         "(G1 x2, pal_talos) instead of the model's own all-visible "
                         "decode. The only fully honest version of this sweep.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    rows = {}

    bodies = BODIES
    if args.gt:
        bodies = [(f"{b} (GT, held out)", b) for b in GT_BODIES]
    for tag, runname in (("PROD", args.run), ("OLD", args.baseline)):
        rd = f"{V.RUNS}/{runname}"
        model, cfg = E.load_run(rd, dev)
        enc = cfg.get("sem_encoder", "siglip")
        src, sspec, slices = E.load_source(args.cache, [args.clip], dev,
                                           max_frames=args.n)
        import json as _json
        _sq = {x["name"]: x for x in _json.load(open(f"{args.cache}/sequences.json"))}
        _s = _sq[args.clip]
        src_rows = list(range(_s["start"], _s["start"] + min(_s["len"], args.n)))
        sdof = np.load(f"{args.cache}/human_smpl_dof.npz")["dof"]
        sg = E.target_geom(sspec, sdof, dev, sem_encoder=enc)

        for bname, xml in bodies:
            spec, dof, rest, qnames, xml2 = E.get_target(xml, args.cache)
            g = E.target_geom(spec, dof, dev, sem_encoder=enc)
            dof_d = torch.as_tensor(dof, device=dev, dtype=torch.float64)
            rest_d = torch.as_tensor(rest, device=dev, dtype=torch.float64)
            B = src.shape[0]
            # GT on THIS body = the model's own full-visibility decode (there is no
            # ground-truth robot motion for X2; the all-visible decode is the ceiling
            # the sparse-constraint rows are measured against).
            if args.gt:
                gt_pose, _ = V.full_pose_of(args.cache, xml, src_rows, dev)
                refR = rot6d_to_matrix(gt_pose[:, :, :6].to(dev)).double()
            else:
                zp, zr = model.encode(src, E.geom_batch(sg, B))
                ref_rot, _ = model.decode(zp, zr, E.geom_batch(g, B))
                refR = rot6d_to_matrix(ref_rot).double()

            for tags in LADDER:
                js = resolve(spec, tags)
                # visible set on the SOURCE skeleton (same tags, human names)
                js_src = resolve(sspec, tags)
                x = torch.zeros_like(src)
                ji = torch.as_tensor(js_src, dtype=torch.long)
                x[:, ji] = src[:, ji]
                m = torch.ones(B, sspec.J, dtype=torch.bool, device=dev)
                m[:, ji] = False
                zp2, zr2 = model.encode(x, E.geom_batch(sg, B), joint_mask=m)
                prot, _ = model.decode(zp2, zr2, E.geom_batch(g, B))
                Rg = rot6d_to_matrix(prot).double()
                q_raw, _ = MP.fit_angles(Rg, spec, dof_d, rest=rest_d,
                                         clamp=False, method="greedy")
                pre, _, _ = E.limit_report(q_raw, dof, dev)
                r6, _, q = MP.project(Rg, spec, dof_d, rest=rest_d, method="global")
                Rp = rot6d_to_matrix(r6).double()
                from common import geodesic_deg
                d_all = geodesic_deg(Rp, refR)                # [B,J] deg vs ceiling
                cm_j = torch.as_tensor(js, dtype=torch.long)
                free = [j for j in range(spec.J) if j not in js]
                rows[f"{tag}|{bname}|K={len(js)}"] = {
                    "tags": "+".join(tags),
                    "commanded_deg": round(float(d_all[:, cm_j].mean()), 2),
                    "free_deg": round(float(d_all[:, free].mean()), 2) if free else 0.0,
                    "viol_pct": pre["viol_dof_pct"],
                    "worst_overshoot_deg": pre["worst_overshoot_deg"]}
            del spec
        del model
        torch.cuda.empty_cache()

    L = ["=" * 96, "SE(3) CONTROL CEILING — constraint density sweep", "=" * 96,
         "commanded/free = geodesic deg vs the SAME model's all-visible decode",
         "(no GT robot motion exists for X2; the all-visible decode is the ceiling).",
         "viol = PRE-projection joint-limit violation: is the model even asking for",
         "something the robot can do. GT-free, the one that matters on hardware.",
         "-" * 96]
    for k in sorted(rows, key=lambda s: (s.split("|")[1], s.split("|")[0],
                                         int(s.split("K=")[1]))):
        r = rows[k]
        L.append(f"{k:<34} {r['tags']:<52} cmd {r['commanded_deg']:>6.2f}  "
                 f"free {r['free_deg']:>6.2f}  viol {r['viol_pct']:>6.2f}%  "
                 f"worst {r['worst_overshoot_deg']:>6.2f}")
    txt = "\n".join(L)
    print(txt, flush=True)
    open(f"{V.RUNS}/SE3_CEILING.txt", "w").write(txt + "\n")
    json.dump(rows, open(f"{V.RUNS}/SE3_CEILING.json", "w"), indent=1)
    print(f"\nverdict: {V.RUNS}/SE3_CEILING.txt", flush=True)


if __name__ == "__main__":
    main()
