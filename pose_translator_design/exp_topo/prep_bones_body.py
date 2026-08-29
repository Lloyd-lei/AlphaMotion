"""JOB 1: fix the soma77 ROOT CONVENTION -> pelvis-rooted body-only cache.

Measured defect (scratchpad g1gt_diag.py): soma77's root is a FLOOR node — reach
229 cm, Hips sits 0.44 reach-units away from the root — and E19 (trained on
pelvis-rooted human_smpl, reach ~99-104 cm) decodes it to garbage
(37.9 cm / 118.9 deg).

Fix = REBASE the skeleton at Hips, then CALIBRATE the rest pose:
  * drop the floor Root node (its only child is Hips);
  * new root local rotation = R_Root @ R_Hips (the global pelvis orientation),
    every other kept joint keeps its BVH local rotation — chains are unchanged
    because everything dropped below Hips is leaf-only subtrees;
  * keep 23 body joints, drop the 48 finger joints (lloyd: fingers masked) plus
    HeadEnd / Jaw / eyes / toe-end leaf nulls;
  * REST-POSE CALIBRATION (measured necessity, not cosmetics): the soma BVH's
    rest has every bone along its local +/-X (mirrored left/right) — with that
    convention E19 self-recon is ~103 deg even after re-rooting.  We re-express
    every joint frame with a fixed per-joint rotation R_cal (the skeleton's
    global rotations at a deterministic standing frame of
    body_check_001__A548), so the stored rest offsets become anatomical
    world-space directions and a standing pose carries near-identity rotations:
        off'[j]  = R_cal[parent(j)] @ off[j]
        R'_l[j]  = R_cal[parent(j)] @ R_l[j] @ R_cal[j]^T   (root: R_g @ R_cal[0]^T)
    FK positions are bit-identical under this change (proven < 1e-4 cm);
  * joint names -> human_smpl's names (LeftForeArm -> L_Elbow, ...) so the
    SigLIP semantic conditioning is in-distribution for E19.

Measured effect (E19 self-recon, 5-clip probe): 103 deg -> 22.7 deg; cross-decode
soma -> unitree_g1_29dof vs GMR GT: 37.9 cm/118.9 deg -> 7-15 cm/25-35 deg.

Result: reach ~111 cm, same convention as human_smpl (root at origin, positions
root-relative in world orientation, Y-up, cm).

Outputs under --out (default cache_bones_body):
  soma_body_spec.npz / soma_body_dof.npz       (ball joints, like soma77's)
  soma_body_local6d.pt   [N,23,6]              (cache-standard local rot6d)
  soma_body_global9d.pt  [N,23,9]              (global rot6d + root-rel pos CM;
                                                encoder input = [r6, pos/reach])
  sequences.json / manifest.json               (per-clip row ranges, reach, stats)

Validation baked in: FK round-trip of the rebased local6d vs the FULL 78-joint
skeleton's FK (must be < 0.1 cm), plus per-clip uprightness (pelvis->Neck1 . +Y).

First batch selection (JOB 2b): deterministic — non-mirror clips from
text_manifest.jsonl (bones_text_manifest.py), sorted by name, round-robin across
the 7 R7 families until --n_clips.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ARDY = '/home/arenalabs/Desktop/"be water, robot"/experiment/ardy'
if ARDY not in sys.path:
    sys.path.insert(0, ARDY)
from common import CACHE_ROOT, SkeletonSpec, matrix_to_rot6d, fk_global   # noqa: E402
from prep_paired import bvh_frame_time, save_spec                         # noqa: E402
from prep_dof_features import ball_dof, rest6d                            # noqa: E402
from bones_text_manifest import FAMILIES, BONES                           # noqa: E402

FIXED = f"{CACHE_ROOT}/cache_paired_fixed"
OUT = f"{CACHE_ROOT}/cache_bones_body"

KEEP = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase",
        "RightLeg", "RightShin", "RightFoot", "RightToeBase"]

# human_smpl name for each kept soma joint (E19's semantic conditioning).
SMPL_NAME = {"Hips": "Pelvis", "Spine1": "Torso", "Spine2": "Spine", "Chest": "Chest",
             "Neck1": "Neck", "Neck2": "Neck", "Head": "Head",
             "LeftShoulder": "L_Thorax", "LeftArm": "L_Shoulder",
             "LeftForeArm": "L_Elbow", "LeftHand": "L_Wrist",
             "RightShoulder": "R_Thorax", "RightArm": "R_Shoulder",
             "RightForeArm": "R_Elbow", "RightHand": "R_Wrist",
             "LeftLeg": "L_Hip", "LeftShin": "L_Knee",
             "LeftFoot": "L_Ankle", "LeftToeBase": "L_Toe",
             "RightLeg": "R_Hip", "RightShin": "R_Knee",
             "RightFoot": "R_Ankle", "RightToeBase": "R_Toe"}
BODY_NAMES = [SMPL_NAME[n] for n in KEEP]

CAL_CLIP = "soma_uniform/bvh/240918/body_check_001__A548.bvh"   # calibration source


def load_soma77_spec():
    d = np.load(f"{FIXED}/soma77_spec.npz", allow_pickle=True)
    return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                        [str(x) for x in d["joint_names"]])


def body_spec(spec77):
    """Pelvis-rooted 23-joint body spec from the 78-joint floor-rooted one."""
    names77 = list(spec77.joint_names)
    ki = [names77.index(n) for n in KEEP]
    old2new = {o: n for n, o in enumerate(ki)}
    par = []
    for o in ki:
        p = int(spec77.parents[o])
        while p >= 0 and p not in old2new:      # skip dropped intermediates
            p = int(spec77.parents[p])
        par.append(old2new.get(p, -1))
    off = spec77.rest_offsets[ki].copy()
    off[0] = 0.0                                # root at origin
    sp = SkeletonSpec("soma_body", par, off, KEEP)
    sp.order_root_first()
    return sp, ki


def rebase_local(jr, names77, ki):
    """[T,78,3,3] BVH locals -> [T,23,3,3] pelvis-rooted locals."""
    iroot, ihips = names77.index("Root"), names77.index("Hips")
    loc = jr[:, ki].clone()
    loc[:, 0] = jr[:, iroot] @ jr[:, ihips]     # global pelvis orientation
    return loc


def standing_calibration(spec77, spec_b, ki, bones_root):
    """R_cal [23,3,3]: the body's global rotations at a deterministic standing
    frame (max uprightness + arms-down score over CAL_CLIP's first 1200 frames)."""
    from ardy.data_processing.skeleton import Skeleton, load_bvh_animation
    bvh = f"{bones_root}/{CAL_CLIP}"
    skl = Skeleton(); skl.load_from_bvh(bvh)
    _, jr = load_bvh_animation(bvh, skl)
    jr = torch.as_tensor(np.asarray(jr), dtype=torch.float32)
    L = rebase_local(jr[np.arange(0, 1200, 5)], list(spec77.joint_names), ki)
    Rg, gp = fk_global(matrix_to_rot6d(L), spec_b)
    kH, kL, kR = KEEP.index("Head"), KEEP.index("LeftHand"), KEEP.index("RightHand")
    up = gp[:, kH, 1] / gp[:, kH].norm(dim=-1).clamp(min=1)
    score = up * 100 - (gp[:, kL, 1] + gp[:, kR, 1]) * 0.5 \
        - (gp[:, kL] + gp[:, kR]).norm(dim=-1) * 0.5
    return Rg[int(score.argmax())], int(score.argmax()) * 5


def calibrated_spec(spec_b, Rcal, root_height=101.28):
    """Rest offsets -> anatomical world dirs at the calibration pose; smpl names.
    Root rest_off = [0, pelvis height, 0] like human_smpl's Pelvis [0, 88.5, 0]
    (FK ignores the root offset, but E19 conditions on it: measured
    20.2 vs 22.7 deg self-recon on the 5-clip probe)."""
    off = torch.as_tensor(spec_b.rest_offsets)
    offp = spec_b.rest_offsets.copy()
    for j in range(1, spec_b.J):
        offp[j] = (Rcal[spec_b.parents[j]] @ off[j]).numpy()
    offp[0] = np.array([0.0, root_height, 0.0], np.float32)
    sp = SkeletonSpec("soma_body", spec_b.parents, offp, BODY_NAMES)
    sp.order_root_first()
    return sp


def calibrate_local(L, Rcal, parents):
    """R'_l[j] = R_cal[par(j)] @ R_l[j] @ R_cal[j]^T (root: par = identity).
    FK positions are invariant; a standing pose becomes near-identity rotations."""
    out = L.clone()
    out[:, 0] = L[:, 0] @ Rcal[0].T
    for j in range(1, L.shape[1]):
        out[:, j] = Rcal[parents[j]] @ L[:, j] @ Rcal[j].T
    return out


def bvh_to_body_local6d(bvh, spec77, ki, Rcal, parents, fps=30.0, max_frames=240):
    """One BVH -> pelvis-rooted CALIBRATED local rot6d [K,23,6], fps-resampled."""
    from ardy.data_processing.skeleton import Skeleton, load_bvh_animation
    skl = Skeleton(); skl.load_from_bvh(bvh)
    assert list(skl.get_bones_names()) == list(spec77.joint_names), bvh
    _, jr = load_bvh_animation(bvh, skl)
    jr = torch.as_tensor(np.asarray(jr), dtype=torch.float32)
    dt = bvh_frame_time(bvh) or (1.0 / 30.0)
    K = max(2, min(int(round(jr.shape[0] * dt * fps)), max_frames))
    idx = np.clip(np.round(np.linspace(0, jr.shape[0] - 1, K)).astype(np.int64),
                  0, jr.shape[0] - 1)
    jr = jr[idx]
    L = calibrate_local(rebase_local(jr, list(spec77.joint_names), ki), Rcal, parents)
    return matrix_to_rot6d(L), jr


def fk_roundtrip_cm(l6_body, jr_full, spec_b, spec77, ki, ihips):
    """Max |FK(body spec) - FK(full skeleton, hips-relative)| in cm."""
    _, gp_b = fk_global(l6_body, spec_b)
    _, gp_f = fk_global(matrix_to_rot6d(jr_full), spec77)
    ref = gp_f[:, ki] - gp_f[:, ihips:ihips + 1]
    return float((gp_b - ref).norm(dim=-1).max())


def pick_first_batch(manifest_path, n_clips):
    """Deterministic, family-spanning: non-mirror clips sorted by name,
    round-robin over the R7 families (+other)."""
    fams = {f: [] for f in FAMILIES + ["other"]}
    with open(manifest_path) as f:
        for line in f:
            r = json.loads(line)
            if not r["is_mirror"]:
                fams[r["family"]].append(r)
    for f in fams:
        fams[f].sort(key=lambda r: r["name"])
    out, i = [], 0
    while len(out) < n_clips and any(fams.values()):
        for f in FAMILIES + ["other"]:
            if fams[f] and len(out) < n_clips:
                out.append(fams[f].pop(0))
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_clips", type=int, default=500)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max_frames", type=int, default=240)
    ap.add_argument("--fk_check_every", type=int, default=25,
                    help="full-skeleton FK round-trip check on every k-th clip")
    ap.add_argument("--shard", type=int, default=None,
                    help="convert only shard i of --n_shards (contiguous split); "
                         "writes soma_body_shard{i}.pt + _shard{i}.json")
    ap.add_argument("--n_shards", type=int, default=3)
    ap.add_argument("--merge", action="store_true",
                    help="merge shard files -> final cache tensors + manifest")
    ap.add_argument("--spec_only", action="store_true",
                    help="rewrite spec/dof/cal npz only (tensors are unaffected "
                         "by spec-side changes like the root rest_off)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    spec77 = load_soma77_spec()
    spec_b, ki = body_spec(spec77)
    ihips = list(spec77.joint_names).index("Hips")
    Rcal, cal_frame = standing_calibration(spec77, spec_b, ki, BONES)
    spec_c = calibrated_spec(spec_b, Rcal,
                             root_height=float(np.linalg.norm(
                                 spec77.rest_offsets[ihips])))
    save_spec(spec_c, f"{args.out}/soma_body_spec.npz")
    np.savez(f"{args.out}/soma_body_cal.npz", Rcal=Rcal.numpy(),
             cal_clip=CAL_CLIP, cal_frame=cal_frame,
             soma_names=np.array(KEEP, dtype=object),
             body_names=np.array(BODY_NAMES, dtype=object))
    D, Rst = ball_dof(spec_c.J)
    np.savez(f"{args.out}/soma_body_dof.npz", dof=D, rest=Rst, rest6d=rest6d(Rst),
             joint_names=np.array(BODY_NAMES, dtype=object),
             slot_joints=np.array(["ball(x,y,z)"] * spec_c.J, dtype=object))
    print(f"[spec] soma_body J={spec_c.J} (from soma77 J={spec77.J}); "
          f"dropped floor Root + 48 fingers + {spec77.J - spec_c.J - 49} leaf nulls; "
          f"rest calibrated at {CAL_CLIP} frame {cal_frame}")
    if args.spec_only:
        print("[spec_only] spec/dof/cal rewritten; tensors untouched")
        return

    clips = pick_first_batch(f"{args.out}/text_manifest.jsonl", args.n_clips)
    print(f"[batch] {len(clips)} clips, families:",
          {f: sum(c['family'] == f for c in clips) for f in FAMILIES + ['other']})

    if args.merge:
        merge_shards(args, spec_c)
        return
    if args.shard is not None:
        a = args.shard * len(clips) // args.n_shards
        b = (args.shard + 1) * len(clips) // args.n_shards
        clips = clips[a:b]
        print(f"[shard {args.shard}/{args.n_shards}] clips {a}:{b}")

    rows, seqs, start = [], [], 0
    fk_max, upright = 0.0, []
    t0 = time.time()
    for i, c in enumerate(clips):
        try:
            l6, jr_full = bvh_to_body_local6d(c["bvh"], spec77, ki, Rcal,
                                              spec_c.parents, args.fps, args.max_frames)
        except Exception as e:
            print("  skip", c["name"], repr(e)[:100]); continue
        if torch.isnan(l6).any():
            print("  skip", c["name"], "NaN"); continue
        if i % args.fk_check_every == 0:
            fk_max = max(fk_max, fk_roundtrip_cm(l6, jr_full, spec_c, spec77, ki, ihips))
        _, gp = fk_global(l6, spec_c)
        up = torch.nn.functional.normalize(gp[:, KEEP.index("Neck1")], dim=-1)[:, 1]
        upright.append(float(up.mean()))
        seqs.append({"name": c["name"], "start": start, "len": int(l6.shape[0]),
                     "family": c["family"], "session": c["session"]})
        start += int(l6.shape[0]); rows.append(l6)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(clips)} rows={start} ({time.time()-t0:.0f}s)", flush=True)

    l6 = torch.cat(rows, 0)
    if args.shard is not None:
        torch.save(l6, f"{args.out}/soma_body_shard{args.shard}.pt")
        json.dump({"seqs": seqs, "fk_max": fk_max, "upright": upright},
                  open(f"{args.out}/_shard{args.shard}.json", "w"))
        print(f"[shard {args.shard}] rows={l6.shape[0]} clips={len(seqs)} "
              f"fk_max={fk_max:.5f} SAVED")
        return
    finalize(args, spec_c, l6, seqs, fk_max, upright, cal_frame)


def finalize(args, spec_c, l6, seqs, fk_max, upright, cal_frame):
    from train_global import build_global
    gr6, gp, reach = build_global(l6, spec_c, "cpu")
    torch.save(l6, f"{args.out}/soma_body_local6d.pt")
    torch.save(torch.cat([gr6, gp], -1), f"{args.out}/soma_body_global9d.pt")
    json.dump(seqs, open(f"{args.out}/sequences.json", "w"))
    up = np.array(upright)
    man = {"embodiment": "soma_body", "rows": int(l6.shape[0]), "J": spec_c.J,
           "clips": len(seqs), "fps": args.fps, "max_frames": args.max_frames,
           "reach_cm": reach, "root": "Hips (pelvis-rooted; floor node dropped)",
           "fingers": "dropped (48 joints) per lloyd; masked-out from the spec",
           "rest_calibration": {"clip": CAL_CLIP, "frame": cal_frame},
           "joint_names": "human_smpl vocabulary (see soma_body_cal.npz)",
           "fk_roundtrip_max_cm": fk_max,
           "uprightness_mean": float(up.mean()),
           "uprightness_p10": float(np.percentile(up, 10)),
           "source": "BONES-SEED soma_uniform BVH, first_batch deterministic"}
    json.dump(man, open(f"{args.out}/manifest.json", "w"), indent=2)
    print(f"[cache] rows={l6.shape[0]} clips={len(seqs)} reach={reach:.1f} cm")
    print(f"[validate] FK round-trip max = {fk_max:.5f} cm  (gate: < 0.1)")
    print(f"[validate] uprightness pelvis->Neck1 . +Y: mean {up.mean():.3f}  "
          f"p10 {np.percentile(up, 10):.3f}  min {up.min():.3f}")
    print("SAVED", args.out)


def merge_shards(args, spec_c):
    rows, seqs, upright, fk_max, start = [], [], [], 0.0, 0
    for i in range(args.n_shards):
        rows.append(torch.load(f"{args.out}/soma_body_shard{i}.pt"))
        d = json.load(open(f"{args.out}/_shard{i}.json"))
        for s in d["seqs"]:
            s = dict(s); s["start"] = start; start += s["len"]; seqs.append(s)
        upright += d["upright"]; fk_max = max(fk_max, d["fk_max"])
    cal_frame = int(np.load(f"{args.out}/soma_body_cal.npz", allow_pickle=True)["cal_frame"])
    finalize(args, spec_c, torch.cat(rows, 0), seqs, fk_max, upright, cal_frame)
    for i in range(args.n_shards):
        os.remove(f"{args.out}/soma_body_shard{i}.pt")
        os.remove(f"{args.out}/_shard{i}.json")


if __name__ == "__main__":
    main()
