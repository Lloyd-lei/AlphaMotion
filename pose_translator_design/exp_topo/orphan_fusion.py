"""orphan_fusion — lloyd 0824: four heterogeneous sources onto ONE orphan body.

Question: with Greenwich alone, can UMI-style / arm / robot / ego-human streams all
land on a robot that has NO data anywhere (Figure stand-in = Apptronik Apollo,
MJCF-only, tilted non-orthogonal hinge triads)?

  A  robot     unitree_g1_29dof real retarget stream (running turn val clip)
  B  ego-proxy human_smpl AMASS pick-up-box val clip (stand-in for lifted ego
               motion; no real ego dataset on this disk)
  C  arm       Rainbow RBY1 wheeled dual-arm, SYNTHESIZED in-limit reach-wave
               (no arm motion data exists on this disk — descriptor is real,
               the trajectory is scripted)
  D  UMI-proxy SAME human clip but ONLY the right wrist enters the encoder
               (m4 v2 sketch pathway) — an end-effector-only stream

Each lane: encode with the source descriptor -> decode with Apollo's descriptor ->
feasibility projection -> mesh video (source capsule tile | Apollo tile) and
numbers: wrist-path deviation vs source (both root-relative, scale-normalized by
reach, reported at source scale) + projection moved deg.

    orphan_fusion.py --run PT_c1dec_reenc_0812 [--lanes ABCD]

Outputs: R4_VIZ/mp4/orphan_fusion/<run>/*.mp4 + runs/ORPHAN_FUSION_<run>.txt
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
import render_newbody as RN                                          # noqa: E402
import greenwich_carnival as GC                                      # noqa: E402
import spatial5 as S5                                                # noqa: E402
from common import rot6d_to_matrix, matrix_to_rot6d, geodesic_deg    # noqa: E402
from train_global import build_global                                # noqa: E402
from PIL import Image, ImageDraw                                     # noqa: E402

FULL = S5.FULL
GMR = S5.GMR
APOLLO = E.MJCF_BODIES["apollo"]
RBY1 = ("/var/tmp/alphamotion/robot_models/menagerie/rainbow_robotics_rby1/"
        "rby1a_1.2_no_gripper.xml")
CLIP_LOCO = "ACCAD__Female1Running_c3d__C14_-__run_turn_right__(90)_stageii"
CLIP_MANIP = "ACCAD__Male1General_c3d__General_A5_-_Pick_Up_Box_stageii"
VIZ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ", "mp4", "orphan_fusion")


def wrist_ids(spec, kws=("wrist", "hand")):
    n = [x.lower() for x in spec.joint_names]
    ids = [i for i, x in enumerate(n) if any(k in x for k in kws)]
    return ids


def load_cached(emb, clip, dev, T):
    seqs = {s["name"]: s for s in json.load(open(f"{FULL}/sequences.json"))}
    s = seqs[clip]
    rows = torch.arange(s["start"], s["start"] + min(s["len"], T))
    spec = V.load_spec(FULL, emb)
    l6 = torch.load(f"{FULL}/{emb}_paired_local6d.pt", mmap=True)[rows]
    gr6, gp, reach = build_global(l6, spec, dev)
    pose = torch.cat([gr6, gp / reach], -1).to(dev).float()
    dof = np.load(f"{FULL}/{emb}_dof.npz")["dof"]
    return pose, spec, dof, gp.numpy(), float(reach)


def synth_rby1(dev, T=150):
    """Scripted dual-arm reach-wave on the REAL RBY1 descriptor, inside its own
    limit boxes; wheels/torso quiet.  q -> local rotations (the same slot-axis
    composition the projector uses) -> local6d -> build_global (training path)."""
    spec, dof, rest, qnames, xml = E.build_from_mjcf(RBY1, "rby1")
    dof_d = torch.as_tensor(dof, dtype=torch.float64)
    rest_d = torch.as_tensor(rest, dtype=torch.float64)
    v = dof_d[:, :18].reshape(-1, 3, 6)
    half, ctr = v[:, :, 4] * np.pi, v[:, :, 5] * np.pi
    names = [x.lower() for x in spec.joint_names]
    q = ctr[None].expand(T, -1, -1).clone()
    t = torch.linspace(0, 4 * np.pi, T)
    rng = np.random.default_rng(7)
    for j, nm in enumerate(names):
        if "arm" not in nm:
            continue
        for k in range(3):
            if half[j, k] < 1e-3:
                continue
            ph, f = rng.uniform(0, 2 * np.pi), rng.choice([0.5, 1.0])
            q[:, j, k] = ctr[j, k] + 0.6 * half[j, k] * torch.sin(f * t + ph)
    A, _ = MP._slot_axes(dof_d)
    Ax = A[None].expand(T, -1, -1, -1)
    L = (MP._rot_axis_angle(Ax[:, :, 0], q[:, :, 0])
         @ MP._rot_axis_angle(Ax[:, :, 1], q[:, :, 1])
         @ MP._rot_axis_angle(Ax[:, :, 2], q[:, :, 2]) @ rest_d[None])
    l6 = matrix_to_rot6d(L.float())
    gr6, gp, reach = build_global(l6, spec, dev)
    pose = torch.cat([gr6, gp / reach], -1).to(dev).float()
    return pose, spec, dof, gp.numpy(), float(reach)


@torch.no_grad()
def decode_metrics(ctx, zp, zr, t):
    rot, _ = ctx.model.decode(zp, zr, E.geom_batch(t["g"], zp.shape[0]))
    Rg = rot6d_to_matrix(rot).double()
    r6, _, q = MP.project(Rg, t["spec"], t["dof_d"], rest=t["rest_d"], method="global")
    moved = float(geodesic_deg(Rg, rot6d_to_matrix(r6)).mean())
    gR, gp = MP.fk_from_angles(q, t["spec"], t["dof_d"], rest=t["rest_d"],
                               root_R=Rg[:, 0])
    return q.cpu().numpy(), gR[:, 0].cpu().numpy(), gp.cpu().numpy(), moved


def lane_video(ctx, src_pos_cm, src_parents, src_label, t, q, rootR, gp, path, header):
    fr = GC.render_tile(t["xml"], t["spec"], t["qnames"], q, rootR, gp,
                        q.shape[0], status="nodataset")
    src_t = S5.capsule_tile(src_pos_cm, src_parents, src_label)
    rows = []
    for i, f_ in enumerate(fr):
        im = Image.fromarray(np.concatenate(
            [src_t[min(i, len(src_t) - 1)], S5.pad_band(f_)], 1)).convert("RGB")
        dr = ImageDraw.Draw(im)
        dr.text((GC.TILE_W + 8, 4), "Apollo (ORPHAN: no data anywhere)",
                font=S5.font(14), fill=(255, 255, 255))
        dr.text((8, 4), header, font=S5.font(14), fill=(255, 230, 160))
        rows.append(np.asarray(im))
    RN.write_mp4(path, np.stack(rows), 20)


def wrist_dev(src_gp, src_reach, src_spec, tgt_gp, tgt_spec, src_kws=("wrist", "hand")):
    """Both paths root-relative cm; compare right-side wrist, scale-normalized,
    reported at SOURCE scale (cm)."""
    si = [i for i in wrist_ids(src_spec, src_kws)
          if src_spec.joint_names[i].lower().startswith(("r", "right"))
          or "right" in src_spec.joint_names[i].lower()] or wrist_ids(src_spec, src_kws)
    ti = [i for i in wrist_ids(tgt_spec)
          if tgt_spec.joint_names[i].lower().startswith(("r", "right"))
          or "right" in tgt_spec.joint_names[i].lower()] or wrist_ids(tgt_spec)
    a = src_gp[:, si[0]] / src_reach
    tr = float(np.linalg.norm(tgt_gp, axis=-1).max())
    b = tgt_gp[:, ti[0]] / tr
    d = np.linalg.norm(a - b, axis=-1) * src_reach
    return (float(d.mean()), float(np.percentile(d, 95)),
            src_spec.joint_names[si[0]], tgt_spec.joint_names[ti[0]])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="PT_c1dec_reenc_0812")
    ap.add_argument("--lanes", default="ABCD")
    ap.add_argument("--T", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ctx = S5.Ctx(a.run, a.device)
    ctx.out = f"{VIZ}/{a.run}"; os.makedirs(ctx.out, exist_ok=True)
    ctx.verdict = f"{V.RUNS}/ORPHAN_FUSION_{a.run}.txt"
    apollo = ctx.target("Apollo (orphan)", APOLLO)
    ctx.say("=" * 88, f"ORPHAN FUSION — 4 source types -> Apollo (MJCF-only, no data)"
            f" | run {a.run}", "=" * 88)

    def run_lane(tag, pose, spec, dof, gp_src, reach, header, vis=None,
                 src_kws=("wrist", "hand")):
        g = E.target_geom(spec, torch.as_tensor(dof), ctx.dev, sem_encoder=ctx.enc)
        if vis is None:
            zp, zr = ctx.model.encode(pose, E.geom_batch(g, pose.shape[0]))
        else:
            zp, zr = S5.sketch_encode(ctx, pose, vis, spec, g)
        q, rootR, gp_t, moved = decode_metrics(ctx, zp, zr, apollo)
        m, p95, sj, tj = wrist_dev(gp_src, reach, spec, gp_t, apollo["spec"], src_kws=src_kws)
        p = f"{ctx.out}/{tag}.mp4"
        lab = header if vis is None else header + " [tile=GT context]"
        lane_video(ctx, gp_src, spec.parents, lab, apollo, q, rootR, gp_t, p, header)
        ctx.say(f"[{tag}] frames={pose.shape[0]} src_reach={reach:.0f}cm "
                f"| wrist dev ({sj}->{tj}): mean {m:.2f}cm p95 {p95:.2f}cm "
                f"| projection moved {moved:.2f}deg", f"  -> {p}")

    if "A" in a.lanes:
        pose, spec, dof, gp, reach = load_cached("unitree_g1_29dof", CLIP_LOCO,
                                                 ctx.dev, a.T)
        run_lane("A_g1_robot", pose, spec, dof, gp, reach,
                 "A: SOURCE robot G1 (real retarget stream)")
    if "B" in a.lanes:
        pose, spec, dof, gp, reach = load_cached("human_smpl", CLIP_MANIP,
                                                 ctx.dev, a.T)
        run_lane("B_human_ego_proxy", pose, spec, dof, gp, reach,
                 "B: SOURCE human pick-up-box (ego-lift proxy)")
    if "C" in a.lanes:
        pose, spec, dof, gp, reach = synth_rby1(ctx.dev, a.T)
        run_lane("C_rby1_arm", pose, spec, dof, gp, reach,
                 "C: SOURCE RBY1 dual-arm (scripted, real descriptor)",
                 src_kws=("arm_4",))
    if "D" in a.lanes:
        pose, spec, dof, gp, reach = load_cached("human_smpl", CLIP_MANIP,
                                                 ctx.dev, a.T)
        names = [n.lower() for n in spec.joint_names]
        wr = [i for i, n in enumerate(names) if n == "r_wrist"]
        run_lane("D_umi_wrist_only", pose, spec, dof, gp, reach,
                 "D: SOURCE right wrist ONLY (UMI proxy, sketch encoder)", vis=wr)


if __name__ == "__main__":
    main()
