"""V7 BRIDGE, ON REAL ROBOTS — mesh video of two clips chained, middle invented.

    v7_bridge_mesh.py --targets x2,adam,g1

Same experiment as v7_bridge.py (two DIFFERENT held-out clips, the middle segment
invented by the model from A's last frame + B's first frame + 125 bits), but drawn as
the actual robot instead of a stick figure, because a stick figure cannot show you
self-collision, a foot through the floor, or an arm folded past its limit.

WHAT MAKES THIS DIFFERENT FROM THE STICK VERSION, and why it is the harder test:
  * Every pose is FEASIBILITY-PROJECTED (manifold_project, method="global") into the
    robot's own joint limits first. What you see is what the robot could execute, not
    the model's wish. A model that cheats by asking for impossible angles loses the
    motion here, visibly.
  * Angles are written into the robot's own MJCF qpos addresses, so the picture is
    MuJoCo's forward kinematics, not our FK re-implementation.
  * Root height is set per frame so the lowest body touches z=0 — feet on the floor.

THE BRIDGE IS MARKED: while the model is inventing, every robot in the frame turns
RED and the banner says so. Grey/coloured = real held-out data.

THREE ROBOTS, deliberately spanning the generalization ladder:
  pnd_adam_lite   IN the training set          — the easy case
  unitree_g1      HELD OUT of V7 training      — zero-shot body, paired data exists
  agibot_x2_ultra IN NO DATASET, EVER          — zero-shot from the URDF alone

Output: R4_VIZ/mp4/v7_bridge_mesh/*.mp4
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
import mujoco as mj                                                  # noqa: E402
import v5_common as V                                                # noqa: E402
import v7_bridge as B7                                               # noqa: E402
import eval_newbody as E                                             # noqa: E402
import render_newbody as RN                                          # noqa: E402
import manifold_project as MP                                        # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402
from train_v5 import load_run, to_states                             # noqa: E402

OUT = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
       'R4_VIZ/mp4/v7_bridge_mesh')
GMR = "/var/tmp/alphamotion/config_harvest/GMR/assets"
TARGETS = {
    "adam": (f"{GMR}/pnd_adam_lite/adam_lite.xml", "pnd_adam_lite (IN training)"),
    "g1":   (f"{GMR}/unitree_g1/g1_mocap_29dof.xml", "unitree_g1 (HELD OUT)"),
    "x2":   ("/var/tmp/alphamotion/robot_models/agibot_x2_urdf/X2_URDF-v1.4.0/"
             "agibot_x2_ultra_mjcf.xml", "agibot_x2_ultra (IN NO DATASET)"),
}
EMB = "human_smpl"
PAIRS = [("walk", "kick"), ("run", "jacks")]
RGBA_N = [[0.55, 0.57, 0.60, 1],      # hard cut  — grey
          [0.42, 0.55, 0.72, 1],      # interp    — blue
          [0.95, 0.72, 0.20, 1],      # jepa      — amber
          [0.30, 0.70, 0.45, 1]]      # maskfsq   — green
RGBA_BRIDGE = [0.82, 0.29, 0.36, 1]   # every robot turns this while inventing


def geom_ids(m, prefix):
    """Every geom belonging to the copy whose bodies carry `prefix`."""
    out = []
    for gi in range(m.ngeom):
        bn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi])) or ""
        if bn.startswith(prefix):
            out.append(gi)
    return np.asarray(out, np.int64)


@torch.no_grad()
def robot_stream(tok, src_ent, tgt_ent, key, kidx, spec, dof_d, rest_d, dev):
    """[T,nk,9] human key states -> (q [T,J,3], rootR [T,3,3], gp [T,J,3] cm) on the
    target robot, AFTER the feasibility projection. Also returns the pre-projection
    limit violation, which is the number that says whether the model was asking for
    something the robot cannot do."""
    rot, _ = V.complete_from_key(tok, src_ent, tgt_ent, key.to(dev).float(), kidx, dev)
    Rg = rot6d_to_matrix(rot.to(dev)).double()
    q_raw, _ = MP.fit_angles(Rg, spec, dof_d, rest=rest_d, clamp=False, method="greedy")
    pre, _, _ = E.limit_report(q_raw, dof_d.cpu().numpy(), dev)
    r6, _, q = MP.project(Rg, spec, dof_d, rest=rest_d, method="global")
    gR, gp = MP.fk_from_angles(q, spec, dof_d, rest=rest_d, root_R=Rg[:, 0])
    return (q.cpu().numpy(), gR[:, 0].cpu().numpy(), gp.cpu().numpy(),
            {"pre_viol_dof_pct": pre["viol_dof_pct"],
             "worst_overshoot_deg": pre["worst_overshoot_deg"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="V7_jepa_0807,V7_maskfsq_0807")
    ap.add_argument("--targets", default="adam,g1,x2")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--la", type=int, default=30)
    ap.add_argument("--lbb", type=int, default=30)
    ap.add_argument("--bridge", type=int, default=30)
    ap.add_argument("--tokens", default="donor", choices=["donor", "interp"],
                    help="donor = tokens from a REAL held-out motion of B's family, "
                         "i.e. what a trained generator would emit (the ceiling). "
                         "interp = tokens from the endpoint straight line (the floor).")
    ap.add_argument("--size", type=int, nargs=2, default=[1600, 700])
    ap.add_argument("--gap", type=float, default=1.25)
    # The copies are spread along +Y, so the camera must look down the X axis for all
    # four to sit at the SAME depth. At the default 150 deg they recede diagonally and
    # the far one renders visibly smaller, which reads as a size difference between
    # arms when there is none.
    ap.add_argument("--azimuth", type=float, default=180)
    ap.add_argument("--elev", type=float, default=-6)
    ap.add_argument("--dist", type=float, default=5.2)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out, exist_ok=True)

    arms = [(a, f"{V.RUNS}/{a}") for a in args.runs.split(",") if a]
    arms = [(a, r) for a, r in arms if os.path.exists(f"{r}/model.pt")]
    models = {a.replace("_0807", ""): load_run(r, dev)[0] for a, r in arms}

    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    tok, _ = V.load_frozen(dev)
    src_ent = V.body_entry(V.AMASS, EMB, dev, kd.reach[EMB])
    kidx = kd.kidx[EMB]
    la, span, lbs = args.la, args.bridge, args.la + args.bridge
    stats_all, paths = {}, []

    for tkey in [t for t in args.targets.split(",") if t]:
        xml, tlabel = TARGETS[tkey]
        print(f"\n=== {tlabel} ===", flush=True)
        try:
            spec, dof, rest, qnames, xml = E.get_target(xml, E.CACHE_AMASS)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[skip] {tkey}: {e}", flush=True); continue
        assert qnames is not None, f"{tkey} needs an MJCF-backed target"
        tg = E.target_geom(spec, dof, dev, sem_encoder=V.frozen_sem_encoder())
        tgt_ent = {"spec": spec, "st": tg["st"], "sem": tg["sem"], "dof": tg["dof"]}
        dof_d = torch.as_tensor(dof, device=dev, dtype=torch.float64)
        rest_d = torch.as_tensor(rest, device=dev, dtype=torch.float64)

        g = np.random.default_rng(2027)
        for famA, famB in PAIRS:
            try:
                sa, _ = kd.sample_starts(1, g, "val", span=args.la, families=[famA])
                sb, _ = kd.sample_starts(1, g, "val", span=args.lbb, families=[famB])
                A = kd.window(EMB, sa, args.la, args.la)[0]
                B = B7.yaw_align(kd.window(EMB, sb, args.lbb, args.lbb)[0], A[-1])
                donor = None
                if args.tokens == "donor":
                    sd, _ = kd.sample_starts(1, g, "val", span=span, families=[famB])
                    donor = kd.window(EMB, sd, span, span)[0]
            except Exception as e:
                print(f"WARN sample {famA}->{famB}: {e}", flush=True); continue

            # ---- the four bridges, as human key states -------------------------
            cur, tgt = A[-1][None].to(dev).float(), B[0][None].to(dev).float()
            kn = torch.ones(1, device=dev)
            m0 = next(iter(models.values()))
            interp = m0.baseline(cur, tgt, kn, span).float()
            seqs = {"hard cut (no model)": A[-1][None].expand(span, -1, -1).clone(),
                    "interp (0 bits)": interp[0].cpu()}
            for nm, m in models.items():
                s = donor[None].to(dev).float() if donor is not None else interp
                e = m.encode(s, s[:, 0], s[:, -1], kn) if donor is not None else \
                    m.encode(interp, cur, tgt, kn)
                o = m.decode(e["path_z"], e["time_z"], cur, tgt, kn, span)
                seqs[nm] = to_states(o, m, cur, tgt, kn).float()[0].cpu()
            full = {k: torch.cat([A, v, B], 0) for k, v in seqs.items()}

            # ---- project each onto THIS robot ----------------------------------
            streams, labels = [], []
            for k, v in full.items():
                try:
                    q, rootR, gp, st = robot_stream(tok, src_ent, tgt_ent, v, kidx,
                                                    spec, dof_d, rest_d, dev)
                except Exception as e:
                    import traceback; traceback.print_exc()
                    print(f"WARN stream {tkey} {k}: {e}", flush=True); continue
                streams.append((q, rootR, gp)); labels.append(k)
                stats_all[f"{tkey} | {famA}->{famB} | {k}"] = st
            if not streams:
                continue

            # ---- MuJoCo scene: one copy per bridge variant ---------------------
            n = len(streams)
            prefixes = [f"m{i}_" for i in range(n)]
            m = RN.build_scene(xml, prefixes, RGBA_N[:n], args.size[0], args.size[1])
            d = mj.MjData(m)
            rend = mj.Renderer(m, height=args.size[1], width=args.size[0])
            cam = mj.MjvCamera()
            cam.lookat[:] = [0, 0, 0.85]
            cam.distance, cam.elevation, cam.azimuth = args.dist, args.elev, args.azimuth
            gid = [geom_ids(m, p) for p in prefixes]
            base_rgba = [m.geom_rgba[x].copy() for x in gid]
            root_adr, slot_adr = [], []
            for p in prefixes:
                fr = [int(m.jnt_qposadr[j]) for j in range(m.njnt)
                      if (mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or "").startswith(p)
                      and m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
                root_adr.append(fr[0])
                tab = -np.ones((spec.J, 3), np.int64)
                for j, ns in enumerate(qnames):
                    for k2, nm2 in enumerate(ns):
                        jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, p + nm2)
                        if jid >= 0:
                            tab[j, k2] = int(m.jnt_qposadr[jid])
                slot_adr.append(tab)
            ys = [(i - (n - 1) / 2) * args.gap for i in range(n)]

            T = min(s[0].shape[0] for s in streams)
            frames = []
            for t in range(T):
                inb = la <= t < lbs
                for i, (q, rootR, gp) in enumerate(streams):
                    m.geom_rgba[gid[i]] = RGBA_BRIDGE if inb else base_rgba[i]
                    pm = gp[t] @ RN.AX / 100.0
                    z = -float(pm[:, 2].min())
                    d.qpos[root_adr[i]:root_adr[i] + 3] = [0.0, ys[i], z]
                    d.qpos[root_adr[i] + 3:root_adr[i] + 7] = RN.mat_to_quat(
                        RN.AX.T @ rootR[t] @ RN.AX)
                    tab = slot_adr[i]
                    for j in range(spec.J):
                        for k2 in range(3):
                            if tab[j, k2] >= 0:
                                d.qpos[tab[j, k2]] = q[t, j, k2]
                mj.mj_forward(m, d)
                rend.update_scene(d, cam)
                img = rend.render().copy()
                phase = ("MODEL BRIDGE - every pose invented" if inb else
                         "clip A (real)" if t < la else "clip B (real)")
                lab_col = (tuple(int(255 * c) for c in RGBA_BRIDGE[:3]) if inb
                           else None)
                labs = [(labels[i], lab_col or tuple(int(255 * c) for c in RGBA_N[i][:3]))
                        for i in range(n)]
                # "hard cut" invents nothing — it FREEZES. Calling both INVENTED would
                # credit the no-model baseline with work it is not doing.
                nts = [(("" if not inb else
                         "FROZEN" if labels[i].startswith("hard cut") else "INVENTED"),
                        (150, 155, 162)) for i in range(n)]
                img = RN.annotate(
                    img, labs,
                    f"{tlabel}  |  {famA}->{famB}  |  f{t}/{T-1}  |  {phase}",
                    f"tok={args.tokens}",
                    [int((0.5 + i) * img.shape[1] / n) for i in range(n)], nts)
                frames.append(img)
            p = f"{args.out}/mesh_{tkey}_{famA}_{famB}_{args.tokens}.mp4"
            RN.write_mp4(p, np.stack(frames), args.fps)
            paths.append(p); print(f"  -> {p}", flush=True)
            del rend, m, d

    json.dump(stats_all, open(f"{V.RUNS}/V7_BRIDGE_MESH_limits.json", "w"), indent=1)
    print("\nPRE-PROJECTION LIMIT VIOLATION (what the model ASKED for):", flush=True)
    for k in sorted(stats_all):
        s = stats_all[k]
        print(f"  {k:<58} viol {s['pre_viol_dof_pct']:>6}%  worst {s['worst_overshoot_deg']:>7}deg",
              flush=True)
    print("\nMP4:", flush=True)
    for p in paths:
        print("  " + p, flush=True)
    json.dump(paths, open(f"{args.out}/_index.json", "w"), indent=1)


if __name__ == "__main__":
    main()
