"""EXAM A — the DOWNSTREAM exam: are the temporal layer's outputs PHYSICALLY
consistent, measured GT-free on the deployment-relevant case (the bridge)?

    exam_a.py

Chain = [clip A real 30f | invented bridge 30f | clip B real 30f] on unitree_g1
(held out of V7 training), same construction as v7_bridge_mesh, donor tokens.
Every variant is feasibility-projected, assembled into the robot's own qpos, and
scored by physeval (inverse dynamics vs motor ratings, foot slip, penetration,
balance, jerk). The REAL segments are identical across variants, so any difference
between rows is the bridge's doing — and the real segments themselves are the
in-video floor: an invented bridge cannot be expected to beat the data it joins.

The OTHER half of Exam A — WBC trackability — stays blocked on FlashSAC/GMT and is
not faked here.
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj                                                  # noqa: E402
import v5_common as V                                                # noqa: E402
import v7_bridge as B7                                               # noqa: E402
import v7_bridge_mesh as BM                                          # noqa: E402
import eval_newbody as E                                             # noqa: E402
import physeval as P                                                 # noqa: E402
from train_v5 import load_run, to_states                             # noqa: E402

EMB = "human_smpl"
G1_XML = ("/var/tmp/alphamotion/config_harvest/GMR/assets/unitree_g1/"
          "g1_mocap_29dof.xml")
PAIRS = [("walk", "kick"), ("run", "jacks")]


def qpos_stream(q, rootR, gp, model, spec, qnames):
    """(q [T,J,3], rootR [T,3,3], gp [T,J,3] cm) -> qpos [T,nq] for the SINGLE-robot
    model: free joint from root pose (lowest body on the floor), hinges by name."""
    fr = [j for j in range(model.njnt) if model.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
    assert len(fr) == 1
    ra = int(model.jnt_qposadr[fr[0]])
    tab = -np.ones((spec.J, 3), np.int64)
    for j, ns in enumerate(qnames):
        for k, nm in enumerate(ns):
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, nm)
            if jid >= 0:
                tab[j, k] = int(model.jnt_qposadr[jid])
    T = q.shape[0]
    qp = np.zeros((T, model.nq))
    for t in range(T):
        pm = gp[t] @ BM.RN.AX / 100.0
        qp[t, ra:ra + 3] = [0.0, 0.0, -float(pm[:, 2].min())]
        qp[t, ra + 3:ra + 7] = BM.RN.mat_to_quat(BM.RN.AX.T @ rootR[t] @ BM.RN.AX)
        for j in range(spec.J):
            for k in range(3):
                if tab[j, k] >= 0:
                    qp[t, tab[j, k]] = q[t, j, k]
    return qp


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="V8_arclen_0807,V7_maskfsq_0807")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    tok, _ = V.load_frozen(dev)
    src_ent = V.body_entry(V.AMASS, EMB, dev, kd.reach[EMB])
    kidx = kd.kidx[EMB]
    models = {a.replace("_0807", ""): load_run(f"{V.RUNS}/{a}", dev)[0]
              for a in args.runs.split(",") if a}

    spec, dof, rest, qnames, xml = E.get_target(G1_XML, E.CACHE_AMASS)
    tg = E.target_geom(spec, dof, dev)
    tgt_ent = {"spec": spec, "st": tg["st"], "sem": tg["sem"], "dof": tg["dof"]}
    dof_d = torch.as_tensor(dof, device=dev, dtype=torch.float64)
    rest_d = torch.as_tensor(rest, device=dev, dtype=torch.float64)
    pmodel = mj.MjModel.from_xml_path(xml)
    cfg = P.PhysEvalConfig()

    la = span = 30
    rows = {}
    g = np.random.default_rng(2027)
    for famA, famB in PAIRS:
        sa, _ = kd.sample_starts(1, g, "val", span=la, families=[famA])
        sb, _ = kd.sample_starts(1, g, "val", span=la, families=[famB])
        A = kd.window(EMB, sa, la, la)[0]
        B = B7.yaw_align(kd.window(EMB, sb, la, la)[0], A[-1])
        sd, _ = kd.sample_starts(1, g, "val", span=span, families=[famB])
        donor = kd.window(EMB, sd, span, span)[0]

        cur, tgt = A[-1][None].to(dev).float(), B[0][None].to(dev).float()
        kn = torch.ones(1, device=dev)
        m0 = next(iter(models.values()))
        interp = m0.baseline(cur, tgt, kn, span).float()
        seqs = {"hard_cut": A[-1][None].expand(span, -1, -1).clone(),
                "interp": interp[0].cpu()}
        for nm, m in models.items():
            d = donor[None].to(dev).float()
            e = m.encode(d, d[:, 0], d[:, -1], kn)
            o = m.decode(e["path_z"], e["time_z"], cur, tgt, kn, span)
            seqs[nm] = to_states(o, m, cur, tgt, kn).float()[0].cpu()

        for k, v in seqs.items():
            full = torch.cat([A, v, B], 0)
            q, rootR, gp, _ = BM.robot_stream(tok, src_ent, tgt_ent, full, kidx,
                                              spec, dof_d, rest_d, dev)
            qp = qpos_stream(q, rootR, gp, pmodel, spec, qnames)
            r_all = P.evaluate(pmodel, qp, cfg, label=f"{famA}->{famB} {k}")
            r_br = P.evaluate(pmodel, qp[la:la + span], cfg,
                              label=f"{famA}->{famB} {k} BRIDGE-ONLY")
            KEYS = ("slip_p50_ms", "slip_p95_ms", "torque_over_pct", "jointlim_pct",
                    "penetration_max_mm", "selfcol_pct_x", "icp_out_frac", "jerk_p95")
            pick = lambda r: {kk: r["headline"].get(kk) for kk in KEYS}
            rows[f"{famA}->{famB} | {k}"] = {"chain": pick(r_all),
                                             "bridge_only": pick(r_br)}
            print(f"{famA}->{famB} {k}: {json.dumps(rows[f'{famA}->{famB} | {k}'])}",
                  flush=True)

    txt = json.dumps(rows, indent=1)
    open(f"{V.RUNS}/EXAMA_verdict.json", "w").write(txt + "\n")
    print(f"\nverdict: {V.RUNS}/EXAMA_verdict.json", flush=True)


if __name__ == "__main__":
    main()
