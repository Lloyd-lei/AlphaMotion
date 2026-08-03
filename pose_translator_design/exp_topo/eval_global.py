"""Detailed eval for the GLOBAL-target AE (v5):
 - per-skeleton global-rot error split into BODY vs FINGER joints (soma has 50 fingers)
 - position error in cm and % reach
 - held-out topology zero-shot
 - per-frame encode+decode latency at batch=1 (real-time feasibility)
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np, torch

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE_ROOT, SkeletonSpec, rot6d_to_matrix, matrix_to_rot6d, fk_global, geodesic_deg
from model import TopoAE
from train_ae import load_skeleton, static_cond
from train_global import build_global

FINGER_KW = ("Hand", "Thumb", "Index", "Middle", "Ring", "Pinky", "Finger")


def make_pose(name, cache, device):
    spec, local6d = load_skeleton(cache, name)
    st = static_cond(spec, device)
    gr6, gp, reach = build_global(local6d, spec, device)
    pose = torch.cat([gr6, gp / reach], -1)
    return spec, st, pose, reach


def eval_one(model, spec, st, pose, reach, device, n=2000):
    idx = torch.arange(min(n, pose.shape[0]))
    B = len(idx)
    b = {"pose_in": pose[idx].to(device),
         "rest_off": st["rest_off"].unsqueeze(0).expand(B, -1, -1),
         "bonelen": st["bonelen"].unsqueeze(0).expand(B, -1),
         "depth": st["depth"].unsqueeze(0).expand(B, -1),
         "hop": st["hop"].unsqueeze(0).expand(B, -1, -1), "kpm": None}
    with torch.no_grad():
        pred, _ = model(b)
    pr = rot6d_to_matrix(pred[..., :6]); gt = rot6d_to_matrix(b["pose_in"][..., :6])
    rot = geodesic_deg(pr, gt)                    # [B,J] full global rotation error
    # SWING error = error in the bone's pointing direction (position-observable part).
    # Residual (full - swing) is the twist-ish component (position-UNobservable).
    axis = torch.nn.functional.normalize(st["rest_off"], dim=-1).to(device)   # [J,3]
    axisB = axis[None].expand(rot.shape[0], -1, -1)
    vp = (pr @ axisB.unsqueeze(-1)).squeeze(-1); vg = (gt @ axisB.unsqueeze(-1)).squeeze(-1)
    cos = (vp * vg).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
    swing = torch.rad2deg(torch.arccos(cos))     # [B,J]
    pos_cm = (pred[..., 6:9] - b["pose_in"][..., 6:9]).norm(dim=-1) * reach  # [B,J]
    finger = np.array([any(k in nm for k in FINGER_KW) for nm in spec.joint_names])
    body = ~finger
    out = {
        "J": spec.J, "reach_cm": round(reach, 1),
        "global_rot_deg_body": round(float(rot[:, body].mean()), 2),
        "swing_deg_body": round(float(swing[:, body].mean()), 2),
        "twist_residual_deg_body": round(float(rot[:, body].mean() - swing[:, body].mean()), 2),
        "pos_cm_body": round(float(pos_cm[:, body].mean()), 2),
        "pct_finger_joints": round(float(finger.mean()) * 100, 1),
    }
    if finger.any():
        out["global_rot_deg_finger"] = round(float(rot[:, finger].mean()), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=f"{CACHE_ROOT}/cache_big")
    ap.add_argument("--run", default=f"{CACHE_ROOT}/runs/v5_global")
    ap.add_argument("--seen", nargs="+", default=["soma77", "g1_30"])
    ap.add_argument("--heldout", default="soma_nohand")
    args = ap.parse_args()
    device = "cuda"
    cfg = json.load(open(f"{args.run}/config.json"))
    model = TopoAE(d_model=cfg["d_model"], n_latent=cfg["n_latent"],
                   use_index=cfg.get("use_index", False), target="global",
                   use_fsq=cfg.get("use_fsq", False)).to(device)
    model.load_state_dict(torch.load(f"{args.run}/model.pt", map_location=device))
    model.eval()

    res = {"seen": {}, "heldout": {}, "latency": {}}
    for name in args.seen:
        spec, st, pose, reach = make_pose(name, args.cache, device)
        res["seen"][name] = eval_one(model, spec, st, pose, reach, device)
        # latency bs=1
        b1 = {"pose_in": pose[:1].to(device),
              "rest_off": st["rest_off"][None], "bonelen": st["bonelen"][None],
              "depth": st["depth"][None], "hop": st["hop"][None], "kpm": None}
        with torch.no_grad():
            for _ in range(10): model(b1)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(200): model(b1)
            torch.cuda.synchronize()
        res["latency"][name] = round((time.time() - t0) / 200 * 1000, 3)
    try:
        spec, st, pose, reach = make_pose(args.heldout, args.cache, device)
        res["heldout"][args.heldout] = eval_one(model, spec, st, pose, reach, device)
    except Exception as e:
        res["heldout"]["error"] = repr(e)[:120]
    print(json.dumps(res, indent=2))
    json.dump(res, open(f"{args.run}/eval_detailed.json", "w"), indent=2)


if __name__ == "__main__":
    main()
