"""Shared data/eval logic for DualTranslator experiments (E0/E5 ablations).

Factored out of train_dual.py so training and the standalone eval harness
(eval_dual.py) run literally the same code path. Behavior-identical to the
FIX_* runs: same load order, same metrics.

RNG contract (load-bearing for split reproduction): the FIX split is
    torch.manual_seed(0) -> [data loading, consumes NO rng] -> torch.randperm(N)
Data loading consumes no RNG only because the SigLIP name-embedding disk cache
(semantics.CACHE) hits; it existed for all FIX-era runs. split_indices() must
therefore be called immediately after manual_seed(0) + load, BEFORE model init.
"""
from __future__ import annotations
import os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import rot6d_to_matrix, geodesic_deg
from model_dual import DualTranslator
from train_ae import static_cond
from train_global import build_global

FINGER_KW = ("Hand", "Thumb", "Index", "Middle", "Ring", "Pinky", "Finger")


def load_paired_data(cache, device):
    """Load a paired cache -> per-embodiment dict. Consumes no RNG (see module doc).
    Manifest schema 2 lists N embodiments; legacy manifests imply [soma77, g1_name]."""
    import json, numpy as np
    from common import SkeletonSpec
    man = json.load(open(f"{cache}/manifest.json"))
    names = man.get("embodiments") or ["soma77", man["g1_name"]]
    paired = {}
    for name in names:
        d = np.load(f"{cache}/{name}_spec.npz", allow_pickle=True)
        spec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], list(d["joint_names"]))
        paired[name] = (spec, torch.load(f"{cache}/{name}_paired_local6d.pt"))
    from semantics import build_name_embeddings, embed_matrix
    lut = build_name_embeddings(sum([list(paired[n][0].joint_names) for n in names], []), device=device)
    data = {}; N = None
    for name in names:
        spec, local6d = paired[name]
        st = static_cond(spec, device)
        gr6, gp, reach = build_global(local6d, spec, device)
        sem = embed_matrix(spec.joint_names, lut).to(device)     # [J,768] SigLIP joint-name semantics
        data[name] = {"spec": spec, "st": st, "pose": torch.cat([gr6, gp / reach], -1).to(device),
                      "reach": reach, "sem": sem}
        N = local6d.shape[0]; print(f"[{name}] J={spec.J} rows={N} reach={reach:.1f}cm sem={tuple(sem.shape)}")
    return data, names, N


def split_indices(N):
    """The FIX train/val split. Call order contract in module docstring."""
    perm = torch.randperm(N); nval = max(128, N // 10)
    return perm[:nval], perm[nval:]


def parse_fsq_levels(s):
    """'8,8,8,5,5,5' -> (8,8,8,5,5,5)"""
    return tuple(int(x) for x in str(s).split(","))


def build_model(cfg, device):
    """Construct DualTranslator from a config dict (train args or {run}/config.json).
    Unknown/missing keys fall back to FIX-era defaults so old configs stay loadable."""
    model = DualTranslator(
        n_latent=cfg.get("n_latent", 32),
        fsq_stages=cfg.get("fsq_stages", 2),
        fsq_levels=parse_fsq_levels(cfg.get("fsq_levels", "8,8,8,5,5,5")),
        single_codebook=cfg.get("single_codebook", False),
    ).to(device)
    return model


def geom(data, name, B):
    st = data[name]["st"]
    return dict(rest_off=st["rest_off"][None].expand(B, -1, -1), bonelen=st["bonelen"][None].expand(B, -1),
                depth=st["depth"][None].expand(B, -1), hop=st["hop"][None].expand(B, -1, -1),
                sem=data[name]["sem"][None].expand(B, -1, -1))


def run_dir(model, data, src, tgt, idx, joint_mask=None, visible_boost=0.0):
    """One direction src->tgt on rows idx. Returns (mean geodesic deg, mean pos in reach units).
    joint_mask [B,J_src]: E0b masked pretraining — encoder sees partial pose, loss stays full-body.
    visible_boost > 0 (same-skeleton directions only): upweight the VISIBLE joints' loss by
    (1+boost) so the completion learns to honor given constraints, not just plausibility."""
    B = len(idx)
    zp, zr = model.encode(data[src]["pose"][idx], geom(data, src, B), joint_mask=joint_mask)
    prot, ppos = model.decode(zp, zr, geom(data, tgt, B))
    gt = data[tgt]["pose"][idx]
    rot_j = geodesic_deg(rot6d_to_matrix(prot), rot6d_to_matrix(gt[..., :6]))   # [B,J]
    pos_j = (ppos - gt[..., 6:9]).norm(dim=-1)                                  # [B,J]
    if visible_boost > 0 and joint_mask is not None and src == tgt:
        w = 1.0 + visible_boost * (~joint_mask).float()
        w = w / w.mean()
        return (rot_j * w).mean(), (pos_j * w).mean()
    return rot_j.mean(), pos_j.mean()


def swing_twist_report(model, data, name, idx):
    d = data[name]; B = len(idx)
    zp, zr = model.encode(d["pose"][idx], geom(data, name, B))
    prot, ppos = model.decode(zp, zr, geom(data, name, B))
    pr = rot6d_to_matrix(prot); gt = rot6d_to_matrix(d["pose"][idx][..., :6])
    a0 = torch.nn.functional.normalize(d["st"]["rest_off"], dim=-1)[None].expand(B, -1, -1)
    full = geodesic_deg(pr, gt)                          # [B,J]
    # swing = bone-direction error (position-observable); residual ~ twist
    vp = (pr @ a0.unsqueeze(-1)).squeeze(-1); vg = (gt @ a0.unsqueeze(-1)).squeeze(-1)
    swing = torch.rad2deg(torch.arccos((vp * vg).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)))
    finger = torch.tensor([any(k in nm for k in FINGER_KW) for nm in d["spec"].joint_names], device=pr.device)
    body = ~finger
    return {"full_rot": round(float(full[:, body].mean()), 1),
            "swing": round(float(swing[:, body].mean()), 1),
            "twist_residual": round(float(full[:, body].mean() - swing[:, body].mean()), 1)}


def evaluate(model, data, names, combos, val_idx):
    """The FIX eval dict: 4-direction rot/pos, swing/twist per embodiment, code alignment.
    code_align_dist generalizes to N embodiments as the mean over unordered pairs
    (identical value to the FIX single-pair number when len(names)==2)."""
    was_training = model.training
    model.eval(); out = {}
    with torch.no_grad():
        for src, tgt in combos:
            rot, pos = run_dir(model, data, src, tgt, val_idx)
            # full names: [:4] truncation collided for soma77/soma_scaled/soma_nohand
            out[f"{src}->{tgt}"] = {"rot": round(float(rot), 1), "pos_cm": round(float(pos * data[tgt]["reach"]), 2)}
        for name in names:                              # swing/twist breakdown (self-recon)
            out[f"ST_{name}"] = swing_twist_report(model, data, name, val_idx)
        # code alignment: how close are codes of the SAME motion across embodiments?
        zs = {n: model.encode(data[n]["pose"][val_idx], geom(data, n, len(val_idx))) for n in names}
        dists, scale = [], None
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                (zp_a, zr_a), (zp_b, zr_b) = zs[names[i]], zs[names[j]]
                dists.append(float(((zp_a - zp_b) ** 2).mean() + ((zr_a - zr_b) ** 2).mean()))
        zp0, zr0 = zs[names[0]]
        out["code_align_dist"] = round(sum(dists) / max(len(dists), 1), 4)
        out["code_scale"] = round(float((zp0 ** 2).mean() + (zr0 ** 2).mean()), 4)
    if was_training:
        model.train()
    return out
