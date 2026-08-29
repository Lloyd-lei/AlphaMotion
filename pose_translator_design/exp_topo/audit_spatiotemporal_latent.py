"""Read-only audit of AlphaMotion's spatial and V7 temporal latent spaces.

This script does not train or mutate either model.  It answers narrower questions:

* What trajectory information is present in each temporal bottleneck after the two
  endpoint states have already been supplied to the decoder?
* Is that information global, time-local, duration-like, or embodiment-specific?
* How stable are aggregate reconstruction claims under clip-level resampling?
* What does the frozen spatial bottleneck retain when all but six key joints are
  hidden, and how much embodiment identity remains linearly decodable?

All predictive probes split by source clip.  Adjacent windows from one clip can
never appear on opposite sides of a probe split.  This is important because the
motion caches contain dense 30 fps trajectories and random frame splits leak nearly
duplicate poses.
"""
from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

import v5_common as V  # noqa: E402
from common import geodesic_deg, rot6d_to_matrix  # noqa: E402
from train_v5 import load_run, orthonormalize  # noqa: E402


RUN_NAMES = ("V7_jepa_0807", "V7_maskfsq_0807")
FAM6 = ("walk", "run", "jacks", "dance", "kick", "punch")
TEMP_BODIES = ("human_smpl", "unitree_h1", "fourier_gr3", "pal_talos",
               "unitree_g1_29dof")
SPATIAL_BODIES = TEMP_BODIES
ALPHAS = np.logspace(-2, 4, 13)


def as_float_dict(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = as_float_dict(v)
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def clip_balanced_windows(kd, split, per_clip, seed, families=None, max_clips=0):
    """Sample unique starts, with every eligible clip receiving equal weight."""
    rng = np.random.default_rng(seed)
    pools = kd.pools(split, V.T_WIN)
    if families is not None:
        keep = set(families)
        pools = [p for p in pools if kd.family[p[0]] in keep]
    if max_clips and len(pools) > max_clips:
        pools = [pools[i] for i in rng.choice(len(pools), max_clips, replace=False)]
    starts, cids = [], []
    for cid, lo, hi in pools:
        possible = hi - lo - V.T_WIN + 1
        take = min(per_clip, possible)
        offsets = rng.choice(possible, take, replace=False)
        starts.extend((lo + offsets).tolist())
        cids.extend([cid] * take)
    order = rng.permutation(len(starts))
    return np.asarray(starts, np.int64)[order], np.asarray(cids, np.int64)[order]


def family_probe_windows(kd, per_clip=8, max_clips=40, seed=29):
    rng = np.random.default_rng(seed)
    starts, cids, labels = [], [], []
    counts = {}
    pools = kd.pools("train", V.T_WIN)
    for fam in FAM6:
        fp = [p for p in pools if kd.family[p[0]] == fam]
        if len(fp) > max_clips:
            fp = [fp[i] for i in rng.choice(len(fp), max_clips, replace=False)]
        counts[fam] = {"clips": len(fp), "windows": 0}
        for cid, lo, hi in fp:
            possible = hi - lo - V.T_WIN + 1
            take = min(per_clip, possible)
            offsets = rng.choice(possible, take, replace=False)
            starts.extend((lo + offsets).tolist())
            cids.extend([cid] * take)
            labels.extend([fam] * take)
            counts[fam]["windows"] += take
    order = rng.permutation(len(starts))
    return (np.asarray(starts, np.int64)[order], np.asarray(cids, np.int64)[order],
            np.asarray(labels)[order], counts)


def linear_baseline(w):
    t = torch.linspace(0.0, 1.0, w.shape[1], dtype=w.dtype)
    u = t[None, :, None, None]
    return orthonormalize((1.0 - u) * w[:, :1] + u * w[:, -1:])


def trajectory_targets(w, reach_cm):
    """Motion summaries not reducible to merely naming the endpoints."""
    w = w.float().cpu()
    pos = w[..., 6:9]
    dp = (pos[:, 1:] - pos[:, :-1]).norm(dim=-1) * reach_cm
    R = rot6d_to_matrix(w[..., :6])
    da = geodesic_deg(R[:, 1:], R[:, :-1])
    base = linear_baseline(w)
    pos_loop = ((pos - base[..., 6:9]).pow(2).sum(-1).mean(1).sqrt()
                * reach_cm)
    rot_loop = geodesic_deg(R, rot6d_to_matrix(base[..., :6])).pow(2).mean(1).sqrt()
    pos_path = dp.sum(1)
    rot_path = da.sum(1)

    ds = V.se3_step(w).numpy()
    bounds = np.round(np.linspace(0, ds.shape[1], 10)).astype(int)
    local = np.stack([ds[:, bounds[i]:bounds[i + 1]].sum(1)
                      for i in range(9)], 1)
    y = np.concatenate([pos_path.numpy(), rot_path.numpy(), pos_loop.numpy(),
                        rot_loop.numpy(), local], 1)
    names = ([f"pos_path_cm:{j}" for j in range(6)]
             + [f"rot_path_deg:{j}" for j in range(6)]
             + [f"pos_loop_cm:{j}" for j in range(6)]
             + [f"rot_loop_deg:{j}" for j in range(6)]
             + [f"local_se3:{i}" for i in range(9)])
    blocks = {
        "position_path": np.arange(0, 6),
        "rotation_path": np.arange(6, 12),
        "position_nonlin": np.arange(12, 18),
        "rotation_nonlin": np.arange(18, 24),
        "local_timing": np.arange(24, 33),
    }
    return y.astype(np.float32), names, blocks


def endpoint_features(w):
    return torch.cat([w[:, 0].reshape(len(w), -1),
                      w[:, -1].reshape(len(w), -1)], -1).float().numpy()


@torch.no_grad()
def encode_temporal(model, kd, emb, starts, device, batch=192, decode=True):
    rows = collections.defaultdict(list)
    reach = kd.reach[emb]
    for lo in range(0, len(starts), batch):
        st = starts[lo:lo + batch]
        w = kd.window(emb, st, V.T_WIN, V.T_WIN).to(device).float()
        cur, tgt = w[:, 0], w[:, -1]
        kn = torch.ones(len(w), device=device)
        e = model.encode(w, cur, tgt, kn)
        for key in ("path_q", "path_f", "time_q", "time_f"):
            rows[key].append(e[key].float().cpu())
        pi, ti = model.token_ints(e)
        rows["path_int"].append(pi.cpu())
        rows["time_int"].append(ti.cpu())
        if decode:
            pred = model.decode(e["path_z"], e["time_z"], cur, tgt, kn,
                                V.T_WIN).float()
            base = model.baseline(cur, tgt, kn, V.T_WIN).float()
            pe = ((pred[..., 6:9] - w[..., 6:9]).norm(dim=-1)
                  .mean((1, 2)) * reach)
            be = ((base[..., 6:9] - w[..., 6:9]).norm(dim=-1)
                  .mean((1, 2)) * reach)
            re = geodesic_deg(rot6d_to_matrix(pred[..., :6]),
                              rot6d_to_matrix(w[..., :6])).mean((1, 2))
            bre = geodesic_deg(rot6d_to_matrix(base[..., :6]),
                               rot6d_to_matrix(w[..., :6])).mean((1, 2))
            gt_energy = w.std(1).mean((1, 2)).clamp_min(1e-8)
            rows["pos_cm"].append(pe.cpu())
            rows["rot_deg"].append(re.cpu())
            rows["baseline_pos_cm"].append(be.cpu())
            rows["baseline_rot_deg"].append(bre.cpu())
            rows["energy_ratio"].append((pred.std(1).mean((1, 2)) / gt_energy).cpu())

            # Causal bottleneck ablations.  Shuffling is the in-distribution test;
            # zeroing is retained as the conventional but OOD diagnostic.
            perm = torch.roll(torch.arange(len(w), device=device), 1)
            p_zero = model.decode(torch.zeros_like(e["path_z"]), e["time_z"],
                                  cur, tgt, kn, V.T_WIN).float()
            p_shuffle = model.decode(e["path_z"][perm], e["time_z"],
                                     cur, tgt, kn, V.T_WIN).float()
            rows["zero_path_pos_cm"].append(
                ((p_zero[..., 6:9] - w[..., 6:9]).norm(dim=-1)
                 .mean((1, 2)) * reach).cpu())
            rows["shuffle_path_pos_cm"].append(
                ((p_shuffle[..., 6:9] - w[..., 6:9]).norm(dim=-1)
                 .mean((1, 2)) * reach).cpu())
            if e["time_z"].shape[1]:
                t_zero = model.decode(e["path_z"], torch.zeros_like(e["time_z"]),
                                      cur, tgt, kn, V.T_WIN).float()
                t_shuffle = model.decode(e["path_z"], e["time_z"][perm],
                                         cur, tgt, kn, V.T_WIN).float()
                rows["zero_time_pos_cm"].append(
                    ((t_zero[..., 6:9] - w[..., 6:9]).norm(dim=-1)
                     .mean((1, 2)) * reach).cpu())
                rows["shuffle_time_pos_cm"].append(
                    ((t_shuffle[..., 6:9] - w[..., 6:9]).norm(dim=-1)
                     .mean((1, 2)) * reach).cpu())
        del w, e
    return {k: torch.cat(v).numpy() for k, v in rows.items()}


def ridge_grouped(feature_sets, y, groups, blocks, n_splits=12, seed=31):
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=seed)
    scores = {name: {"overall": [], **{b: [] for b in blocks}}
              for name in feature_sets}
    alphas = {name: [] for name in feature_sets}
    for tr, te in splitter.split(y, groups=groups):
        for name, X in feature_sets.items():
            pipe = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            pipe.fit(X[tr], y[tr])
            p = pipe.predict(X[te])
            scores[name]["overall"].append(
                r2_score(y[te], p, multioutput="variance_weighted"))
            for block, idx in blocks.items():
                scores[name][block].append(
                    r2_score(y[te][:, idx], p[:, idx], multioutput="variance_weighted"))
            alphas[name].append(float(pipe[-1].alpha_))
    out = {}
    for name, by_block in scores.items():
        out[name] = {}
        for block, vals in by_block.items():
            vals = np.asarray(vals)
            out[name][block] = {
                "median_r2": float(np.median(vals)),
                "group_split_p025": float(np.quantile(vals, 0.025)),
                "group_split_p975": float(np.quantile(vals, 0.975)),
            }
        out[name]["ridge_alpha_median"] = float(np.median(alphas[name]))
    return out


def bootstrap_group_difference(a, b, groups, n_boot=5000, seed=43):
    """Clip-level paired bootstrap of mean(a-b)."""
    gids = np.unique(groups)
    ga, gb = [], []
    for g in gids:
        m = groups == g
        ga.append(float(np.mean(a[m])))
        gb.append(float(np.mean(b[m])))
    ga, gb = np.asarray(ga), np.asarray(gb)
    d = ga - gb
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = d[rng.integers(0, len(d), len(d))].mean()
    return {
        "mean_delta": float(d.mean()),
        "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "n_clips": int(len(gids)),
    }


def discrete_usage(path_int, time_int, levels=5):
    def entropy(x):
        count = np.bincount(x.astype(np.int64), minlength=levels).astype(np.float64)
        p = count[count > 0] / max(count.sum(), 1.0)
        return float(-(p * np.log2(p)).sum())

    def bank(D):
        if D.shape[1] == 0:
            return None
        by_reg = []
        for r in range(D.shape[1]):
            bits = sum(entropy(D[:, r, k]) for k in range(D.shape[2]))
            unique = len(np.unique(D[:, r], axis=0))
            by_reg.append({"entropy_bits": bits, "unique_codes": int(unique)})
        return {"registers": by_reg,
                "entropy_bits_total": float(sum(x["entropy_bits"] for x in by_reg)),
                "nominal_bits_total": float(D.shape[1] * D.shape[2] * math.log2(levels))}

    return {"path": bank(path_int), "time": bank(time_int)}


def classifier_grouped(X, labels, groups, n_splits=4, seed=53):
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    vals = []
    for tr, te in sg.split(X, labels, groups):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0),
        )
        clf.fit(X[tr], labels[tr])
        vals.append(balanced_accuracy_score(labels[te], clf.predict(X[te])))
    return {"balanced_accuracy_mean": float(np.mean(vals)),
            "balanced_accuracy_std": float(np.std(vals)),
            "folds": [float(x) for x in vals]}


def body_classifier(X, labels, motion_groups, n_splits=10, seed=59):
    vals = []
    split = GroupShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=seed)
    for tr, te in split.split(X, labels, motion_groups):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2500, C=1.0))
        clf.fit(X[tr], labels[tr])
        vals.append(balanced_accuracy_score(labels[te], clf.predict(X[te])))
    return {"balanced_accuracy_mean": float(np.mean(vals)),
            "balanced_accuracy_std": float(np.std(vals)),
            "chance": 1.0 / len(np.unique(labels))}


def body_classifier_highdim(X, labels, motion_groups, n_splits=8, seed=61):
    vals = []
    split = GroupShuffleSplit(n_splits=n_splits, test_size=0.25, random_state=seed)
    for tr, te in split.split(X, labels, motion_groups):
        n_comp = min(96, len(tr) - 1, X.shape[1] - 1)
        clf = make_pipeline(
            StandardScaler(with_mean=False),
            TruncatedSVD(n_components=n_comp, random_state=seed),
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=1.0),
        )
        clf.fit(X[tr], labels[tr])
        vals.append(balanced_accuracy_score(labels[te], clf.predict(X[te])))
    return {"balanced_accuracy_mean": float(np.mean(vals)),
            "balanced_accuracy_std": float(np.std(vals)),
            "chance": 1.0 / len(np.unique(labels))}


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    # The primal dxd expression is wasteful for E19's 3328 scalar code.  The
    # mathematically identical sample-Gram form stays nxn (n is a few hundred).
    K, L = X @ X.T, Y @ Y.T
    return float((K * L).sum() /
                 max(np.linalg.norm(K, "fro") * np.linalg.norm(L, "fro"), 1e-12))


def cosine_rows(X, Y):
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Y = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    return (X * Y).sum(1)


def crossbody_stats(features, endpoints):
    bodies = list(features)
    cka = np.eye(len(bodies))
    same_cos, diff_cos, digit_agree = {}, {}, {}
    for i, a in enumerate(bodies):
        for j in range(i + 1, len(bodies)):
            b = bodies[j]
            X, Y = features[a], features[b]
            cka[i, j] = cka[j, i] = linear_cka(X, Y)
            same = cosine_rows(X, Y)
            diff = cosine_rows(X, np.roll(Y, 1, axis=0))
            key = f"{a}__{b}"
            same_cos[key] = float(np.median(same))
            diff_cos[key] = float(np.median(diff))
            labels = np.r_[np.ones(len(same)), np.zeros(len(diff))]
            digit_agree[key] = float((X == Y).mean())
            same_cos[key + "_auc"] = float(roc_auc_score(labels, np.r_[same, diff]))
    X = np.concatenate([features[b] for b in bodies])
    E = np.concatenate([endpoints[b] for b in bodies])
    labels = np.concatenate([[b] * len(features[b]) for b in bodies])
    groups = np.tile(np.arange(len(next(iter(features.values())))), len(bodies))
    return {
        "bodies": bodies,
        "cka": cka.tolist(),
        "same_motion_cosine_median": same_cos,
        "exact_scalar_agreement": digit_agree,
        "body_probe_latent": (body_classifier_highdim(X, labels, groups)
                              if X.shape[1] > 512
                              else body_classifier(X, labels, groups)),
        "body_probe_endpoints": body_classifier(E, labels, groups),
    }


def local_incremental_matrix(endpoints, q, local_y, groups, seed=67):
    """Delta R2(endpoint + one register -> one time bin) over endpoint alone."""
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(split.split(endpoints, groups=groups))
    base_scores = []
    for t in range(local_y.shape[1]):
        base = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
        base.fit(endpoints[tr], local_y[tr, t])
        base_scores.append(r2_score(local_y[te, t], base.predict(endpoints[te])))
    mat = np.zeros((q.shape[1], local_y.shape[1]))
    for r in range(q.shape[1]):
        X = np.concatenate([endpoints, q[:, r]], 1)
        for t in range(local_y.shape[1]):
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            m.fit(X[tr], local_y[tr, t])
            mat[r, t] = r2_score(local_y[te, t], m.predict(X[te])) - base_scores[t]
    return mat


@torch.no_grad()
def rate_probe(model, kd, emb, starts, device, batch=128):
    rates = (30, 45, 60, 90, 120)
    banks = {k: [] for k in ("path_q", "path_f", "time_q", "time_f", "path_int")}
    for T in rates:
        cur = collections.defaultdict(list)
        fidx = V.uniform_fidx(V.T_WIN, T)
        for lo in range(0, len(starts), batch):
            st = starts[lo:lo + batch]
            w = kd.window_fidx(emb, st, fidx, V.T_WIN).to(device).float()
            kn = torch.ones(len(w), device=device)
            e = model.encode(w, w[:, 0], w[:, -1], kn)
            pi, _ = model.token_ints(e)
            for key in ("path_q", "path_f", "time_q", "time_f"):
                cur[key].append(e[key].float().cpu())
            cur["path_int"].append(pi.cpu())
        for key in banks:
            banks[key].append(torch.cat(cur[key]).numpy())

    ref = rates.index(60)
    agreement, cosine = {}, {}
    for i, T in enumerate(rates):
        agreement[str(T)] = float((banks["path_int"][i] ==
                                   banks["path_int"][ref]).mean())
        cosine[str(T)] = float(np.median(cosine_rows(
            banks["path_q"][i].reshape(len(starts), -1),
            banks["path_q"][ref].reshape(len(starts), -1))))

    labels = np.repeat(np.asarray(rates), len(starts))
    groups = np.tile(np.arange(len(starts)), len(rates))

    def rate_acc(key):
        if banks[key][0].shape[1] == 0:
            return None
        X = np.concatenate([x.reshape(len(starts), -1) for x in banks[key]])
        vals = []
        split = GroupShuffleSplit(n_splits=8, test_size=0.25, random_state=71)
        for tr, te in split.split(X, labels, groups):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2500, C=1.0))
            clf.fit(X[tr], labels[tr])
            vals.append(balanced_accuracy_score(labels[te], clf.predict(X[te])))
        return {"balanced_accuracy_mean": float(np.mean(vals)),
                "balanced_accuracy_std": float(np.std(vals)), "chance": 0.2}

    time_agreement, time_cosine = None, None
    if banks["time_q"][0].shape[1]:
        time_agreement, time_cosine = {}, {}
        for i, T in enumerate(rates):
            time_agreement[str(T)] = float(
                (banks["time_q"][i] == banks["time_q"][ref]).mean())
            time_cosine[str(T)] = float(np.median(cosine_rows(
                banks["time_q"][i].reshape(len(starts), -1),
                banks["time_q"][ref].reshape(len(starts), -1))))

    return {"rates": rates, "path_digit_agreement_vs_60": agreement,
            "path_cosine_vs_60": cosine,
            "time_scalar_agreement_vs_60": time_agreement,
            "time_cosine_vs_60": time_cosine,
            "decode_rate_from_path_q": rate_acc("path_q"),
            "decode_rate_from_path_f": rate_acc("path_f"),
            "decode_rate_from_time_q": rate_acc("time_q"),
            "decode_rate_from_time_f": rate_acc("time_f")}


@torch.no_grad()
def encode_spatial_body(tok, kd, emb, rows, device, batch=128, sparse=False):
    entry = V.body_entry(V.AMASS, emb, device, reach=kd.reach[emb])
    pose, reach = V.full_pose_of(V.AMASS, emb, rows, device,
                                 reach=kd.reach[emb])
    qbanks, endpoints = [], []
    recon = collections.defaultdict(list)
    key_idx = kd.kidx[emb]
    for lo in range(0, len(rows), batch):
        x = pose[lo:lo + batch].to(device).float()
        B, J = x.shape[:2]
        geom = V.geom_for(entry, B)
        mask = None
        if sparse:
            mask = torch.ones(B, J, dtype=torch.bool, device=device)
            mask[:, key_idx] = False
        zp, zr, (cp, cr) = tok.encode(x, geom, joint_mask=mask, return_codes=True)
        qbanks.append(torch.cat([cp, cr], 1).float().cpu())
        endpoints.append(x[:, key_idx].reshape(B, -1).cpu())
        if sparse:
            rr, pp = tok.decode(zp, zr, geom)
            vis = ~mask
            p_err = (pp - x[..., 6:9]).norm(dim=-1) * reach
            r_err = geodesic_deg(rot6d_to_matrix(rr), rot6d_to_matrix(x[..., :6]))
            recon["hidden_pos_cm"].append(p_err[~vis].cpu())
            recon["visible_pos_cm"].append(p_err[vis].cpu())
            recon["hidden_rot_deg"].append(r_err[~vis].cpu())
            recon["visible_rot_deg"].append(r_err[vis].cpu())
    del pose, entry
    gc.collect()
    out = {"q": torch.cat(qbanks).numpy(),
           "endpoints": torch.cat(endpoints).numpy()}
    if sparse:
        out["reconstruction"] = {k: {"mean": float(torch.cat(v).mean()),
                                      "median": float(torch.cat(v).median())}
                                 for k, v in recon.items()}
    return out


def spatial_audit(kd_human, starts, cids, device, semantic_cache):
    # Never let this read-only audit rewrite the shared semantic cache.  Prime a
    # private copy with the union of all audited names so every later body lookup is
    # a cache hit (semantics.build_name_embeddings otherwise overwrites, not merges).
    import semantics as S
    semantic_cache = Path(semantic_cache)
    semantic_cache.parent.mkdir(parents=True, exist_ok=True)
    shared = Path(S.CACHES["siglip"])
    if not semantic_cache.exists() and shared.exists():
        shutil.copy2(shared, semantic_cache)
    S.CACHES["siglip"] = str(semantic_cache)
    all_names = []
    for emb in SPATIAL_BODIES:
        all_names.extend(V.load_spec(V.AMASS, emb).joint_names)
    S.build_name_embeddings(list(dict.fromkeys(all_names)), device=device)

    tok, cfg = V.load_frozen(device)
    rows = starts + V.T_WIN // 2
    by_body, by_body_pos, by_body_rot, endpoints = {}, {}, {}, {}
    human_full = None
    for emb in SPATIAL_BODIES:
        kd = kd_human if emb == "human_smpl" else V.KeyData(
            V.AMASS, [emb], device="cpu", verbose=False)
        z = encode_spatial_body(tok, kd, emb, rows, device)
        n_stream = z["q"].shape[1] // 2
        by_body[emb] = z["q"].reshape(len(rows), -1)
        by_body_pos[emb] = z["q"][:, :n_stream].reshape(len(rows), -1)
        by_body_rot[emb] = z["q"][:, n_stream:].reshape(len(rows), -1)
        endpoints[emb] = z["endpoints"]
        if emb == "human_smpl":
            human_full = z
        del z
        gc.collect()

    sparse = encode_spatial_body(tok, kd_human, "human_smpl", rows, device,
                                 sparse=True)
    full_q = human_full["q"].reshape(len(rows), -1)
    sparse_q = sparse["q"].reshape(len(rows), -1)
    n_stream = human_full["q"].shape[1] // 2
    full_pos = human_full["q"][:, :n_stream].reshape(len(rows), -1)
    full_rot = human_full["q"][:, n_stream:].reshape(len(rows), -1)
    sparse_pos = sparse["q"][:, :n_stream].reshape(len(rows), -1)
    sparse_rot = sparse["q"][:, n_stream:].reshape(len(rows), -1)
    same = cosine_rows(full_q, sparse_q)
    mask_stability = {
        "full_vs_six_key_cosine_median": float(np.median(same)),
        "full_vs_six_key_cka": linear_cka(full_q, sparse_q),
        "exact_scalar_agreement": float((full_q == sparse_q).mean()),
        "position_stream": {
            "cosine_median": float(np.median(cosine_rows(full_pos, sparse_pos))),
            "cka": linear_cka(full_pos, sparse_pos),
            "exact_scalar_agreement": float((full_pos == sparse_pos).mean()),
        },
        "rotation_stream": {
            "cosine_median": float(np.median(cosine_rows(full_rot, sparse_rot))),
            "cka": linear_cka(full_rot, sparse_rot),
            "exact_scalar_agreement": float((full_rot == sparse_rot).mean()),
        },
        "pos_vs_rot_population_cka_full_pose": linear_cka(full_pos, full_rot),
        "six_key_reconstruction": sparse["reconstruction"],
    }
    out = {"config": cfg,
           "crossbody": crossbody_stats(by_body, endpoints),
           "crossbody_position_stream": crossbody_stats(by_body_pos, endpoints),
           "crossbody_rotation_stream": crossbody_stats(by_body_rot, endpoints),
           "mask_stability": mask_stability,
           "sampling": {"n_frames": len(rows), "n_clips": len(np.unique(cids)),
                        "split": "temporal validation clips; one midpoint/frame sample"}}
    del tok
    torch.cuda.empty_cache()
    return out


def data_audit(kd):
    out = {}
    for split in ("train", "val"):
        clips, wins = collections.Counter(), collections.Counter()
        for cid, lo, hi in kd.pools(split, V.T_WIN):
            fam = kd.family[cid]
            clips[fam] += 1
            wins[fam] += hi - lo - V.T_WIN + 1
        out[split] = {"clips": dict(clips), "possible_windows": dict(wins)}
    out["split_overlap_clips"] = 0
    out["split_method"] = "random clip IDs, seed=123, not family-stratified"
    out["spatial_training_eval_warning"] = (
        "dual_common.split_indices uses random frame-level splitting; neighboring "
        "frames from one clip can cross train/validation. Temporal KeyData is clip-level.")
    return out


def make_figures(result, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    orange = "#ff8000"  # ArenaLabs Papaya Orange
    gray = "#747474"
    # Incremental endpoint-conditioned R2 by target block.
    blocks = ["position_path", "rotation_path", "position_nonlin",
              "rotation_nonlin", "local_timing"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(blocks))
    width = 0.36
    for i, arm in enumerate(("jepa", "maskfsq")):
        reg = result["temporal"][arm]["regression"]
        vals = [reg["endpoint+all_q"][b]["median_r2"]
                - reg["endpoint"][b]["median_r2"] for b in blocks]
        ax.bar(x + (i - 0.5) * width, vals, width,
               color=orange if arm == "jepa" else gray, label=arm)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [b.replace("_", "\n") for b in blocks])
    ax.set_ylabel("incremental grouped-test R2 beyond endpoints")
    ax.set_title("What the quantized temporal latent adds beyond endpoint conditioning")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "temporal_incremental_r2.png", dpi=180)
    plt.close(fig)

    # JEPA factorization: the nominal time seat may carry more than duration.
    jr = result["temporal"]["jepa"]["regression"]
    labels = ["endpoints", "+ path q", "+ time q", "+ path & time q"]
    vals = [jr[k]["overall"]["median_r2"] for k in
            ("endpoint", "endpoint+path_q", "endpoint+time_q", "endpoint+all_q")]
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.bar(np.arange(4), vals, color=[gray, gray, orange, orange])
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylim(min(0, min(vals) - 0.05), 1.0)
    ax.set_ylabel("grouped-test R2, all trajectory summaries")
    ax.set_title("JEPA's single 'time' register is a global motion-summary channel")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(outdir / "jepa_factorization.png", dpi=180)
    plt.close(fig)

    # Locality maps.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for ax, arm in zip(axes, ("jepa", "maskfsq")):
        mat = np.asarray(result["temporal"][arm]["local_incremental_r2"])
        vmax = max(abs(mat.min()), abs(mat.max()), 1e-4)
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{arm}: one register's added R2")
        ax.set_xlabel("trajectory time bin")
        ax.set_ylabel("FSQ path register")
        fig.colorbar(im, ax=ax, fraction=0.035)
    fig.tight_layout()
    fig.savefig(outdir / "temporal_locality.png", dpi=180)
    plt.close(fig)

    # Rate preservation.
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rates = result["temporal"]["jepa"]["rate"]["rates"]
    for arm, color, marker in (("jepa", orange, "o"), ("maskfsq", gray, "s")):
        d = result["temporal"][arm]["rate"]["path_digit_agreement_vs_60"]
        ax.plot(rates, [d[str(t)] for t in rates], marker=marker, color=color,
                label=arm)
    ax.set_xlabel("frames used to traverse the same 60-frame path")
    ax.set_ylabel("path FSQ scalar agreement vs 60-frame view")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    ax.set_title("Rate invariance of the discrete path code")
    fig.tight_layout()
    fig.savefig(outdir / "rate_invariance.png", dpi=180)
    plt.close(fig)

    # Data split family support.
    fams = sorted(set(result["data"]["train"]["clips"])
                  | set(result["data"]["val"]["clips"]))
    tr = [result["data"]["train"]["clips"].get(f, 0) for f in fams]
    va = [result["data"]["val"]["clips"].get(f, 0) for f in fams]
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    xx = np.arange(len(fams))
    ax.bar(xx - 0.18, tr, 0.36, color=gray, label="train")
    ax.bar(xx + 0.18, va, 0.36, color=orange, label="validation")
    ax.set_yscale("log")
    ax.set_xticks(xx, fams, rotation=35, ha="right")
    ax.set_ylabel("clips (log scale)")
    ax.set_title("The temporal validation split is clip-clean but not family-stratified")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "data_split_support.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(EXP.parent / "audits" / "DEEP_AUDIT_0807"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--val-per-clip", type=int, default=4)
    ap.add_argument("--crossbody-clips", type=int, default=240)
    ap.add_argument("--rate-clips", type=int, default=160)
    ap.add_argument("--spatial-clips", type=int, default=240)
    ap.add_argument("--skip-spatial", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.manual_seed(17)
    np.random.seed(17)

    kd = V.KeyData(V.AMASS, ["human_smpl"], device="cpu", verbose=False)
    starts, cids = clip_balanced_windows(kd, "val", args.val_per_clip, 17)
    w = kd.window("human_smpl", starts, V.T_WIN, V.T_WIN)
    endpoints = endpoint_features(w)
    y, target_names, blocks = trajectory_targets(w, kd.reach["human_smpl"])
    data = data_audit(kd)
    data["audit_sampling"] = {
        "n_windows": len(starts), "n_unique_windows": len(np.unique(starts)),
        "n_clips": len(np.unique(cids)), "windows_per_clip_max": args.val_per_clip,
        "duplicate_window_count": int(len(starts) - len(np.unique(starts))),
    }

    fam_starts, fam_cids, fam_labels, fam_counts = family_probe_windows(kd)
    fam_w = kd.window("human_smpl", fam_starts, V.T_WIN, V.T_WIN)
    fam_endpoints = endpoint_features(fam_w)

    temporal, encoded = {}, {}
    models = {}
    for run in RUN_NAMES:
        arm = "maskfsq" if "maskfsq" in run else "jepa"
        print(f"[temporal] loading {run}", flush=True)
        model, cfg = load_run(f"{V.RUNS}/{run}", dev)
        models[arm] = model
        z = encode_temporal(model, kd, "human_smpl", starts, dev)
        encoded[arm] = z
        q = z["path_q"].reshape(len(starts), -1)
        f = z["path_f"].reshape(len(starts), -1)
        tq = z["time_q"].reshape(len(starts), -1)
        tf = z["time_f"].reshape(len(starts), -1)
        all_q = np.concatenate([q, tq], 1)
        all_f = np.concatenate([f, tf], 1)
        feature_sets = {
            "endpoint": endpoints,
            "path_q": q,
            "path_f": f,
            "endpoint+path_q": np.concatenate([endpoints, q], 1),
            "endpoint+path_f": np.concatenate([endpoints, f], 1),
            "endpoint+all_q": np.concatenate([endpoints, all_q], 1),
            "endpoint+all_f": np.concatenate([endpoints, all_f], 1),
        }
        if tq.shape[1]:
            feature_sets |= {
                "time_q": tq,
                "time_f": tf,
                "endpoint+time_q": np.concatenate([endpoints, tq], 1),
                "endpoint+time_f": np.concatenate([endpoints, tf], 1),
            }
        reg = ridge_grouped(feature_sets, y, cids, blocks)
        local = local_incremental_matrix(endpoints, z["path_q"],
                                         y[:, blocks["local_timing"]], cids)

        fz = encode_temporal(model, kd, "human_smpl", fam_starts, dev, decode=False)
        fq = np.concatenate([fz["path_q"].reshape(len(fam_starts), -1),
                             fz["time_q"].reshape(len(fam_starts), -1)], 1)
        family_probe = {
            "endpoint": classifier_grouped(fam_endpoints, fam_labels, fam_cids),
            "latent_q": classifier_grouped(fq, fam_labels, fam_cids),
            "endpoint+latent_q": classifier_grouped(
                np.concatenate([fam_endpoints, fq], 1), fam_labels, fam_cids),
            "support": fam_counts,
            "scope": ("representation-accessibility probe on model-training clips; "
                      "probe folds are held out by clip, but this is not model OOD"),
        }
        reconstruction = {
            "pos_cm_mean": float(z["pos_cm"].mean()),
            "rot_deg_mean": float(z["rot_deg"].mean()),
            "baseline_pos_cm_mean": float(z["baseline_pos_cm"].mean()),
            "baseline_rot_deg_mean": float(z["baseline_rot_deg"].mean()),
            "energy_ratio_mean": float(z["energy_ratio"].mean()),
            "model_minus_baseline_pos_bootstrap": bootstrap_group_difference(
                z["pos_cm"], z["baseline_pos_cm"], cids),
        }
        for key in ("zero_path_pos_cm", "shuffle_path_pos_cm",
                    "zero_time_pos_cm", "shuffle_time_pos_cm"):
            if key in z:
                reconstruction[key + "_mean"] = float(z[key].mean())
                reconstruction[key + "_minus_correct_bootstrap"] = (
                    bootstrap_group_difference(z[key], z["pos_cm"], cids))

        temporal[arm] = {
            "config": cfg,
            "regression": reg,
            "local_incremental_r2": local.tolist(),
            "family_probe": family_probe,
            "reconstruction": reconstruction,
            "discrete_usage": discrete_usage(z["path_int"], z["time_int"]),
        }
        print(f"[temporal] {arm} regression and family probe complete", flush=True)

    temporal["paired_arm_difference"] = {
        "mask_minus_jepa_pos_cm": bootstrap_group_difference(
            encoded["maskfsq"]["pos_cm"], encoded["jepa"]["pos_cm"], cids),
        "mask_minus_jepa_rot_deg": bootstrap_group_difference(
            encoded["maskfsq"]["rot_deg"], encoded["jepa"]["rot_deg"], cids),
        "mask_minus_jepa_energy_ratio": bootstrap_group_difference(
            encoded["maskfsq"]["energy_ratio"], encoded["jepa"]["energy_ratio"], cids),
    }

    # Same path at multiple frame rates.
    rate_starts, _ = clip_balanced_windows(kd, "val", 1, 79,
                                           max_clips=args.rate_clips)
    for arm, model in models.items():
        print(f"[rate] {arm}", flush=True)
        temporal[arm]["rate"] = rate_probe(model, kd, "human_smpl", rate_starts,
                                            dev)

    # Paired motions on five embodiments, including one held-out G1.
    cb_starts, cb_cids = clip_balanced_windows(kd, "val", 1, 83,
                                               max_clips=args.crossbody_clips)
    for arm, model in models.items():
        features, eps = {}, {}
        for emb in TEMP_BODIES:
            print(f"[crossbody temporal] {arm} / {emb}", flush=True)
            kdb = kd if emb == "human_smpl" else V.KeyData(
                V.AMASS, [emb], device="cpu", verbose=False)
            zz = encode_temporal(model, kdb, emb, cb_starts, dev, decode=False)
            features[emb] = zz["path_q"].reshape(len(cb_starts), -1)
            ww = kdb.window(emb, cb_starts, V.T_WIN, V.T_WIN)
            eps[emb] = endpoint_features(ww)
            if emb != "human_smpl":
                del kdb
            gc.collect()
        temporal[arm]["crossbody"] = crossbody_stats(features, eps)

    partial = {
        "scope": {
            "models_mutated": False,
            "checkpoints": [f"{V.RUNS}/{x}/model.pt" for x in RUN_NAMES],
            "temporal_claim": ("Grouped statistical probes of what existing latents "
                               "encode; not an architecture selection verdict."),
        },
        "data": data,
        "target_names": target_names,
        "temporal": temporal,
        "spatial": None,
    }
    with open(outdir / "temporal_partial.json", "w") as f:
        json.dump(as_float_dict(partial), f, indent=2, ensure_ascii=False)

    spatial = None
    if not args.skip_spatial:
        sp_starts, sp_cids = clip_balanced_windows(kd, "val", 1, 89,
                                                   max_clips=args.spatial_clips)
        print("[spatial] frozen E19 audit", flush=True)
        # Free temporal models before loading E19 and full-pose tensors.
        models.clear()
        torch.cuda.empty_cache()
        spatial = spatial_audit(kd, sp_starts, sp_cids, dev,
                                outdir / "siglip_names_audit.pt")

    result = {
        "scope": {
            "models_mutated": False,
            "checkpoints": [f"{V.RUNS}/{x}/model.pt" for x in RUN_NAMES]
                           + [f"{V.FROZEN}/model.pt"],
            "temporal_claim": ("Grouped statistical probes of what existing latents "
                               "encode; not an architecture selection verdict."),
            "elapsed_seconds": time.time() - t0,
        },
        "data": data,
        "target_names": target_names,
        "temporal": temporal,
        "spatial": spatial,
    }
    result = as_float_dict(result)
    with open(outdir / "audit.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    make_figures(result, outdir)
    with open(outdir / "README.txt", "w") as f:
        f.write("Read-only AlphaMotion spatial/temporal latent audit.\n")
        f.write(f"Elapsed: {result['scope']['elapsed_seconds']:.1f}s\n")
        f.write("Primary numbers: audit.json\n")
        f.write("Figures: temporal_incremental_r2.png, jepa_factorization.png, "
                "temporal_locality.png, rate_invariance.png, "
                "data_split_support.png\n")
    print(f"[done] {outdir / 'audit.json'}", flush=True)


if __name__ == "__main__":
    main()
