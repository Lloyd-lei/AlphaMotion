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
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import rot6d_to_matrix, geodesic_deg
from model_dual import DualTranslator
from train_ae import static_cond
from train_global import build_global

FINGER_KW = ("Hand", "Thumb", "Index", "Middle", "Ring", "Pinky", "Finger")


class PoseStore:
    """0809 full-13k surgery: 5.1M rows x 496 joints no longer fit resident on GPU.
    Rows live fp16 on CPU; indexing gathers the batch and returns fp32 on `device` —
    every consumer does data[n]["pose"][idx], so this is a drop-in. ~36MB/step of
    PCIe traffic at bs 256 x 8 combos, invisible next to the forward pass."""

    def __init__(self, pose_cpu16, device):
        self.t = pose_cpu16
        self.device = device

    @property
    def shape(self):
        return self.t.shape

    def __len__(self):
        return self.t.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx) and idx.device.type != "cpu":
            idx = idx.cpu()
        return self.t[idx].to(self.device, non_blocking=True).float()


def load_paired_data(cache, device, max_rows=0, sem_encoder="siglip", pose_device=None):
    """Load a paired cache -> per-embodiment dict. Consumes no RNG (see module doc).
    Manifest schema 2 lists N embodiments; legacy manifests imply [soma77, g1_name].
    max_rows > 0 truncates every embodiment to the first max_rows frames — SMOKE TESTS
    ONLY (it changes N and therefore the split); 0 = the real, unchanged path.
    sem_encoder picks the frozen joint-name text encoder ("siglip" 768 / "qwen3" 1024);
    its disk cache must already exist or the load consumes RNG and breaks the split.
    `cache` may be a comma list of caches with identical embodiment sets — rows are
    concatenated in the listed order (lafan,amass_full for the 0809 arms).
    pose_device: "gpu" = legacy resident-on-GPU fp32; "cpu16" = PoseStore; None = auto
    (cpu16 once total rows exceed 2.5M, printed loudly)."""
    import json, numpy as np, gc
    from common import SkeletonSpec
    caches = [c for c in str(cache).split(",") if c]
    man = json.load(open(f"{caches[0]}/manifest.json"))
    names = man.get("embodiments") or ["soma77", man["g1_name"]]
    for c in caches[1:]:
        m2 = json.load(open(f"{c}/manifest.json"))
        n2 = m2.get("embodiments") or ["soma77", m2["g1_name"]]
        assert set(n2) == set(names), f"embodiment mismatch: {c}"
    specs = {}
    for name in names:
        d = np.load(f"{caches[0]}/{name}_spec.npz", allow_pickle=True)
        specs[name] = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                                   list(d["joint_names"]))
    from semantics import build_name_embeddings, embed_matrix
    lut = build_name_embeddings(sum([list(specs[n].joint_names) for n in names], []),
                                device=device, encoder=sem_encoder)
    if pose_device is None:
        pose_device = os.environ.get("POSE_DEVICE") or None   # smoke-test override
    if pose_device is None:
        rows0 = sum(torch.load(f"{c}/{names[0]}_paired_local6d.pt", mmap=True).shape[0]
                    for c in caches)
        pose_device = "cpu16" if (rows0 if not max_rows else min(rows0, max_rows)) > 2_500_000 \
            else "gpu"
        if pose_device == "cpu16":
            print(f"[load_paired_data] {rows0} rows -> pose resident fp16 on CPU "
                  f"(PoseStore)", flush=True)
    data = {}; N = None
    for name in names:
        spec = specs[name]
        l6 = torch.cat([torch.load(f"{c}/{name}_paired_local6d.pt") for c in caches]) \
            if len(caches) > 1 else torch.load(f"{caches[0]}/{name}_paired_local6d.pt")
        local6d = l6[:max_rows].contiguous() if max_rows else l6
        st = static_cond(spec, device)
        gr6, gp, reach = build_global(local6d, spec, device)
        sem = embed_matrix(spec.joint_names, lut).to(device)     # [J,D] joint-name text semantics
        pose = torch.cat([gr6, gp / reach], -1)
        entry = {"spec": spec, "st": st,
                 "pose": PoseStore(pose.half(), device) if pose_device == "cpu16"
                 else pose.to(device),
                 "reach": reach, "sem": sem}
        dof_path = f"{caches[0]}/{name}_dof.npz"                 # mechanical spec (axes + limits)
        if os.path.exists(dof_path):
            entry["dof"] = torch.as_tensor(np.load(dof_path)["dof"], dtype=torch.float32, device=device)
            from dof_head import build_kin                       # E25: precomputed hinge kinematics
            entry["kin"] = build_kin(caches[0], name, spec, device=device)
        data[name] = entry
        N = local6d.shape[0]; print(f"[{name}] J={spec.J} rows={N} reach={reach:.1f}cm sem={tuple(sem.shape)}")
        del l6, local6d, gr6, gp, pose
        gc.collect()
    return data, names, N


def split_indices(N, max_val=5000):
    """The FIX train/val split. Call order contract in module docstring.
    max_val caps eval size for big caches (251k rows -> 25k val OOMed the one-shot
    eval encode); FIX-era caches (48k -> 4800) are below the cap, bitwise unchanged."""
    perm = torch.randperm(N); nval = min(max(128, N // 10), max_val)
    return perm[:nval], perm[nval:]


def parse_fsq_levels(s):
    """'8,8,8,5,5,5' -> (8,8,8,5,5,5)"""
    return tuple(int(x) for x in str(s).split(","))


def build_model(cfg, device):
    """Construct DualTranslator from a config dict (train args or {run}/config.json).
    Unknown/missing keys fall back to FIX-era defaults so old configs stay loadable."""
    from semantics import SEM_DIM
    model = DualTranslator(
        d_model=cfg.get("d_model", 256),
        n_heads=cfg.get("n_heads", 8),
        enc_layers=cfg.get("enc_layers", 4),
        dec_layers=cfg.get("dec_layers", 4),
        n_latent=cfg.get("n_latent", 32),
        fsq_stages=cfg.get("fsq_stages", 2),
        fsq_levels=parse_fsq_levels(cfg.get("fsq_levels", "8,8,8,5,5,5")),
        single_codebook=cfg.get("single_codebook", False),
        n_judges=cfg.get("n_judges", 0),
        use_dof=cfg.get("use_dof", False),
        rot_head=cfg.get("rot_head", "free"),        # E25 arm A ("angles"); default = every prior run
        # E27: text-encoder width. Configs with no "sem_encoder" key -> 768, i.e. SigLIP.
        sem_dim=SEM_DIM[cfg.get("sem_encoder", "siglip")],
        # E28: residual decode of encoder-visible joints. Missing key -> False -> the
        # model has no extra modules and behaves bit-for-bit like every prior run.
        vis_residual=cfg.get("vis_residual", False),
                          rot_only=cfg.get("rot_only", False),
    ).to(device)
    return model


def geom(data, name, B):
    st = data[name]["st"]
    g = dict(rest_off=st["rest_off"][None].expand(B, -1, -1), bonelen=st["bonelen"][None].expand(B, -1),
             depth=st["depth"][None].expand(B, -1), hop=st["hop"][None].expand(B, -1, -1),
             sem=data[name]["sem"][None].expand(B, -1, -1))
    g["spec"] = data[name]["spec"]        # static; rot_only decode needs it for FK
    g["reach"] = data[name]["reach"]
    if "dof" in data[name]:
        g["dof"] = data[name]["dof"][None].expand(B, -1, -1)
        g["kin"] = data[name]["kin"]         # static, not batch-expanded (E25)
    return g


def fk_pos_torch(rot6d, spec):
    """DIFFERENTIABLE FK from GLOBAL rot6d -> joint positions (root at origin).

    Same recipe as v5_common.fk_pos (the numpy yardstick every reported position
    comes through), written in torch so the SAME quantity can enter the loss.
    Added 0812: the audit showed the 5.0-weighted position loss trains hpos,
    whose output is discarded at render time, leaving hrot — the head that
    actually determines the picture — supervised only by a per-joint geodesic
    that is blind to lever arms. This closes that loop."""
    R = rot6d_to_matrix(rot6d)                                  # [B,J,3,3]
    par = list(spec.parents)
    off = torch.as_tensor(np.asarray(spec.rest_offsets),
                          dtype=R.dtype, device=R.device)       # [J,3]
    B, J = R.shape[0], R.shape[1]
    cols = [None] * J
    order = sorted(range(J), key=lambda j: _depth(par, j))      # parents first
    for j in order:
        pj = par[j]
        if pj < 0:
            cols[j] = R.new_zeros(B, 3)
        else:
            cols[j] = cols[pj] + torch.einsum("bik,k->bi", R[:, pj], off[j])
    return torch.stack(cols, 1)                                 # [B,J,3]


def _depth(par, j):
    d = 0
    while par[j] >= 0:
        j = par[j]; d += 1
    return d


def run_dir(model, data, src, tgt, idx, joint_mask=None, visible_boost=0.0, dof_loss=False,
            extras=None, tail_alpha=0.0, arm_boost=0.0, ee_boost=0.0,
            fk_loss=False):
    """One direction src->tgt on rows idx. Returns (mean geodesic deg, mean pos in reach units).
    extras (N3f --var_floor): pass a dict and it gains "var_floor" = the hinge
    max(0, std_gt - std_decoded) on the per-joint/per-dim std ACROSS THE BATCH of the
    decoded pose (rot6d + pos together), the anti-amplitude-compression penalty whose
    target is the GT batch's own variance. None (default) = bit-exact prior behavior.
    joint_mask [B,J_src]: E0b masked pretraining — encoder sees partial pose, loss stays full-body.
    visible_boost > 0 (same-skeleton directions only): upweight the VISIBLE joints' loss by
    (1+boost) so the completion learns to honor given constraints, not just plausibility.
    dof_loss (E25 arm B, TRAINING ONLY): before the geodesic, project prediction and target
    local rotations onto each joint's hinge manifold, so the loss never charges the model
    for components the mechanism cannot realize. Genuine 3-DOF joints and the root pass
    through untouched. Eval never sets this — the yardstick stays the raw global geodesic."""
    B = len(idx)
    zp, zr = model.encode(data[src]["pose"][idx], geom(data, src, B), joint_mask=joint_mask)
    obs = vis = None
    if getattr(model, "vis_residual", False) and src == tgt:
        # E28: SAME-SKELETON decode only — the decoder may lean on the pose the encoder
        # actually saw (final = observed + delta). joint_mask is SOURCE-side, so for a
        # cross-body direction (src != tgt) there is no observed target pose and the
        # residual path must stay off: translation still runs purely from the codes,
        # which keeps the codes honest (no identity shortcut can leak across bodies).
        obs = data[src]["pose"][idx]
        vis = ~joint_mask if joint_mask is not None else \
            torch.ones(B, obs.shape[1], dtype=torch.bool, device=obs.device)
    prot, ppos = model.decode(zp, zr, geom(data, tgt, B), obs=obs, vis=vis)
    gt = data[tgt]["pose"][idx]
    if extras is not None:
        # N3f: "match the data's variance" — decoded per-joint/dim std over the
        # batch must not fall below the GT batch's own (regression-to-mean cure).
        pv = torch.cat([prot, ppos], -1)                                    # [B,J,9]
        extras["var_floor"] = torch.relu(gt.std(0).detach() - pv.std(0)).mean()
    Rp, Rt = rot6d_to_matrix(prot), rot6d_to_matrix(gt[..., :6])
    if dof_loss and "kin" in data[tgt]:
        from dof_head import project_global6d
        kin = data[tgt]["kin"]
        Rp = project_global6d(Rp, kin)
        Rt = project_global6d(Rt, kin).detach()
    rot_j = geodesic_deg(Rp, Rt)                                               # [B,J]
    if fk_loss:
        # 0812: the position term is computed by running FK on the PREDICTED
        # rotations instead of reading the discarded hpos head, so the metric
        # error (which knows about lever arms) supervises the head that renders.
        # Same units as before: the cached pose stores gp / reach.
        fkp = fk_pos_torch(prot, data[tgt]["spec"]) / data[tgt]["reach"]
        fkg = fk_pos_torch(gt[..., :6], data[tgt]["spec"]) / data[tgt]["reach"]
        pos_j = (fkp - fkg).norm(dim=-1)                                        # [B,J]
    else:
        pos_j = (ppos - gt[..., 6:9]).norm(dim=-1)                              # [B,J]
    if arm_boost > 0 or ee_boost > 0:
        # decoder post-train (0811): static per-joint pressure on the arm chain.
        # ee_boost hits the wrist/hand POSITION term only — the 0811 probes
        # showed ~17% of the wrist p95 tail is per-frame recoverable and it is
        # positional. mean-1 normalized; fingers excluded; composes with
        # tail_alpha below.
        w2 = data[tgt].get("_arm_w2")
        if w2 is None:
            KW = ("shoulder", "elbow", "wrist", "hand", "arm", "clavicle",
                  "collar", "scap", "bicep")
            EE = ("wrist", "hand", "palm")
            nl = [n.lower() for n in data[tgt]["spec"].joint_names]
            fin = [any(k.lower() in n for k in FINGER_KW) for n in nl]
            m = torch.tensor([any(k in n for k in KW) and not f
                              for n, f in zip(nl, fin)], device=rot_j.device)
            me = torch.tensor([any(k in n for k in EE) and not f
                               for n, f in zip(nl, fin)], device=rot_j.device)
            aw = 1.0 + arm_boost * m.float()
            pw = 1.0 + arm_boost * m.float() + ee_boost * me.float()
            w2 = (aw / aw.mean(), pw / pw.mean())
            data[tgt]["_arm_w2"] = w2
        rot_j = rot_j * w2[0][None]
        pos_j = pos_j * w2[1][None]
    if visible_boost > 0 and joint_mask is not None and src == tgt:
        w = 1.0 + visible_boost * (~joint_mask).float()
        w = w / w.mean()
        return (rot_j * w).mean(), (pos_j * w).mean()
    if tail_alpha > 0:
        # 0809 tail cure (backfist verdict): a mean objective lets the model abandon
        # low-density pose regions. Reweight each sample-joint by its OWN detached
        # relative error ^ alpha (mean-1 normalized), so hard poses buy gradient in
        # proportion to how badly they are being fit. TRAINING ONLY — evaluate()
        # never passes tail_alpha; the yardstick stays the plain mean.
        wr = (rot_j.detach() / (rot_j.detach().mean() + 1e-9)) ** tail_alpha
        wp = (pos_j.detach() / (pos_j.detach().mean() + 1e-9)) ** tail_alpha
        return (rot_j * (wr / wr.mean())).mean(), (pos_j * (wp / wp.mean())).mean()
    return rot_j.mean(), pos_j.mean()


def amplitude_report(model, data, dirs, pools, n_win=24, T=90, seed=0):
    """AMPLITUDE RATIO — how much of the real motion's range survives decoding.

    Added 0813 after lloyd watched a punch render and said the robot barely
    moved. Mean position error cannot see this: collapsing toward the average
    pose is a LOW-error strategy, so a model can score well while producing
    motion that is visibly dead. Measured on contiguous windows (the metric is
    temporal; evaluate()'s random rows cannot express it).

    ratio = std_over_time(decoded FK positions) / std_over_time(GT FK positions),
    per joint, averaged. 1.0 = the real range; 0.2 = a body that barely moves.
    """
    g = np.random.default_rng(seed)
    ok = [(a, b) for a, b in pools if b - a >= T + 2]
    starts = [int(g.integers(a, b - T)) for a, b in
              (ok[i] for i in g.choice(len(ok), n_win, replace=False))]
    ARM = ("shoulder", "elbow", "wrist", "arm", "hand", "clavicle")
    LEG = ("hip", "knee", "ankle", "foot", "leg", "toe", "shank")
    out = {}
    was = model.training
    model.eval()
    with torch.no_grad():
        for src, tgt in dirs:
            spec = data[tgt]["spec"]
            nl = [x.lower() for x in spec.joint_names]
            am = torch.tensor([any(k in x for k in ARM) for x in nl])
            lm = torch.tensor([any(k in x for k in LEG) for x in nl])
            acc = {"all": [], "arm": [], "leg": []}
            for s0 in starts:
                rows = torch.arange(s0, s0 + T)
                p9 = data[src]["pose"][rows]
                prot, _ = model.decode(*model.encode(p9, geom(data, src, T)),
                                       geom(data, tgt, T))
                fp = fk_pos_torch(prot, spec).cpu()
                fg = fk_pos_torch(data[tgt]["pose"][rows][..., :6], spec).cpu()
                sp_, sg = fp.std(0).norm(dim=-1), fg.std(0).norm(dim=-1)
                for k, msk in (("all", None), ("arm", am), ("leg", lm)):
                    a_ = sp_ if msk is None else sp_[msk]
                    b_ = sg if msk is None else sg[msk]
                    if len(a_):
                        acc[k].append(float(a_.mean() / (b_.mean() + 1e-9)))
            out[f"{src}->{tgt}"] = {k: round(float(np.mean(v)), 3)
                                    for k, v in acc.items() if v}
    if was:
        model.train()
    return out


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
        if getattr(model, "judge", None) is not None:    # E14: per-embodiment critic readout
            # judge indices follow TRAIN-time embodiment order, which need not match this
            # cache's names (e.g. evaluating an 18-judge model on the 2-body BONES cache)
            sub = val_idx[:1024]; jr = {}
            n_j = model.judge.net[-1].out_features
            for n in names:
                zp_j, zr_j = model.encode(data[n]["pose"][sub], geom(data, n, len(sub)))
                act = torch.sigmoid(model.judge(zp_j, zr_j)).mean(0)
                top = int(act.argmax())
                self_i = names.index(n) if len(names) == n_j else None
                jr[n] = {"self": round(float(act[self_i]), 3) if self_i is not None else None,
                         "top_idx": top, "top_val": round(float(act[top]), 3)}
            out["judges"] = jr
    if was_training:
        model.train()
    return out
