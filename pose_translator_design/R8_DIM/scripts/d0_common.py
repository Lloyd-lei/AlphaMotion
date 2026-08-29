"""R8: intrinsic-dimension estimators + raw-motion loaders. All GPU, no_grad."""
import os, sys, json, re
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo'
R8 = '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/R8_DIM'
SP = os.path.dirname(os.path.abspath(__file__))
ROOT = "/var/tmp/alphamotion/topo_exp"
AMASS = f"{ROOT}/cache_xwbc_amass"
LAFAN = f"{ROOT}/cache_xwbc_lafan"
DEV = "cuda"

sys.path.insert(0, EXP)

# ---- palette (same as R7) ----
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"
GRID = "#e1e0d9"; BASE = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": BASE, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "figure.dpi": 150, "font.family": "sans-serif",
})


def savefig(fig, name):
    p = os.path.join(R8, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)
    return p


# ================= estimators =================
@torch.no_grad()
def knn_dists(X, k=21, block=1024, group=None, exclude_same_group=True):
    """X [N,D] float32 cuda. Returns kNN distances [N,k] (self excluded).
    If group given and exclude_same_group, neighbours with the same group id are masked out
    (kills the temporal-duplicate / same-clip artefact)."""
    N = X.shape[0]
    out = torch.empty(N, k, device=X.device)
    Xn2 = (X * X).sum(1)
    for i in range(0, N, block):
        xb = X[i:i + block]
        d2 = Xn2[i:i + block, None] + Xn2[None, :] - 2.0 * (xb @ X.T)
        d2.clamp_(min=0)
        idx = torch.arange(i, min(i + block, N), device=X.device)
        d2[torch.arange(xb.shape[0], device=X.device), idx] = float("inf")
        if group is not None and exclude_same_group:
            same = group[i:i + block, None] == group[None, :]
            d2masked = d2.masked_fill(same, float("inf"))
            # keep rows that would lose everything (shouldn't happen)
            d2 = d2masked
        vals, _ = torch.topk(d2, k, dim=1, largest=False)
        out[i:i + block] = vals.sqrt()
    return out


def twonn(dk):
    """Facco et al. 2017. dk [N,>=2] sorted NN distances."""
    r1 = dk[:, 0].double(); r2 = dk[:, 1].double()
    good = (r1 > 1e-12) & (r2 > r1 * (1 + 1e-9))
    mu = (r2[good] / r1[good]).cpu().numpy()
    mu = np.sort(mu)
    n = len(mu)
    F = np.arange(1, n + 1) / (n + 1.0)
    keep = int(n * 0.90)                      # standard: drop the top 10% tail
    x = np.log(mu[:keep]); y = -np.log(1 - F[:keep])
    d = float((x * y).sum() / (x * x).sum())  # least squares through origin
    return d


def mle_levina_bickel(dk, k):
    """Levina-Bickel MLE with MacKay-Ghahramani inverse averaging. dk [N,>=k]."""
    d = dk[:, :k].double()
    Tk = d[:, k - 1:k]
    Tj = d[:, :k - 1]
    ok = (Tj > 1e-12).all(1) & (Tk[:, 0] > 1e-12)
    r = torch.log(Tk[ok] / Tj[ok]).mean(1)    # = 1/m_k  (inverse dim per point)
    return float(1.0 / r.mean().item())


@torch.no_grad()
def corr_dim(X, group=None, n_pairs=1_500_000, lo=0.02, hi=0.20, seed=0):
    """Grassberger-Procaccia: slope of log C(r) vs log r in a small-r scaling band.
    Pairs sampled at random; same-group pairs dropped."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    N = X.shape[0]
    i = torch.randint(0, N, (n_pairs,), generator=g)
    j = torch.randint(0, N, (n_pairs,), generator=g)
    m = i != j
    i, j = i[m].to(X.device), j[m].to(X.device)
    if group is not None:
        m2 = group[i] != group[j]
        i, j = i[m2], j[m2]
    CH = max(1, int(4e7 // max(X.shape[1], 1)))     # cap the temp buffer at ~40M floats
    ds = []
    for s in range(0, i.numel(), CH):
        ds.append((X[i[s:s + CH]] - X[j[s:s + CH]]).norm(dim=1))
    d = torch.cat(ds); del ds
    d = d[d > 0]
    sub = d[torch.randperm(d.numel(), device=d.device)[:200000]].float()
    q = torch.quantile(sub, torch.tensor([lo, hi], device=d.device))
    r_lo, r_hi = q[0].item(), q[1].item()
    rs = np.exp(np.linspace(np.log(r_lo), np.log(r_hi), 20))
    C = np.array([(d < r).float().mean().item() for r in rs])
    ok = C > 0
    x = np.log(rs[ok]); y = np.log(C[ok])
    A = np.vstack([x, np.ones_like(x)]).T
    slope = np.linalg.lstsq(A, y, rcond=None)[0][0]
    return float(slope), (rs, C)


def pca_stats(X):
    """X [N,D] cuda float32. Returns dict with participation ratio + variance dims."""
    Xc = X - X.mean(0, keepdim=True)
    N, D = Xc.shape
    # eigenvalues of covariance via SVD (use gram matrix if D >> N)
    with torch.no_grad():
        if D > N:
            G = (Xc @ Xc.T).double()
            ev = torch.linalg.eigvalsh(G).flip(0).clamp(min=0) / (N - 1)
        else:
            C = (Xc.T @ Xc).double() / (N - 1)
            ev = torch.linalg.eigvalsh(C).flip(0).clamp(min=0)
    ev = ev.cpu().numpy()
    tot = ev.sum()
    cum = np.cumsum(ev) / tot
    pr = float(tot ** 2 / (ev ** 2).sum())
    return {"pr": pr,
            "d90": int(np.searchsorted(cum, 0.90) + 1),
            "d95": int(np.searchsorted(cum, 0.95) + 1),
            "d99": int(np.searchsorted(cum, 0.99) + 1),
            "ambient": int(D), "eig": ev}


@torch.no_grad()
def all_estimators(X, group=None, exclude_same_group=True, kmax=21, tag=""):
    X = X.to(DEV).float().contiguous()
    p = pca_stats(X)
    dk = knn_dists(X, k=kmax, group=group, exclude_same_group=exclude_same_group)
    finite = torch.isfinite(dk).all(1)
    dk = dk[finite]
    res = {
        "n": int(X.shape[0]), "ambient": p["ambient"],
        "pca_pr": round(p["pr"], 2), "pca_d90": p["d90"], "pca_d95": p["d95"], "pca_d99": p["d99"],
        "twonn": round(twonn(dk), 2),
        "mle_k5": round(mle_levina_bickel(dk, 5), 2),
        "mle_k10": round(mle_levina_bickel(dk, 10), 2),
        "mle_k20": round(mle_levina_bickel(dk, 20), 2),
    }
    cd, _ = corr_dim(X, group=group if exclude_same_group else None)
    res["corrdim"] = round(cd, 2)
    if tag:
        res["tag"] = tag
    return res, p["eig"]


# ================= raw motion loading =================
_SPEC_CACHE = {}


def load_spec(cache, body):
    from common import SkeletonSpec
    key = (cache, body)
    if key not in _SPEC_CACHE:
        d = np.load(f"{cache}/{body}_spec.npz", allow_pickle=True)
        _SPEC_CACHE[key] = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                                        list(d["joint_names"]))
    return _SPEC_CACHE[key]


_L6D_CACHE = {}


def get_local6d(cache, body):
    key = (cache, body)
    if key not in _L6D_CACHE:
        _L6D_CACHE.clear()          # keep at most one 700MB tensor resident
        _L6D_CACHE[key] = torch.load(f"{cache}/{body}_paired_local6d.pt", map_location="cpu")
    return _L6D_CACHE[key]


@torch.no_grad()
def fk_positions(cache, body, rows, chunk=8192):
    """-> root-relative global joint positions [n,J,3] (cm, world-oriented),
    root global rotation [n,3,3], joint names."""
    from common import fk_global
    spec = load_spec(cache, body)
    local6d = get_local6d(cache, body)
    local6d = local6d[torch.as_tensor(rows)]
    outs, roots = [], []
    for i in range(0, local6d.shape[0], chunk):
        gR, gp = fk_global(local6d[i:i + chunk].to(DEV), spec)
        outs.append(gp.cpu()); roots.append(gR[:, 0].cpu())
    del local6d
    return torch.cat(outs), torch.cat(roots), list(spec.joint_names)


def _yaw(rootR):
    """Heading angle from the root's global rotation (works for any skeleton)."""
    f = rootR[:, :, 0]                       # root local +x axis, in world
    return torch.atan2(f[:, 2], f[:, 0])


def _rot_y(p, th):
    """Rotate points p [..., 3] about +Y by angle th (broadcastable to p[...,0])."""
    c, s = torch.cos(th), torch.sin(th)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return torch.stack([x * c + z * s, y, -x * s + z * c], -1)


def canon_heading(gp, rootR):
    """Rotate each frame about +Y so the root faces a canonical direction. gp [n,J,3]."""
    th = -_yaw(rootR)
    return _rot_y(gp, th[:, None]), th


def canon_heading_window(gpw, rootR0):
    """gpw [n,T,J,3] canonicalised by the window's FIRST frame heading (turning preserved)."""
    th = -_yaw(rootR0)
    return _rot_y(gpw, th[:, None, None])


def seq_table(cache):
    seqs = json.load(open(f"{cache}/sequences.json"))
    return seqs


def sample_frames(cache, n_target, seed=0, clip_subset=None):
    """Evenly spread frame rows across clips. Returns rows [n], clipid [n], names."""
    seqs = seq_table(cache)
    idxs = range(len(seqs)) if clip_subset is None else clip_subset
    rows, cid = [], []
    tot = sum(seqs[i]["len"] for i in idxs)
    stride = max(1, tot // n_target)
    for i in idxs:
        s = seqs[i]
        r = np.arange(s["start"], s["start"] + s["len"], stride)
        rows.append(r); cid.append(np.full(len(r), i, np.int32))
    rows = np.concatenate(rows); cid = np.concatenate(cid)
    if len(rows) > n_target:
        g = np.random.default_rng(seed)
        sel = np.sort(g.choice(len(rows), n_target, replace=False))
        rows, cid = rows[sel], cid[sel]
    return rows, cid, [s["name"] for s in seqs]


def sample_windows(cache, T, n_target, seed=0, clip_subset=None):
    """Window start rows spread across clips. Returns starts [n], clipid [n]."""
    seqs = seq_table(cache)
    idxs = list(range(len(seqs))) if clip_subset is None else list(clip_subset)
    valid = [(i, seqs[i]["len"] - T) for i in idxs if seqs[i]["len"] > T]
    tot = sum(v for _, v in valid)
    if tot == 0:
        return np.array([]), np.array([])
    stride = max(1, tot // n_target)
    starts, cid = [], []
    for i, L in valid:
        s = seqs[i]["start"]
        r = np.arange(s, s + L, stride)
        starts.append(r); cid.append(np.full(len(r), i, np.int32))
    starts = np.concatenate(starts); cid = np.concatenate(cid)
    if len(starts) > n_target:
        g = np.random.default_rng(seed)
        sel = np.sort(g.choice(len(starts), n_target, replace=False))
        starts, cid = starts[sel], cid[sel]
    return starts, cid


@torch.no_grad()
def window_features(cache, body, starts, T, drop_root=True, joints=None):
    """Load window data. Returns [n, T*(J-1)*3] float32 (cm/reach), heading-canonicalised
    at the window's first frame."""
    grid = starts[:, None] + np.arange(T)[None, :]
    need = np.unique(grid.ravel())
    gp, rootR, jn = fk_positions(cache, body, need)
    lut = np.searchsorted(need, grid)
    gpw = gp[torch.as_tensor(lut)]                       # [n,T,J,3]
    gpw = canon_heading_window(gpw, rootR[torch.as_tensor(lut[:, 0])])
    reach = gp.norm(dim=-1).max().item()
    if joints is not None:
        gpw = gpw[:, :, joints, :]
    elif drop_root:
        gpw = gpw[:, :, 1:, :]
    return (gpw / reach).reshape(gpw.shape[0], -1).float(), jn


@torch.no_grad()
def frame_features(cache, body, rows, drop_root=True, canon=True, joints=None):
    gp, rootR, jn = fk_positions(cache, body, rows)
    reach = gp.norm(dim=-1).max().item()
    if canon:
        gp, _ = canon_heading(gp, rootR)
    if joints is not None:
        gp = gp[:, joints, :]
    elif drop_root:
        gp = gp[:, 1:, :]
    return (gp / reach).reshape(gp.shape[0], -1).float(), jn


# ---- action families (same regex as R7) ----
FAMILIES = ["dance", "jump", "run", "walk", "sit", "reach"]
_PATS = {
    "dance": r"danc|salsa|cha_?cha|tango|ballet|jazz|samba|swing_danc",
    "jump": r"jump|hop|leap|skip",
    "run": r"run|jog|sprint",
    "walk": r"walk|stroll|march|stride",
    "sit": r"sit|chair|stool",
    "reach": r"reach|pick|grasp|grab|place",
}


def action_family(name):
    low = name.lower()
    for f in FAMILIES:
        if re.search(_PATS[f], low):
            return f
    return "other"


# ---- body-part joint groups (indices into the 22-joint SMPL-ish skeleton) ----
def part_indices(jn):
    def ix(names):
        return [jn.index(n) for n in names if n in jn]
    return {
        "legs": ix(["L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "L_Toe", "R_Toe"]),
        "arms": ix(["L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"]),
        "torso_head": ix(["Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "R_Thorax"]),
        "hands_wrists": ix(["L_Wrist", "R_Wrist"]),
        "whole": list(range(1, len(jn))),
    }
