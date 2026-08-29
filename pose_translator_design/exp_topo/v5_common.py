"""ARCH V5 temporal layer — shared machinery for the five V5 scripts.

Spec: pose_translator_design/ARCH_V5_TEMPORAL.md (lloyd, 2026-08-06).

The one-line thesis this file encodes: the temporal layer does NOT model the whole
body. It models the SE3 trajectory of ~6 KEY JOINTS (root, both wrists, both ankles,
head); the other joints are completed per frame by the FROZEN spatial layer
(E19_K128_0805) through its masked-encode path. Speed is not a hand-set gear — it is
whatever the CUMULATIVE ARC-LENGTH (path-integral) curve says, quantized by its own
FSQ codebook, separate from the path codebook.

What lives here
  * KEY-JOINT resolution for ANY skeleton (name patterns first, rest-pose geometry
    as the body-agnostic fallback — pal_talos's joints are literally called body07)
  * KeyData: cache -> per-embodiment key-joint SE3 arrays [N, nk, 9], clip bounds,
    the SAME clip-level train/val split rng(123) that temporal_common.StreamGroup
    uses, so every V5 number is comparable with P1's
  * the path integral: SE3 arc-length increments and the cumulative "distance
    travelled vs time" curve, differentiable (chordal rotation distance, not acos)
  * the masked spatial completion path: key joints -> frozen E19 -> full body on ANY
    target skeleton, which is the assumption v5_probe_keyjoints.py is there to falsify
  * motion-energy / jitter metrics, all computed on FK-from-ROTATION positions —
    the raw decoder position stream is never used (that bug cost a day, see p1_fkfix)

Coordinates: the caches are Y-up, centimetres, positions ROOT-RELATIVE (build_global
puts the root at the origin), and stored divided by `reach`. So by default "motion"
throughout V5 means ARTICULATED motion; global translation is not in this
representation at all and no default V5 number claims to measure it.

ROOT-VEL SIDE CHANNEL (--root_vel, 2026-08-06): the hole above is now closeable.
{cache}/{emb}_root_world.pt (built by _amass_work/extract_root_world.py) carries the
world root trajectory the cache builds dropped. KeyData(root_vel=True) appends a 7th
key "joint": identity rot6d + the root LINEAR VELOCITY expressed in the HEADING
frame (world velocity rotated by inverse root yaw -> x=lateral, y=vertical,
z=forward; heading CHANGES stay in the root rotation channel), normalized by LEG
LENGTH (sole-to-pelvis from the rest pose — Froude scaling, so speeds compare
across body sizes; units = leg-lengths/second). se3_step / arclen_curve /
pathint_loss / riemann_timing_loss integrate it with weight `lam_v` (|v|/fps per
step = exact world root path length at native rate), and motion_energy /
jitter_stats accept the world root to measure locomotion. Default off = bit-exact
previous behavior; zeroing the channel reproduces the root-relative numbers exactly.
Caveat: the channel stores the GT instantaneous velocity at the sampled frame, so
under warp/resample readouts its integral measures the ORIGINAL traversal, not the
warped one — validated uses are native-rate windows (losses vs GT, metrics).
"""
from __future__ import annotations
import os, sys, json, math, re, glob
import numpy as np
import torch
import torch.nn.functional as F

EXP = os.path.dirname(os.path.abspath(__file__))
if EXP not in sys.path:
    sys.path.insert(0, EXP)

from common import SkeletonSpec, rot6d_to_matrix, matrix_to_rot6d, geodesic_deg  # noqa: E402
from train_global import build_global                                            # noqa: E402

ROOT = "/var/tmp/alphamotion/topo_exp"
AMASS = os.environ.get("V5_AMASS", f"{ROOT}/cache_xwbc_amass")
LAFAN = f"{ROOT}/cache_xwbc_lafan"
# The frozen spatial layer. NOT used anywhere in temporal TRAINING (the key states
# come from the cache's local6d, pure geometry) — only at decode/eval/render time,
# to expand 6 key joints into a full body on the target skeleton. So swapping it
# changes every eval and every video, and changes no training.
FROZEN = os.environ.get("V5_FROZEN", f"{ROOT}/runs/E19_K128_0805")
STREAM_ROOT = f"{ROOT}/e13_stream"
KEYCACHE = f"{ROOT}/v5_keycache"
RUNS = f"{ROOT}/runs"
VERDICT = f"{RUNS}/V5_verdict_0806.txt"
FIGS = f"{RUNS}/V5_figs"
VIZ_OUT = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
           'R4_VIZ/mp4/v5')
GEN_ROOT = '/home/arenalabs/Desktop/"be water, robot"/GENMO/outputs/investor_demo'

FPS = 30.0
T_WIN = 60                                     # 2 s @ 30 fps, same window as P1
SPLIT_SEED = 123                               # temporal_common.StreamGroup's val split

# reference numbers from the spec / today's measurements, printed next to ours so a
# regression is visible without opening another file
REF = {
    "constraint5_pos_cm_g1": 0.80,     # 5-point SE3 constraint, single frame, G1 vs GT
    "path_speed_cos_current": 0.644,   # same path fast vs slow, current architecture
    "path_speed_cos_accept": 0.95,     # acceptance line
    "path_speed_cos_unrelated": 0.246, # incoherent-motion control
    "knn_body_leak_p1": 0.141,         # P1 temporal-token body identifiability
    "wrist_intrinsic_dim": 2.65,       # R8 per-joint intrinsic dim, wrists
    "leg_intrinsic_dim": 0.84,         # R8 per-joint intrinsic dim, legs
    "spectral_cover_run": 0.665,       # run: energy fraction inside the token bandwidth
}


# --------------------------------------------------------------------------- scratchpad
def scratchpad():
    """Where the validated MuJoCo render machinery lives (showcase_all / carnival /
    agibot / p1_verdict). It is session-scoped scratch, so find it rather than
    hard-coding one session id; $V5_SCRATCH overrides."""
    env = os.environ.get("V5_SCRATCH")
    if env and os.path.exists(f"{env}/showcase_all.py"):
        return env
    pats = ["/tmp/claude-1000/-home-arenalabs-Desktop--be-water--robot-/*/scratchpad",
            "/tmp/claude-*/*/scratchpad"]
    hits = []
    for p in pats:
        hits += [d for d in glob.glob(p) if os.path.exists(f"{d}/showcase_all.py")]
    if not hits:
        raise RuntimeError("no scratchpad with showcase_all.py found; set $V5_SCRATCH")
    return sorted(hits, key=os.path.getmtime)[-1]


# --------------------------------------------------------------------------- families
FAM_RE = [
    ("jacks", re.compile(r"jumping_?jack|jumping jack", re.I)),
    ("run",   re.compile(r"(?<![a-z])run|jog|sprint", re.I)),
    ("walk",  re.compile(r"walk|stroll|march", re.I)),
    ("dance", re.compile(r"danc|salsa|cha_?cha|tango|ballet|samba|zeibekiko|"
                         r"maleviziotikos", re.I)),
    ("kick",  re.compile(r"kick", re.I)),
    ("punch", re.compile(r"punch|hook|jab|backfist|uppercut|boxing", re.I)),
    ("jump",  re.compile(r"jump|hop|leap", re.I)),
    ("stand", re.compile(r"stand|still|idle|posing|t_?pose", re.I)),
]
# the 6 families the probes and the verdict table must cover (spec §1 / task)
V5_FAMILIES = ["walk", "run", "jacks", "dance", "kick", "punch"]


def family_of(name):
    for fam, rx in FAM_RE:
        if rx.search(name):
            return fam
    return "other"


# --------------------------------------------------------------------------- key joints
KEY_PATTERNS = [
    ("head",    [r"^head$", r"head", r"neck"]),
    ("l_wrist", [r"^l_wrist$", r"left_wrist", r"wrist.*left", r"left.*hand",
                 r"left_end_effector", r"l_?hand"]),
    ("r_wrist", [r"^r_wrist$", r"right_wrist", r"wrist.*right", r"right.*hand",
                 r"right_end_effector", r"r_?hand"]),
    ("l_ankle", [r"^l_ankle$", r"left_ankle", r"ankle.*left", r"left_foot",
                 r"foot.*left", r"anklepitchleft"]),
    ("r_ankle", [r"^r_ankle$", r"right_ankle", r"ankle.*right", r"right_foot",
                 r"foot.*right", r"anklepitchright"]),
]


def rest_positions(spec):
    """Rest-pose global joint positions [J,3] (cm, Y-up, root at origin) — identity
    rotations everywhere, i.e. just the offsets accumulated down the tree."""
    off = np.asarray(spec.rest_offsets, np.float64)
    p = np.zeros((spec.J, 3))
    for j in range(spec.J):
        k = int(spec.parents[j])
        if k >= 0:
            p[j] = p[k] + off[j]
    return p


def _geometric_key(spec):
    """Body-agnostic fallback: pick extremities off the REST pose. head = highest
    joint; ankles = lowest joint on each side; wrists = most lateral joint above the
    root. Works for pal_talos ('body00'...'body32') where names carry nothing."""
    p = rest_positions(spec)
    x, y = p[:, 0], p[:, 1]
    J = spec.J
    root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
    idx = {"head": int(np.argmax(y))}
    for tag, sgn in (("l_ankle", +1), ("r_ankle", -1)):
        cand = [j for j in range(J) if j != root and np.sign(x[j]) == sgn]
        cand = cand or [j for j in range(J) if j != root]
        idx[tag] = int(min(cand, key=lambda j: y[j]))
    for tag, sgn in (("l_wrist", +1), ("r_wrist", -1)):
        cand = [j for j in range(J) if j != root and np.sign(x[j]) == sgn
                and y[j] > y[root]]
        cand = cand or [j for j in range(J) if j != root and np.sign(x[j]) == sgn]
        cand = cand or [j for j in range(J) if j != root]
        idx[tag] = int(max(cand, key=lambda j: abs(x[j])))
    return idx


def key_joints(spec, verbose=False):
    """-> (idx [6] int list, names [6], tags [6]). Order is ALWAYS
    (root, head, l_wrist, r_wrist, l_ankle, r_ankle) so the 6x9 state vector means the
    same thing on every skeleton — that is what makes the cross-embodiment InfoNCE and
    the cross-embodiment token transfer legal."""
    names = [n.lower() for n in spec.joint_names]
    root = int(np.where(np.asarray(spec.parents) < 0)[0][0])
    geo = _geometric_key(spec)
    out, used = {"root": root}, {root}
    for tag, pats in KEY_PATTERNS:
        hit = None
        for pat in pats:
            cands = [j for j, n in enumerate(names)
                     if re.search(pat, n) and j not in used]
            if cands:
                # deepest match wins: 'left_wrist_yaw_link' over 'left_wrist_roll_link'
                hit = max(cands, key=lambda j: _depth(spec, j))
                break
        if hit is None:
            hit = geo[tag]
            if hit in used:                        # degenerate tiny skeleton
                hit = max(range(spec.J), key=lambda j: (j not in used, _depth(spec, j)))
        out[tag] = int(hit); used.add(int(hit))
    tags = ["root", "head", "l_wrist", "r_wrist", "l_ankle", "r_ankle"]
    idx = [out[t] for t in tags]
    if verbose:
        print(f"[key joints] {spec.name}: " +
              ", ".join(f"{t}={spec.joint_names[j]}" for t, j in zip(tags, idx)))
    return idx, [spec.joint_names[j] for j in idx], tags


_DEPTH = {}


def _depth(spec, j):
    key = (id(spec), j)
    if key not in _DEPTH:
        d, k = 0, int(spec.parents[j])
        while k >= 0:
            d += 1; k = int(spec.parents[k])
        _DEPTH[key] = d
    return _DEPTH[key]


PREFER_FIXED_SPEC = os.environ.get("V5_FIXED_SPEC", "1") not in ("0", "false", "no")
_FIXED_SEEN = set()


def load_spec(cache, emb):
    """Skeleton spec, preferring `{emb}_spec_fixed.npz` when one exists.

    runs/DATAFIX_talos_g123.md §4.1 recommended exactly this ("在加载处加'有
    _fixed 先用 _fixed'的逻辑") as the alternative to overwriting the caches.
    Only two embodiments have a _fixed spec:
      pal_talos          — joint_names body00..body32 replaced by the REAL link names
                           recovered from the source npz (validated: parents 33/33,
                           FK 0.00 mm on 23 bodies). This matters here because
                           key_joints() resolves head/wrists/ankles BY NAME first;
                           with body00-style names every key joint fell through to the
                           rest-pose geometric fallback.
      unitree_g1_23dof   — byte-identical content (only its *_dof_fixed.npz differs,
                           and no V5/V6/V7 path reads dof).
    V5_FIXED_SPEC=0 restores the previous (unfixed) behaviour bit for bit.
    NOTE unitree_h2 has the same body00.. disease and is NOT fixed (DATAFIX §4.3).
    """
    f = f"{cache}/{emb}_spec_fixed.npz"
    if PREFER_FIXED_SPEC and os.path.exists(f):
        if (cache, emb) not in _FIXED_SEEN:
            _FIXED_SEEN.add((cache, emb))
            print(f"[spec] {emb}: using {os.path.basename(f)} (DATAFIX_talos_g123)",
                  flush=True)
    else:
        f = f"{cache}/{emb}_spec.npz"
    d = np.load(f, allow_pickle=True)
    return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                        list(d["joint_names"]))


# --------------------------------------------------------------------------- root world
IDENT6 = np.array([1., 0., 0., 0., 1., 0.], np.float32)   # rot6d of the identity


def load_root_world(cache, emb):
    """World root trajectory [N,3] float32 cm, Y-up ardy basis, row-aligned with
    {emb}_paired_local6d.pt. Built by _amass_work/extract_root_world.py."""
    f = f"{cache}/{emb}_root_world.pt"
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"{f} missing — run _amass_work/extract_root_world.py for this cache")
    return torch.load(f)


def leg_length_cm(spec, root_world=None):
    """Sole-to-pelvis, cm — the Froude length scale for the root-vel channel.

    Preferred estimator: MEDIAN world root height over the cache (root_world
    [N,3] cm; the source data has the floor at y=0, so the median frame — mostly
    standing — reads sole-to-pelvis directly). Measured 2026-08-06, it agrees
    across the AMASS and LAFAN caches to ~5 cm per embodiment. The spec rest pose
    is only the fallback because it is NOT trustworthy everywhere: booster_t1's
    rest_offsets[0,1] is 168 cm (an MJCF default, ~2.5x the robot's real ~66 cm
    pelvis height) and the LAFAN human's identity-rotation rest pose does not
    extend the legs (leg reads 10.5 cm)."""
    if root_world is not None:
        return max(10.0, float(np.median(np.asarray(root_world)[:, 1])))
    h = float(np.asarray(spec.rest_offsets)[0, 1])
    lo = -float(rest_positions(spec)[:, 1].min())
    return max(10.0, min(h, lo) if lo > 10.0 else h)


def heading_yaw(R, fwd_col=2, fb_col=0):
    """Root yaw about +Y from global root rotations [...,3,3] -> radians.

    fwd_col = which BODY column is "forward". This is NOT the same on every
    embodiment: robots (Isaac x-forward, AX-conjugated) and the AMASS human read
    forward on column 2 (up on column 1), but the LAFAN human's pelvis convention
    puts forward on column 0 (up on column 2) — measured 2026-08-06 by aligning
    each column's horizontal projection with the horizontal root velocity over all
    >30 cm/s frames (AMASS human .69/-.03/-.39 by col2; LAFAN human .55 by col0).
    Use detect_fwd_col to measure it; hardcoding col2 silently breaks LAFAN.
    Where forward is near-vertical (flips, lying) fall back to fb_col."""
    f = R[..., :, fwd_col]
    h = torch.sqrt(f[..., 0] ** 2 + f[..., 2] ** 2)
    g = R[..., :, fb_col]
    use_g = (h < 1e-3).unsqueeze(-1)
    f = torch.where(use_g, g, f)
    return torch.atan2(f[..., 0], f[..., 2])


def detect_fwd_col(R, root_world, clips, fps=FPS, min_cm_s=30.0):
    """-> (fwd_col, fb_col, scores[3]). fwd_col = the body column whose horizontal
    direction best tracks the horizontal world root velocity over moving frames;
    the near-vertical column (|mean y| > 0.8) is excluded. fb_col = the remaining
    third column (used when forward points vertical)."""
    p = root_world.double()
    v = torch.zeros_like(p)
    for c in clips:
        s, e = c["start"], c["start"] + c["len"]
        if c["len"] > 1:
            v[s:e - 1] = (p[s + 1:e] - p[s:e - 1]) * fps
    v[:, 1] = 0.0
    sp = v.norm(dim=-1)
    m = sp > min_cm_s
    if m.sum() < 100:                              # nothing moves: keep convention
        return 2, 0, [0.0, 0.0, 0.0]
    scores, upness = [], []
    for col in range(3):
        f = R[..., :, col].clone(); f[..., 1] = 0.0
        f = f / (f.norm(dim=-1, keepdim=True) + 1e-9)
        scores.append(float(((f * v).sum(-1) / (sp + 1e-9))[m].mean()))
        upness.append(abs(float(R[..., 1, col].mean())))
    cand = [c for c in range(3) if upness[c] < 0.8] or [0, 1, 2]
    fwd = max(cand, key=lambda c: scores[c])
    up = max(range(3), key=lambda c: upness[c])
    fb = next(c for c in range(3) if c not in (fwd, up)) if up != fwd else (fwd + 1) % 3
    return fwd, fb, [round(s, 3) for s in scores]


def yaw_to_world(yaw):
    """R_y(yaw) [...,3,3]: heading frame -> world (columns: lateral, up, forward)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, z, s], -1), torch.stack([z, o, z], -1),
                        torch.stack([-s, z, c], -1)], -2)


def root_vel_channel(root_world, clips, root_rot6d, leg_cm, fps=FPS,
                     fwd_col=2, fb_col=0):
    """-> [N,9] float32: IDENT6 rot slots + heading-frame root linear velocity in
    LEG-LENGTHS/SECOND (x=lateral, y=vertical, z=forward).

    Velocity is the clip-aware forward difference v_t = (p[t+1]-p[t])*fps (last
    frame of each clip repeats the previous), so sum(|v|)/fps over a native-rate
    window is EXACTLY the world root path length. Expressing it in the heading
    frame removes yaw (which the root rotation channel already carries), making
    forward/lateral/vertical egocentric; dividing by leg length is the Froude
    normalization that makes speeds comparable across body sizes.
    fwd_col/fb_col: which body columns are forward/fallback — embodiment-dependent,
    see detect_fwd_col/heading_yaw."""
    p = root_world.double()
    v = torch.zeros_like(p)
    for c in clips:
        s, e = c["start"], c["start"] + c["len"]
        if c["len"] > 1:
            v[s:e - 1] = (p[s + 1:e] - p[s:e - 1]) * fps
            v[e - 1] = v[e - 2]
    yaw = heading_yaw(rot6d_to_matrix(root_rot6d.double()), fwd_col, fb_col)
    vh = torch.einsum("nab,na->nb", yaw_to_world(yaw), v)   # R^T v = einsum ab,a->b
    vh = (vh / float(leg_cm)).float()
    out = torch.zeros(root_world.shape[0], 9, dtype=torch.float32)
    out[:, :6] = torch.from_numpy(IDENT6)
    out[:, 6:9] = vh
    return out


# --------------------------------------------------------------------------- data
class KeyData:
    """Key-joint SE3 states for one paired cache, one or more embodiments.

    All embodiments of a paired cache share the row index, so window (clip, offset)
    pairs are the SAME motion on different bodies — that is the InfoNCE positive and
    the cross-embodiment leakage probe's input.

    key[emb] is float32 [N, 6, 9]: global rot6d (6) + root-relative position / reach
    (3). The root's position slots are identically 0 by construction (build_global
    puts the root at the origin); they are kept so the state layout is uniform.

    root_vel=True (--root_vel) appends a 7th channel -> [N, 7, 9]: IDENT6 rot slots
    + heading-frame root linear velocity in leg-lengths/s (see root_vel_channel).
    Also exposes root_world[emb] [N,3] cm and leg_cm[emb]. Default False keeps every
    existing number bit-exact.
    """

    def __init__(self, cache=AMASS, embs=("human_smpl",), device="cpu",
                 max_clips=0, verbose=True, root_vel=False):
        self.cache = cache
        self.embs = list(embs)
        self.root_vel = bool(root_vel)
        seqs = json.load(open(f"{cache}/sequences.json"))
        if max_clips:
            seqs = seqs[:max_clips]
        self.clips = [{"name": s["name"], "start": int(s["start"]), "len": int(s["len"])}
                      for s in seqs]
        self.family = [family_of(c["name"]) for c in self.clips]
        self.key, self.spec, self.kidx, self.knames, self.reach = {}, {}, {}, {}, {}
        self.root_world, self.leg_cm, self.fwd_col = {}, {}, {}
        for e in self.embs:
            self._load_emb(e, device, verbose)
        C = len(self.clips)
        rng = np.random.default_rng(SPLIT_SEED)
        self.val_clips = set(rng.permutation(C)[: max(1, C // 10)].tolist())
        if verbose:
            print(f"[KeyData] {os.path.basename(cache)} clips={C} "
                  f"val={len(self.val_clips)} embs={self.embs}", flush=True)

    def _load_emb(self, emb, device, verbose):
        spec = load_spec(self.cache, emb)
        idx, knames, _ = key_joints(spec, verbose=verbose)
        tag = os.path.basename(self.cache)
        cf = f"{KEYCACHE}/{tag}_{emb}.npz"
        key = None
        if os.path.exists(cf):
            z = np.load(cf)
            if list(z["kidx"]) == idx:
                key = torch.from_numpy(z["key"])
                self.reach[emb] = float(z["reach"])
        if key is None:
            l6 = torch.load(f"{self.cache}/{emb}_paired_local6d.pt")
            gr6, gp, reach = build_global(l6, spec, device)
            key = torch.cat([gr6[:, idx], gp[:, idx] / reach], -1).float().contiguous()
            os.makedirs(KEYCACHE, exist_ok=True)
            np.savez(cf, key=key.numpy(), kidx=np.asarray(idx), reach=reach)
            self.reach[emb] = float(reach)
            if verbose:
                print(f"  [{emb}] rows={key.shape[0]} reach={reach:.1f}cm -> {cf}",
                      flush=True)
        if self.root_vel:
            rw = load_root_world(self.cache, emb)
            assert rw.shape[0] == key.shape[0], (emb, rw.shape, key.shape)
            # under max_clips only the listed clip spans get velocities; rows past
            # the prefix stay zero and are never sampled — same rule as `key` rows
            leg = leg_length_cm(spec, root_world=rw)
            R0 = rot6d_to_matrix(key[:, 0, :6].double())
            fwd, fb, scores = detect_fwd_col(R0, rw, self.clips)
            vel = root_vel_channel(rw, self.clips, key[:, 0, :6], leg,
                                   fwd_col=fwd, fb_col=fb)
            key = torch.cat([key, vel[:, None, :]], 1).contiguous()
            self.root_world[emb] = rw; self.leg_cm[emb] = leg; self.fwd_col[emb] = fwd
            if verbose:
                print(f"  [{emb}] +root_vel channel (leg={leg:.1f}cm, "
                      f"fwd_col={fwd} scores={scores})", flush=True)
        self.key[emb] = key
        self.spec[emb] = spec; self.kidx[emb] = idx; self.knames[emb] = knames

    # -------------------------------------------------------------- window sampling
    def pools(self, split, span):
        out = []
        for c, cl in enumerate(self.clips):
            if cl["len"] < span:
                continue
            if (c in self.val_clips) == (split == "val"):
                out.append((c, cl["start"], cl["start"] + cl["len"]))
        assert out, f"no {split} clip >= {span} frames"
        return out

    def sample_starts(self, n, g, split="train", span=T_WIN, families=None):
        """-> (starts int64 [n], clip ids [n])."""
        pools = self.pools(split, span)
        if families:
            f = [p for p in pools if self.family[p[0]] in families]
            pools = f or pools
        pick = g.integers(0, len(pools), n)
        st, ci = [], []
        for i in pick:
            c, s, e = pools[i]
            st.append(s + int(g.integers(0, e - s - span + 1))); ci.append(c)
        return np.asarray(st, np.int64), np.asarray(ci, np.int64)

    def window_idx(self, emb, starts, idx):
        """starts [n] + frame offsets idx [T] -> float32 [n,T,nk,9]."""
        pos = np.asarray(starts)[:, None] + np.asarray(idx, np.int64)[None]
        return self.key[emb][torch.from_numpy(pos.reshape(-1))].reshape(
            len(starts), len(idx), -1, 9).clone()

    def window(self, emb, starts, T_out, span=None):
        """The SAME raw span read out at T_out frames -> float32 [n,T_out,nk,9].

        span fixed + T_out varying is the ONLY honest "same path, different speed"
        view: the trajectory traced is bit-identical and only the number of frames
        spent traversing it changes. P1 instead varied the raw span, which changes
        the path as well — one reason its path/speed entanglement measured 0.644.
        """
        return self.window_idx(emb, starts, uniform_idx(span or T_out, T_out))

    def window_fidx(self, emb, starts, fidx, span):
        """FIX A (V5b): FRACTIONAL-time readout -> float32 [n,T,nk,9].

        Linear interp on positions, linear interp + re-orthonormalization on the 6D
        rotations. Valid because inter-frame rotations are small: measured
        2026-08-06 on 2000 AMASS train windows, p50 = 1.1 deg, p99 = 17.6 deg,
        p99.9 = 33 deg per frame @30 fps (lerp+GS tracks slerp closely well past
        that); the >90 deg tail is <0.1% of frames, where nearest-neighbor frame
        duplication was strictly worse anyway.
        """
        fidx = np.asarray(fidx, np.float64)
        i0 = np.floor(fidx).astype(np.int64)
        i1 = np.minimum(i0 + 1, span - 1)
        a = torch.from_numpy(fidx - i0).float()[None, :, None, None]
        st = np.asarray(starts)[:, None]
        k0 = self.key[emb][torch.from_numpy((st + i0[None]).reshape(-1))]
        k1 = self.key[emb][torch.from_numpy((st + i1[None]).reshape(-1))]
        n, T = len(starts), len(fidx)
        k = (1 - a) * k0.reshape(n, T, -1, 9) + a * k1.reshape(n, T, -1, 9)
        return orthonormalize6d(k)

    def rows_for(self, starts, T_out, span=None):
        return (np.asarray(starts)[:, None] +
                uniform_idx(span or T_out, T_out)[None])


def uniform_idx(span, T_out):
    """Even (nearest-frame) readout of a `span`-frame path at T_out samples."""
    if span == T_out:
        return np.arange(T_out, dtype=np.int64)
    return np.round(np.linspace(0, span - 1, T_out)).astype(np.int64)


def orthonormalize6d(s):
    """Project the rotation half of a key-joint state onto SO(3), re-emit as 6D."""
    return torch.cat([matrix_to_rot6d(rot6d_to_matrix(s[..., :6])), s[..., 6:9]], -1)


def uniform_fidx(span, T_out):
    """FIX A (V5b): CONTINUOUS fractional-time uniform readout — no rounding.

    uniform_idx's np.round makes the resample nearest-neighbor: stretching a path
    DUPLICATES frames, so the velocity profile becomes a staircase (0-jump-0-jump)
    and the "same path at a different speed" training pairs genuinely differ in
    content. Path-token invariance across such pairs is then only achievable by
    discarding content — the measured V5 content-share collapse 0.91 -> 0.14.
    Feed this to KeyData.window_fidx, which interpolates.
    """
    return np.linspace(0.0, span - 1, T_out)


def warp_fidx(span, T_out, rng, n_ctrl=3, strength=1.0):
    """FIX A (V5b): fractional-time version of warp_idx — same monotone warp law,
    but returning CONTINUOUS indices so window_fidx can interpolate instead of
    duplicating frames."""
    u = np.sort(rng.random(n_ctrl))
    x = np.concatenate([[0.0], u, [1.0]])
    y = np.sort(rng.random(n_ctrl))
    y = np.concatenate([[0.0], (1 - strength) * u + strength * y, [1.0]])
    t = np.linspace(0.0, 1.0, T_out)
    return np.clip(np.interp(t, x, y) * (span - 1), 0.0, span - 1)


def warp_idx(span, T_out, rng, n_ctrl=3, strength=1.0):
    """NON-UNIFORM monotone time warp of the same `span`-frame path -> idx [T_out].

    A uniform resample is the easy half of the speed problem. The hard half — and
    the exact shape of lloyd's acceptance recipe ("resample the first 30 frames of a
    60-frame window into 15 and append the rest") — is a warp that runs fast through
    part of the path and slow through the rest. The PATH is still bit-identical; only
    the timing law changes. Training the path token to be invariant to THIS is what
    makes the invariance non-trivial rather than an artefact of the architecture.
    """
    u = np.sort(rng.random(n_ctrl))
    x = np.concatenate([[0.0], u, [1.0]])
    y = np.sort(rng.random(n_ctrl))
    y = np.concatenate([[0.0], (1 - strength) * u + strength * y, [1.0]])
    t = np.linspace(0.0, 1.0, T_out)
    return np.clip(np.round(np.interp(t, x, y) * (span - 1)), 0, span - 1).astype(np.int64)


# --------------------------------------------------------------------------- path integral
NK_VEL = 7        # a 7-channel key state carries the root-vel side channel last


def se3_step(key, lam=0.5, lam_v=1.0, fps=FPS):
    """Per-frame SE3 geodesic increments of a key-joint trajectory.

    key [...,T,nk,9] -> [...,T-1]. Per joint the step is
        ds_j = sqrt( ||dp||^2 + (lam * ||dR||_F)^2 )
    with ||dR||_F the CHORDAL rotation distance (= 2*sqrt(2)*sin(theta/2)), summed
    over joints. Chordal rather than acos-geodesic on purpose: it is smooth and
    finite-gradient everywhere, and it is monotone in the true angle, so the
    arc-length CURVE (which is all the loss looks at) is unchanged in shape.
    Positions are in reach units, so lam=0.5 makes one radian worth half a reach.

    ROOT-VEL: when key has NK_VEL(7) channels (KeyData root_vel=True), the last
    channel is the heading-frame root velocity in leg-lengths/s, NOT a joint. It is
    excluded from the dp/dR sums and contributes lam_v * |v_t| / fps per step —
    at native rate that is exactly the world root path length in leg-lengths, so
    locomotion (invisible in the root-relative channels) enters the integral.
    A zeroed channel reproduces the 6-channel numbers exactly.
    """
    vel = None
    if key.shape[-2] == NK_VEL:
        vel = key[..., :-1, -1, 6:9]                # v_t, frames 0..T-2 (fwd diff)
        key = key[..., :-1, :]                      # drop the vel channel (joint dim)
    R = rot6d_to_matrix(key[..., :6])
    dR = (R[..., 1:, :, :, :] - R[..., :-1, :, :, :]).flatten(-2).norm(dim=-1)
    dp = (key[..., 1:, :, 6:9] - key[..., :-1, :, 6:9]).norm(dim=-1)
    ds = torch.sqrt(dp ** 2 + (lam * dR) ** 2 + 1e-12).sum(-1)
    if vel is not None:
        ds = ds + (lam_v / fps) * vel.norm(dim=-1)
    return ds


def arclen_curve(key, lam=0.5, lam_v=1.0):
    """Cumulative "distance travelled vs time" curve [...,T], starting at 0.
    Endpoint = total path length (走对路); shape = the timing law (速度对)."""
    ds = se3_step(key, lam, lam_v)
    return torch.cat([torch.zeros_like(ds[..., :1]), ds.cumsum(-1)], -1)


def pathint_loss(pred, gt, lam=0.5, w_end=1.0, beta=0.25, lam_v=1.0):
    """MAIN LOSS: cumulative arc-length curve matching (ARCH_V5 §4).

    Both curves are divided by the GT endpoint (scale-free), then compared with a
    SMOOTH-L1 rather than a plain MSE: an untrained decoder emits near-random
    rotations whose arc length is orders of magnitude above GT, and a quadratic there
    produces a four-digit loss that swamps every auxiliary for the first thousand
    steps. Huber keeps the gradient bounded without changing the optimum.

    Endpoint term = path length (went the right way); curve shape = timing law (at the
    right speed); the shortest-path property between fixed endpoints is what makes
    smoothness fall out instead of being regularized in.

    Returns (loss, endpoint_relative_error) — the second is a diagnostic, not a loss.
    Read it against static_pathint_anchor(gt): that is the score of a prediction that
    does not move at all, so anything at or above it means the model is frozen.
    """
    cg = arclen_curve(gt, lam, lam_v)
    cp = arclen_curve(pred, lam, lam_v)
    end = cg[..., -1:].clamp_min(1e-3)
    a, b = cp / end, cg / end
    shape = F.smooth_l1_loss(a, b, beta=beta)
    epl = F.smooth_l1_loss(a[..., -1], b[..., -1], beta=beta)
    ep = ((cp[..., -1] - cg[..., -1]).abs() / end[..., 0]).mean()
    return shape + w_end * epl, ep


def riemann_timing_loss(pred, gt, lam=0.5, deadzone=0.02, beta=0.1, w_end=0.1,
                        lam_v=1.0):
    """FIX B (V5b, lloyd's design): per-SEGMENT increment matching.

    pathint_loss matches the NORMALIZED CUMULATIVE curve, so timing errors CANCEL:
    a fast first half plus a slow second half lands on the same curve endpoint and
    much of the same shape. Here the per-frame SE3 geodesic steps ds_t are matched
    directly, Riemann-sum style, with a DEADZONE ("整个过程也要最小，但不要太严格，
    允许出一点错"): |ds_pred - ds_gt| below the GT increment noise floor costs
    nothing. deadzone default 0.02 reach units = the measured p10 of GT |ds|
    (2026-08-06, 2000 AMASS train windows: p5 0.009 / p10 0.020 / p50 0.174 /
    mean 0.218). Huber (smooth-L1, small beta) above the deadzone keeps
    untrained-decoder blowups from swamping the auxiliaries, same argument as
    pathint_loss. The TOTAL path length (cumulative endpoint) is kept as a
    separate LOW-weight relative term so the overall budget still has to be right
    without re-opening the cancellation hole.

    Returns (loss, endpoint_relative_error) like pathint_loss; the second is a
    diagnostic only.
    """
    dsp, dsg = se3_step(pred, lam, lam_v), se3_step(gt, lam, lam_v)
    mag = F.relu((dsp - dsg).abs() - deadzone)
    seg = F.smooth_l1_loss(mag, torch.zeros_like(mag), beta=beta)
    end_g = dsg.sum(-1).clamp_min(1e-3)
    epl = F.smooth_l1_loss(dsp.sum(-1) / end_g, torch.ones_like(end_g), beta=0.25)
    ep = ((dsp.sum(-1) - dsg.sum(-1)).abs() / end_g).mean()
    return seg + w_end * epl, ep


def static_pathint_anchor(gt, lam=0.5, **kw):
    """The score pathint_loss gives a prediction that never moves — computed exactly
    by feeding it a window frozen at frame 0, so it can never drift from the loss."""
    return float(pathint_loss(gt[..., :1, :, :].expand_as(gt), gt, lam, **kw)[0])


def key_se3_err(pred, gt, reach=1.0):
    """-> (pos cm, rot deg) mean over frames/joints. Diagnostic, not a loss.
    A root-vel 7th channel is excluded (its slots are leg-lengths/s, not reach units)."""
    if gt.shape[-2] == NK_VEL:
        pred, gt = pred[..., :6, :], gt[..., :6, :]
    p = (pred[..., 6:9] - gt[..., 6:9]).norm(dim=-1).mean() * reach
    r = geodesic_deg(rot6d_to_matrix(pred[..., :6]), rot6d_to_matrix(gt[..., :6])).mean()
    return float(p), float(r)


# --------------------------------------------------------------------------- motion metrics
def fk_pos(rot6d, spec):
    """global rot6d [T,J,6] -> RIGID-BONE positions [T,J,3] cm.

    The raw decoder position stream is never used anywhere in V5 (p1_fkfix: rendering
    it gave rubber bones and ~12 cm error). Every position in every V5 number and
    every V5 frame comes through here.
    """
    R = rot6d_to_matrix(torch.as_tensor(np.asarray(rot6d), dtype=torch.float64)).numpy()
    par = np.asarray(spec.parents); off = np.asarray(spec.rest_offsets, np.float64)
    p = np.zeros((R.shape[0], len(par), 3))
    for j in range(len(par)):
        k = int(par[j])
        if k >= 0:
            p[:, j] = p[:, k] + np.einsum("tab,b->ta", R[:, k], off[j])
    return p


def motion_energy(p, root_world=None):
    """[T,J,3] cm -> mean over joints of the per-joint temporal std (cm).
    Identical definition to p1_motion_metrics so P1 and V5 numbers are comparable.

    root_world [T,3] cm (--root_vel side channel): measure in the WORLD frame, so
    locomotion counts. Default None = root-relative = bit-exact previous number."""
    p = np.asarray(p)
    if root_world is not None:
        p = p + np.asarray(root_world)[:, None, :]
    d = p - p.mean(0, keepdims=True)
    return float(np.sqrt((d ** 2).sum(-1).mean(0)).mean())


def jitter_stats(p, fps=FPS, root_world=None):
    """Frame-to-frame jump distribution of a position trajectory [T,J,3] cm.

    The per-frame spatial completion has NO temporal prior, so jitter is the expected
    failure mode and has to be quantified, not asserted away:
      step_*    : ||p_t - p_{t-1}|| per joint, cm  (median / p95 / max)
      accel     : ||p_{t+1} - 2p_t + p_{t-1}|| mean, cm  — the sharp one; smooth motion
                  has small acceleration even when it is fast, jitter does not
      rev_frac  : fraction of steps that REVERSE direction (cos < 0 with the previous
                  step). Real motion is locally smooth (~0.1-0.2); white jitter -> 0.5

    root_world [T,3] cm: evaluate in the world frame (locomotion included);
    default None keeps the root-relative numbers bit-exact.
    """
    p = np.asarray(p)
    if root_world is not None:
        p = p + np.asarray(root_world)[:, None, :]
    d = np.linalg.norm(np.diff(p, axis=0), axis=-1)                 # [T-1,J]
    a = np.linalg.norm(np.diff(p, n=2, axis=0), axis=-1)            # [T-2,J]
    v = np.diff(p, axis=0)
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9
    u = v / n
    cos = (u[1:] * u[:-1]).sum(-1)
    return {"step_med_cm": float(np.median(d)), "step_p95_cm": float(np.quantile(d, .95)),
            "step_max_cm": float(d.max()), "accel_mean_cm": float(a.mean()),
            "accel_p95_cm": float(np.quantile(a, .95)),
            "rev_frac": float((cos < 0).mean())}


# --------------------------------------------------------------------------- frozen layer
def load_frozen(device, run=FROZEN):
    from tokenize_stream import load_tokenizer
    tok, cfg = load_tokenizer(run, device)
    return tok, cfg


def frozen_sem_encoder():
    """The text tower the FROZEN spatial run was trained with. SigLIP is 768-wide,
    Qwen3 is 1024 — feeding the wrong one is a shape crash at best, silent garbage
    conditioning at worst (0809: V5_FROZEN can now point at a Qwen-based run)."""
    try:
        import json as _json
        return _json.load(open(f"{FROZEN}/config.json")).get("sem_encoder", "siglip")
    except Exception:
        return "siglip"


def body_entry(cache, emb, device, reach=None, encoder=None):
    """Geometry + semantics conditioning for one body, WITHOUT loading its pose rows.
    (tokenize_stream.load_body FKs the whole cache just to get `reach`; when the reach
    is already known that is 30 s of pure waste per call.)"""
    from train_ae import static_cond
    from semantics import build_name_embeddings, embed_matrix
    spec = load_spec(cache, emb)
    lut = build_name_embeddings(list(spec.joint_names), device=device,
                                encoder=encoder or frozen_sem_encoder())
    e = {"spec": spec, "st": static_cond(spec, device),
         "sem": embed_matrix(spec.joint_names, lut).to(device)}
    dof_path = f"{cache}/{emb}_dof.npz"
    if os.path.exists(dof_path):
        e["dof"] = torch.as_tensor(np.load(dof_path)["dof"], dtype=torch.float32,
                                   device=device)
    if reach is not None:
        e["reach"] = float(reach)
    return e


def geom_for(entry, B):
    st = entry["st"]
    g = dict(rest_off=st["rest_off"][None].expand(B, -1, -1),
             bonelen=st["bonelen"][None].expand(B, -1),
             depth=st["depth"][None].expand(B, -1),
             hop=st["hop"][None].expand(B, -1, -1),
             sem=entry["sem"][None].expand(B, -1, -1))
    if "dof" in entry:
        g["dof"] = entry["dof"][None].expand(B, -1, -1)
    return g


@torch.no_grad()
def complete_from_key(tok, src_entry, tgt_entry, key_state, kidx, device, bs=256):
    """THE CORE ASSUMPTION, as code.

    key_state [n,nk,9] on the SOURCE skeleton -> mask every joint except `kidx` ->
    frozen E19 encoder -> decode on the TARGET skeleton -> (rot6d [n,J_t,6],
    pos [n,J_t,3] in reach units).

    Masked joints feed the encoder a learned mask token, so whatever sits in their
    slots of `x` is irrelevant (model_dual._trunk); zeros are used.
    A root-vel 7th channel (KeyData root_vel=True) is dropped here: the frozen
    spatial layer knows nothing about world velocity.
    """
    if key_state.shape[-2] == len(kidx) + 1:
        key_state = key_state[..., :len(kidx), :]
    J = src_entry["spec"].J
    ki = torch.as_tensor(kidx, dtype=torch.long)
    outs_r, outs_p = [], []
    for i in range(0, key_state.shape[0], bs):
        k = key_state[i:i + bs].to(device).float()
        b = k.shape[0]
        x = torch.zeros(b, J, 9, device=device)
        x[:, ki] = k
        m = torch.ones(b, J, dtype=torch.bool, device=device)
        m[:, ki] = False
        zp, zr = tok.encode(x, geom_for(src_entry, b), joint_mask=m)
        rot, pos = tok.decode(zp, zr, geom_for(tgt_entry, b))
        outs_r.append(rot.cpu()); outs_p.append(pos.cpu())
    return torch.cat(outs_r), torch.cat(outs_p)


@torch.no_grad()
def full_pose_of(cache, emb, rows, device, reach=None):
    """GT full-body pose [n,J,9] for cache rows (rot6d + pos/reach)."""
    spec = load_spec(cache, emb)
    l6 = torch.load(f"{cache}/{emb}_paired_local6d.pt")
    l6 = l6[torch.as_tensor(np.asarray(rows).reshape(-1), dtype=torch.long)]
    gr6, gp, r = build_global(l6, spec, device)
    r = r if reach is None else float(reach)
    return torch.cat([gr6, gp / r], -1), r


# --------------------------------------------------------------------------- misc
def add_root_vel_flag(ap):
    """The one switch for the locomotion side channel. Default OFF = bit-exact
    pre-root_vel behavior (6 key channels, root-relative everything)."""
    ap.add_argument("--root_vel", action="store_true",
                    help="append the heading-frame root-velocity channel "
                         "(leg-lengths/s) to KeyData and the path integral")
    return ap


def bit_budget(n_path, n_time, levels, T=T_WIN, fps=FPS):
    """Exact bit budget. P1's reference: 8 content registers + 1 duration seat of the
    same 6x5 FSQ = 125.38 bit / 2 s window = 62.7 bit/s. V5 must be printed next to
    it every run so the arms are never compared at different rates by accident."""
    per = float(sum(math.log2(int(l)) for l in levels))
    win = T / fps
    return {"bits_per_register": round(per, 3), "vocab_per_register": int(np.prod(levels)),
            "n_path": n_path, "n_time": n_time,
            "path_bits_per_window": round(per * n_path, 2),
            "time_bits_per_window": round(per * n_time, 2),
            "total_bits_per_window": round(per * (n_path + n_time), 2),
            "window_seconds": round(win, 3),
            "total_bits_per_s": round(per * (n_path + n_time) / win, 2),
            "p1_reference_bits_per_s": 62.69,
            "temporal_compression_x": round(T / max(n_path, 1), 2)}


def prefix_budgets(n_path):
    b, out = 1, []
    while b <= n_path:
        out.append(b); b *= 2
    if out[-1] != n_path:
        out.append(n_path)
    return tuple(out)


def fig_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb", "text.color": "#0b0b0b",
        "axes.labelcolor": "#52514e", "xtick.color": "#898781", "ytick.color": "#898781",
        "axes.edgecolor": "#c3c2b7", "axes.grid": True, "grid.color": "#e1e0d9",
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "figure.dpi": 140,
        "font.family": "sans-serif"})
    return plt


SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]


def loud(lines, mark="!!!", width=100):
    """Print a falsification banner that survives a 2000-line log."""
    import textwrap
    wrapped = []
    for l in lines:
        wrapped += textwrap.wrap(l, width - len(mark) - 1) or [""]
    bar = (mark * (width // len(mark) + 1))[:width]
    print("\n" + bar, flush=True)
    for l in wrapped:
        print(f"{mark} {l}", flush=True)
    print(bar + "\n", flush=True)
