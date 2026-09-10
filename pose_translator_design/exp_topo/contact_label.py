"""Greenwich-v2 contact-label data layer.

For BOTH xwbc caches (lafan, amass_full) x ALL 18 embodiments (human_smpl + 17
robots), label every frame x every joint with

    gap          [N,J] float16   world height above the embodiment's floor (m)
    contact_soft [N,J] float16   sigmoid((thr-gap)/tau) * sigmoid((v_thr-v)/tau_v)
    contact      [N,J] bool      after hysteresis + 0.16 s despeckle

World reconstruction (verified): fk_global(local6d, spec) -> root-relative cm,
world = (gp + root_world[:,None,:]) / 100  -> metres, ardy basis, **Y is up**.

FLOOR, per embodiment (lloyd 0829):
  T-pose = identity local rotations FK'd through rest_offsets.  Standing that
  T-pose on the ground puts its LOWEST joint on the floor, so the T-pose floor
  reference is h0_tpose = 0 and the T-pose contributes
     j_low : which joint defines the floor
     D     : root -> j_low depth = the embodiment's own length scale
  The joint CENTRE is not the sole, so the operating floor is measured from the
  data as h0 = median over clips of (per-clip p5 of the per-frame lowest joint).
  h0 - h0_tpose = h0 is reported per corpus per embodiment: that IS the gauge
  discrepancy (sole clearance + retarget offset).  Nothing is shifted silently.

Thresholds derive from the embodiment's own scale, no hand-tuned constants:
  thr = FRAC * D  (FRAC = 0.04), tau = thr/3, v_thr = 0.10 m/s, tau_v = v_thr/3
Hysteresis + min-segment machinery mirrors amctrl/horizon/pack.py::_foot_schedule.

GAUGE HAZARD: the two caches ship DIFFERENT human_smpl skeletons under the same
name (rest offsets differ by 46.74 cm; LAFAN's is in a bone-local +Z rest
convention, so its identity-FK T-pose is NOT a standing pose).  Every corpus is
loaded and labelled in its own pass with its own spec; nothing is cached by name.
"""
from __future__ import annotations
import os, sys, json, time, argparse, re
import numpy as np
import torch
from scipy.special import expit

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
from common import SkeletonSpec, fk_global

ROOT = "/var/tmp/alphamotion/topo_exp"
CACHES = {"lafan": f"{ROOT}/cache_xwbc_lafan", "amass": f"{ROOT}/cache_xwbc_amass_full"}
FPS = 30.0
FRAC_THR = 0.05          # contact margin as a fraction of D (the T-pose leg length)
V_THR = 0.10             # m/s, horizontal
MIN_SEG_S = 0.16         # despeckle, matches pack.py FOOT_MIN_SEG (8 frames @50fps)
SITE_FRAC = 0.15         # joints closer than this fraction of D are ONE contact site
CLIP_FLOOR_PCT = 15.0    # per-clip floor = this percentile of the per-frame lowest joint
                         # (15 mirrors pack.py's z_on = percentile(z,15); a sweep over
                         #  5/10/15/25 x 0.04/0.05 is in CONTACT_LABEL_DATA.md)
UP = 1                   # ardy Y-up

# ---- 脚底点 (lloyd 0831) --------------------------------------------------
# 17 台机器人里 10 台的骨架在脚踝就结束, 底下没有脚/趾连杆, 于是"最低关节高度"量的是
# 踝轴中心, 离真实脚底还差 3.5-16.8cm (阈值只有 2.5-4.9cm)。更糟的是脚底最低点在踝的
# 前后方数厘米处, 这条水平力臂把**脚的俯仰角**直接翻译成了接触判定 —— 常数偏移修不掉。
# 解法: 从 MJCF 名字无关地量出"静止姿态下全局最低的网格顶点"在其所属 body 局部系里的
# 坐标, FK 时按该关节的全局旋转带出来:  sole = joint + R_global @ off_local
# 缓存里本来就有每关节的全局旋转 (fk_global 的第一个返回值, 原先被丢掉)。
SOLE_JSON = "/home/arenalabs/tmp/sole_offsets3.json"
SOLE_ALIAS = {"unitree_g1_29dof": "unitree_g1", "unitree_g1_23dof": "unitree_g1"}
USE_SOLE = os.environ.get("AM_SOLE", "1") != "0"     # 对照臂: AM_SOLE=0 退回关节口径

# ---- session floor correction (lloyd 0831) ------------------------------
# A capture session (one subject in one sub-dataset) shares one floor calibration.
# Measured on AMASS/g1_29dof: 288 sessions, 90% inside a 2.11 cm band, median
# within-session sigma 0.51 cm -- a session's floor is a stable constant, while a
# few sessions sit 3-7 cm high (KIT__950: 25 clips at 4.83 cm, sigma 0.41).  Those
# sessions label as "never touches the ground" while the subject is standing still.
# Grouping by session (not by clip) keeps the estimate non-circular: a per-clip
# percentile forces contact into every clip by construction, a per-session one is
# measured across dozens of different motions.
# The floor is estimated from the POOLED frames of a session (p2 of the lowest
# joint), not from a median of per-clip percentiles: a per-clip percentile tracks
# what the motion happens to do, so on LAFAN -- 5 subjects x 15 wildly different
# activities -- it flagged 3 of 5 sessions as incoherent, which is nonsense.  The
# pooled p2 is the level the body actually keeps returning to.
# The test is ONE-SIDED.  A session sitting BELOW h0 is penetration, which is a
# property of the source data (LAFAN reaches -9 cm) and not a floor error -- h0 is
# already the corpus median.  Only a floor sitting HIGH manufactures false
# "never touches the ground", so only that is corrected.
SESS_MIN_FRAMES = 2000   # fewer pooled frames than this -> no estimate, leave as is
SESS_OFF_FRAC   = 0.50   # floor above this fraction of thr = suspect
SESS_WALL_FRAC  = 0.50   # (p15-p2) below this fraction of thr = a real wall, so shift;
                         # otherwise it is high with no wall -> untrusted, no supervision


# ----------------------------------------------------------------- families
FAMILY_RE = [
    ("mixed",  r"multiple_?action"),
    ("obstacle", r"obstacle|hurdle|step_?over|stair"),
    ("crawl",  r"crawl|prone|on_?all_?fours|allfours|knee_?walk"),
    ("ground", r"lie|lying|lay_?down|laydown|floor|ground|roll_?over|rolling|situp|sit_?up|pushup|push_?up|yoga|supine"),
    ("fall",   r"fall|trip|stumble|collaps|faint|slip|knock"),
    ("getup",  r"get_?up|getup|stand_?up|standup|rise|recover"),
    ("push",   r"push|shove|stumble"),
    ("fight",  r"kick|punch|box|fight|martial|karate|combat|strike|kung|sword|fenc"),
    ("dance",  r"danc|ballet|waltz|tango|salsa|hip.?hop|charleston|capoeira|breakdance"),
    ("run",    r"run|sprint|jog|dash"),
    ("jump",   r"jump|hop|leap|skip|bound|vault"),
    ("sit",    r"sit|kneel|crouch|squat|chair"),
    ("manip",  r"pick.?up|lift|throw|catch|carry|hold|grab|handle|place|put.?down|drink|eat|box_?lift"),
    ("gesture",r"wave|point|clap|nod|bow|shake|scratch|salute|greet|talk|conversation"),
    ("stretch",r"stretch|bend|rotat|twist|reach|yawn|arch|tilt|sway|swing|balance"),
    ("sport",  r"golf|tennis|basketball|soccer|football|baseball|climb|swim|skate|ski|surf|row|bowl|split|aiming|shoot"),
    ("walk",   r"walk|stroll|pace|march|step|stride|amble|wander|cross|locomot"),
    ("turn",   r"turn|circle|zig.?zag|navigate|move"),
    ("stand",  r"stand|idle|rest|wait|neutral|tpose|t_?pose"),
]
FAMILY_RE = [(k, re.compile(p, re.I)) for k, p in FAMILY_RE]


def session(clip_name: str) -> str:
    """Capture session = the unit that shares one floor calibration.
    AMASS  'KIT__950__walk01_stageii'  -> 'KIT__950'
    LAFAN  'aiming1_subject1'          -> 'subject1'
    Anything else falls back to the whole name (= its own session, so it is never
    pooled with clips it has nothing to do with)."""
    if "__" in clip_name:
        p = clip_name.split("__")
        return "__".join(p[:2]) if len(p) >= 3 else p[0]
    m = re.search(r"(subject\w+)", clip_name, re.I)
    return m.group(1) if m else clip_name


def family(clip_name: str) -> str:
    t = clip_name.replace("__", " ").replace("_", " ")
    for k, rex in FAMILY_RE:
        if rex.search(t):
            return k
    return "other"


# ------------------------------------------------------------ limb grouping
def limb_groups(joint_names):
    """joint name -> coarse limb group, so 'how many limbs touch' is comparable
    across skeletons whose feet carry 1 (ankle) or 3 (ankle/roll/toe) joints."""
    out = []
    for n in joint_names:
        s = n.lower()
        side = ""
        if re.search(r"(^|[_\W])l($|[_\W])|left|(^|_)l_|_l$|_l_|_l\d", s): side = "L"
        if re.search(r"(^|[_\W])r($|[_\W])|right|(^|_)r_|_r$|_r_|_r\d", s): side = "R"
        if re.search(r"foot|ankle|toe|sole|leg_?[lr]?[56]", s):     part = "foot"
        elif re.search(r"knee|shin|calf|shank|leg_?[lr]?4", s):      part = "knee"
        elif re.search(r"hand|wrist|palm|finger|gripper|claw", s):   part = "hand"
        elif re.search(r"elbow|forearm", s):                         part = "elbow"
        elif re.search(r"shoulder|upperarm|arm", s):                 part = "shoulder"
        elif re.search(r"hip|thigh|leg_?[lr]?[123]", s):             part = "hip"
        elif re.search(r"head|neck|imu|mocap", s):                   part = "head"
        else:                                                        part = "trunk"
        out.append(f"{side}_{part}" if side and part not in ("trunk", "head") else part)
    return out


def contact_sites(spec, D):
    """Skeleton-agnostic contact SITES: two joints are the same physical contact
    site when they are parent/child and the bone between them is shorter than
    SITE_FRAC * D.  Name-free (pal_talos / unitree_h2 ship anonymous body00..N),
    so 'how many places touch the ground' is comparable across all 18 rigs.
    A foot's ankle_pitch/ankle_roll/toe chain collapses to one site; the two feet
    stay separate.  Returns (site_id per joint [J], n_sites)."""
    J = spec.J
    off = np.linalg.norm(spec.rest_offsets, axis=-1) / 100.0          # metres
    par = np.arange(J)
    def find(a):
        while par[a] != a: par[a] = par[par[a]]; a = par[a]
        return a
    for j in range(1, J):
        p = int(spec.parents[j])
        if p >= 0 and off[j] < SITE_FRAC * D:
            a, b = find(j), find(p)
            if a != b: par[a] = b
    root = np.array([find(j) for j in range(J)])
    uq, sid = np.unique(root, return_inverse=True)
    return sid, len(uq)


# ----------------------------------------------------------------- helpers
def load_spec(cache, emb):
    d = np.load(f"{cache}/{emb}_spec.npz", allow_pickle=True)
    return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], list(d["joint_names"]))


def tpose(spec):
    """identity local rotations -> root-relative joint heights (metres)."""
    ident = torch.tensor([1., 0, 0, 0, 1, 0]).view(1, 1, 6).repeat(1, spec.J, 1)
    with torch.no_grad():
        _, gp = fk_global(ident, spec)
    return (gp[0].numpy() / 100.0)


def hysteresis(enter, leave, breaks):
    """state[t] = enter ? 1 : leave ? 0 : state[t-1]; reset at each clip start.
    Vectorised as a forward-fill of the last decisive event."""
    N, J = enter.shape
    ev = np.where(enter, 1, np.where(leave, 0, -1)).astype(np.int8)
    ev[breaks] = np.where(enter[breaks], 1, 0)          # clip start is always decisive
    idx = np.where(ev >= 0, np.arange(N, dtype=np.int64)[:, None], -1)
    np.maximum.accumulate(idx, axis=0, out=idx)
    # a clip must not inherit the previous clip's state: clamp idx to >= clip start
    cs = np.zeros(N, np.int64); cs[breaks] = breaks
    np.maximum.accumulate(cs, out=cs)
    idx = np.maximum(idx, cs[:, None])
    return np.take_along_axis(ev, idx, axis=0) == 1


def despeckle(c, breaks, min_seg):
    """Absorb runs shorter than min_seg: fill short gaps first, then drop short
    contacts (pack.py order).  Runs spanning a whole clip are never flipped."""
    N, J = c.shape
    isbrk = np.zeros(N, bool); isbrk[breaks] = True
    ends = np.zeros(N, bool); ends[np.asarray(breaks[1:]) - 1] = True; ends[N - 1] = True
    for val in (False, True):
        for j in range(J):
            col = c[:, j]
            chg = np.empty(N, bool); chg[0] = True
            np.not_equal(col[1:], col[:-1], out=chg[1:]); chg |= isbrk
            st = np.flatnonzero(chg); en = np.append(st[1:], N)
            m = (col[st] == val) & ((en - st) < min_seg) & ~(isbrk[st] & ends[en - 1])
            for a, b in zip(st[m], en[m]):
                col[a:b] = not val
    return c


def sole_offsets(emb, spec):
    """-> {joint_index: off_local_ardy[3]} 或 {} (没量到 = 退回关节口径)。
    body 名 -> 关节下标按 spec.joint_names 精确匹配; 匹配不上就跳过该侧, 不猜。"""
    if not USE_SOLE or not os.path.exists(SOLE_JSON):
        return {}
    tbl = json.load(open(SOLE_JSON))
    ent = tbl.get(SOLE_ALIAS.get(emb, emb))
    if not ent:
        return {}
    names = list(spec.joint_names)
    out = {}
    for side in ("a", "b"):
        chain = ent.get(side) or []
        # 沿 MJCF 祖先链上溯, 取**第一个**在 spec 里存在的 body: build_merge 会丢掉
        # 一部分 body (toeLeft / LINK_FOOT_L 之类), 只按最末端 body 匹配的话 17 台里
        # 有 9 台整台退回关节口径。偏移已在每个祖先自己的局部系里量好, 换个祖先就是
        # 换一条记录, 不做任何近似。
        for c in chain:
            if c["body"] in names:
                out[names.index(c["body"])] = np.asarray(c["off_ardy"], np.float32)
                break
    return out


def run_emb(cache, emb, seqs, dev, chunk_rows=400_000):
    """-> dict of arrays + header."""
    t0 = time.time()
    spec = load_spec(cache, emb)
    J = spec.J
    SOFF = sole_offsets(emb, spec)
    L = torch.load(f"{cache}/{emb}_paired_local6d.pt", mmap=True)
    RW = torch.load(f"{cache}/{emb}_root_world.pt", mmap=True).float()
    N = L.shape[0]
    assert RW.shape[0] == N, (emb, L.shape, RW.shape)
    starts = np.array([s["start"] for s in seqs], np.int64)
    lens = np.array([s["len"] for s in seqs], np.int64)
    assert starts[-1] + lens[-1] == N, (emb, starts[-1] + lens[-1], N)

    Y = np.empty((N, J), np.float16)
    V = np.empty((N, J), np.float16)
    dt = 1.0 / FPS
    # chunk on whole-clip boundaries so finite differences never cross a clip
    ci = 0
    while ci < len(seqs):
        cj = ci
        while cj < len(seqs) and (starts[cj] + lens[cj] - starts[ci]) < chunk_rows:
            cj += 1
        cj = max(cj, ci + 1)
        s0, s1 = int(starts[ci]), int(starts[cj - 1] + lens[cj - 1])
        b = L[s0:s1].float().to(dev, non_blocking=True)
        with torch.no_grad():
            gr_, gp = fk_global(b, spec)
            w = (gp + RW[s0:s1].to(dev)[:, None, :]) / 100.0        # [n,J,3] metres
            # 脚底点: 把该关节的接触参考点从关节中心挪到脚底最低顶点。gr_ 是全局旋转,
            # 局部偏移已在 ardy 基 (prep_gmr_cache 的约定 R'=AX R AXᵀ 使局部系同步置换,
            # 所以 v_world = R'_ardy @ v_local_ardy 直接成立)。单位: 偏移是米, w 也是米。
            for j_, off_ in SOFF.items():
                w[:, j_] = w[:, j_] + torch.einsum(
                    "nab,b->na", gr_[:, j_], torch.as_tensor(off_, device=dev))
            v = torch.gradient(w[:, :, [0, 2]], spacing=dt, dim=0)[0].norm(dim=-1)
            for k in range(ci, cj):                                  # one-sided at clip edges
                a, e = int(starts[k] - s0), int(starts[k] + lens[k] - s0)
                if e - a >= 2:
                    v[a] = (w[a + 1][:, [0, 2]] - w[a][:, [0, 2]]).norm(dim=-1) / dt
                    v[e - 1] = (w[e - 1][:, [0, 2]] - w[e - 2][:, [0, 2]]).norm(dim=-1) / dt
                else:
                    v[a:e] = 0
            Y[s0:s1] = w[:, :, UP].half().cpu().numpy()
            V[s0:s1] = v.half().cpu().numpy()
        ci = cj
    t_fk = time.time() - t0

    # ---- T-pose floor + scale
    pr = tpose(spec)
    yr = pr[:, UP]
    j_low = int(np.argmin(yr)); D_tpose = float(-yr.min())
    m = Y.min(1).astype(np.float32)                                   # per-frame lowest joint
    rootY = RW[:, UP].numpy() / 100.0
    D_data = float(np.percentile(rootY - m, 99))
    tpose_ok = bool(D_tpose > 0.5 * D_data)
    D = D_tpose if tpose_ok else D_data
    thr = FRAC_THR * D; tau = thr / 3.0; tau_v = V_THR / 3.0
    sid, nsit = contact_sites(spec, D)
    # ---- operating floor h0 (data), per clip then median
    fl = np.array([np.percentile(m[s:s + l], CLIP_FLOOR_PCT) for s, l in zip(starts, lens)], np.float32)
    h0 = float(np.median(fl))
    # the joint the DATA actually stands on (lowest corpus-median height).  Equals
    # the T-pose j_low on every rig whose T-pose is valid; it is the only usable
    # floor joint when it is not (LAFAN human_smpl).  Used for the walk cross-check.
    j_floor = int(np.argmin(np.median(Y[::37].astype(np.float32), axis=0)))

    # ---- session floor correction (lloyd 0831).  One capture session shares one
    # floor calibration; estimating it from >= SESS_MIN_CLIPS clips of DIFFERENT
    # motions keeps it non-circular (a per-clip percentile would force contact into
    # every clip by construction).  Three outcomes per session:
    #   |off| <= SESS_OFF_FRAC*thr        -> leave on the corpus floor h0
    #   coherent and off within reason    -> shift that session's floor by off
    #   sigma too large, or off > 0.25*D  -> UNTRUSTED: no contact supervision
    # A floor mis-calibration is small next to the body; an offset above a quarter
    # of the leg length is not a calibration error, so we refuse to invent a floor.
    sess = np.array([session(str(q.get("name") or q.get("clip") or i))
                     for i, q in enumerate(seqs)])
    h0_clip = np.full(len(seqs), h0, np.float32); trust_clip = np.ones(len(seqs), bool)
    sess_stat = {}; gmin_all = Y.min(1).astype(np.float32) - h0
    for sname in np.unique(sess):
        k = np.flatnonzero(sess == sname)
        idx = np.concatenate([np.arange(starts[i], starts[i] + lens[i]) for i in k])
        if idx.size < SESS_MIN_FRAMES:
            sess_stat[str(sname)] = dict(n=int(len(k)), floor_m=0.0, wall_m=0.0, act="few")
            continue
        p2, p15 = np.percentile(gmin_all[idx], [2, 15])
        floor = float(p2); wall = float(p15 - p2)
        if floor <= SESS_OFF_FRAC * thr:
            act = "keep"                       # on the floor, or below it (penetration
                                               # is a property of the data, not a floor
                                               # error: h0 is already the corpus median)
        elif wall < SESS_WALL_FRAC * thr:
            h0_clip[k] = h0 + floor; act = "shift"      # a sharp wall 4 cm up IS a floor
        else:
            trust_clip[k] = False; act = "untrusted"    # high and no wall -> cannot say
        sess_stat[str(sname)] = dict(n=int(len(k)), floor_m=floor, wall_m=wall, act=act)
    n_shift = sum(1 for v in sess_stat.values() if v["act"] == "shift")
    n_untr = sum(1 for v in sess_stat.values() if v["act"] == "untrusted")
    trust = np.repeat(trust_clip, lens)

    # control arm: what the labels would be if each clip carried its own floor
    fl_row = np.repeat(fl, lens)
    ctrl_lo = (Y.astype(np.float32) - fl_row[:, None]) < thr
    # control arm 2: the pre-0831 labels, on the single corpus floor
    gap_global = (Y.astype(np.float32) - h0)
    gap = (Y.astype(np.float32) - np.repeat(h0_clip, lens)[:, None])
    vv = V.astype(np.float32)
    soft = expit((thr - gap) / tau) * expit((V_THR - vv) / tau_v)
    enter = (gap < thr) & (vv < V_THR)
    leave = (gap > thr + 0.5 * thr) | (vv > 2 * V_THR)
    breaks = starts
    ms = int(round(MIN_SEG_S * FPS))
    con = despeckle(hysteresis(enter, leave, breaks), breaks, ms)
    # CONTROL ARM (lloyd 0829 discipline): the same machinery with the SPEED GATE
    # REMOVED.  The speed gate is what the brief specified, but it throws away
    # 25-77% of the joints the height gate admits, worst exactly in the ground /
    # crawl regime and worst on the rigs whose retarget floats (pal_talos keeps
    # only 23% there), and it leaves joints that are BELOW the floor unlabelled.
    # Both channels ship; the report carries the numbers for each.
    con_low = despeckle(hysteresis(gap < thr, gap > 1.5 * thr, breaks), breaks, ms)

    hdr = dict(
        embodiment=emb, corpus=os.path.basename(cache), J=J, N=int(N), fps=FPS,
        joint_names=spec.joint_names, limb_groups=limb_groups(spec.joint_names),
        site_id=sid.tolist(), n_sites=int(nsit), site_frac=SITE_FRAC,
        j_low=j_low, j_low_name=spec.joint_names[j_low],
        j_floor=j_floor, j_floor_name=spec.joint_names[j_floor],
        D_tpose_m=D_tpose, D_data_p99_m=D_data, tpose_valid=tpose_ok, D_used_m=D,
        h0_tpose_m=0.0, h0_m=h0, gauge_offset_m=h0,
        clip_floor_p5_median_m=h0, clip_floor_p5_iqr_m=float(np.percentile(fl, 75) - np.percentile(fl, 25)),
        thr_m=thr, thr_frac_of_D=FRAC_THR, tau_m=tau, v_thr_ms=V_THR, tau_v_ms=tau_v,
        min_seg_frames=int(round(MIN_SEG_S * FPS)), min_seg_s=MIN_SEG_S,
        rule="c = sigmoid((thr-gap)/tau)*sigmoid((v_thr-vxy)/tau_v); binary = hysteresis+despeckle",
        sole_applied=sorted(int(k) for k in SOFF),
        sole_joint_names=[spec.joint_names[k] for k in sorted(SOFF)],
        sole_drop_cm=[round(float(-SOFF[k][UP] * 100), 2) for k in sorted(SOFF)],
        sess_min_frames=SESS_MIN_FRAMES, sess_off_frac=SESS_OFF_FRAC,
        sess_wall_frac=SESS_WALL_FRAC,
        n_sessions=int(len(sess_stat)), n_sess_shift=n_shift, n_sess_untrusted=n_untr,
        n_clips_untrusted=int((~trust_clip).sum()), n_clips=int(len(seqs)),
        frac_frames_untrusted=float((~trust).mean()),
        sessions=sess_stat,
        fk_seconds=round(t_fk, 1),
    )
    # pre-0831 labels on the single corpus floor, kept as the change-audit arm
    con_low_global = despeckle(hysteresis(gap_global < thr, gap_global > 1.5 * thr,
                                          breaks), breaks, ms)
    return dict(gap=gap.astype(np.float16), contact_soft=soft.astype(np.float16),
                contact=con, contact_low=con_low, clip_floor=fl, header=hdr, spec=spec, Y=Y, V=V,
                ctrl=(ctrl_lo & (vv < V_THR)), starts=starts, lens=lens,
                trust=trust, clip_h0=h0_clip, trust_clip=trust_clip,
                contact_low_global=con_low_global)


def label_cache(cache, embs=None, out_dir=None, dev="cuda", log=print):
    """Label one cache dir.  cache: the *_paired_local6d.pt / *_spec.npz directory.
    embs: list of embodiment names, or None/["all"] for the manifest's full list.
    out_dir: where <emb>_contact.npz goes (default: alongside the cache).
    Returns {emb: stats dict}.  Import this from a sibling process to split work."""
    out_dir = out_dir or cache
    os.makedirs(out_dir, exist_ok=True)
    seqs = json.load(open(f"{cache}/sequences.json"))
    man = json.load(open(f"{cache}/manifest.json"))
    if not embs or list(embs) == ["all"]:
        embs = man["embodiments"]
    fams = np.array([family(s["name"]) for s in seqs])
    starts = np.array([s["start"] for s in seqs], np.int64)
    lens = np.array([s["len"] for s in seqs], np.int64)
    stats = {}
    for e in embs:
        t0 = time.time()
        r = run_emb(cache, e, seqs, dev)
        hdr = r["header"]
        hdr["cache"] = cache
        out = f"{out_dir}/{e}_contact.npz"
        np.savez(out, contact_soft=r["contact_soft"], contact=r["contact"],
                 contact_low=r["contact_low"], gap=r["gap"],
                 clip_floor=r["clip_floor"], header=json.dumps(hdr),
                 trust=r["trust"], clip_h0=r["clip_h0"], trust_clip=r["trust_clip"],
                 contact_low_global=r["contact_low_global"])
        sz = os.path.getsize(out) / 1e9
        st = clip_stats(r, fams, starts, lens)
        cl_, cg_, tr_ = r["contact_low"], r["contact_low_global"], r["trust"]
        st.update(untrusted_frac=float((~tr_).mean()),
                  n_sess_shift=hdr["n_sess_shift"], n_sess_untrusted=hdr["n_sess_untrusted"],
                  low_rate_before=float(cg_.mean()), low_rate_after=float(cl_.mean()),
                  low_rate_after_trusted=float(cl_[tr_].mean()) if tr_.any() else 0.0,
                  frames_flipped_on=float((cl_ & ~cg_).mean()),
                  frames_flipped_off=float((~cl_ & cg_).mean()))
        st.update({k: hdr[k] for k in ("J", "N", "D_used_m", "D_tpose_m", "D_data_p99_m",
                                       "tpose_valid", "j_low_name", "j_floor_name",
                                       "h0_m", "gauge_offset_m",
                                       "thr_m", "v_thr_ms", "clip_floor_p5_iqr_m")})
        st.update(size_gb=round(sz, 3), seconds=round(time.time() - t0, 1),
                  fk_seconds=hdr["fk_seconds"], npz=out)
        stats[e] = st
        log(f"[{os.path.basename(cache)}/{e}] J={hdr['J']} D={hdr['D_used_m']:.3f} "
            f"thr={hdr['thr_m']*100:.1f}cm h0={hdr['h0_m']*100:.2f}cm "
            f"rate={st['contact_rate']*100:.1f}% any={st['any_rate']*100:.1f}% "
            f"ge3={st['ge3_joint']*100:.2f}% ge3L={st['ge3_limb']*100:.2f}% "
            f"{sz:.2f}GB {st['seconds']:.0f}s")
        del r
    return stats


def clip_stats(r, fams, starts, lens):
    """Contact statistics for one labelled embodiment."""
    con = r["contact"]; hdr = r["header"]; N, J = con.shape
    nc = con.sum(1).astype(np.int16)
    groups = np.array(hdr["site_id"])
    uq = sorted(set(groups.tolist()))
    gmat = np.stack([con[:, groups == g].any(1) for g in uq], 1)     # [N,G]
    gname = {g: "+".join(np.array(hdr["joint_names"])[groups == g][:3].tolist())
             + ("+..." if (groups == g).sum() > 3 else "") for g in uq}
    ng = gmat.sum(1).astype(np.int16)
    cl = r.get("contact_low")
    lo_extra = dict()
    if cl is not None:
        nl = cl.sum(1); gl = np.stack([cl[:, groups == g].any(1) for g in uq], 1).sum(1)
        pen = r["gap"].astype(np.float32) < 0
        lo_extra = dict(low_contact_rate=float(cl.mean()), low_any_rate=float((nl > 0).mean()),
                        low_ge3_site=float((gl >= 3).mean()), low_mean_sites=float(gl.mean()),
                        penetration_rate=float(pen.mean()),
                        penetration_labelled=float((pen & con).sum() / max(pen.sum(), 1)),
                        penetration_labelled_low=float((pen & cl).sum() / max(pen.sum(), 1)))
    hist_j = np.bincount(nc, minlength=9)[:9].astype(np.int64)
    hist_j = np.append(hist_j[:8], int((nc >= 8).sum()))
    hist_l = np.bincount(ng, minlength=9)[:9].astype(np.int64)
    hist_l = np.append(hist_l[:8], int((ng >= 8).sum()))
    per_group = {gname[g]: float(gmat[:, i].mean()) for i, g in enumerate(uq)}
    per_joint = {n: float(con[:, i].mean()) for i, n in enumerate(hdr["joint_names"])}
    byfam = {}
    for f in sorted(set(fams.tolist())):
        idx = np.flatnonzero(fams == f)
        rows = np.concatenate([np.arange(starts[i], starts[i] + lens[i]) for i in idx])
        c = con[rows]; n_ = c.sum(1); g_ = np.stack([c[:, groups == g].any(1) for g in uq], 1).sum(1)
        cl_ = cl[rows] if cl is not None else None
        byfam[f] = dict(clips=int(len(idx)), frames=int(len(rows)),
                        rate=float(c.mean()), any=float((n_ > 0).mean()),
                        low_any=float(cl_.any(1).mean()) if cl_ is not None else None,
                        low_ge3_site=float((np.stack([cl_[:, groups == g].any(1) for g in uq], 1).sum(1) >= 3).mean()) if cl_ is not None else None,
                        ge3_joint=float((n_ >= 3).mean()), ge3_limb=float((g_ >= 3).mean()),
                        mean_joints=float(n_.mean()), mean_limbs=float(g_.mean()))
    # walk cross-check: height of the floor-defining joint on labelled-contact frames
    jl = hdr.get("j_floor", hdr["j_low"]); Y = r["Y"][:, jl].astype(np.float32); cl = con[:, jl]
    wid = np.flatnonzero((fams == "walk"))
    if len(wid):
        wr = np.concatenate([np.arange(starts[i], starts[i] + lens[i]) for i in wid])
        sel = wr[cl[wr]]
        walk_h = float(np.median(Y[sel])) if len(sel) else float("nan")
    else:
        walk_h = float("nan")
    ctrl = r.get("ctrl")
    ctrl_any = float(ctrl.any(1).mean()) if ctrl is not None else float("nan")
    return dict(contact_rate=float(con.mean()), ctrl_perclip_floor_any=ctrl_any, any_rate=float((nc > 0).mean()),
                ge3_joint=float((nc >= 3).mean()), ge3_limb=float((ng >= 3).mean()),
                ge4_joint=float((nc >= 4).mean()), ge4_limb=float((ng >= 4).mean()),
                mean_joints=float(nc.mean()), mean_limbs=float(ng.mean()),
                hist_joints=hist_j.tolist(), hist_limbs=hist_l.tolist(),
                n_sites=len(uq), site_names=[gname[g] for g in uq], per_group=per_group, per_joint=per_joint, by_family=byfam,
                walk_stance_h_m=walk_h, walk_minus_h0_m=walk_h - hdr["h0_m"], **lo_extra)


def augment_low(path):
    """Add the speed-gate-free control channel to an EXISTING <emb>_contact.npz.
    Needs only the stored `gap` + `clip_floor`, so a finished AMASS run can be
    upgraded without re-reading the 4 GB local6d tensor."""
    z = np.load(path, allow_pickle=True)
    if "contact_low" in z.files:
        return False
    hdr = json.loads(str(z["header"])); gap = z["gap"].astype(np.float32)
    cache = hdr.get("cache") or CACHES[{"cache_xwbc_lafan": "lafan"}.get(hdr["corpus"], "amass")]
    seqs = json.load(open(f"{cache}/sequences.json"))
    breaks = np.array([s["start"] for s in seqs], np.int64)
    thr = hdr["thr_m"]; ms = int(round(MIN_SEG_S * FPS))
    cl = despeckle(hysteresis(gap < thr, gap > 1.5 * thr, breaks), breaks, ms)
    np.savez(path, contact_soft=z["contact_soft"], contact=z["contact"], contact_low=cl,
             gap=z["gap"], clip_floor=z["clip_floor"], header=z["header"])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", nargs="*", default=None,
                    help="paths to existing <emb>_contact.npz; add contact_low and exit")
    ap.add_argument("--cache", required=False,
                    help="cache dir, or one of: " + ", ".join(CACHES))
    ap.add_argument("--emb", nargs="+", default=["all"])
    ap.add_argument("--out", default=None, help="npz output dir (default: --cache)")
    ap.add_argument("--stats", default=None, help="write per-embodiment stats json here")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.augment is not None:
        for p_ in a.augment:
            print(f"{'added' if augment_low(p_) else 'already had'} contact_low: {p_}")
        return
    cache = CACHES.get(a.cache, a.cache)
    t0 = time.time()
    stats = label_cache(cache, a.emb, a.out, a.device)
    if a.stats:
        os.makedirs(os.path.dirname(os.path.abspath(a.stats)), exist_ok=True)
        json.dump(stats, open(a.stats, "w"), indent=1)
        print(f"stats -> {a.stats}")
    tot = sum(s["N"] for s in stats.values())
    print(f"DONE {len(stats)} embs, {tot} frame-rows, {time.time()-t0:.0f}s, "
          f"{sum(s['size_gb'] for s in stats.values()):.2f} GB")


if __name__ == "__main__":
    main()
