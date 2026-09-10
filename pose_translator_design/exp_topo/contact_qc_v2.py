"""Visual acceptance gate for the SESSION-FLOOR-CORRECTED contact labels (lloyd 0831).

Reads the labels from a STAGING dir (--npz) while reading the skeletons and the
motion from the read-only production cache.  Nothing is ever written into the
cache dirs.

What is drawn, per joint, as a translucent sphere of radius = that rig's own thr:

  RED      in `contact`               (height gate + 0.10 m/s speed gate agreed)
  ORANGE   in `contact_low` only      (height gate OK, SPEED GATE vetoed it)
                                       -> this IS supervised: contact_low is the
                                          channel lloyd chose for training
  WHITE    near the floor, admitted by neither gate
  MAGENTA  the frame is UNTRUSTED (trust=False): its session was high off the
           floor with no wall, so we refuse to supervise it at all.  Magenta is
           NOT "no contact", it is "we do not know".
  small dot  joint far from the floor, no marker

The checkerboard plane is always drawn at the floor the LABELLER used for that
clip, i.e. clip_h0 after the session correction (`before` sheets use the old
global h0 instead, which is the point of the comparison).  Numbers next to the
balls are the gap in cm above that floor.
"""
from __future__ import annotations
import os, sys, json, argparse, collections
import numpy as np, torch

EXP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP)
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont
from common import fk_global
import contact_label as CL
from contact_qc import add_checker_floor, FONT, FONTB
from prep_gmr_cache import AX

torch.set_grad_enabled(False)
STAGE = "/home/arenalabs/tmp/cl_v2"
OUT = "/home/arenalabs/tmp/cl_v2_qc"

C_CON = [0.95, 0.15, 0.15, 0.32]      # red    - contact
C_VET = [1.00, 0.60, 0.02, 0.46]      # orange - speed-vetoed (still in contact_low)
C_CND = [1.00, 1.00, 1.00, 0.22]      # white  - near floor, not admitted
C_UNT = [0.85, 0.25, 0.95, 0.40]      # magenta- untrusted, no supervision
LEGEND = ("legend: every ball has radius = thr (that rig's contact threshold), the number is the gap in cm above the "
          "labelled floor.   RED = in `contact`   |   ORANGE = height gate OK but the 0.10 m/s SPEED GATE vetoed it "
          "(still supervised: `contact_low` is the TRAINING channel)   |   WHITE = near the floor, admitted by neither   |   "
          "MAGENTA = UNTRUSTED frame: no contact supervision at all (this is 'we do not know', NOT 'no contact')")


class Panel:
    def __init__(self, w=460, h=460):
        sp = mj.MjSpec()
        add_checker_floor(sp)
        sp.visual.global_.offwidth = max(w, 1280)
        sp.visual.global_.offheight = max(h, 1280)
        self.m = sp.compile(); self.d = mj.MjData(self.m)
        mj.mj_forward(self.m, self.d)
        self.ren = mj.Renderer(self.m, h, w)
        self.cam = mj.MjvCamera(); self.opt = mj.MjvOption()
        self.opt.label = mj.mjtLabel.mjLABEL_NONE

    def draw(self, P, parents, thr, con, vet, cand, dist, untrusted=False,
             azim=125, elev=-12, dist_scale=0.95):
        lo, hi = P.min(0), P.max(0); mid = (lo + hi) / 2
        self.cam.lookat[:] = [mid[0], mid[1], max(0.30, mid[2])]
        self.cam.distance = max(1.5, dist_scale * float(np.linalg.norm(hi - lo)) + 0.35)
        self.cam.elevation, self.cam.azimuth = elev, azim
        self.ren.update_scene(self.d, self.cam, self.opt)
        sc = self.ren.scene

        def add(gt, size, pos, rgba, label=""):
            if sc.ngeom >= sc.maxgeom: return None
            g = sc.geoms[sc.ngeom]; sc.ngeom += 1
            mj.mjv_initGeom(g, gt, np.asarray(size, np.float64), np.asarray(pos, np.float64),
                            np.eye(3).flatten(), np.asarray(rgba, np.float32))
            if label: g.label = label
            return g

        for j, p in enumerate(parents):
            if p < 0: continue
            g = add(mj.mjtGeom.mjGEOM_CAPSULE, [0, 0, 0], np.zeros(3), [0.25, 0.72, 0.85, 1])
            if g is not None:
                mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, 0.012, P[p], P[j])
        nmark = int(con.sum() + vet.sum() + cand.sum())
        for j in range(len(parents)):
            if untrusted:
                # everything the height gate WOULD have admitted, drawn as unknown
                if con[j] or vet[j] or cand[j]:
                    add(mj.mjtGeom.mjGEOM_SPHERE, [thr] * 3, P[j], C_UNT)
                    add(mj.mjtGeom.mjGEOM_SPHERE, [thr * 0.4] * 3, P[j], [0.9, 0.3, 1.0, 1.0],
                        f"{dist[j]*100:.1f}?")
                else:
                    add(mj.mjtGeom.mjGEOM_SPHERE, [0.014] * 3, P[j], [0.75, 0.45, 0.9, 1.0])
                continue
            if con[j]:
                add(mj.mjtGeom.mjGEOM_SPHERE, [thr] * 3, P[j], C_CON)
                add(mj.mjtGeom.mjGEOM_SPHERE, [thr * 0.35] * 3, P[j], [1.0, 0.1, 0.1, 1.0],
                    f"{dist[j]*100:.1f}")
            elif vet[j]:
                add(mj.mjtGeom.mjGEOM_SPHERE, [thr] * 3, P[j], C_VET)
                add(mj.mjtGeom.mjGEOM_SPHERE, [thr * 0.45] * 3, P[j], [1.0, 0.62, 0.0, 1.0],
                    f"{dist[j]*100:.1f}")
            elif cand[j]:
                add(mj.mjtGeom.mjGEOM_SPHERE, [thr] * 3, P[j], C_CND)
                add(mj.mjtGeom.mjGEOM_SPHERE, [thr * 0.3] * 3, P[j], [0.95, 0.95, 0.95, 1.0],
                    f"{dist[j]*100:.1f}" if nmark <= 8 else "")
            else:
                add(mj.mjtGeom.mjGEOM_SPHERE, [0.014] * 3, P[j], [1.0, 0.6, 0.15, 1.0])
        return self.ren.render().copy()


def caption(img, lines, colors=None):
    pad = 46 + 15 * (len(lines) - 1)
    out = Image.new("RGB", (img.shape[1], img.shape[0] + pad), (12, 12, 16))
    out.paste(Image.fromarray(img), (0, pad))
    dr = ImageDraw.Draw(out)
    f0 = ImageFont.truetype(FONTB, 15); f1 = ImageFont.truetype(FONT, 12)
    dr.text((7, 4), lines[0], (255, 255, 255), font=f0)
    for i, t in enumerate(lines[1:]):
        dr.text((7, 23 + 15 * i), t, (colors or {}).get(i + 1, (185, 190, 200)), font=f1)
    return np.asarray(out)


def tile(imgs, cols, path, title, legend=LEGEND, sub=None):
    if not imgs: return None
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    hdr = 62 + (16 if sub else 0)
    sheet = Image.new("RGB", (cols * w, rows * h + hdr), (8, 8, 11))
    dr = ImageDraw.Draw(sheet)
    dr.text((10, 7), title, (255, 255, 255), font=ImageFont.truetype(FONTB, 16))
    y = 28
    if sub:
        dr.text((10, y), sub, (255, 220, 120), font=ImageFont.truetype(FONT, 12)); y += 16
    for i, ln in enumerate(_wrap(legend, cols * w - 20)):
        dr.text((10, y + 14 * i), ln, (170, 180, 195), font=ImageFont.truetype(FONT, 11))
    for i, im in enumerate(imgs):
        sheet.paste(Image.fromarray(im), ((i % cols) * w, hdr + (i // cols) * h))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path)
    print(f"  sheet -> {path}  ({len(imgs)} cells)", flush=True)
    return path


def _wrap(text, px, fs=11):
    f = ImageFont.truetype(FONT, fs)
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if f.getlength(t) > px and cur:
            lines.append(cur); cur = w_
        else:
            cur = t
    if cur: lines.append(cur)
    return lines[:3]


def write_mp4(path, frames, fps=30):
    """rot's imageio is on the PyAV plugin, which (a) rejects `quality=8` and
    (b) fails inside avcodec_open2(libx264) when a yuv420p frame has an ODD
    height or width -- which a tile of 391 px cells silently produces.  Pad to
    even here rather than constraining every caller's cell size."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    h, w = frames[0].shape[:2]
    if h % 2 or w % 2:
        H, W = h + h % 2, w + w % 2
        pad = []
        for f in frames:
            g = np.zeros((H, W, 3), np.uint8); g[:h, :w] = f; pad.append(g)
        frames = pad
    try:
        iio.mimwrite(path, frames, fps=fps, quality=8)
    except TypeError:                     # PyAV plugin: no `quality` kwarg
        iio.mimwrite(path, frames, fps=fps, codec="libx264", out_pixel_format="yuv420p")
    print(f"  video -> {path}  ({len(frames)} frames, "
          f"{os.path.getsize(path)/1e6:.1f} MB)", flush=True)
    return path


# ------------------------------------------------------------------ data
class Emb2:
    """One embodiment: skeleton + motion from the READ-ONLY cache, labels from staging."""

    def __init__(self, cache, npz_dir, emb, seqs):
        self.emb = emb
        self.spec = CL.load_spec(cache, emb)
        z = np.load(f"{npz_dir}/{emb}_contact.npz", allow_pickle=True)
        self.hdr = json.loads(str(z["header"]))
        self.con = z["contact"]; self.low = z["contact_low"]; self.gap = z["gap"]
        self.lowg = z["contact_low_global"] if "contact_low_global" in z.files else self.low
        self.trust = z["trust"] if "trust" in z.files else np.ones(len(self.con), bool)
        self.trust_clip = z["trust_clip"] if "trust_clip" in z.files else None
        self.clip_h0 = z["clip_h0"] if "clip_h0" in z.files else None
        self.L = torch.load(f"{cache}/{emb}_paired_local6d.pt", mmap=True)
        self.RW = torch.load(f"{cache}/{emb}_root_world.pt", mmap=True).float()
        self.sid = np.array(self.hdr["site_id"]); self.nsite = self.hdr["n_sites"]
        self.thr = self.hdr["thr_m"]; self.h0 = self.hdr["h0_m"]
        self.starts = np.array([s["start"] for s in seqs], np.int64)
        self.lens = np.array([s["len"] for s in seqs], np.int64)
        self.h0row = (np.repeat(self.clip_h0, self.lens) if self.clip_h0 is not None
                      else np.full(len(self.con), self.h0, np.float32))

    def clip_of(self, r):
        return int(np.searchsorted(self.starts, r, "right") - 1)

    def sites(self, rows, arr=None):
        c = (self.con if arr is None else arr)[rows]
        return np.stack([c[:, self.sid == g].any(1) for g in range(self.nsite)], 1).sum(1)

    def world(self, rows, floor=None, dev="cuda"):
        """[n,J,3] MuJoCo (Z-up) with the labelled floor at z=0."""
        rows = np.asarray(rows)
        b = self.L[rows].float().to(dev)
        _, gp = fk_global(b, self.spec)
        w = (gp.cpu() + self.RW[rows][:, None, :]) / 100.0
        f = self.h0row[rows] if floor is None else np.broadcast_to(np.asarray(floor, np.float32), (len(rows),))
        w[:, :, CL.UP] -= torch.as_tensor(np.ascontiguousarray(f), dtype=w.dtype)[:, None]
        return w.numpy() @ AX


def cell(E, P, row, corpus, clip, fam, panel, extra=None, mode="after", **kw):
    """mode: after = shipped labels (session floor); before = pre-0831 global-h0 labels."""
    if mode == "after":
        gap = E.gap[row].astype(np.float32)
        con, low = E.con[row], E.low[row]
        untr = not bool(E.trust[row])
        floor_txt = f"floor=clip_h0 {E.h0row[row]*100:+.2f}cm"
    else:
        gap = E.gap[row].astype(np.float32) + E.h0row[row] - E.h0
        low = E.lowg[row]; con = E.con[row] & low
        untr = False
        floor_txt = f"floor=GLOBAL h0 {E.h0*100:+.2f}cm"
    vet = low & ~con
    cand = (~con) & (~vet) & (gap < 4 * E.thr)
    img = panel.draw(P, E.spec.parents, E.thr, con, vet, cand, gap, untrusted=untr, **kw)
    ns = int(np.stack([low[E.sid == g].any() for g in range(E.nsite)]).sum())
    l3 = (f"UNTRUSTED - NO CONTACT SUPERVISION   ({int(low.sum())}j would have been labelled)"
          if untr else
          f"contact {int(con.sum())}j + speed-vetoed {int(vet.sum())} = contact_low(TRAIN) "
          f"{int(low.sum())}j / {ns} sites")
    lines = [f"{E.emb}   [{corpus}]", f"{clip[:54]}  f{row - E.starts[E.clip_of(row)]}  fam={fam}",
             l3, f"thr={E.thr*100:.1f}cm  {floor_txt}  gap_min={gap.min()*100:+.1f}cm"]
    cols = {2: (215, 110, 255) if untr else (250, 120, 120), 3: (150, 200, 230)}
    if extra:
        lines.append(extra); cols[4] = (255, 210, 90)
    return caption(img, lines, cols)


# =====================================================================
#  DELIVERABLE 2 : full coverage -- every family x every embodiment
# =====================================================================
STRATA = (("airborne", "fewest contact sites in the family (flight / hover)"),
          ("typical", "the family's MEDIAN contact-site count"),
          ("loaded", "top decile of contact sites (ground / crouch)"))


def pick_family_rows(cache, npz_dir, seqs, fams, ufam, ref, rng, k=3):
    """One pick per STRATUM per family, chosen ONCE on the reference rig so all 17
    embodiments show the SAME instant (the caches are row-aligned).

    Picking only the top quartile -- what the first pass did -- lands every cell on
    a body lying on the floor, because LAFAN's clips are 4-minute mixed-content
    recordings and the max-contact frame of a clip called `walk3` is the subject
    on the ground.  Sampling the airborne / median / loaded strata instead shows
    what the labeller does across the whole range, which is what a gate needs."""
    E = Emb2(cache, npz_dir, ref, seqs)
    out = {}
    for f in ufam:
        cl = np.flatnonzero(fams == f)
        # pool frames from up to 8 different clips of this family
        use = rng.choice(cl, size=min(8, len(cl)), replace=False)
        rows, ns = [], []
        for ci in use:
            s0, ln = int(E.starts[ci]), int(E.lens[ci])
            pr = s0 + rng.choice(ln, size=min(ln, 1500), replace=False)
            rows.append(pr); ns.append(E.sites(pr, E.low))
        rows = np.concatenate(rows); ns = np.concatenate(ns)
        q = np.percentile(ns, [10, 50, 90])
        picks = []
        for si, tgt in enumerate(q[:k]):
            pool = rows[np.abs(ns - tgt) <= (0.5 if si else 0.001)]
            if not len(pool):
                pool = rows[np.argsort(np.abs(ns - tgt))[:200]]
            r = int(rng.choice(pool))
            picks.append((int(np.searchsorted(E.starts, r, "right") - 1), r,
                          f"{STRATA[si][0]} ({int(round(tgt))} sites on {ref})"))
        out[f] = picks
    del E
    return out


def pick_family_windows(cache, npz_dir, seqs, fams, ufam, ref, rng, nfr=90):
    """one contiguous window per family, for the videos."""
    E = Emb2(cache, npz_dir, ref, seqs)
    out = {}
    for f in ufam:
        cl = np.flatnonzero(fams == f)
        # prefer a clip long enough for the window, median length among those
        ok = cl[E.lens[cl] >= nfr]
        cand = ok if len(ok) else cl
        ci = int(cand[np.argsort(E.lens[cand])[len(cand) // 2]])
        s0, ln = int(E.starts[ci]), int(E.lens[ci])
        n = min(nfr, ln)
        # window with the most site-variation (so it is not a static stretch)
        best, bs = s0, -1
        for st in range(s0, s0 + ln - n + 1, max(1, (ln - n) // 12 + 1)):
            v = float(np.std(E.sites(np.arange(st, st + n), E.low)))
            if v > bs: bs, best = v, st
        out[f] = (ci, np.arange(best, best + n))
    del E
    return out


def run_coverage(corp, cache, npz_dir, out, embs, seqs, fams, ufam, seed=0,
                 nfr=90, cell=300, sheet_cell=460, cols=6, videos=True):
    rng = np.random.default_rng(seed)
    ref = "human_smpl" if "human_smpl" in embs else embs[0]
    rows = pick_family_rows(cache, npz_dir, seqs, fams, ufam, ref, rng)
    wins = pick_family_windows(cache, npz_dir, seqs, fams, ufam, ref, rng, nfr) if videos else {}
    print(f"[cov/{corp}] {len(ufam)} families x {len(embs)} embodiments, "
          f"ref={ref}, {'videos on' if videos else 'sheets only'}", flush=True)
    Psheet = Panel(sheet_cell, sheet_cell)
    Pvid = Panel(cell, cell) if videos else None
    grid = {}                        # (emb, fam, k) -> image
    vid = collections.defaultdict(dict)   # fam -> emb -> [frames]
    for e in embs:
        E = Emb2(cache, npz_dir, e, seqs)
        for f in ufam:
            for k, (ci, r, strat) in enumerate(rows[f]):
                P = E.world(np.array([r]))[0]
                grid[(e, f, k)] = cell_wrap(E, P, r, corp, seqs[ci]["name"],
                                            f"{f}  [{strat}]", Psheet)
            if videos:
                ci, rr = wins[f]
                W = E.world(rr)
                vid[f][e] = [cell_wrap(E, W[t], int(rr[t]), corp, seqs[ci]["name"], f, Pvid)
                             for t in range(len(rr))]
        print(f"[cov/{corp}] {e} rendered "
              f"({'untrusted frames ' + format(float((~E.trust).mean())*100, '.2f') + '%'})", flush=True)
        del E
    made = []
    for f in ufam:
        for k in range(len(rows[f])):
            ims = [grid[(e, f, k)] for e in embs if (e, f, k) in grid]
            ci, r, strat = rows[f][k]
            made.append(tile(ims, cols, f"{out}/sheets/by_family/{corp}_{f}_{STRATA[k][0]}.png",
                             f"{corp} / family={f} / stratum={STRATA[k][0]} — {STRATA[k][1]} — the SAME instant on all {len(ims)} embodiments",
                             sub=f"clip={seqs[ci]['name']}   row={r}   session={CL.session(seqs[ci]['name'])}   pick={strat}"))
    for e in embs:
        ims = [grid[(e, f, k)] for f in ufam for k in range(len(rows[f])) if (e, f, k) in grid]
        made.append(tile(ims, 3, f"{out}/sheets/by_embodiment/{corp}_{e}.png",
                         f"{corp} / {e} — ALL {len(ufam)} motion families x 3 contact strata (airborne | typical | loaded), {len(ims)} cells"))
    if videos:
        for f in ufam:
            ci, rr = wins[f]
            frames = []
            for t in range(len(rr)):
                cells = [vid[f][e][t] for e in embs]
                h, w = cells[0].shape[:2]
                nrow = (len(cells) + cols - 1) // cols
                canvas = np.zeros((nrow * h, cols * w, 3), np.uint8)
                for i, c in enumerate(cells):
                    canvas[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = c
                frames.append(canvas)
            made.append(write_mp4(f"{out}/videos/{corp}_fam_{f}.mp4", frames))
    return [m for m in made if m]


def cell_wrap(E, P, row, corp, clip, fam, panel, **kw):
    return cell(E, P, row, corp, clip, fam, panel, **kw)


# =====================================================================
#  DELIVERABLE 3 : did the SHIFT invent contact during real flight?
# =====================================================================
def sess_report(npz_dir, embs):
    """-> {session: {emb: act}} plus the per-emb floor/wall."""
    acts = collections.defaultdict(dict); det = collections.defaultdict(dict)
    for e in embs:
        h = json.loads(str(np.load(f"{npz_dir}/{e}_contact.npz", allow_pickle=True)["header"]))
        for k, v in h["sessions"].items():
            acts[k][e] = v["act"]; det[k][e] = v
    return acts, det


def run_beforeafter(corp, cache, npz_dir, out, embs, seqs, fams, want_sess, rng,
                    nfr=110, cellpx=330, per_sess_frames=4, video_embs=None):
    """Side-by-side BEFORE (contact_low_global on the old global h0) vs AFTER
    (contact_low on the session-corrected floor), plus a video, for each session
    the correction touched.  Flight phases are hunted deliberately: the frames
    picked are the ones with the LARGEST gain, plus the airborne extremes."""
    names = np.array([s["name"] for s in seqs])
    sess = np.array([CL.session(n) for n in names])
    made = []
    P = Panel(cellpx, cellpx)
    acts, det = sess_report(npz_dir, embs)
    for sname in want_sess:
        if sname not in acts: 
            print(f"[ba] {sname} absent, skipped", flush=True); continue
        cidx = np.flatnonzero(sess == sname)
        for e in embs:
            a = acts[sname].get(e)
            if a not in ("shift", "untrusted"):
                continue
            E = Emb2(cache, npz_dir, e, seqs)
            rows = np.concatenate([np.arange(E.starts[c], E.starts[c] + E.lens[c]) for c in cidx])
            lo = E.lowg[rows].any(1); la = E.low[rows].any(1)
            gained = rows[la & ~lo]
            gmin_a = E.gap[rows].astype(np.float32).min(1)
            gmin_b = gmin_a + E.h0row[rows] - E.h0
            d = det[sname][e]
            tag = (f"{a.upper()}  session floor {d['floor_m']*100:+.2f}cm  "
                   f"wall {d['wall_m']*100:.2f}cm  thr {E.thr*100:.1f}cm  "
                   f"({len(cidx)} clips)")
            # frame picks: the biggest gains + the most airborne frames in the session
            pick = []
            if a == "shift" and len(gained):
                pick += list(rng.choice(gained, size=min(per_sess_frames - 1, len(gained)),
                                        replace=False))
            pick.append(int(rows[int(np.argmax(gmin_b))]))          # peak of the flight
            ims = []
            for r in [int(x) for x in pick]:
                ci = E.clip_of(r)
                for mode in ("before", "after"):
                    fl = E.h0 if mode == "before" else None
                    W = E.world(np.array([r]), floor=fl)[0]
                    ex = (f"BEFORE (pre-0831, global h0)" if mode == "before"
                          else f"AFTER (session floor)  |  {tag}")
                    ims.append(cell_wrap(E, W, r, corp, names[ci], fams[ci], P, extra=ex, mode=mode))
            made.append(tile(ims, 2, f"{out}/beforeafter/{corp}_{sname}_{e}.png",
                             f"{corp} / session {sname} / {e} — LEFT before (contact_low_global, global h0)   RIGHT after (contact_low, session floor)",
                             sub=tag + f"   |   frames gaining contact: {la.sum()-lo.sum()} of {len(rows)} "
                                       f"({100*(la.sum()-lo.sum())/len(rows):.2f}%)   "
                                       f"most-airborne frame in session: lowest joint {gmin_b.max()*100:.1f}cm above the OLD floor"))
            if video_embs is not None and e not in video_embs:
                print(f"[ba] {sname} / {e} / {a}: gained {int((la&~lo).sum())} frames "
                      f"({100*float((la&~lo).mean()):.2f}%)  [sheet only]", flush=True)
                del E; continue
            # a video over the clip with the most gain (or the longest, if untrusted)
            per = np.array([int((E.low[E.starts[c]:E.starts[c]+E.lens[c]].any(1) &
                                 ~E.lowg[E.starts[c]:E.starts[c]+E.lens[c]].any(1)).sum())
                            for c in cidx])
            ci = int(cidx[int(np.argmax(per))] if per.max() > 0 else cidx[int(np.argmax(E.lens[cidx]))])
            s0, ln = int(E.starts[ci]), int(E.lens[ci])
            n = min(nfr, ln); st = s0 + max(0, (ln - n) // 2)
            rr = np.arange(st, st + n)
            Wb = E.world(rr, floor=E.h0); Wa = E.world(rr)
            fr = []
            for t in range(n):
                cb = cell_wrap(E, Wb[t], int(rr[t]), corp, names[ci], fams[ci], P,
                               extra="BEFORE (pre-0831, global h0)", mode="before")
                ca = cell_wrap(E, Wa[t], int(rr[t]), corp, names[ci], fams[ci], P,
                               extra=f"AFTER  |  {tag}", mode="after")
                fr.append(np.concatenate([cb, ca], 1))
            made.append(write_mp4(f"{out}/beforeafter/{corp}_{sname}_{e}.mp4", fr))
            print(f"[ba] {sname} / {e} / {a}: gained {int((la&~lo).sum())} frames "
                  f"({100*float((la&~lo).mean()):.2f}%)", flush=True)
            del E
    return [m for m in made if m]


def flight_audit(corp, cache, npz_dir, out, embs, seqs, want_sess):
    """NUMERIC companion to the flight rendering: for every frame a shift GAINED,
    how high was the lowest joint above the OLD (global) floor?  A gain at
    +15 cm would be contact invented in mid-air; a gain at +4 cm is the wall."""
    names = np.array([s["name"] for s in seqs])
    sess = np.array([CL.session(n) for n in names])
    acts, det = sess_report(npz_dir, embs)
    rep = {}
    for e in embs:
        E = Emb2(cache, npz_dir, e, seqs)
        sh = [s for s in acts if acts[s].get(e) == "shift"]
        if not sh:
            del E; continue
        cidx = np.flatnonzero(np.isin(sess, sh))
        rows = np.concatenate([np.arange(E.starts[c], E.starts[c] + E.lens[c]) for c in cidx])
        lo = E.lowg[rows].any(1); la = E.low[rows].any(1)
        g = rows[la & ~lo]
        if not len(g):
            del E; continue
        gmin_b = (E.gap[g].astype(np.float32).min(1) + E.h0row[g] - E.h0) * 100
        allb = (E.gap[rows].astype(np.float32).min(1) + E.h0row[rows] - E.h0) * 100
        rep[e] = dict(thr_cm=E.thr * 100, n_shift_sess=len(sh), n_frames=int(len(rows)),
                      n_gained=int(len(g)), frac_gained=float(len(g) / len(rows)),
                      gained_height_cm=dict(p50=float(np.percentile(gmin_b, 50)),
                                            p95=float(np.percentile(gmin_b, 95)),
                                            p99=float(np.percentile(gmin_b, 99)),
                                            max=float(gmin_b.max())),
                      session_airborne_p99_cm=float(np.percentile(allb, 99)),
                      session_airborne_max_cm=float(allb.max()),
                      gained_above_2thr=float((gmin_b > 2 * E.thr * 100).mean()),
                      gained_above_3thr=float((gmin_b > 3 * E.thr * 100).mean()))
        print(f"[flight] {e:20s} shift-sessions={len(sh):3d} gained {len(g):7d} frames "
              f"({100*len(g)/len(rows):5.2f}%)  height above OLD floor: "
              f"p50={rep[e]['gained_height_cm']['p50']:5.2f} p99={rep[e]['gained_height_cm']['p99']:5.2f} "
              f"max={rep[e]['gained_height_cm']['max']:6.2f} cm  (thr={E.thr*100:.1f}, "
              f"session max airborne={allb.max():.0f} cm)   >3thr: "
              f"{100*rep[e]['gained_above_3thr']:.2f}%", flush=True)
        del E
    os.makedirs(out, exist_ok=True)
    json.dump(rep, open(f"{out}/{corp}_flight_audit.json", "w"), indent=1)
    return rep


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="amass")
    ap.add_argument("--npz", default=None, help="staging dir (default %s/<cache>)" % STAGE)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--do", default="coverage", help="coverage,beforeafter,flight")
    ap.add_argument("--drop", default="pal_talos")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nfr", type=int, default=90)
    ap.add_argument("--no_videos", action="store_true")
    ap.add_argument("--sessions", default="")
    ap.add_argument("--speed_clip", default="aiming")
    ap.add_argument("--speed_mode", default="static", choices=["static", "veto"])
    ap.add_argument("--ba_video_embs",
                    default="human_smpl,unitree_g1_29dof,unitree_h1_2,fourier_gr3,engineai_pm01",
                    help="before/after VIDEOS only for these rigs (sheets are made for every "
                         "rig the correction touched); 'all' for everything")
    a = ap.parse_args()
    corp_key = a.cache
    cache = CL.CACHES.get(a.cache, a.cache)
    corp = os.path.basename(cache)
    npz_dir = a.npz or f"{STAGE}/{corp_key}"
    seqs = json.load(open(f"{cache}/sequences.json"))
    fams = np.array([CL.family(s["name"]) for s in seqs])
    ufam = sorted(set(fams.tolist()))
    drop = set(x for x in a.drop.split(",") if x)
    embs = [e for e in json.load(open(f"{cache}/manifest.json"))["embodiments"]
            if e not in drop and os.path.exists(f"{npz_dir}/{e}_contact.npz")]
    rng = np.random.default_rng(a.seed)
    made = []
    todo = a.do.split(",")
    if "coverage" in todo:
        made += run_coverage(corp, cache, npz_dir, a.out, embs, seqs, fams, ufam,
                             seed=a.seed, nfr=a.nfr, videos=not a.no_videos)
    if "flight" in todo:
        flight_audit(corp, cache, npz_dir, a.out, embs, seqs, None)
    if "speedgate" in todo:
        made += speed_probe(corp, cache, npz_dir, a.out,
                            [x for x in ("human_smpl", "unitree_g1_29dof", "unitree_h1_2")
                             if x in embs], seqs, clip_substr=a.speed_clip, nfr=a.nfr,
                            mode=a.speed_mode)
    if "beforeafter" in todo:
        acts, _ = sess_report(npz_dir, embs)
        want = ([s for s in a.sessions.split(",") if s] or
                sorted({s for s in acts if any(v in ("shift", "untrusted")
                                               for v in acts[s].values())}))
        ve = (None if a.ba_video_embs == "all"
              else set(x for x in a.ba_video_embs.split(",") if x))
        made += run_beforeafter(corp, cache, npz_dir, a.out, embs, seqs, fams, want, rng,
                                nfr=a.nfr, video_embs=ve)
    json.dump(made, open(f"{a.out}/{corp}_{a.do.replace(',','_')}_files.json", "w"), indent=1)
    print(f"\n{len(made)} files written under {a.out}")



# =====================================================================
#  SPEED-GATE PROBE : is the 0.10 m/s veto catching real motion or jitter?
# =====================================================================
def speed_probe(corp, cache, npz_dir, out, embs, seqs, clip_substr="aiming",
                nfr=90, cellpx=380, cols=3, mode="static"):
    """Render the stretch where the speed gate is MOST at risk of being wrong:
    the window with the LOWEST mean foot speed (mode='static') in the clips whose
    name contains `clip_substr`.  Every marked joint is labelled
    `gap_cm / speed_cm_per_s`, so lloyd can read straight off the picture whether
    an ORANGE (vetoed) foot is actually moving or is retarget jitter sitting just
    over the 10 cm/s line."""
    from common import fk_global as _fk
    names = [s["name"] for s in seqs]
    hit = [i for i, n in enumerate(names) if clip_substr.lower() in n.lower()]
    if not hit:
        print(f"[speed] no clip matching {clip_substr!r}", flush=True); return []
    P = Panel(cellpx, cellpx)
    made = []
    for e in embs:
        E = Emb2(cache, npz_dir, e, seqs)
        lg = np.array(E.hdr["limb_groups"])
        feet = np.flatnonzero((lg == "L_foot") | (lg == "R_foot"))
        if not len(feet):
            feet = np.arange(E.spec.J)
        best = None
        for ci in hit:
            s0, ln = int(E.starts[ci]), int(E.lens[ci])
            if ln < nfr: continue
            rows = np.arange(s0, s0 + ln)
            W = E.world(rows)
            v = np.linalg.norm(np.gradient(W[:, :, :2], 1 / CL.FPS, axis=0), axis=-1)
            if mode == "veto":
                # the window carrying the MOST speed-gate vetoes on the feet
                nv = (E.low[rows][:, feet] & ~E.con[rows][:, feet]).sum(1).astype(np.float32)
                k = np.convolve(nv, np.ones(nfr) / nfr, "valid"); t = int(np.argmax(k))
                score = -float(k[t])
            else:
                fs = v[:, feet].mean(1)
                k = np.convolve(fs, np.ones(nfr) / nfr, "valid")
                t = int(np.argmin(k)); score = float(k[t])
            if best is None or score < best[0]:
                best = (score, ci, s0 + t, W[t:t + nfr], v[t:t + nfr],
                        float(v[t:t + nfr][:, feet].mean()))
        if best is None:
            del E; continue
        _sc, ci, r0, W, V, mfs = best
        fr = []
        for t in range(nfr):
            r = r0 + t
            gap = E.gap[r].astype(np.float32); con = E.con[r]
            vet = E.low[r] & ~con
            cand = (~con) & (~vet) & (gap < 4 * E.thr)
            img = P.draw(W[t], E.spec.parents, E.thr, con, vet, cand, gap,
                         untrusted=not bool(E.trust[r]))
            fr.append(caption(img, [
                f"{e}   [{corp}]   SPEED-GATE PROBE",
                f"{names[ci][:48]}  f{r - E.starts[ci]}   {nfr}-frame window "
                f"({'most STATIC' if mode=='static' else 'most VETOED'} window; mean foot speed {mfs*100:.1f} cm/s)",
                f"contact {int(con.sum())}j   SPEED-VETOED {int(vet.sum())}j   "
                f"contact_low {int(E.low[r].sum())}j",
                f"foot speed now: " + "  ".join(
                    f"{E.spec.joint_names[j][:14]}={V[t, j]*100:.0f}" for j in feet[:4])
                + f"  cm/s   (gate = 10 cm/s)"],
                {2: (250, 120, 120), 3: (255, 200, 90)}))
        made.append(write_mp4(f"{out}/speedgate/{corp}_{clip_substr}_{mode}_{e}.mp4", fr))
        # a still sheet of the 6 frames with the most vetoes in that window
        nv = np.array([int((E.low[r0 + t] & ~E.con[r0 + t]).sum()) for t in range(nfr)])
        sel = np.argsort(-nv)[:6]
        made.append(tile([fr[t] for t in sorted(sel)], cols,
                         f"{out}/speedgate/{corp}_{clip_substr}_{mode}_{e}.png",
                         f"{corp} / {e} / {names[ci]} — the 6 most-vetoed frames of the {mode} window",
                         sub=f"mean foot speed over the window {mfs*100:.1f} cm/s; the speed gate fires at 10 cm/s. "
                             f"ORANGE = vetoed. Read the bottom line for each foot's actual speed."))
        print(f"[speed] {e}: static window mean foot speed {mfs*100:.1f} cm/s, "
              f"max vetoes in a frame {nv.max()}", flush=True)
        del E
    return [m for m in made if m]

if __name__ == "__main__":
    main()
