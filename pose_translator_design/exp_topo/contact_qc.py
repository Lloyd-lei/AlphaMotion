"""Visual QC for the Greenwich-v2 contact labels.

Renders the labelled skeleton over the checker floor that the LABELLER used
(the plane is drawn exactly at the embodiment's own floor h0, so what you see is
what the labeller measured against) and, at every candidate joint, draws a
TRANSLUCENT SPHERE OF RADIUS = thr (the contact distance threshold) plus the gap
in cm.  A joint whose sphere touches the floor is exactly a joint the height gate
admits; the colour then says whether the velocity gate + hysteresis agreed.

  red translucent sphere     labelled in contact                        (gap printed)
  orange translucent sphere  low enough, but the 0.10 m/s SPEED GATE vetoed it
  white translucent sphere   near the floor, not admitted by either gate
  small orange dot           joint, no marker (far from the floor)
  cyan bone                  skeleton

Checker floor + lighting recipe copied from exp_topo/tri_embod_ground.py.
MuJoCo is Z-up and the caches are ardy Y-up, so positions go through @ AX.
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
from prep_gmr_cache import AX          # v_ardy = AX @ v_mj  ->  p_mj = p_ardy @ AX

torch.set_grad_enabled(False)
W = H = 460
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def add_checker_floor(sp):
    tx = sp.add_texture()
    tx.name = "_checker"; tx.type = mj.mjtTexture.mjTEXTURE_2D
    tx.builtin = mj.mjtBuiltin.mjBUILTIN_CHECKER
    tx.rgb1 = [0.2, 0.3, 0.45]; tx.rgb2 = [0.85, 0.85, 0.9]
    tx.mark = mj.mjtMark.mjMARK_EDGE; tx.markrgb = [0.1, 0.1, 0.1]
    tx.width = 512; tx.height = 512
    mt = sp.add_material()
    mt.name = "_checker_mat"; mt.texrepeat = [1, 1]; mt.texuniform = True; mt.reflectance = 0
    mt.textures[mj.mjtTextureRole.mjTEXROLE_RGB] = "_checker"
    g = sp.worldbody.add_geom(name="_floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[30, 30, 0.1])
    g.material = "_checker_mat"; g.contype = 1; g.conaffinity = 1
    sky = sp.add_texture()
    sky.name = "_sky"; sky.type = mj.mjtTexture.mjTEXTURE_SKYBOX
    sky.builtin = mj.mjtBuiltin.mjBUILTIN_GRADIENT
    sky.rgb1 = [0.55, 0.7, 0.9]; sky.rgb2 = [0.9, 0.93, 0.97]
    sky.width = 512; sky.height = 512
    sun = sp.worldbody.add_light(name="_sun", pos=[0, -1.5, 3.5], dir=[0, 0.3, -1],
                                 diffuse=[0.7, 0.7, 0.7], castshadow=True)
    sun.type = mj.mjtLightType.mjLIGHT_DIRECTIONAL
    sp.visual.global_.offwidth = max(W, 1280); sp.visual.global_.offheight = max(H, 1280)
    sp.visual.headlight.ambient = [0.35, 0.35, 0.35]
    sp.visual.headlight.diffuse = [0.5, 0.5, 0.5]


class Panel:
    def __init__(self, w=W, h=H):
        sp = mj.MjSpec(); add_checker_floor(sp)
        self.m = sp.compile(); self.d = mj.MjData(self.m)
        mj.mj_forward(self.m, self.d)
        self.ren = mj.Renderer(self.m, h, w)
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        self.opt.label = mj.mjtLabel.mjLABEL_NONE      # our geoms carry their own

    def draw(self, P, parents, thr, contact, veto, cand, dist, azim=125, elev=-12):
        """P [J,3] world metres in MUJOCO frame with the labelled floor at z=0."""
        lo = P.min(0); hi = P.max(0); mid = (lo + hi) / 2
        self.cam.lookat[:] = [mid[0], mid[1], max(0.30, mid[2])]
        self.cam.distance = max(1.5, 0.95 * float(np.linalg.norm(hi - lo)) + 0.35)
        self.cam.elevation, self.cam.azimuth = elev, azim
        self.ren.update_scene(self.d, self.cam, self.opt)
        sc = self.ren.scene
        def add(gtype, size, pos, rgba, label=""):
            if sc.ngeom >= sc.maxgeom: return None
            g = sc.geoms[sc.ngeom]; sc.ngeom += 1
            mj.mjv_initGeom(g, gtype, np.asarray(size, np.float64), np.asarray(pos, np.float64),
                            np.eye(3).flatten(), np.asarray(rgba, np.float32))
            if label: g.label = label
            return g
        for j, p in enumerate(parents):
            if p < 0: continue
            g = add(mj.mjtGeom.mjGEOM_CAPSULE, [0, 0, 0], np.zeros(3), [0.25, 0.72, 0.85, 1])
            if g is not None:
                mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, 0.012, P[p], P[j])
        nmark = int(contact.sum() + veto.sum() + cand.sum())
        for j in range(len(parents)):
            if contact[j]:
                add(mj.mjtGeom.mjGEOM_SPHERE, [thr] * 3, P[j], [0.95, 0.15, 0.15, 0.32])
                add(mj.mjtGeom.mjGEOM_SPHERE, [thr * 0.35] * 3, P[j], [1.0, 0.1, 0.1, 1.0],
                    f"{dist[j]*100:.1f}")
            elif veto[j]:
                add(mj.mjtGeom.mjGEOM_SPHERE, [thr] * 3, P[j], [1.0, 0.60, 0.02, 0.46])
                add(mj.mjtGeom.mjGEOM_SPHERE, [thr * 0.45] * 3, P[j], [1.0, 0.62, 0.0, 1.0],
                    f"{dist[j]*100:.1f}")
            elif cand[j]:
                add(mj.mjtGeom.mjGEOM_SPHERE, [thr] * 3, P[j], [1.0, 1.0, 1.0, 0.22])
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
        c = (colors or {}).get(i + 1, (185, 190, 200))
        dr.text((7, 23 + 15 * i), t, c, font=f1)
    return np.asarray(out)


def tile(imgs, cols, path, title):
    if not imgs: return
    rows = (len(imgs) + cols - 1) // cols
    h, w = imgs[0].shape[:2]
    narrow = True                       # title on line 1, legend on line 2 (never clipped)
    hdr = 60
    sheet = Image.new("RGB", (cols * w, rows * h + hdr), (8, 8, 11))
    dr = ImageDraw.Draw(sheet)
    tf = ImageFont.truetype(FONTB, 19 if not narrow else 15)
    dr.text((10, 8), title if not narrow else title[:200], (255, 255, 255), font=tf)
    lg = ("legend  (every ball has radius = thr, the contact threshold; number = gap in cm above the floor):   "
          "RED = in `contact` (official channel)   |   "
          "ORANGE = height gate OK, 0.10 m/s SPEED GATE vetoed  ->  NOT in `contact` but IS in "
          "`contact_low`, the channel chosen for TRAINING   |   WHITE = near the floor, admitted by neither")
    dr.text((10, 34) if narrow else (max(10, cols * w - 1400), 11), lg,
            (170, 180, 195), font=ImageFont.truetype(FONT, 11 if narrow else 12))
    for i, im in enumerate(imgs):
        sheet.paste(Image.fromarray(im), ((i % cols) * w, hdr + (i // cols) * h))
    sheet.save(path)
    print(f"  sheet -> {path}  ({len(imgs)} cells)", flush=True)


# ------------------------------------------------------------------ data
class Emb:
    def __init__(self, cache, emb):
        self.emb = emb
        self.spec = CL.load_spec(cache, emb)
        z = np.load(f"{cache}/{emb}_contact.npz", allow_pickle=True)
        self.hdr = json.loads(str(z["header"]))
        self.con = z["contact"]; self.gap = z["gap"]
        self.low = z["contact_low"] if "contact_low" in z.files else self.con
        self.L = torch.load(f"{cache}/{emb}_paired_local6d.pt", mmap=True)
        self.RW = torch.load(f"{cache}/{emb}_root_world.pt", mmap=True).float()
        self.sid = np.array(self.hdr["site_id"])
        self.nsite = self.hdr["n_sites"]
        self.thr = self.hdr["thr_m"]; self.h0 = self.hdr["h0_m"]
        self.clip_floor = z["clip_floor"] if "clip_floor" in z.files else None

    def sites(self, rows):
        c = self.con[rows]
        return np.stack([c[:, self.sid == g].any(1) for g in range(self.nsite)], 1).sum(1)

    def world(self, rows, floor=None):
        """[n,J,3] MuJoCo frame (Z-up) with the labelled floor at z = 0.
        `floor` overrides the global h0 (used for the per-clip-floor comparison)."""
        b = self.L[rows].float().cuda()
        _, gp = fk_global(b, self.spec)
        w = (gp.cpu() + self.RW[rows][:, None, :]) / 100.0
        w[:, :, CL.UP] -= (self.h0 if floor is None else floor)
        return (w.numpy() @ AX)

    def low_at_floor(self, ci, starts, lens, floor):
        """contact_low (the TRAINING channel: height gate + hysteresis + despeckle,
        NO speed gate) recomputed for clip `ci` against an arbitrary floor."""
        s0, ln = int(starts[ci]), int(lens[ci])
        g = self.gap[s0:s0 + ln].astype(np.float32) + self.h0 - float(floor)
        br = np.array([0], np.int64)          # `breaks` is an INDEX array of clip starts
        ms = int(round(CL.MIN_SEG_S * CL.FPS))
        return g, CL.despeckle(CL.hysteresis(g < self.thr, g > 1.5 * self.thr, br), br, ms)


def frame_cell(E, row, corpus, clip, fam, panel):
    P = E.world(np.array([row]))[0]
    gap = E.gap[row].astype(np.float32); con = E.con[row]
    veto = E.low[row] & ~con
    cand = (~con) & (~veto) & (gap < 4 * E.thr)
    ns = int(E.sites(np.array([row]))[0])
    img = panel.draw(P, E.spec.parents, E.thr, con, veto, cand, gap)
    return caption(img, [
        f"{E.emb}   [{corpus}]",
        f"{clip}  f{row}   family={fam}",
        f"contact {int(con.sum())}j / {ns} sites   +speed-vetoed {int(veto.sum())}  "
        f"=> contact_low(TRAIN) {int(con.sum()) + int(veto.sum())}j   "
        f"thr={E.thr*100:.1f}cm  h0={E.h0*100:.2f}cm",
    ], {2: (250, 120, 120)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="lafan")
    ap.add_argument("--out", default="/home/arenalabs/tmp/contact_qc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--videos", action="store_true")
    ap.add_argument("--extra", action="store_true",
                    help="two diagnostic sheets: (a) below-floor frames of a negative-h0 rig "
                         "next to its twin, (b) clips that never register a contact")
    ap.add_argument("--pen_pair", default="unitree_g1_23dof,unitree_g1_29dof")
    ap.add_argument("--hover_embs", default="tienkung,unitree_h1,unitree_h1_2")
    ap.add_argument("--drop", default="pal_talos",
                    help="embodiments never rendered (bad URDF etc.)")
    a = ap.parse_args()
    cache = CL.CACHES.get(a.cache, a.cache)
    corpus = os.path.basename(cache)
    os.makedirs(a.out, exist_ok=True)
    seqs = json.load(open(f"{cache}/sequences.json"))
    embs = json.load(open(f"{cache}/manifest.json"))["embodiments"]
    have = [e for e in embs if os.path.exists(f"{cache}/{e}_contact.npz")]
    if len(have) != len(embs):
        print(f"[qc] WARNING  {len(embs)-len(have)} of {len(embs)} embodiments have NO "
              f"<emb>_contact.npz in {cache} and are SKIPPED: "
              f"{[e for e in embs if e not in have]}", flush=True)
    drop = [x for x in a.drop.split(",") if x]
    if any(e in have for e in drop):
        print(f"[qc] dropping on request: {[e for e in drop if e in have]}", flush=True)
    embs = [e for e in have if e not in drop]
    if not embs:
        raise SystemExit(f"[qc] no <emb>_contact.npz under {cache}")
    fams = np.array([CL.family(s["name"]) for s in seqs])
    ufam = sorted(set(fams.tolist()))
    rng = np.random.default_rng(a.seed)
    panel = Panel()
    grid = {}          # (emb, key) -> image
    picks = []

    # The 18 caches are ROW-ALIGNED, so the family cells use the SAME clip and the
    # SAME frame for every embodiment: one instant, 18 rigs, directly comparable.
    ref_name = "human_smpl" if "human_smpl" in embs else embs[0]
    ref = Emb(cache, ref_name)
    print(f"[qc] family reference embodiment = {ref_name}", flush=True)
    fam_rows = {}
    for f in ufam:
        # a frame that is CHARACTERISTIC of the family: random among the frames whose
        # human_smpl contact-site count sits in the family clip's top quartile (an
        # argmax pick lands on a freak frame; a uniform pick lands on flight phase).
        ci = int(rng.choice(np.flatnonzero(fams == f)))
        s0, ln = seqs[ci]["start"], seqs[ci]["len"]
        cand_rows = s0 + rng.choice(ln, size=min(ln, 3000), replace=False)
        ns = ref.sites(cand_rows)
        q = np.percentile(ns, 75)
        pool = cand_rows[ns >= max(q, 1)]
        r = int(rng.choice(pool if len(pool) else cand_rows))
        fam_rows[f] = (ci, r)
    del ref

    for e in embs:
        E = Emb(cache, e)
        for f in ufam:
            ci, r = fam_rows[f]
            grid[(e, f"fam:{f}")] = frame_cell(E, r, corpus, seqs[ci]["name"], f, panel)
            picks.append((e, f, seqs[ci]["name"], r, int(E.sites(np.array([r]))[0])))
        probe = rng.choice(E.con.shape[0], size=min(E.con.shape[0], 60000), replace=False)
        ns = E.sites(probe)
        for k, sel in (("1", ns == 1), ("2", ns == 2), ("3", ns == 3), ("4+", ns >= 4)):
            idx = probe[sel]
            if not len(idx):
                continue
            r = int(rng.choice(idx))
            ci = int(np.searchsorted([q["start"] for q in seqs], r, "right") - 1)
            grid[(e, f"n:{k}")] = frame_cell(E, r, corpus, seqs[ci]["name"], fams[ci], panel)
        print(f"[qc] {e} rendered", flush=True)
        del E
    # --- sheets by embodiment
    d1 = f"{a.out}/by_embodiment"; os.makedirs(d1, exist_ok=True)
    for e in embs:
        keys = [f"fam:{f}" for f in ufam] + [f"n:{k}" for k in ("1", "2", "3", "4+")]
        ims = [grid[(e, k)] for k in keys if (e, k) in grid]
        tile(ims, 5, f"{d1}/{corpus}_{e}.png",
             f"{corpus} / {e} — one frame per motion family + contact-count strata (1/2/3/4+ sites)")
    # --- sheets by family
    d2 = f"{a.out}/by_family"; os.makedirs(d2, exist_ok=True)
    for f in ufam:
        ims = [grid[(e, f"fam:{f}")] for e in embs if (e, f"fam:{f}") in grid]
        tile(ims, 6, f"{d2}/{corpus}_{f}.png", f"{corpus} / family={f} — all {len(ims)} labelled embodiments")
    for k in ("1", "2", "3", "4+"):
        ims = [grid[(e, f"n:{k}")] for e in embs if (e, f"n:{k}") in grid]
        tile(ims, 6, f"{d2}/{corpus}_count_{k}.png",
             f"{corpus} / exactly {k} simultaneous contact site(s) — all {len(ims)} labelled embodiments")
    json.dump([dict(emb=p[0], family=p[1], clip=p[2], row=p[3], sites=p[4]) for p in picks],
              open(f"{a.out}/{corpus}_qc_picks.json", "w"), indent=1)

    if a.extra:
        extra_sheets(cache, corpus, seqs, fams, embs, a, rng)

    if a.videos:
        make_videos(cache, corpus, seqs, fams, a.out, avail=embs)


# ------------------------------------------------------- diagnostic sheets
def _cell(panel, E, row, corpus, clip, fam, extra, extra_col=(255, 210, 90)):
    P = E.world(np.array([row]))[0]
    gap = E.gap[row].astype(np.float32); con = E.con[row]
    veto = E.low[row] & ~con
    cand = (~con) & (~veto) & (gap < 4 * E.thr)
    img = panel.draw(P, E.spec.parents, E.thr, con, veto, cand, gap)
    ns = int(E.sites(np.array([row]))[0])
    return caption(img, [
        f"{E.emb}   [{corpus}]",
        f"{clip[:52]}  f{row}  family={fam}",
        f"contacts {int(con.sum())}j / {ns} sites   vetoed {int(veto.sum())}   "
        f"thr={E.thr*100:.1f}cm  h0={E.h0*100:.2f}cm",
        extra,
    ], {2: (250, 120, 120), 3: extra_col})


def extra_sheets(cache, corpus, seqs, fams, embs, a, rng):
    """(a) the below-floor frames of a negative-h0 rig, beside its twin;
       (b) clips that never register a single contact."""
    panel = Panel()
    d = f"{a.out}/by_family"; os.makedirs(d, exist_ok=True)
    starts = np.array([s["start"] for s in seqs]); lens = np.array([s["len"] for s in seqs])
    def ci_of(r): return int(np.searchsorted(starts, r, "right") - 1)

    # ---- (a) penetration -------------------------------------------------
    pen = [e for e in a.pen_pair.split(",") if e in embs]
    if len(pen) == 2:
        A = Emb(cache, pen[0]); B = Emb(cache, pen[1])
        n = A.con.shape[0]
        probe = np.sort(rng.choice(n, size=min(n, 150000), replace=False))
        # ardy-world height of the lowest joint, measured against the CACHE floor y=0
        ylow = A.gap[probe].astype(np.float32).min(1) + A.h0
        nlab = A.con[probe].sum(1)
        miss = probe[(ylow < 0) & (nlab == 0)]      # under the floor, labelled nothing
        hit = probe[(ylow < 0) & (nlab > 0)]        # under the floor, labelled
        pick = (list(rng.choice(miss, size=min(5, len(miss)), replace=False)) +
                list(rng.choice(hit, size=min(3, len(hit)), replace=False)))
        print(f"[qc] penetration sheet: {pen[0]}  below-floor frames "
              f"{100*np.mean(ylow < 0):.1f}% of probe, of which "
              f"{100*np.mean(nlab[ylow < 0] > 0):.1f}% carry a label", flush=True)
        ims = []
        for r in sorted(int(x) for x in pick):
            ci = ci_of(r)
            for E in (A, B):
                y = float(E.gap[r].astype(np.float32).min() + E.h0)
                tag = "UNDER THE CACHE FLOOR" if y < 0 else "above y=0"
                ims.append(_cell(panel, E, r, corpus, seqs[ci]["name"], fams[ci],
                                 f"lowest joint y={y*100:+.2f}cm  ({tag})   "
                                 f"gap_min={float(E.gap[r].astype(np.float32).min())*100:+.2f}cm",
                                 (255, 120, 120) if y < 0 else (150, 220, 150)))
        tile(ims, 4, f"{d}/{corpus}_penetration_{pen[0]}.png",
             f"{corpus} / {pen[0]} (h0={A.h0*100:+.2f}cm) vs its twin {pen[1]} "
             f"(h0={B.h0*100:+.2f}cm) — SAME rows, frames whose lowest joint is below the cache floor y=0. "
             f"Top 5 pairs = {pen[0]} labelled NOTHING there; last 3 = it did label.")
        del A, B

    # ---- (b) clips that never touch --------------------------------------
    ims = []
    for e in [x for x in a.hover_embs.split(",") if x in embs]:
        E = Emb(cache, e)
        any_j = E.con.any(1)
        per = np.add.reduceat(any_j.astype(np.int32), starts)
        zero = np.flatnonzero(per == 0)
        if not len(zero):
            del E; continue
        order = zero[np.argsort(-lens[zero])][:60]         # the longest silent clips
        for ci in [int(x) for x in rng.choice(order, size=min(3, len(order)), replace=False)]:
            r = int(starts[ci] + lens[ci] // 2)
            cf = float(E.clip_floor[ci]) if E.clip_floor is not None else float("nan")
            gmin = float(E.gap[r].astype(np.float32).min())
            ims.append(_cell(panel, E, r, corpus, seqs[ci]["name"], fams[ci],
                             f"ZERO CONTACT over all {lens[ci]} frames   "
                             f"gap_min={gmin*100:+.1f}cm (thr={E.thr*100:.1f}cm)   "
                             f"clip floor {cf*100:+.2f}cm vs h0 {E.h0*100:+.2f}cm"
                             if cf == cf else
                             f"ZERO CONTACT over all {lens[ci]} frames   gap_min={gmin*100:+.1f}cm",
                             (120, 200, 255)))
        print(f"[qc] hover sheet: {e}  {len(zero)} silent clips "
              f"({100*len(zero)/len(seqs):.1f}%)", flush=True)
        del E
    if ims:
        tile(ims, 3, f"{d}/{corpus}_hover_zero_contact.png",
             f"{corpus} — clips that register ZERO contact from first frame to last "
             f"(middle frame of the longest such clips)")

    # ---- (c) global h0 floor vs the clip's own clip_floor ----------------
    ims = []
    for e in [x for x in a.hover_embs.split(",") if x in embs]:
        E = Emb(cache, e)
        any_j = E.con.any(1)
        per = np.add.reduceat(any_j.astype(np.int32), starts)
        cf = np.asarray(E.clip_floor, np.float32)
        off = cf - E.h0                                   # how far the clip floats
        bad = np.flatnonzero((per == 0) & (lens >= 60))
        if not len(bad):
            del E; continue
        bad = bad[np.argsort(-off[bad])][:3]              # the worst offenders
        for ci in [int(x) for x in bad]:
            r = int(starts[ci] + lens[ci] // 2)
            src = seqs[ci]["name"].split("__")[0]
            g2, low2 = E.low_at_floor(ci, starts, lens, cf[ci])
            k = r - int(starts[ci])
            # LEFT: the floor the labels actually used
            P = E.world(np.array([r]))[0]
            gap = E.gap[r].astype(np.float32); con = E.con[r]; veto = E.low[r] & ~con
            cand = (~con) & (~veto) & (gap < 4 * E.thr)
            ims.append(caption(
                panel.draw(P, E.spec.parents, E.thr, con, veto, cand, gap),
                [f"{e}   [{corpus}]   src={src}",
                 f"{seqs[ci]['name'][:52]}  f{k}/{lens[ci]}",
                 f"FLOOR = GLOBAL h0 = {E.h0*100:+.2f}cm   (what the shipped labels use)",
                 f"contact {int(con.sum())}j  contact_low {int(E.low[r].sum())}j   "
                 f"clip total {int(per[ci])} contact-frames   gap_min={gap.min()*100:+.1f}cm"],
                {2: (255, 170, 90), 3: (250, 120, 120)}))
            # RIGHT: the clip's own floor, contact_low recomputed against it
            P2 = E.world(np.array([r]), floor=float(cf[ci]))[0]
            g = g2[k]; c2 = low2[k]
            cnd = (~c2) & (g < 4 * E.thr)
            ims.append(caption(
                panel.draw(P2, E.spec.parents, E.thr, c2, np.zeros_like(c2), cnd, g),
                [f"{e}   [{corpus}]   src={src}",
                 f"{seqs[ci]['name'][:52]}  f{k}/{lens[ci]}",
                 f"FLOOR = THIS CLIP'S OWN clip_floor = {cf[ci]*100:+.2f}cm  "
                 f"(h0{off[ci]*100:+.2f}cm)",
                 f"contact_low RECOMPUTED: {int(c2.sum())}j here, "
                 f"{int(low2.any(1).sum())}/{lens[ci]} frames in the clip   "
                 f"gap_min={g.min()*100:+.1f}cm"],
                {2: (140, 220, 255), 3: (150, 230, 150)}))
        print(f"[qc] per-clip-floor sheet: {e} worst offsets "
              f"{[round(float(off[c])*100, 1) for c in bad]} cm", flush=True)
        del E
    if ims:
        tile(ims, 2, f"{d}/{corpus}_perclip_floor_ab.png",
             f"{corpus} — silent clips rendered TWICE: left = global h0 (shipped labels), "
             f"right = that clip's own clip_floor with contact_low recomputed against it. "
             f"RED on the right = would be labelled if the floor were per-clip.")


def make_videos(cache, corpus, seqs, fams, out, embs=("human_smpl", "unitree_g1_29dof"),
                avail=None):
    vd = f"{out}/videos"; os.makedirs(vd, exist_ok=True)
    if avail is not None:
        embs = tuple(e for e in embs if e in avail) or tuple(avail[:2])
    want = [(f, f) for f in ("walk", "ground", "fall", "getup", "sit")
            if (fams == f).any()][:3]
    panel = Panel(560, 560)
    E = {e: Emb(cache, e) for e in embs}
    for tag, fam in want:
        cands = np.flatnonzero(fams == fam)
        ci = int(cands[int(np.argmax([seqs[c]["len"] for c in cands]))])   # longest clip
        s0, ln = seqs[ci]["start"], seqs[ci]["len"]
        n = min(300, ln); off = ln // 3
        rows = np.arange(s0 + off, s0 + off + n)
        frames = []
        for t, r in enumerate(rows):
            cells = []
            for e in embs:
                Ee = E[e]
                P = Ee.world(np.array([r]))[0]
                gap = Ee.gap[r].astype(np.float32); con = Ee.con[r]
                veto = Ee.low[r] & ~con
                cand = (~con) & (~veto) & (gap < 4 * Ee.thr)
                img = panel.draw(P, Ee.spec.parents, Ee.thr, con, veto, cand, gap)
                ns = int(Ee.sites(np.array([r]))[0])
                cells.append(caption(img, [f"{e}  [{corpus}]",
                                           f"{seqs[ci]['name']}  f{r-s0}  family={fam}",
                                           f"contacts {int(con.sum())}j / {ns} sites  "
                                           f"speed-vetoed {int(veto.sum())}  "
                                           f"thr={Ee.thr*100:.1f}cm"], {2: (250, 120, 120)}))
            frames.append(np.concatenate(cells, 1))
        p = f"{vd}/{corpus}_{tag}.mp4"
        try:                                    # imageio-ffmpeg plugin
            iio.mimwrite(p, frames, fps=30, quality=8)
        except TypeError:                       # pyav plugin: no `quality` kwarg
            iio.mimwrite(p, frames, fps=30, codec="libx264",
                         out_pixel_format="yuv420p")
        print(f"  video -> {p}  ({len(frames)} frames)", flush=True)


if __name__ == "__main__":
    main()
