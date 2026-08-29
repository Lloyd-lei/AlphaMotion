"""THE CARNIVAL — one production video: motion x embodiment x task generalization.

    carnival.py

Four chapters, separated by title cards, concatenated into ONE mp4:

  Ch.1  ONE MOTION, EVERY BODY   the same val clip decoded from the same 125 bits
        onto every robot whose MJCF builds — in-training bodies, the held-out G1,
        and X2 which exists in no dataset. Tiled grid, every tile labeled with its
        generalization status.
  Ch.2  THE CHAIN               walk -> kick -> jacks: three UNSEEN clips chained by
        two model-invented bridges (donor tokens). Robots turn RED while the motion
        is invented. Three bodies side by side: adam (trained) / G1 (held out) /
        X2 (no dataset).
  Ch.3  THE CLOCK               same 125 bits, three speeds (0.5x/1x/2x) on X2;
        finished variants freeze.
  Ch.4  THE SKETCH              4 keyframes -> full kick on G1, next to the sketch
        it came from (token retrieval: right family 6/6).

Rendering: the same contract as every mesh video in this project — feasibility
projection into each robot's own limits, MuJoCo FK, feet on the floor.

Output: R4_VIZ/mp4/CARNIVAL.mp4 (+ per-chapter files alongside)
"""
from __future__ import annotations
import os, sys, glob, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
import mujoco as mj                                                  # noqa: E402
import v5_common as V                                                # noqa: E402
import v7_bridge as B7                                               # noqa: E402
import v7_bridge_mesh as BM                                          # noqa: E402
import eval_newbody as E                                             # noqa: E402
import render_newbody as RN                                          # noqa: E402
from train_v5 import load_run, to_states                             # noqa: E402

OUTD = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
        'R4_VIZ/mp4')
EMB = "human_smpl"
GMR = "/var/tmp/alphamotion/config_harvest/GMR/assets"
FPS, CANVAS = 20, (1680, 920)
RED = [0.82, 0.29, 0.36, 1]

# (key, xml, label, status)  status: TRAINED / HELD OUT / NO DATASET
ROSTER = [
    ("x2",     BM.TARGETS["x2"][0],                    "AgiBot X2",   "NO DATASET"),
    ("g1",     BM.TARGETS["g1"][0],                    "Unitree G1",  "HELD OUT"),
    ("adam",   BM.TARGETS["adam"][0],                  "PND Adam",    "TRAINED"),
    ("t1",     f"{GMR}/booster_t1/T1_locomotion.xml",  "Booster T1",  "TRAINED"),
    ("h1",     f"{GMR}/unitree_h1/h1_with_hand.xml",   "Unitree H1",  "TRAINED"),
    ("kuavo",  f"{GMR}/kuavo_s45/biped_s45_collision.xml", "Kuavo S45", "TRAINED"),
    ("hi",     f"{GMR}/hightorque_hi/hi_25dof.xml",    "HighTorque Hi", "TRAINED"),
    ("h1_2",   f"{GMR}/unitree_h1_2/h1_2_handless.xml", "Unitree H1-2", "TRAINED"),
]
ST_COL = {"TRAINED": (150, 155, 162), "HELD OUT": (240, 190, 80),
          "NO DATASET": (235, 100, 110)}


class Body:
    def __init__(self, key, xml, label, status, dev):
        self.key, self.label, self.status = key, label, status
        spec, dof, rest, qnames, xml2 = E.get_target(xml, E.CACHE_AMASS)
        assert qnames is not None
        g = E.target_geom(spec, dof, dev)
        self.spec, self.qnames, self.xml = spec, qnames, xml2
        self.ent = {"spec": spec, "st": g["st"], "sem": g["sem"], "dof": g["dof"]}
        self.dof = torch.as_tensor(dof, device=dev, dtype=torch.float64)
        self.rest = torch.as_tensor(rest, device=dev, dtype=torch.float64)


def solo_frames(body, streams_rgba, T, size, dist=3.1, red_mask=None):
    """streams_rgba: [(stream, rgba)]. One scene, n copies, returns raw frames."""
    n = len(streams_rgba)
    prefixes = [f"m{i}_" for i in range(n)]
    m = RN.build_scene(body.xml, prefixes, [c for _, c in streams_rgba],
                       size[0], size[1])
    d = mj.MjData(m)
    rend = mj.Renderer(m, height=size[1], width=size[0])
    cam = mj.MjvCamera()
    cam.lookat[:] = [0, 0, 0.8]
    cam.distance, cam.elevation, cam.azimuth = dist, -6, 180
    gid = [BM.geom_ids(m, p) for p in prefixes]
    base = [m.geom_rgba[x].copy() for x in gid]
    root_adr, slot_adr = [], []
    for p in prefixes:
        # Some vendor MJCFs leave the free joint UNNAMED (booster_t1, unitree_h1,
        # kuavo), so the name-prefix search comes back empty. The BODY carrying the
        # joint is always prefixed by mjSpec.attach, so fall back to body names.
        fr = [int(m.jnt_qposadr[j]) for j in range(m.njnt)
              if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE
              and ((mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or "").startswith(p)
                   or (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY,
                                     int(m.jnt_bodyid[j])) or "").startswith(p))]
        root_adr.append(fr[0])
        tab = -np.ones((body.spec.J, 3), np.int64)
        for j, ns in enumerate(body.qnames):
            for k2, nm2 in enumerate(ns):
                jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, p + nm2)
                if jid >= 0:
                    tab[j, k2] = int(m.jnt_qposadr[jid])
        slot_adr.append(tab)
    gap = 1.05
    ys = [(i - (n - 1) / 2) * gap for i in range(n)]
    frames = []
    for t in range(T):
        inb = bool(red_mask is not None and red_mask(t))
        for i, ((q, rootR, gp), _) in enumerate(streams_rgba):
            m.geom_rgba[gid[i]] = RED if inb else base[i]
            tt = min(t, q.shape[0] - 1)
            pm = gp[tt] @ RN.AX / 100.0
            d.qpos[root_adr[i]:root_adr[i] + 3] = [0.0, ys[i], -float(pm[:, 2].min())]
            d.qpos[root_adr[i] + 3:root_adr[i] + 7] = RN.mat_to_quat(
                RN.AX.T @ rootR[tt] @ RN.AX)
            tab = slot_adr[i]
            for j in range(body.spec.J):
                for k2 in range(3):
                    if tab[j, k2] >= 0:
                        d.qpos[tab[j, k2]] = q[tt, j, k2]
        mj.mj_forward(m, d)
        rend.update_scene(d, cam)
        frames.append(rend.render().copy())
    del rend, m, d
    return frames


def label_tile(img, title, sub, sub_col):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    W, H = im.size
    f = lambda s, b=True: ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if b else ""), s)
    dr.rectangle([0, H - 46, W, H], fill=(15, 15, 17))
    dr.text((10, H - 42), title, font=f(18), fill=(232, 234, 238))
    dr.text((10, H - 21), sub, font=f(14, False), fill=sub_col)
    return np.asarray(im)


def canvas_pad(fr):
    W, H = CANVAS
    h, w = fr.shape[:2]
    out = np.zeros((H, W, 3), np.uint8); out[:] = 12
    y, x = (H - h) // 2, (W - w) // 2
    out[y:y + h, x:x + w] = fr[:H, :W]
    return out


def banner(fr, text):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(fr)
    dr = ImageDraw.Draw(im)
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    dr.rectangle([0, 0, im.size[0], 46], fill=(15, 15, 17))
    dr.text((16, 8), text, font=f, fill=(235, 236, 240))
    return np.asarray(im)


def title_card(lines, secs=2.0):
    from PIL import Image, ImageDraw, ImageFont
    W, H = CANVAS
    im = Image.new("RGB", (W, H), (12, 12, 14))
    dr = ImageDraw.Draw(im)
    fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 54)
    fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26)
    y = H // 2 - 40 * len(lines)
    for i, ln in enumerate(lines):
        f = fb if i == 0 else fs
        w = dr.textlength(ln, font=f)
        dr.text(((W - w) / 2, y), ln, font=f, fill=(235, 236, 240))
        y += 74 if i == 0 else 40
    return [np.asarray(im)] * int(secs * FPS)


@torch.no_grad()
def main():
    dev = "cuda"
    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    tok, _ = V.load_frozen(dev)
    src = V.body_entry(V.AMASS, EMB, dev, kd.reach[EMB])
    kidx = kd.kidx[EMB]
    model = load_run(f"{V.RUNS}/V7_maskfsq_0807", dev)[0]     # the leading arm
    kn = torch.ones(1, device=dev)
    T = V.T_WIN
    g = np.random.default_rng(2027)
    bodies = []
    for key, xml, label, status in ROSTER:
        try:
            bodies.append(Body(key, xml, label, status, dev))
            print(f"[roster] {label:16s} OK  ({status})", flush=True)
        except Exception as e:
            print(f"[roster] {label:16s} SKIP: {type(e).__name__} {e}", flush=True)
    B = {b.key: b for b in bodies}

    def stream(body, seq):
        return BM.robot_stream(tok, src, body.ent, seq, kidx, body.spec,
                               body.dof, body.rest, dev)[:3]

    def dec(w, cur, tgt, T_out=None):
        e = model.encode(w, cur, tgt, kn)
        o = model.decode(e["path_z"], e["time_z"], cur, tgt, kn, T_out or w.shape[1])
        return to_states(o, model, cur, tgt, kn).float()[0].cpu()

    chapters = []

    # ---- Ch.1 one motion, every body --------------------------------------
    st, _ = kd.sample_starts(1, g, "val", span=T, families=["kick"])
    w = kd.window(EMB, st, T, T).to(dev).float()
    rec = dec(w, w[:, 0], w[:, -1])
    tiles = []
    tile_sz = (CANVAS[0] // 4, (CANVAS[1] - 46) // 2)
    for b in bodies[:8]:
        try:
            fr = solo_frames(b, [(stream(b, rec), [0.6, 0.62, 0.66, 1])], T, tile_sz,
                             dist=2.9)
            fr = [label_tile(x, b.label, b.status, ST_COL[b.status]) for x in fr]
            tiles.append(fr)
        except Exception as e:
            print(f"[ch1] {b.label} skip: {e}", flush=True)
    n = len(tiles)
    blank = np.full((tile_sz[1], tile_sz[0], 3), 12, np.uint8)
    while len(tiles) % 4:                       # rows must be equal width
        tiles.append([blank] * T)
    rows = [tiles[i:i + 4] for i in range(0, len(tiles), 4)]
    ch1 = []
    for t in range(T):
        rimgs = [np.concatenate([c[t] for c in r], 1) for r in rows if r]
        fr = np.concatenate(rimgs, 0)
        ch1.append(banner(canvas_pad(fr),
                          "Ch.1  ONE TOKEN STREAM, EVERY BODY — same 125 bits, "
                          f"unseen kick, {n} robots"))
    chapters.append((title_card(["ONE TOKEN STREAM, EVERY BODY",
                                 "the same 125 bits decoded onto every robot",
                                 "yellow = held out of training, red = in no dataset"]),
                     ch1))

    # ---- Ch.2 the chain ----------------------------------------------------
    la = 30
    sa, _ = kd.sample_starts(1, g, "val", span=la, families=["walk"])
    sb, _ = kd.sample_starts(1, g, "val", span=la, families=["kick"])
    sc, _ = kd.sample_starts(1, g, "val", span=la, families=["jacks"])
    A = kd.window(EMB, sa, la, la)[0]
    Bw = B7.yaw_align(kd.window(EMB, sb, la, la)[0], A[-1])
    Cw = B7.yaw_align(kd.window(EMB, sc, la, la)[0], Bw[-1])
    def bridge(a_end, b_start, fam):
        sd, _ = kd.sample_starts(1, g, "val", span=la, families=[fam])
        d0 = kd.window(EMB, sd, la, la)[None][0].to(dev).float()
        cur, tgt = a_end[None].to(dev).float(), b_start[None].to(dev).float()
        e = model.encode(d0, d0[:, 0], d0[:, -1], kn)
        o = model.decode(e["path_z"], e["time_z"], cur, tgt, kn, la)
        return to_states(o, model, cur, tgt, kn).float()[0].cpu()
    chain = torch.cat([A, bridge(A[-1], Bw[0], "kick"), Bw,
                       bridge(Bw[-1], Cw[0], "jacks"), Cw], 0)
    Tc = chain.shape[0]
    red = lambda t: (30 <= t < 60) or (90 <= t < 120)
    tiles2 = []
    for key in ("adam", "g1", "x2"):
        if key not in B:
            continue
        b = B[key]
        fr = solo_frames(b, [(stream(b, chain), [0.6, 0.62, 0.66, 1])], Tc,
                         (CANVAS[0] // 3, CANVAS[1] - 46), dist=3.2, red_mask=red)
        fr = [label_tile(x, b.label, b.status, ST_COL[b.status]) for x in fr]
        tiles2.append(fr)
    ch2 = []
    for t in range(Tc):
        fr = np.concatenate([c[t] for c in tiles2], 1)
        phase = ("INVENTED BRIDGE" if red(t) else
                 "walk (real)" if t < 30 else "kick (real)" if t < 90 else
                 "jacks (real)")
        ch2.append(banner(canvas_pad(fr),
                          f"Ch.2  THE CHAIN — walk>kick>jacks, bridges invented | "
                          f"f{t}  {phase}"))
    chapters.append((title_card(["THE CHAIN",
                                 "three unseen clips joined by two invented bridges",
                                 "robots turn RED while the model is inventing"]),
                     ch2))

    # ---- Ch.3 the clock ----------------------------------------------------
    if "x2" in B:
        b = B["x2"]
        st3, _ = kd.sample_starts(1, g, "val", span=T, families=["kick"])
        w3 = kd.window(EMB, st3, T, T).to(dev).float()
        cur, tgt = w3[:, 0], w3[:, -1]
        outs = [("0.5x", dec(w3, cur, tgt, 30)), ("1x", dec(w3, cur, tgt, 60)),
                ("2x", dec(w3, cur, tgt, 120))]
        tiles3 = []
        for lab, s in outs:
            fr = solo_frames(b, [(stream(b, s), [0.6, 0.62, 0.66, 1])], 120,
                             (CANVAS[0] // 3, CANVAS[1] - 46), dist=3.0)
            L = s.shape[0]
            fr = [label_tile(x, f"{lab}  ({L} frames)",
                             "DONE - holding" if t >= L else f"{min(t, L)}/{L}",
                             (150, 155, 162) if t >= L else (240, 240, 240))
                  for t, x in enumerate(fr)]
            tiles3.append(fr)
        ch3 = [banner(canvas_pad(np.concatenate([c[t] for c in tiles3], 1)),
                      "Ch.3  THE CLOCK — identical 125 bits, three speeds (X2, "
                      "no dataset)") for t in range(120)]
        chapters.append((title_card(["THE CLOCK",
                                     "the same tokens, decoded at three speeds",
                                     "a variant that finishes simply freezes"]), ch3))

    # ---- Ch.4 the sketch ---------------------------------------------------
    if "g1" in B:
        b = B["g1"]
        st4, _ = kd.sample_starts(1, g, "val", span=T, families=["kick"])
        w4 = kd.window(EMB, st4, T, T).to(dev).float()
        kf = np.round(np.linspace(0, T - 1, 4)).astype(int)
        segs = []
        for i in range(3):
            L = kf[i + 1] - kf[i] + 1
            s = model.baseline(w4[:, kf[i]], w4[:, kf[i + 1]], kn, L)
            segs.append(s[:, :-1] if i < 2 else s)
        sk = torch.cat(segs, 1)
        fill = dec(sk, sk[:, 0], sk[:, -1])
        trip = [("GT (never shown)", w4[0].cpu()), ("4-keyframe sketch", sk[0].cpu()),
                ("model fill", fill)]
        tiles4 = []
        for lab, s in trip:
            fr = solo_frames(b, [(stream(b, s), [0.6, 0.62, 0.66, 1])], T,
                             (CANVAS[0] // 3, CANVAS[1] - 46), dist=3.0)
            fr = [label_tile(x, lab, "", (150, 155, 162)) for x in fr]
            tiles4.append(fr)
        ch4 = [banner(canvas_pad(np.concatenate([c[t] for c in tiles4], 1)),
                      "Ch.4  THE SKETCH — 4 keyframes in, full kick out (G1, held "
                      "out); token retrieval: right family 6/6")
               for t in range(T)]
        chapters.append((title_card(["THE SKETCH",
                                     "four keyframes become a full motion",
                                     "and its token retrieves the right dataset family"]),
                         ch4))

    # ---- concat ------------------------------------------------------------
    allf = []
    for card, ch in chapters:
        allf += card + ch
    RN.write_mp4(f"{OUTD}/CARNIVAL.mp4", np.stack(allf), FPS)
    print(f"\nCARNIVAL: {OUTD}/CARNIVAL.mp4  ({len(allf)} frames)", flush=True)


if __name__ == "__main__":
    main()
