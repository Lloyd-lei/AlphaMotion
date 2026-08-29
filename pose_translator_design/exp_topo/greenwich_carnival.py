"""GREENWICH CARNIVAL — 20 embodiments, one codebook. Two productions.

    greenwich_carnival.py --video 1   # same UNSEEN motion on all 20 bodies
    greenwich_carnival.py --video 2   # SE(3): pin both wrists, CLAP, all 20 bodies

Named by lloyd 2026-08-09: PROD_spatial_0808 = "Greenwich" — the prime meridian
every longitude is measured from; here, the one codebook every body decodes from.

COLOR CODE (label band under every tile; lloyd: "数据集没出现过的标黄、红、蓝"):
  BLUE    trained      the body's paired data was in Greenwich's training set
  YELLOW  held out     paired data exists in the cache but was excluded (--holdout)
  RED     no dataset   the body appears in NO dataset anywhere; built from its
                       vendor MJCF at eval time. The hard zero-shot.

VIDEO 1  one AMASS clip (Greenwich trained on the LAFAN cache, so EVERY AMASS clip
         is motion-unseen), encoded ONCE from the human, decoded to all 20.
VIDEO 2  SE(3) constraint carnival: the encoder sees ONLY {root, both wrists} of
         SOMA__soma_subject2__clap_002 (a real clap — found in AMASS, no kimodo
         needed); everything else is a mask token the model must invent, then
         feasibility-projected into each robot's own limits.

Pipeline per tile = the only sanctioned one: encode -> decode to target skeleton ->
greedy+LM feasibility projection -> MuJoCo qpos -> EGL render. Fixed-base vendor
models (GR2 etc.) are lifted by their standing height via the attach frame.
Output: R4_VIZ/mp4/greenwich/*.mp4  (5x4 grid, 1600x1240)
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
import mujoco as mj                                                  # noqa: E402
import v5_common as V                                                # noqa: E402
import eval_newbody as E                                             # noqa: E402
import render_newbody as RN                                          # noqa: E402
import manifold_project as MP                                        # noqa: E402
from common import rot6d_to_matrix                                   # noqa: E402

OUT = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/'
       'R4_VIZ/mp4/greenwich')
GMR = "/var/tmp/alphamotion/config_harvest/GMR/assets"
RM = "/var/tmp/alphamotion/robot_models"
# 0812: cache_xwbc_amass (the old 4000-clip build) was deleted for disk;
# it had also stopped being "motion-unseen" once cache_xwbc_amass_full
# (13281 clips) superseded it as the training corpus. Point at the real
# held-out build instead, which is what the comment always intended.
SRC_CACHE = "/var/tmp/alphamotion/topo_exp/cache_amass_heldout"  # motion-unseen source

# (name, xml, status)   status: trained / heldout / nodataset
ROSTER = [
    ("Booster K1",   f"{GMR}/booster_k1/K1_serial.xml",              "trained"),
    ("Booster T1",   f"{GMR}/booster_t1/T1_locomotion.xml",          "trained"),
    ("Booster T1-29",f"{GMR}/booster_t1_29dof/t1_mocap.xml",         "trained"),
    ("Fourier GR3",  f"{GMR}/fourier_gr3v2_1_1/mjcf/gr3v2_1_1.xml",  "trained"),
    ("Fourier N1",   f"{GMR}/fourier_n1/n1_mocap.xml",               "trained"),
    ("HighTorque Hi",f"{GMR}/hightorque_hi/hi_25dof.xml",            "trained"),
    ("Kuavo S45",    f"{GMR}/kuavo_s45/biped_s45_collision.xml",     "trained"),
    ("PND Adam",     f"{GMR}/pnd_adam_lite/adam_lite.xml",           "trained"),
    ("TienKung",     f"{GMR}/tienkung/mjcf/tienkung.xml",                 "trained"),
    ("Unitree H1",   f"{GMR}/unitree_h1/h1_with_hand.xml",           "trained"),
    ("Unitree H1-2", f"{GMR}/unitree_h1_2/h1_2_handless.xml",        "trained"),
    ("Unitree G1",   f"{GMR}/unitree_g1/g1_mocap_29dof.xml",         "heldout"),
    ("G1 official",  f"{RM}/unitree_ros_official/robots/g1_description/g1_29dof.xml",
                                                                      "heldout"),
    ("PAL Talos",    f"{GMR}/pal_talos/talos.xml",                   "heldout"),
    ("AgiBot X2",    f"{RM}/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml",
                                                                      "nodataset"),
    ("Fourier GR2",  f"{RM}/Wiki-GRx-Models_FFTAI/GRX/GR2/gr2v3_8_7/mjcf/gr2v3_8_7.xml",
                                                                      "nodataset"),
    ("Apollo",       f"{RM}/menagerie/apptronik_apollo/apptronik_apollo.xml",
                                                                      "nodataset"),
    ("Galaxea R1Pro",f"{GMR}/galaxea_r1pro/r1_pro.xml",              "nodataset"),
    ("Stanford Toddy", f"{GMR}/stanford_toddy/toddy_mocap.xml",      "nodataset"),
]
COL = {"trained": (110, 170, 235), "heldout": (240, 190, 60),
       "nodataset": (235, 90, 90)}
TAG = {"trained": "TRAINED", "heldout": "HELD OUT", "nodataset": "NO DATASET"}
TILE_W, TILE_H, BAND = 384, 336, 44


STATUS_RGBA = {"trained": [0.45, 0.58, 0.85, 1.0],
               "heldout": [0.88, 0.72, 0.25, 1.0],
               "nodataset": [0.87, 0.34, 0.34, 1.0]}


def render_tile(xml, spec, qnames, q, rootR, gp, T, status="trained"):
    """One robot, one solo scene -> T frames. The ROBOT ITSELF is tinted by its
    generalization status (lloyd: label colour alone was unreadable), and the camera
    distance is normalized by the robot's standing height so a toddler-sized body
    and a 1.8 m body fill the tile equally."""
    m = RN.build_scene(xml, ["m0_"], [STATUS_RGBA[status]], TILE_W, TILE_H + BAND,
                       offsets=[0.0])
    d = mj.MjData(m)
    mj.mj_forward(m, d)
    h = float(max(d.xpos[1:, 2].max(), 0.4)) if m.nbody > 1 else 1.0
    rend = mj.Renderer(m, height=TILE_H, width=TILE_W)
    cam = mj.MjvCamera()
    cam.lookat[:] = [0, 0, 0.55 * h]
    cam.distance, cam.elevation, cam.azimuth = max(1.6, 2.45 * h), -8, 180
    fr = [j for j in range(m.njnt) if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE]
    ra = int(m.jnt_qposadr[fr[0]]) if fr else -1
    tab = -np.ones((spec.J, 3), np.int64)
    for j, ns in enumerate(qnames):
        for k, nm in enumerate(ns):
            jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "m0_" + nm)
            if jid >= 0:
                tab[j, k] = int(m.jnt_qposadr[jid])
    frames = []
    for t in range(T):
        if ra >= 0:
            pm = gp[t] @ RN.AX / 100.0
            d.qpos[ra:ra + 3] = [0.0, 0.0, -float(pm[:, 2].min())]
            d.qpos[ra + 3:ra + 7] = RN.mat_to_quat(RN.AX.T @ rootR[t] @ RN.AX)
        for j in range(spec.J):
            for k in range(3):
                if tab[j, k] >= 0:
                    d.qpos[tab[j, k]] = q[t, j, k]
        mj.mj_forward(m, d)
        rend.update_scene(d, cam)
        frames.append(rend.render().copy())
    del rend, m, d
    return frames


def label(img, name, status, extra=""):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (TILE_W, TILE_H + BAND), (14, 14, 16))
    im.paste(Image.fromarray(img), (0, 0))
    dr = ImageDraw.Draw(im)
    f = lambda s, b=True: ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if b else ""), s)
    dr.text((8, TILE_H + 4), name, font=f(15), fill=(232, 234, 238))
    dr.text((8, TILE_H + 22), TAG[status] + extra, font=f(12, False), fill=COL[status])
    return np.asarray(im)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=int, default=1, choices=(1, 2))
    ap.add_argument("--run", default="PROD_spatial_0808")
    ap.add_argument("--src_bones", action="store_true",
                    help="source motion from cache_bones_body (out-of-corpus) "
                         "instead of the AMASS cache")
    ap.add_argument("--clip", default="")
    ap.add_argument("--max_frames", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(OUT, exist_ok=True)
    clip = args.clip or ("Transitions__mazen_c3d__dance_jumpinplace_stageii"
                         if args.video == 1 else "SOMA__soma_subject2__clap_002_stageii")

    model, cfg = E.load_run(f"{V.RUNS}/{args.run}", dev)
    enc = cfg.get("sem_encoder", "siglip")
    if args.src_bones:
        import json as _j
        BON = "/var/tmp/alphamotion/topo_exp/cache_bones_body"
        man = _j.load(open(f"{BON}/manifest.json"))
        seqs = {x["name"]: x for x in _j.load(open(f"{BON}/sequences.json"))}
        if clip not in seqs:
            clip = sorted(seqs, key=lambda k: -seqs[k]["len"])[3]
            print(f"[bones src] using clip {clip}", flush=True)
        sq = seqs[clip]
        import numpy as _np
        d = _np.load(f"{BON}/soma_body_spec.npz", allow_pickle=True)
        from common import SkeletonSpec
        sspec = SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                             [str(x) for x in d["joint_names"]])
        g9 = torch.load(f"{BON}/soma_body_global9d.pt")[
            sq["start"]:sq["start"] + min(sq["len"], args.max_frames)]
        src = torch.cat([g9[..., :6], g9[..., 6:9] / man["reach_cm"]],
                        -1).to(dev).float()
        sdof = _np.load(f"{BON}/soma_body_dof.npz")["dof"]
    else:
        src, sspec, _ = E.load_source(SRC_CACHE, [clip], dev,
                                      max_frames=args.max_frames)
        sdof = np.load(f"{SRC_CACHE}/human_smpl_dof.npz")["dof"]
    sg = E.target_geom(sspec, sdof, dev, sem_encoder=enc)
    B = src.shape[0]
    print(f"clip {clip}  frames {B}", flush=True)

    jm = None
    if args.video == 2:
        # SE(3) constraint: encoder sees ONLY root + both wrists (the clap track).
        names = [n.lower() for n in sspec.joint_names]
        root = int(np.where(np.asarray(sspec.parents) < 0)[0][0])
        vis = [root] + [i for i, n in enumerate(names) if "wrist" in n]
        x = torch.zeros_like(src)
        vi = torch.as_tensor(sorted(set(vis)), dtype=torch.long)
        x[:, vi] = src[:, vi]
        jm = torch.ones(B, sspec.J, dtype=torch.bool, device=dev)
        jm[:, vi] = False
        src_in = x
        print(f"SE3 constraint cells: {[sspec.joint_names[i] for i in vi.tolist()]}",
              flush=True)
    else:
        src_in = src

    zp, zr = model.encode(src_in, E.geom_batch(sg, B), joint_mask=jm)

    # ---- tile 0: THE SOURCE. FK the human key input and draw it as capsules ----
    hp = V.fk_pos(src[..., :6].cpu(), sspec) / 100.0            # [B,J,3] m, Y-up
    hp = hp @ RN.AX                                             # -> MuJoCo Z-up
    hp[..., 2] -= hp[..., 2].min(axis=(1))[:, None] - 0.02
    spec_h = mj.MjSpec()
    spec_h.visual.global_.offwidth = TILE_W
    spec_h.visual.global_.offheight = TILE_H + BAND
    spec_h.worldbody.add_light(pos=[0, -2, 3], dir=[0, 0.5, -1])
    gpl = spec_h.worldbody.add_geom()
    gpl.type = mj.mjtGeom.mjGEOM_PLANE; gpl.size = [8, 8, 0.1]
    gpl.rgba = [0.9, 0.9, 0.9, 1]
    mh = spec_h.compile()
    dh = mj.MjData(mh); mj.mj_forward(mh, dh)
    rh = mj.Renderer(mh, height=TILE_H, width=TILE_W)
    camh = mj.MjvCamera(); camh.lookat[:] = [0, 0, 0.85]
    camh.distance, camh.elevation, camh.azimuth = 3.4, -8, 180
    pares = np.asarray(sspec.parents)
    hum_frames = []
    for t in range(B):
        rh.update_scene(dh, camh)
        sc = rh.scene
        for j in range(sspec.J):
            if pares[j] < 0:
                continue
            if sc.ngeom >= sc.maxgeom:
                break
            gg = sc.geoms[sc.ngeom]; sc.ngeom += 1
            mj.mjv_initGeom(gg, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.eye(3).flatten(),
                            np.array([0.25, 0.25, 0.28, 1], np.float32))
            mj.mjv_connector(gg, mj.mjtGeom.mjGEOM_CAPSULE, 0.032,
                             hp[t, pares[j]], hp[t, j])
        hum_frames.append(rh.render().copy())
    del rh
    from PIL import Image, ImageDraw, ImageFont
    def _hl(img):
        im = Image.new("RGB", (TILE_W, TILE_H + BAND), (14, 14, 16))
        im.paste(Image.fromarray(img), (0, 0))
        dr = ImageDraw.Draw(im)
        f = lambda s2, b=True: ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if b else ""), s2)
        dr.text((8, TILE_H + 4), "SOURCE", font=f(16), fill=(255, 255, 255))
        dr.text((8, TILE_H + 24), "human (SMPL), the input", font=f(12, False),
                fill=(200, 200, 205))
        return np.asarray(im)
    tiles, order = {"__src__": [_hl(f) for f in hum_frames]}, ["__src__"]
    for name, xml, status in ROSTER:
        try:
            spec, dof, rest, qnames, xml2 = E.get_target(xml, E.CACHE_AMASS)
            g = E.target_geom(spec, dof, dev, sem_encoder=enc)
            dd = torch.as_tensor(dof, device=dev, dtype=torch.float64)
            rr = torch.as_tensor(rest, device=dev, dtype=torch.float64)
            prot, _ = model.decode(zp, zr, E.geom_batch(g, B))
            R = rot6d_to_matrix(prot).double()
            q_raw, _ = MP.fit_angles(R, spec, dd, rest=rr, clamp=False, method="greedy")
            pre, _, _ = E.limit_report(q_raw, dof, dev)
            r6, _, q = MP.project(R, spec, dd, rest=rr, method="global")
            gR, gp = MP.fk_from_angles(q, spec, dd, rest=rr, root_R=R[:, 0])
            fr = render_tile(xml2, spec, qnames, q.cpu().numpy(),
                             gR[:, 0].cpu().numpy(), gp.cpu().numpy(), B,
                             status=status)
            extra = f"  viol {pre['viol_dof_pct']:.1f}%"
            tiles[name] = [label(f, name, status, extra) for f in fr]
            order.append(name)
            print(f"  [{len(order):>2}/20] {name:<16} viol {pre['viol_dof_pct']:>5.2f}%",
                  flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"  SKIP {name}: {ex}", flush=True)

    n = len(order)
    cols = 5
    rows_n = (n + cols - 1) // cols
    blank = np.full((TILE_H + BAND, TILE_W, 3), 14, np.uint8)
    frames = []
    for t in range(B):
        rws = []
        for r in range(rows_n):
            row = [tiles[order[r * cols + c]][t] if r * cols + c < n else blank
                   for c in range(cols)]
            rws.append(np.concatenate(row, 1))
        img = np.concatenate(rws, 0)
        from PIL import Image, ImageDraw, ImageFont
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im)
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        hdr = ("GREENWICH | BLUE=trained YELLOW=held-out RED=no-dataset | " +
               ("unseen motion: dance" if args.video == 1
                else "SE(3): ONLY root+both wrists given (clap) — rest invented") +
               f" | f{t}/{B-1}")
        dr.rectangle([0, 0, im.size[0], 34], fill=(10, 10, 12))
        dr.text((12, 6), hdr, font=fb, fill=(235, 236, 240))
        frames.append(np.asarray(im))
    p = f"{OUT}/greenwich_{'motion' if args.video == 1 else 'se3clap'}.mp4"
    RN.write_mp4(p, np.stack(frames), 20)
    print(f"\n-> {p}", flush=True)


if __name__ == "__main__":
    main()
