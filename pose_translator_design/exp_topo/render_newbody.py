"""Side-by-side mesh video of N models decoding the SAME clip onto the SAME robot.

Every panel is the FEASIBILITY-PROJECTED pose (manifold_project, method="global"),
i.e. what the robot could actually execute — so what you see is not a model's raw
wish but the executable result, which is the only fair comparison when the models
differ in how badly they violate the joint limits.

Angles come straight out of the projection (q per DOF slot) and are written into the
robot's own MJCF qpos addresses, so the render is MuJoCo's forward kinematics, not
our FK re-implementation. Root height is set per frame so the lowest body touches z=0.

Usage:
  python render_newbody.py --runs CTRL=/path/run1,SYNTH=/path/run2 \
      --target apollo --clip MoSh__50020__salsa_stageii --out x.mp4
"""
from __future__ import annotations
import os, sys, json, argparse, subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
import mujoco as mj
import eval_newbody as E
import manifold_project as MP
from common import rot6d_to_matrix

AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])      # MuJoCo Z-up -> cache Y-up
COLS = [(250, 160, 60), (90, 210, 110), (150, 155, 165)]
RGBA = [[0.95, 0.55, 0.15, 1], [0.25, 0.75, 0.35, 1], [0.55, 0.57, 0.60, 1]]


def mat_to_quat(R):
    """[...,3,3] -> wxyz."""
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(R).as_quat(scalar_first=True)


def build_scene(xml, prefixes, rgbas, offwidth, offheight, offsets=None):
    """offsets [n] y-metres. FIXED-BASE vendor models (Fourier GR2/GR3 have no free
    joint) cannot be separated through qpos, so their copies are placed by the attach
    FRAME instead; floating-base models keep the qpos path and get frame pos 0."""
    s = mj.MjSpec()
    try:
        s.visual.global_.offwidth = offwidth
        s.visual.global_.offheight = offheight
    except AttributeError:
        pass
    s.visual.headlight.ambient = [0.4, 0.4, 0.4]
    s.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    probe = mj.MjModel.from_xml_path(xml)
    fixed_base = not any(probe.jnt_type[j] == mj.mjtJoint.mjJNT_FREE
                         for j in range(probe.njnt))
    for i, p in enumerate(prefixes):
        child = mj.MjSpec.from_file(xml)
        f = s.worldbody.add_frame()
        if fixed_base and offsets is not None:
            # lift so the lowest body clears the floor (probe the rest pose once)
            pd = mj.MjData(probe); mj.mj_forward(probe, pd)
            lift = float(-pd.xpos[:, 2].min()) + 0.02
            f.pos = [0.0, float(offsets[i]), lift]
        f.attach_body(child.worldbody.first_body(), p, "")
    s.worldbody.add_light(pos=[0, -2, 3], dir=[0, 0.5, -1])
    g = s.worldbody.add_geom()
    g.type = mj.mjtGeom.mjGEOM_PLANE
    g.size = [8, 8, 0.1]
    g.rgba = [0.9, 0.9, 0.9, 1]
    m = s.compile()
    # tint by BODY name: most MJCF geoms are unnamed, so a geom-name test would
    # only catch a handful of them (it caught Apollo's hands and nothing else).
    for p, c in zip(prefixes, rgbas):
        for i in range(m.ngeom):
            b = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])) or ""
            if b.startswith(p):
                m.geom_matid[i] = -1
                m.geom_rgba[i] = c
    return m


def write_mp4(path, frames, fps):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "26",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    p.stdin.close()
    assert p.wait() == 0, "ffmpeg failed"


def annotate(img, labels, header, sub, xs, notes):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    W, H = im.size

    def font(sz, bold=True):
        p = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else "")
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            return ImageFont.load_default()
    fl, fh, fs = font(21), font(26), font(16, False)
    d.rectangle([0, 0, W, 42], fill=(18, 18, 20))
    d.rectangle([0, H - 66, W, H], fill=(18, 18, 20))
    d.text((14, 9), header, font=fh, fill=(232, 234, 238))
    d.text((W - 14 - d.textlength(sub, font=fs), 14), sub, font=fs, fill=(150, 155, 162))
    for (txt, col), x in zip(labels, xs):
        d.text((x - d.textlength(txt, font=fl) / 2, H - 60), txt, font=fl, fill=col)
    for (txt, col), x in zip(notes, xs):
        d.text((x - d.textlength(txt, font=fs) / 2, H - 34), txt, font=fs, fill=col)
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="LABEL=path,LABEL=path")
    ap.add_argument("--target", default="apollo")
    ap.add_argument("--clip", default="MoSh__50020__salsa_stageii")
    ap.add_argument("--src_cache", default=E.CACHE_AMASS)
    ap.add_argument("--max_frames", type=int, default=200)
    ap.add_argument("--size", type=int, nargs=2, default=[1500, 620])
    ap.add_argument("--gap", type=float, default=1.15)
    ap.add_argument("--azimuth", type=float, default=150)
    ap.add_argument("--elev", type=float, default=-8)
    ap.add_argument("--dist", type=float, default=3.4)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)

    runs = [x.split("=", 1) for x in args.runs.split(",")]
    spec, dof, rest, qnames, xml = E.get_target(args.target, E.CACHE_AMASS)
    assert qnames is not None, "render needs an MJCF-backed target"
    src, sspec, _ = E.load_source(args.src_cache, [args.clip], device,
                                  max_frames=args.max_frames)
    # Per-run geoms: each run's text tower decides the semantic width, so a
    # SigLIP run and a Qwen run in the same figure need their own embeddings.
    sdof = np.load(f"{args.src_cache}/human_smpl_dof.npz")["dof"]
    def geoms_for(enc):
        return (E.target_geom(sspec, sdof, device, sem_encoder=enc),
                E.target_geom(spec, dof, device, sem_encoder=enc))
    geo_cache = {}
    dof_d = torch.as_tensor(dof, device=device, dtype=torch.float64)
    rest_d = torch.as_tensor(rest, device=device, dtype=torch.float64)

    streams, stats = [], []
    for label, run in runs:
        model, mcfg = E.load_run(run, device)
        enc = mcfg.get("sem_encoder", "siglip")
        if enc not in geo_cache:
            geo_cache[enc] = geoms_for(enc)
        sg, g = geo_cache[enc]
        with torch.no_grad():
            zp, zr = model.encode(src, E.geom_batch(sg, src.shape[0]))
            prot, _ = model.decode(zp, zr, E.geom_batch(g, src.shape[0]))
        Rg = rot6d_to_matrix(prot).double()
        q_raw, _ = MP.fit_angles(Rg, spec, dof_d, rest=rest_d, clamp=False, method="greedy")
        pre, _, _ = E.limit_report(q_raw, dof, device)
        r6, _, q = MP.project(Rg, spec, dof_d, rest=rest_d, method="global")
        from common import geodesic_deg
        moved = float(geodesic_deg(Rg, rot6d_to_matrix(r6)).mean())
        gR, gp = MP.fk_from_angles(q, spec, dof_d, rest=rest_d, root_R=Rg[:, 0])
        streams.append((q.cpu().numpy(), gR[:, 0].cpu().numpy(), gp.cpu().numpy()))
        stats.append(dict(label=label, run=run,
                          pre_viol_dof_pct=pre["viol_dof_pct"],
                          worst=pre["worst_overshoot_deg"], moved=round(moved, 2)))
        print(label, stats[-1], flush=True)
        del model
        torch.cuda.empty_cache()

    n = len(streams)
    prefixes = [f"m{i}_" for i in range(n)]
    ys = [(i - (n - 1) / 2) * args.gap for i in range(n)]
    m = build_scene(xml, prefixes, RGBA[:n], args.size[0], args.size[1], offsets=ys)
    d = mj.MjData(m)
    rend = mj.Renderer(m, height=args.size[1], width=args.size[0])
    cam = mj.MjvCamera()
    cam.lookat[:] = [0, 0, 0.85]
    cam.distance = args.dist
    cam.elevation = args.elev
    cam.azimuth = args.azimuth
    # per-copy address tables, resolved by PREFIXED joint name (attaching N copies
    # renumbers every qpos address, so the single-robot table cannot be reused)
    root_adr, slot_adr = [], []
    for p in prefixes:
        # Free joints are often UNNAMED in vendor MJCFs (Fourier GR2/GR3), so the
        # name-prefix test finds nothing; the BODY is always prefixed by attach, so
        # fall back to that. Some vendor models are FIXED-BASE (no free joint at all)
        # -- attach then gives the copy no root dof, and we place it by the frame.
        fr = [int(m.jnt_qposadr[j]) for j in range(m.njnt)
              if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE
              and ((mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, j) or "").startswith(p)
                   or (mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY,
                                     int(m.jnt_bodyid[j])) or "").startswith(p))]
        assert len(fr) <= 1, (p, fr)
        root_adr.append(fr[0] if fr else -1)      # -1 = fixed base, skip root writes
        tab = -np.ones((spec.J, 3), np.int64)
        for j, ns in enumerate(qnames):
            for k, nm in enumerate(ns):
                jid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, p + nm)
                if jid >= 0:                      # vendor models may omit some slots
                    tab[j, k] = int(m.jnt_qposadr[jid])
        slot_adr.append(tab)
    T = streams[0][0].shape[0]
    frames = []
    for t in range(T):
        for i, (q, rootR, gp) in enumerate(streams):
            ra = root_adr[i]
            if ra < 0:
                # FIXED-BASE vendor model: no free joint to write, and the body sits
                # wherever the MJCF left it -- for Fourier that is base_link at z=0
                # with the legs BELOW the floor plane, which is why the first GR2
                # render looked legless. Lift the whole copy by its own standing
                # height via the attach frame instead (done once, in build_scene).
                pass
            else:
                pm = gp[t] @ AX / 100.0                            # Y-up cm -> mj m
                z = -float(pm[:, 2].min())
                d.qpos[ra:ra + 3] = [0.0, ys[i], z]
                d.qpos[ra + 3:ra + 7] = mat_to_quat(AX.T @ rootR[t] @ AX)
            tab = slot_adr[i]
            for j in range(spec.J):
                for k in range(3):
                    if tab[j, k] >= 0:
                        d.qpos[tab[j, k]] = q[t, j, k]
        mj.mj_forward(m, d)
        rend.update_scene(d, cam)
        frames.append(rend.render().copy())
    frames = np.stack(frames)

    # panel centres straight from each copy's tint (no camera-projection guessing)
    mid = frames[len(frames) // 2].astype(np.int16)
    xs = []
    for i in range(n):
        c = np.array(RGBA[i][:3]) * 255
        msk = (np.abs(mid - c[None, None]) < 34).all(-1)
        xs.append(float(np.nonzero(msk)[1].mean()) if msk.sum() > 200
                  else args.size[0] * (i + 0.5) / n)
    labels = [(s["label"], COLS[i]) for i, s in enumerate(stats)]
    notes = [(f"viol {s['pre_viol_dof_pct']:.1f}%  proj {s['moved']:.1f}°", COLS[i])
             for i, s in enumerate(stats)]
    hdr = f"{os.path.basename(args.target)}  —  {args.clip[:44]}   (both panels feasibility-projected)"
    out = np.stack([annotate(f, labels, hdr, "zero-shot decode from human_smpl", xs, notes)
                    for f in frames])
    write_mp4(args.out, out, args.fps)
    json.dump(stats, open(args.out.replace(".mp4", "") + "_stats.json", "w"), indent=2)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
