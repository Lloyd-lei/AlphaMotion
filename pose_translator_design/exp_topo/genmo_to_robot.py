"""lloyd's 0814 route: GENMO is the generator, AlphaMotion is the cross-embodiment
layer.  This is the second half — take GENMO's SMPL output and put it on robots.

    genmo_to_robot.py --clip /var/tmp/alphamotion/genmo_bridge/boxing.pt

GENMO can do BOTH video->motion and text->motion, so the same pipe serves two
purposes: a prompt front-end today, and (the bigger half) a data engine — any
human video becomes motion we can retarget to 20 bodies.

The motion here came out of a VIDEO, so it is motion Greenwich has never seen
in any form; nothing about this clip is in the paired cache.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
import mujoco as mj                                              # noqa: E402
import v5_common as V                                            # noqa: E402
import eval_newbody as E                                         # noqa: E402
import manifold_project as MP                                    # noqa: E402
import render_newbody as RN                                      # noqa: E402
import greenwich_carnival as GC                                  # noqa: E402
from common import rot6d_to_matrix                               # noqa: E402
from spatial5 import pad_band                                     # noqa: E402
from PIL import Image, ImageDraw, ImageFont                      # noqa: E402

RUNS = V.RUNS
RM = "/var/tmp/alphamotion/robot_models"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ", "mp4", "genmo")
# paths + generalisation status straight from greenwich_carnival's roster, so
# the "heldout" label on G1 stays honest: it is a body Greenwich never trained on
ROBOTS = [(n, x, st) for n, x, st in GC.ROSTER
          if n in ("Unitree G1", "Unitree H1", "Fourier GR3", "Booster T1")]


def font(s, b=True):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"
                              % ("-Bold" if b else ""), s)


def human_tile(hp_cm, spec, T):
    """SOURCE panel: the human GENMO recovered, as 3-D capsules (never a stick
    figure — lloyd's standing rule), so every robot tile has a reference."""
    hp = hp_cm / 100.0
    hp = hp @ RN.AX
    hp[..., 2] -= hp[..., 2].min(axis=1)[:, None] - 0.02
    sh = mj.MjSpec()
    sh.visual.global_.offwidth = GC.TILE_W
    sh.visual.global_.offheight = GC.TILE_H + GC.BAND
    sh.worldbody.add_light(pos=[0, -2, 3], dir=[0, 0.5, -1])
    g = sh.worldbody.add_geom()
    g.type = mj.mjtGeom.mjGEOM_PLANE; g.size = [8, 8, 0.1]
    g.rgba = [0.9, 0.9, 0.9, 1]
    mh = sh.compile(); dh = mj.MjData(mh); mj.mj_forward(mh, dh)
    r = mj.Renderer(mh, height=GC.TILE_H, width=GC.TILE_W)
    cam = mj.MjvCamera(); cam.lookat[:] = [0, 0, 0.85]
    cam.distance, cam.elevation, cam.azimuth = 3.4, -8, 180
    par = np.asarray(spec.parents)
    frames = []
    for t in range(T):
        r.update_scene(dh, cam)
        sc = r.scene
        for j in range(spec.J):
            if par[j] < 0 or sc.ngeom >= sc.maxgeom:
                continue
            gg = sc.geoms[sc.ngeom]; sc.ngeom += 1
            mj.mjv_initGeom(gg, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.eye(3).flatten(),
                            np.array([0.25, 0.25, 0.28, 1], np.float32))
            mj.mjv_connector(gg, mj.mjtGeom.mjGEOM_CAPSULE, 0.032,
                             hp[t, par[j]], hp[t, j])
        frames.append(r.render().copy())
    del r
    out = []
    for f in frames:
        im = Image.new("RGB", (GC.TILE_W, GC.TILE_H + GC.BAND), (14, 14, 16))
        im.paste(Image.fromarray(f), (0, 0))
        d = ImageDraw.Draw(im)
        d.text((8, GC.TILE_H + 4), "SOURCE", font=font(16), fill=(255, 255, 255))
        d.text((8, GC.TILE_H + 24), "GENMO output (SMPL) — the input",
               font=font(12, False), fill=(200, 200, 205))
        out.append(np.asarray(im))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="/var/tmp/alphamotion/genmo_bridge/boxing.pt")
    ap.add_argument("--spatial", default="PROD_c1_91k_dim20_tail_0810")
    ap.add_argument("--max_frames", type=int, default=105)
    ap.add_argument("--tag", default="boxing")
    ap.add_argument("--note", default="motion recovered from VIDEO by GENMO")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(OUT, exist_ok=True)

    d = torch.load(args.clip, map_location="cpu", weights_only=False)
    gr6 = d["rot6d"][:args.max_frames]                      # GLOBAL rot6d
    T = gr6.shape[0]
    sp = np.load(f"{os.environ['V5_AMASS']}/human_smpl_spec.npz", allow_pickle=True)
    sspec = E.SkeletonSpec(str(sp["name"]), sp["parents"], sp["rest_offsets"],
                           [str(x) for x in sp["joint_names"]])

    class S:
        parents = sp["parents"]; rest_offsets = sp["rest_offsets"]
    gp = torch.from_numpy(V.fk_pos(gr6.numpy(), S)).float()  # [T,22,3] cm root-rel
    reach = float(gp.norm(dim=-1).max())
    src = torch.cat([gr6, gp / reach], -1).to(dev).float()   # [T,22,9]
    print(f"frames {T}  reach {reach:.1f} cm", flush=True)

    model, cfg = E.load_run(f"{RUNS}/{args.spatial}", dev)
    enc = cfg.get("sem_encoder")
    sdof = np.load(f"{os.environ['V5_AMASS']}/human_smpl_dof.npz")["dof"]
    sg = E.target_geom(sspec, sdof, dev, sem_encoder=enc)
    zp, zr = model.encode(src, E.geom_batch(sg, T))           # encoded ONCE
    print("encoded; decoding to robots...", flush=True)

    tiles = [human_tile(gp.numpy(), sspec, T)]
    labels = [("SOURCE", "")]
    for name, xml, status in ROBOTS:
        spec_t, dof_t, rest_t, qn_t, xml_t = E.get_target(xml, E.CACHE_AMASS)
        g_t = E.target_geom(spec_t, dof_t, dev, sem_encoder=enc)
        rot, _ = model.decode(zp, zr, E.geom_batch(g_t, T))
        Rg = rot6d_to_matrix(rot).double()
        dof_d = torch.as_tensor(dof_t, device=dev, dtype=torch.float64)
        rest_d = torch.as_tensor(rest_t, device=dev, dtype=torch.float64)
        _, _, qj = MP.project(Rg, spec_t, dof_d, rest=rest_d, method="global")
        gR, gpos = MP.fk_from_angles(qj, spec_t, dof_d, rest=rest_d,
                                     root_R=Rg[:, 0])
        rt = GC.render_tile(xml_t, spec_t, qn_t, qj.cpu().numpy(),
                            gR[:, 0].cpu().numpy(), gpos.cpu().numpy(),
                            T, status=status)
        tiles.append([pad_band(f) if f.shape[0] != tiles[0][0].shape[0] else f
                      for f in rt])
        labels.append((name, status))
        print(f"  {name} done", flush=True)

    L = min(len(t) for t in tiles)
    frames = []
    for t in range(L):
        row = np.concatenate([ti[t] for ti in tiles], 1)
        im = Image.fromarray(row).convert("RGB")
        dr = ImageDraw.Draw(im)
        dr.text((8, 6), "GENMO -> Greenwich -> robots", font=font(15),
                fill=(255, 230, 150))
        dr.text((8, 26), args.note + "  |  encoded ONCE, decoded to every body"
                "  |  no GT", font=font(11, False), fill=(190, 200, 215))
        frames.append(np.asarray(im))
    p = f"{OUT}/genmo_{args.tag}_to_robots.mp4"
    RN.write_mp4(p, np.stack(frames), 30)
    print("->", p, flush=True)


if __name__ == "__main__":
    main()
