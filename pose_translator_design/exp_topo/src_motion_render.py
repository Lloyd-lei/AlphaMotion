"""Render the RAW input motion (full-body human SMPL) for the speed-token demo clips.

    src_motion_render.py

lloyd, 2026-08-09: "我得知道你原来输入的原动作是什么样啊" — the speed-token videos'
"GT" panel is the human KEY states pushed through Greenwich onto the robot; the true
original input (full 22-joint SMPL clip) was never shown. This renders it: same seed,
same sampling order as speed_token_rollout.py, so the windows are the EXACT rows the
demo encoded. Capsule render lifted from greenwich_carnival's SOURCE tile.

Output: R4_VIZ/mp4/exam_c_mesh/source_{fam}_smpl.mp4
"""
from __future__ import annotations
import os, sys, subprocess
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES",
                      "/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
import mujoco as mj                                                  # noqa: E402
from PIL import Image, ImageDraw, ImageFont                          # noqa: E402
import v5_common as V                                                # noqa: E402
import v7_bridge_mesh as BM                                          # noqa: E402
from speed_token_rollout import EMB, FAMS                            # noqa: E402

W, H, BAND, FPS = 900, 700, 46, 20
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "R4_VIZ", "mp4", "exam_c_mesh")


def main():
    T = V.T_WIN
    kd = V.KeyData(V.AMASS, [EMB], device="cpu")
    spec = kd.spec[EMB]
    pares = np.asarray(spec.parents)
    l6 = torch.load(f"{V.AMASS}/{EMB}_paired_local6d.pt", map_location="cpu")
    g = np.random.default_rng(23)                       # SAME seed/order as the demo

    sp = mj.MjSpec()
    sp.visual.global_.offwidth = W; sp.visual.global_.offheight = H
    sp.worldbody.add_light(pos=[0, -2, 3], dir=[0, 0.5, -1])
    gpl = sp.worldbody.add_geom()
    gpl.type = mj.mjtGeom.mjGEOM_PLANE; gpl.size = [8, 8, 0.1]
    gpl.rgba = [0.9, 0.9, 0.9, 1]
    m = sp.compile(); d = mj.MjData(m); mj.mj_forward(m, d)
    rend = mj.Renderer(m, height=H, width=W)
    cam = mj.MjvCamera(); cam.lookat[:] = [0, 0, 0.85]
    cam.distance, cam.elevation, cam.azimuth = 3.2, -8, 180
    font = lambda s, b=True: ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if b else ""), s)

    for fam in FAMS:
        st, cid = kd.sample_starts(1, g, "val", span=T, families=[fam])
        s0 = int(st[0]); cname = kd.clips[int(cid[0])]["name"]
        hp = np.asarray(V.fk_pos(l6[s0:s0 + T], spec)) / 100.0      # [T,J,3] m, Y-up
        hp = hp @ BM.RN.AX                                          # -> Z-up
        hp[..., 2] -= hp[..., 2].min(axis=1)[:, None] - 0.02
        frames = []
        for t in range(T):
            rend.update_scene(d, cam)
            sc = rend.scene
            for j in range(spec.J):
                if pares[j] < 0 or sc.ngeom >= sc.maxgeom:
                    continue
                gg = sc.geoms[sc.ngeom]; sc.ngeom += 1
                mj.mjv_initGeom(gg, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                np.zeros(3), np.eye(3).flatten(),
                                np.array([0.25, 0.25, 0.28, 1], np.float32))
                mj.mjv_connector(gg, mj.mjtGeom.mjGEOM_CAPSULE, 0.032,
                                 hp[t, pares[j]], hp[t, j])
            im = Image.new("RGB", (W, H + BAND), (14, 14, 16))
            im.paste(Image.fromarray(rend.render().copy()), (0, 0))
            dr = ImageDraw.Draw(im)
            dr.text((10, H + 4), f"SOURCE input | {fam} | full-body human (SMPL) "
                    f"| f{t}/{T - 1}", font=font(16), fill=(255, 255, 255))
            dr.text((10, H + 24), cname, font=font(12, False), fill=(200, 200, 205))
            frames.append(np.asarray(im))
        fin = f"{OUT}/source_{fam}_smpl.mp4"
        BM.RN.write_mp4(fin, np.stack(frames), FPS)
        print(f"[{fam}] {cname}\n  -> {fin}", flush=True)


if __name__ == "__main__":
    main()
