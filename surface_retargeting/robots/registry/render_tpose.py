"""Render every inventoried robot in its proposed UMR T-pose (front view, floor-aligned). Visual QA only."""
import json, os, sys, numpy as np, mujoco
from PIL import Image, ImageDraw
OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(f'{OUT}/robots_raw.json'))
tiles = []
for r in R:
    if not r.get('load_ok'):
        continue
    m = mujoco.MjModel.from_xml_path(r['mjcf'])
    d = mujoco.MjData(m)
    q = m.qpos0.copy()
    for jn, v in (r.get('umr_tpose_qpos_proposed') or {}).items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        q[m.jnt_qposadr[j]] = v
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    # shift root so lowest point at z=0 (free joint) for framing
    if r.get('has_freejoint'):
        d.qpos[2] -= r.get('min_z_qpos0_m', 0.0)
        mujoco.mj_forward(m, d)
    H = r.get('height_tpose_m') or 1.5
    vis = [g for g in range(m.ngeom) if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0 and m.geom_group[g] != 3]
    opt = mujoco.MjvOption()
    for gi in range(6):
        opt.geomgroup[gi] = 1
    opt.geomgroup[3] = 0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [d.qpos[0] if r.get('has_freejoint') else 0, d.qpos[1] if r.get('has_freejoint') else 0, H / 2]
    cam.distance = max(H, r.get('armspan_tpose_m') or H) * 1.9
    cam.azimuth, cam.elevation = 180 - 25, -8
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, 360)
    m.vis.global_.offheight = max(m.vis.global_.offheight, 420)
    with mujoco.Renderer(m, 420, 360) as ren:
        ren.update_scene(d, cam, opt)
        img = Image.fromarray(ren.render())
    dr = ImageDraw.Draw(img)
    dr.text((6, 6), f"{r['name']}\nH={H:.2f}m nq={r['nq']} {r['hand_type']}", fill=(255, 255, 0))
    img.save(f"{OUT}/png/{r['name']}.png")
    tiles.append(img)
    print(r['name'], flush=True)
cols = 7
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new('RGB', (cols * 360, rows * 420))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 360, (i // cols) * 420))
sheet.save(f'{OUT}/png/contact_sheet_tpose.png')
