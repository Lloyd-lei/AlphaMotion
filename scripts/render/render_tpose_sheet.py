"""Render every staged robot in its final T-pose (front + top view) into one contact sheet.
Output: DATA/robots/tpose_contact_sheet.png"""
import json, os, sys
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco, numpy as np
from PIL import Image, ImageDraw
from surface_retargeting.robots import robots as R
from surface_retargeting.paths import ROBOTS

W, H = 300, 340
tiles = []
for rec in R.load_inventory():
    name = rec['name']
    st = json.load(open(ROBOTS / name / 'staging.json'))
    m = mujoco.MjModel.from_xml_path(st['staged_mjcf']); d = mujoco.MjData(m)
    R.set_tpose(m, d, st['tpose'], root_z=0.0)
    # lift to floor
    geoms = R._surface_geoms(m)
    zmin = min((m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:m.mesh_vertadr[m.geom_dataid[g]] + m.mesh_vertnum[m.geom_dataid[g]]]
                @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min() for g in geoms if m.geom_type[g] == 7)
    d.qpos[2] -= zmin; mujoco.mj_forward(m, d)
    hgt = rec.get('height_tpose_m') or 1.5
    opt = mujoco.MjvOption()
    for gi in range(6):
        opt.geomgroup[gi] = 1
    opt.geomgroup[3] = 0
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W); m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    imgs = []
    for az, el in ((180, -5), (180, -89)):           # front (looking along -x), top
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [d.qpos[0], d.qpos[1], hgt * 0.55]
        cam.distance = max(hgt, rec.get('armspan_tpose_m') or hgt) * 1.8
        cam.azimuth, cam.elevation = az, el
        with mujoco.Renderer(m, H, W) as ren:
            ren.update_scene(d, cam, opt); imgs.append(Image.fromarray(ren.render()))
    tile = Image.new('RGB', (2 * W, H)); tile.paste(imgs[0], (0, 0)); tile.paste(imgs[1], (W, 0))
    dr = ImageDraw.Draw(tile)
    mt = st['tpose_metrics_final']
    txt = (f"{name}  {'FIXED' if st['tpose_fixed'] else 'draft/UMR'}  pass={mt['pass']}\n"
           f"L arm {mt['L']['upper_arm_deg']:.0f}/{mt['L']['forearm_deg']:.0f} deg tip dz {mt['L']['tip_minus_shoulder_cm'][2]:+.1f}cm"
           f" dx {mt['L']['tip_minus_shoulder_cm'][0]:+.1f}\nR arm {mt['R']['upper_arm_deg']:.0f}/{mt['R']['forearm_deg']:.0f} deg"
           f" tip dz {mt['R']['tip_minus_shoulder_cm'][2]:+.1f}cm dx {mt['R']['tip_minus_shoulder_cm'][0]:+.1f}\n"
           f"excl {len(st['exclusions'])} flagged {len(st['flagged_nonadjacent_rest_overlaps'])}")
    dr.text((5, 5), txt, fill=(255, 255, 0))
    # shoulder-height guide line on the front view
    tiles.append(tile)
    print(name, flush=True)
cols = 3
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new('RGB', (cols * 2 * W, rows * H), (30, 30, 30))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * 2 * W, (i // cols) * H))
sheet.save(ROBOTS / 'tpose_contact_sheet.png')
print('saved', ROBOTS / 'tpose_contact_sheet.png', sheet.size)
