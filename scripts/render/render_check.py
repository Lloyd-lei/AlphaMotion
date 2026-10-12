"""Visual QA: 12-frame grid (whole clip), each cell = human slots | UMR robot | post-pass robot (side view, dense
points).  Red = ground-contact points, orange = object-contact points, blue = penetrating points, grey = object.
Optional --mp4 renders a window as video.
Usage: python scripts/render_check.py <clip> <robot> [--mp4 START END]"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageDraw
import mujoco, numpy as np
from surface_retargeting.robots import surface as S
from surface_retargeting.strict import geometry as G
from surface_retargeting.retarget import pipeline as P
from surface_retargeting.paths import OUT, RUNS, ROBOTS

ap = argparse.ArgumentParser(); ap.add_argument('clip'); ap.add_argument('robot')
ap.add_argument('--mp4', nargs=2, type=int, default=None)
a = ap.parse_args()
o = np.load(OUT / a.clip / f'{a.robot}.npz', allow_pickle=True)
prov = json.loads(str(o['provenance']))
Hs = np.load(prov['human_slots'])['source_slots']
umr = np.load(prov['umr_out'], allow_pickle=True)
if len(Hs) != len(o['qpos']):
    Hs = Hs[umr['frame_ids']]
floor = float(o['floor_offset_removed'])
Hs = Hs - np.array([0, 0, floor], np.float32)
m = mujoco.MjModel.from_xml_path(str(o['robot_xml'])); surf = S.load_surface(a.robot)
geo = G.RobotGeometry(m, surf, 0.0)
hreg, _ = P.human_slot_regions(a.robot, prov['source_kind'], prov['source_data'], prov['seq_key'], str(Path(prov['umr_out']).with_suffix('.umr.log')))
T = len(o['qpos'])
has_obj = 'object_pos' in o.files
if has_obj:
    from surface_retargeting.strict.objects import load_umr_hoi_object
    obj, _ = load_umr_hoi_object(str(umr['source_object_dir']), umr['frame_ids'], float(umr['smpl_scale'][0]))
    obj.pos = obj.pos - np.array([0, 0, floor])


W, H, SC = 640, 400, 190.0           # cell pixels, pixels per metre


def draw_pts(img, X, col, x0):
    u = ((X[:, 0] - x0) * SC).astype(int); v = (H - 30 - X[:, 2] * SC).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    arr = np.asarray(img).copy()
    arr[v[ok], u[ok]] = col[ok]
    return Image.fromarray(arr)


def colours(X, reg, gc, oc):
    c = np.full((len(X), 3), 140, np.uint8)
    c[gc[reg]] = [220, 30, 30]
    c[oc[reg]] = [255, 140, 0]
    c[X[:, 2] < -0.0005] = [30, 60, 240]
    return c


def cell(t):
    img = Image.new('RGB', (W, H), (250, 250, 250))
    ocz = np.zeros(13, bool)
    hx = Hs[t][:, 0].mean()
    x0 = hx - 0.5
    hc = o['human_ground_contact'][t]; hoc = o['human_object_contact'][t] if has_obj else ocz
    panels = [(Hs[t], hreg, hc, hoc, 0.0)]
    for k, (q, gc, oc) in enumerate(((o['qpos_umr'][t], ocz, ocz),
                                     (o['qpos'][t], o['ground_contact'][t], o['object_contact'][t] if has_obj else ocz))):
        geo.set(q.astype(float)); X = geo.dense_world()
        X = X - [X[:, 0].mean() - hx, 0, 0]
        panels.append((X, surf['region'].astype(int), gc, oc, 1.1 * (k + 1)))
    for X, reg, gc, oc, dx in panels:
        if has_obj:
            op = obj.surface_points_world(t, 600)
            img = draw_pts(img, op + [dx, 0, 0], np.full((len(op), 3), 60, np.uint8), x0)
        img = draw_pts(img, X + [dx, 0, 0], colours(X, reg, gc, oc), x0)
    d = ImageDraw.Draw(img)
    d.line([(0, H - 30), (W, H - 30)], fill=(0, 0, 0))
    d.text((5, 5), f't={t} human | UMR raw | post-pass   gL{int(o["ground_contact"][t][S.RID["L_foot"]])} '
                   f'gR{int(o["ground_contact"][t][S.RID["R_foot"]])} flag{int(o["flags"][t])}', fill=(0, 0, 0))
    return img


fr = np.linspace(0, T - 1, 12).astype(int)
sheet = Image.new('RGB', (4 * W, 3 * H), (255, 255, 255))
for i, t in enumerate(fr):
    sheet.paste(cell(t), ((i % 4) * W, (i // 4) * H))
ImageDraw.Draw(sheet).text((10, 3 * H - 14), f'{a.clip} -> {a.robot}  red=ground contact  orange=object contact  '
                           'blue=below floor  dark=object', fill=(0, 0, 0))
outp = OUT / a.clip / f'{a.robot}_grid.png'
sheet.save(outp); print('saved', outp)
if a.mp4:
    import imageio.v2 as imageio
    vp = OUT / a.clip / f'{a.robot}_{a.mp4[0]}-{a.mp4[1]}.mp4'
    fps = float(o['fps'])
    step = max(1, int(round(fps / 30)))
    w = imageio.get_writer(vp, fps=fps / step)
    for t in range(a.mp4[0], min(a.mp4[1], T), step):
        w.append_data(np.asarray(cell(t)))
    w.close(); print('saved', vp)
