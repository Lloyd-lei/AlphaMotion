"""Render (pos_m, quat_xyzw, qj) clips on the vendor MJCF with a floor; hstack panels; 12-frame grid."""
import os, sys, subprocess
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, imageio.v2 as iio
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ruler import XML, hinge_names
def render(robot, pos, quat, qj, path, title, fps=30, W=960, H=540):
    spec = mujoco.MjSpec.from_file(XML[robot])
    for g in list(spec.worldbody.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE: spec.delete(g)
    for l in list(spec.worldbody.lights): spec.delete(l)
    spec.worldbody.add_geom(name='_floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[40, 40, 0.1], rgba=[0.86, 0.87, 0.9, 1])
    spec.worldbody.add_light(name='_sun', pos=[3, -4, 8], dir=[-0.3, 0.45, -1], castshadow=True)
    spec.visual.global_.offwidth = W; spec.visual.global_.offheight = H
    m = spec.compile(); d = mujoco.MjData(m); r = mujoco.Renderer(m, height=H, width=W); cam = mujoco.MjvCamera()
    h = float(pos[:, 2].max()) * 2.0 + 0.6; cam.distance = 2.1 * h; cam.elevation = -14; cam.azimuth = 135
    w = iio.get_writer(path, fps=fps, codec='libx264', quality=8, macro_block_size=1); T = len(pos)
    for t in range(T):
        d.qpos[:3] = pos[t]; d.qpos[3:7] = quat[t][[3, 0, 1, 2]]; d.qpos[7:] = qj[t]; mujoco.mj_forward(m, d)
        cam.lookat[:] = [pos[t, 0], pos[t, 1], 0.5 * h]; r.update_scene(d, camera=cam); img = Image.fromarray(r.render()); ImageDraw.Draw(img).text((16, 12), f'{title}   frame {t}/{T}', fill=(20, 20, 20)); w.append_data(np.asarray(img))
    w.close(); r.close()
def stack(paths, out, grid_path=None):
    args = ['ffmpeg', '-y', '-loglevel', 'error'] + sum([['-i', p] for p in paths], []) + ['-filter_complex', ''.join(f'[{i}:v]' for i in range(len(paths))) + f'hstack=inputs={len(paths)}[v]', '-map', '[v]', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', out]
    subprocess.run(args, check=True)
    if grid_path:
        rd = iio.get_reader(out); n = rd.count_frames(); idx = np.linspace(0, n - 1, 12).round().astype(int); fr = rd.get_data(0); Wt = fr.shape[1]; rows_h = int(270 * Wt / 960 / len(paths))
        sheet = Image.new('RGB', (min(Wt, 1920), rows_h * 12 * min(Wt, 1920) // Wt), (5, 5, 5)); sw = sheet.width; sh = sheet.height // 12
        for k, i in enumerate(idx): sheet.paste(Image.fromarray(rd.get_data(int(i))).resize((sw, sh)), (0, k * sh))
        rd.close(); sheet.save(grid_path, quality=85)
