"""Side-by-side render: human source surface point cloud (UMR source slots, scaled) | robot retarget.
Contact points (z < thr) drawn red on both; robot contact derived from its own surface point cloud.
python render_umr.py --result R.npz --src-slots S.npz --start 600 --end 1050 --out out.mp4"""
import argparse, sys, numpy as np, mujoco, imageio, trimesh
sys.path.insert(0, '/var/tmp/alphamotion/umr_factory/umr/UMR/scripts')
from mujoco_geom_surface import surface_geom_ids, geom_local_mesh
ap = argparse.ArgumentParser(); ap.add_argument('--result'); ap.add_argument('--src-slots'); ap.add_argument('--start', type=int, default=0)
ap.add_argument('--end', type=int, default=-1); ap.add_argument('--out'); ap.add_argument('--thr', type=float, default=0.01)
ap.add_argument('--offset', type=float, default=1.1); ap.add_argument('--w', type=int, default=1280); ap.add_argument('--h', type=int, default=720)
ap.add_argument('--title', default='')
a = ap.parse_args()
R = np.load(a.result, allow_pickle=True); q = R['qpos']; fps = float(R['fps'][0])
S = np.load(a.src_slots)['source_slots']
end = len(q) if a.end < 0 else a.end
spec = mujoco.MjSpec.from_file(str(R['robot_xml']))
m = spec.compile(); m.vis.global_.offwidth = max(a.w, 640); m.vis.global_.offheight = max(a.h, 480); d = mujoco.MjData(m)
# robot surface sample (for red contact dots)
gids = [int(g) for g in surface_geom_ids(m, 'auto')]
loc, gof = [], []
for g in gids:
    v, f = geom_local_mesh(m, g); tm = trimesh.Trimesh(np.asarray(v), np.asarray(f), process=False)
    n = max(1, int(round(6000 * tm.area / 1.0))) ; p, _ = trimesh.sample.sample_surface(tm, min(n, 400), seed=0); loc.append(p); gof.append(np.full(len(p), g))
loc = np.concatenate(loc); gof = np.concatenate(gof)
r = mujoco.Renderer(m, a.h, a.w, max_geom=20000)
cam = mujoco.MjvCamera(); cam.distance = 4.2; cam.azimuth = 180; cam.elevation = -10
sub = np.arange(0, S.shape[1], 3)
w = imageio.get_writer(a.out, fps=fps, codec='libx264', quality=7, macro_block_size=1)
def add_sphere(scn, p, rad, rgba):
    if scn.ngeom >= scn.maxgeom: return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, np.array([rad, 0, 0]), np.asarray(p, np.float64), np.eye(3).ravel(), np.asarray(rgba, np.float32))
    scn.ngeom += 1
for t in range(a.start, end):
    d.qpos[:] = q[t]; mujoco.mj_forward(m, d)
    root = d.qpos[:3].copy()
    off = np.array([0, a.offset, 0.0])  # human drawn at +y (left of screen when azimuth=90 looks along -y? camera at +x side)
    cam.lookat[:] = [root[0], root[1] + a.offset / 2, 0.75]
    r.update_scene(d, cam); scn = r.scene
    hs = S[t]
    con = hs[:, 2] < a.thr
    for p in hs[sub][~con[sub]]: add_sphere(scn, p + off, 0.008, (0.3, 0.6, 1.0, 0.9))
    for p in hs[con]: add_sphere(scn, p + off, 0.010, (1.0, 0.1, 0.1, 1.0))
    P = np.einsum('nij,nj->ni', d.geom_xmat[gof].reshape(-1, 3, 3), loc) + d.geom_xpos[gof]
    for p in P[P[:, 2] < a.thr]: add_sphere(scn, p, 0.010, (1.0, 0.1, 0.1, 1.0))
    img = r.render()
    w.append_data(img)
w.close(); print('saved', a.out, end - a.start, 'frames')
