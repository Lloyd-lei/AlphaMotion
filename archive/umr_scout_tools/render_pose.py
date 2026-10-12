import sys, json, numpy as np, mujoco, imageio.v3 as iio
xml, tpose_json, out = sys.argv[1:4]
tp = json.load(open(tpose_json))
m = mujoco.MjModel.from_xml_path(xml); d = mujoco.MjData(m)
for j, v in tp.items():
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]] = v
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 480, 480)
imgs = []
for az in (180, 90):
    cam = mujoco.MjvCamera(); cam.lookat[:] = d.qpos[:3] if m.jnt_type[0]==0 else 0; cam.distance = 2.6; cam.azimuth = az; cam.elevation = -5
    r.update_scene(d, cam); imgs.append(r.render())
iio.imwrite(out, np.concatenate(imgs, 1))
