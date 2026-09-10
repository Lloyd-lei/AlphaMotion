"""Kinematic playback of decoded qpos pairs (lloyd 0830: '给我看看 greenwich 解码出来的
躺下/蹲下/起身').  Renders A (Greenwich decode) and B (native retarget) side by side --
pure qpos playback, no physics, so what you see is exactly what the codec emits.
  usage: render_qpos_pair.py <xml> <qposA.npy> <qposB.npy> <out.mp4> [fps] [stride]
"""
import sys, numpy as np, mujoco as mj, imageio.v2 as iio

xml, qa_p, qb_p, out = sys.argv[1:5]
fps = int(sys.argv[5]) if len(sys.argv) > 5 else 30
stride = int(sys.argv[6]) if len(sys.argv) > 6 else 1
spec = mj.MjSpec.from_file(xml)
if not any(j.type == mj.mjtJoint.mjJNT_FREE for j in spec.joints):
    spec.bodies[1].add_freejoint()
spec.worldbody.add_geom(name="_floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], rgba=[0.85, 0.87, 0.9, 1])
spec.worldbody.add_light(name="_sun", pos=[0, -1, 3], dir=[0, 0.25, -1], diffuse=[0.7, 0.7, 0.7], castshadow=True)
model = spec.compile(); data = mj.MjData(model)
ren = mj.Renderer(model, 480, 640)
cam = mj.MjvCamera(); cam.distance, cam.elevation = 3.2, -14

def frames(qp):
    F = []
    for i in range(0, len(qp), stride):
        data.qpos[:] = qp[i]; mj.mj_forward(model, data)
        cam.lookat[:] = data.qpos[:3] + np.array([0, 0, 0.3])
        cam.azimuth = 135
        ren.update_scene(data, cam); F.append(ren.render().copy())
    return F

qa, qb = np.load(qa_p), np.load(qb_p)
T = min(len(qa), len(qb))
FA, FB = frames(qa[:T]), frames(qb[:T])
w = iio.get_writer(out, fps=fps, codec="libx264", quality=7)
for a, b in zip(FA, FB):
    w.append_data(np.concatenate([a, b], axis=1))     # 左 A=Greenwich | 右 B=native
w.close()
print("WROTE", out, T, "frames")
