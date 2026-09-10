"""Three-panel kinematic playback: decoded | refined | native.
usage: render_qpos_triple.py <xml> <qposA.npy> <qposR.npy> <qposB.npy> <out.mp4> [fps] [stride]
"""
import sys, numpy as np, mujoco as mj, imageio.v2 as iio

xml, qa_p, qr_p, qb_p, out = sys.argv[1:6]
fps = int(sys.argv[6]) if len(sys.argv) > 6 else 30
stride = int(sys.argv[7]) if len(sys.argv) > 7 else 1
spec = mj.MjSpec.from_file(xml)
if not any(j.type == mj.mjtJoint.mjJNT_FREE for j in spec.joints):
    spec.bodies[1].add_freejoint()
spec.worldbody.add_light(name="_sun", pos=[0, -1, 3], dir=[0, 0.25, -1],
                         diffuse=[0.7, 0.7, 0.7], castshadow=True)
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

qa, qr, qb = np.load(qa_p), np.load(qr_p), np.load(qb_p)
T = min(len(qa), len(qr), len(qb))
FA, FR, FB = frames(qa[:T]), frames(qr[:T]), frames(qb[:T])
w = iio.get_writer(out, fps=fps, codec="libx264", quality=7)
for a, r, b in zip(FA, FR, FB):
    w.append_data(np.concatenate([a, r, b], axis=1))  # 左 decoded | 中 refined | 右 native
w.close()
print("WROTE", out, T, "frames")
