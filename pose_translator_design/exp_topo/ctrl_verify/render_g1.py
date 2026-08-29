"""Side-by-side replay: A (Greenwich) | B (xwbc native) on the G1 MJCF, lateral follow camera."""
import os, sys, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj, imageio.v2 as imageio
XML = "/var/tmp/alphamotion/config_harvest/GMR/assets/unitree_g1/g1_mocap_29dof.xml"
OUT = "/var/tmp/alphamotion/ctrl_verify/"; VID = "/home/arenalabs/Desktop/\"be water, robot\"/pose_translator_design/R4_VIZ/mp4/ctrl_verify/"
os.makedirs(VID, exist_ok=True)
sp = mj.MjSpec.from_file(XML); pass
sp.worldbody.add_geom(name="_floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[30, 30, 0.1], rgba=[0.9, 0.9, 0.92, 1])
sp.worldbody.add_light(name="_sun", pos=[0, -2, 4], dir=[0, 0.3, -1], diffuse=[0.6, 0.6, 0.6], castshadow=True)
model = sp.compile(); model.vis.global_.offwidth = 1280; model.vis.global_.offheight = 720
model.vis.headlight.ambient[:] = 0.4; model.vis.headlight.diffuse[:] = 0.6
data = mj.MjData(model)
W, H = 640, 560
def render(qp, floor_off, label):
    rend = mj.Renderer(model, height=H, width=W); cam = mj.MjvCamera()
    heading = np.degrees(np.arctan2(qp[-1, 1] - qp[0, 1], qp[-1, 0] - qp[0, 0])) if np.linalg.norm(qp[-1, :2] - qp[0, :2]) > 0.3 else 0.0
    cam.azimuth = heading + 90; cam.elevation = -12; cam.distance = 3.2
    frames = []
    for t in range(len(qp)):
        data.qpos[:] = qp[t]; data.qpos[2] -= floor_off; mj.mj_forward(model, data)
        cam.lookat[:] = [qp[t, 0], qp[t, 1], 0.8]
        rend.update_scene(data, cam); frames.append(rend.render())
    del rend; return np.stack(frames)
def floor_offset(qp):
    """lowest foot-mesh vertex over the clip (so both tracks stand on the plane)."""
    feet = [g for g in range(model.ngeom) if model.geom_bodyid[g] > 0 and "ankle_roll" in mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])]
    lo = 1e9
    for t in range(0, len(qp), 3):
        data.qpos[:] = qp[t]; mj.mj_kinematics(model, data)
        for g in feet:
            if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]; a = model.mesh_vertadr[mid]; n = model.mesh_vertnum[mid]
                v = model.mesh_vert[a:a+n] @ data.geom_xmat[g].reshape(3,3).T + data.geom_xpos[g]; lo = min(lo, v[:, 2].min())
    return lo
import cv2
for name in sys.argv[1:]:
    qA = np.load(f"{OUT}tracks_g1/{name}__qposA.npy"); qB = np.load(f"{OUT}tracks_g1/{name}__qposB.npy")
    fA = render(qA, floor_offset(qA), "A"); fB = render(qB, floor_offset(qB), "B")
    out = []
    for a, b in zip(fA, fB):
        a = a.copy(); b = b.copy()
        cv2.putText(a, "A  human -> Greenwich -> G1", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
        cv2.putText(b, "B  xwbc native G1 retarget", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        out.append(np.concatenate([a, b], 1))
    p = f"{VID}{name[:40]}__G1_A_vs_B.mp4"
    with imageio.get_writer(p, format="FFMPEG", fps=30, codec="libx264", quality=7, pixelformat="yuv420p") as w:
        for f in out: w.append_data(f)
    print("->", p)
