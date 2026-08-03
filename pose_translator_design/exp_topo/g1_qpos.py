"""Real-mesh G1 rendering: convert our g1_30 GLOBAL rot6d stream to MuJoCo qpos
for the actual Unitree G1 model (STL meshes), validate against GT, render GIFs.

The g1_30 cache was built by ardy.data_processing.skeleton.load_csv_g1_animation:
  local_rot_j (ardy Y-up frame) = M @ (Q_body_j @ Rot(axis_j, theta_j)) @ M^-1
where M = mujoco->ardy axis permutation (x,y,z)_mj -> (y,z,x)_ardy, Q_body_j is the
XML <body quat> (baked INTO the local rotation, not the rest pose), axis_j the raw
XML hinge axis. This module inverts that exactly:
  theta_j = twist about axis_j of  Q_j^T @ M^T @ (R_parent^T R_j) @ M
Root free joint: quat from M^T R_root M, position fixed at standing height.

Usage (validate GT first, then model output):
  python g1_qpos.py --mode gtcheck   --seq_idx 12
  python g1_qpos.py --run .../runs/FIX_translate --mode translate --seq_idx 12
  python g1_qpos.py --run .../runs/FIX_translate --mode recon     --seq_idx 12
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
from scipy.spatial.transform import Rotation
from common import rot6d_to_matrix
from dual_common import load_paired_data, build_model, geom

ARDY_G1_XML = ('/home/arenalabs/Desktop/"be water, robot"/experiment/ardy/'
               'ardy/assets/skeletons/g1skel34/xml/g1.xml')
M_MJ2ARDY = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])   # v_ardy = M @ v_mj
STAND_H = 0.793
R4_VIZ = ('/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/R4_VIZ')
COL_GT = [0.55, 0.57, 0.60, 1]
COL_PRED = [0.95, 0.55, 0.15, 1]


class G1QposMapper:
    """Precomputes per-joint hinge axis + body static quat from the G1 XML."""

    def __init__(self, spec, xml=ARDY_G1_XML):
        self.spec = spec
        self.xml = xml
        model = mj.MjModel.from_xml_path(xml)
        J = spec.J
        self.axes = np.zeros((J, 3))
        self.body_R = np.tile(np.eye(3), (J, 1, 1))        # XML <body quat> as matrix
        self.mj_names = []
        for j, name in enumerate(spec.joint_names):
            if spec.parents[j] < 0:
                self.mj_names.append("floating_base_joint")
                continue
            jname = name.replace("_skel", "_joint")
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
            assert jid >= 0, f"joint {jname} not in {xml}"
            assert np.allclose(model.jnt_pos[jid], 0)
            self.mj_names.append(jname)
            self.axes[j] = model.jnt_axis[jid]
            bid = model.jnt_bodyid[jid]
            self.body_R[j] = Rotation.from_quat(model.body_quat[bid], scalar_first=True).as_matrix()

    def angles_from_global(self, g6):
        """g6 [T,J,6] global rot6d (ardy Y-up) ->
        (angles [T,J] rad, root_quat wxyz [T,4] mujoco frame, residual_deg [T,J]).
        residual = how far the local rotation is from a pure hinge rotation
        (GT ~ 0; model output = off-manifold part, dropped by the twist projection)."""
        R = rot6d_to_matrix(torch.as_tensor(np.asarray(g6), dtype=torch.float32)).numpy()
        T, J = R.shape[:2]
        Mv = M_MJ2ARDY
        angles = np.zeros((T, J)); resid = np.zeros((T, J)); root_quat = None
        for j in range(J):
            p = int(self.spec.parents[j])
            if p < 0:
                Rmj = Mv.T @ R[:, j] @ Mv
                root_quat = Rotation.from_matrix(Rmj).as_quat(scalar_first=True)
                continue
            L = np.einsum("tji,tjk->tik", R[:, p], R[:, j])          # R_p^T @ R_j
            Rax = self.body_R[j].T @ (Mv.T @ L @ Mv)                 # strip frame + body quat
            q = Rotation.from_matrix(Rax).as_quat(scalar_first=True)
            th = 2 * np.arctan2(q[:, 1:] @ self.axes[j], q[:, 0])
            th = (th + np.pi) % (2 * np.pi) - np.pi
            angles[:, j] = th
            Rfit = Rotation.from_rotvec(np.outer(th, self.axes[j])).as_matrix()
            cos = (np.einsum("tji,tji->t", Rfit, Rax) - 1) / 2
            resid[:, j] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        return angles, root_quat, resid


# ---------------------------------------------------------------- scene ----
def quat_z_to(v):
    v = v / (np.linalg.norm(v) + 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    c = float(v @ z); axis = np.cross(z, v); s = np.linalg.norm(axis)
    axis = axis / s if s > 1e-6 else np.array([1.0, 0.0, 0.0])
    ang = np.arctan2(s, c)
    return np.concatenate([[np.cos(ang / 2)], axis * np.sin(ang / 2)])


def build_scene(xml, robots, capsules):
    """robots: [(prefix,)] mesh G1 copies. capsules: [(tag, spec, rgba)] free-capsule
    skeletons (one capsule per bone, posed later from world joint positions)."""
    scene = mj.MjSpec()
    try:
        scene.visual.global_.offwidth = 1920; scene.visual.global_.offheight = 1080
    except AttributeError:
        pass
    scene.visual.headlight.ambient = [0.4, 0.4, 0.4]
    scene.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    for (pre,) in robots:
        child = mj.MjSpec.from_file(xml)
        f = scene.worldbody.add_frame()
        f.attach_body(child.worldbody.first_body(), pre, "")
    for tag, cspec, rgba in capsules:
        for j, p in enumerate(cspec.parents):
            if p < 0:
                continue
            L = max(float(np.linalg.norm(cspec.rest_offsets[j])) / 100.0, 1e-3)
            b = scene.worldbody.add_body(name=f"{tag}_b{j}")
            jt = b.add_freejoint()
            jt.name = f"{tag}_b{j}"
            g = b.add_geom()
            g.type = mj.mjtGeom.mjGEOM_CAPSULE
            g.size[0] = 0.02
            g.fromto = [0, 0, 0, 0, 0, L]
            g.rgba = rgba
    scene.worldbody.add_light(pos=[0, -2, 3], dir=[0, 0.5, -1])
    g = scene.worldbody.add_geom()
    g.type = mj.mjtGeom.mjGEOM_PLANE; g.size = [8, 8, 0.1]; g.rgba = [0.9, 0.9, 0.9, 1]
    return scene.compile()


def set_robot(model, data, prefix, mapper, angles_t, root_quat_t, root_pos):
    for j, name in enumerate(mapper.mj_names):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, prefix + name)
        adr = model.jnt_qposadr[jid]
        if int(mapper.spec.parents[j]) < 0:
            data.qpos[adr:adr + 3] = root_pos
            data.qpos[adr + 3:adr + 7] = root_quat_t
        else:
            data.qpos[adr] = angles_t[j]


def set_capsules(model, data, tag, cspec, pos_world):
    """pos_world [J,3] world joint positions (mujoco frame, meters)."""
    for j, p in enumerate(cspec.parents):
        if p < 0:
            continue
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{tag}_b{j}")
        adr = model.jnt_qposadr[jid]
        a, b = pos_world[p], pos_world[j]
        d = b - a
        if np.linalg.norm(d) < 1e-8:
            d = np.array([0.0, 0.0, 1e-6])
        data.qpos[adr:adr + 3] = a
        data.qpos[adr + 3:adr + 7] = quat_z_to(d)


def ardy_to_mj_pos(pos_cm):
    """[...,3] ardy Y-up cm -> mujoco Z-up meters (root-relative)."""
    return (pos_cm / 100.0) @ M_MJ2ARDY            # v_mj = M^T v_ardy  (row-vec form)


# ---------------------------------------------------------------- data ----
def model_out(model, data, src, tgt, idx):
    with torch.no_grad():
        zp, zr = model.encode(data[src]["pose"][idx], geom(data, src, len(idx)))
        prot, ppos = model.decode(zp, zr, geom(data, tgt, len(idx)))
    return prot.cpu().numpy(), (ppos * data[tgt]["reach"]).cpu().numpy()


def gt_out(data, name, idx):
    p = data[name]["pose"][idx].cpu().numpy()
    return p[..., :6], p[..., 6:9] * float(data[name]["reach"])


def numeric_check(mapper, angles, root_quat, gt_pos_cm):
    """FK the qpos through MuJoCo, compare body positions vs cache GT positions."""
    model = mj.MjModel.from_xml_path(mapper.xml)
    data = mj.MjData(model)
    errs = []
    bids = []
    for name in mapper.spec.joint_names:
        bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name.replace("_skel", "_link")
                            if name != "waist_pitch_skel" else "torso_link")
        if bid < 0 and name == "pelvis_skel":
            bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "pelvis")
        assert bid >= 0, name
        bids.append(bid)
    for t in range(len(angles)):
        set_robot(model, data, "", mapper, angles[t], root_quat[t], [0, 0, STAND_H])
        mj.mj_forward(model, data)
        p = data.xpos[bids] - data.xpos[bids[0]]           # root-relative, mj frame
        gt = ardy_to_mj_pos(gt_pos_cm[t])
        errs.append(np.linalg.norm(p - gt, axis=-1))
    e = np.array(errs) * 100
    return float(e.mean()), float(e.max())


# ---------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/var/tmp/alphamotion/topo_exp/runs/FIX_translate")
    ap.add_argument("--cache", default="/var/tmp/alphamotion/topo_exp/cache_paired_seq")
    ap.add_argument("--mode", choices=["gtcheck", "translate", "recon"], default="gtcheck")
    ap.add_argument("--seq_idx", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=120)
    ap.add_argument("--stride", type=int, default=2, help="30fps -> gif at 30/stride fps")
    ap.add_argument("--size", type=int, nargs=2, default=[960, 540])
    ap.add_argument("--azimuth", type=float, default=180)
    ap.add_argument("--gap", type=float, default=1.0, help="lateral panel separation (m); 0 = overlay")
    ap.add_argument("--out", default=R4_VIZ)
    ap.add_argument("--frame_png", default=None, help="also dump this frame index as png")
    args = ap.parse_args()
    device = "cuda"
    os.makedirs(args.out, exist_ok=True)

    data, names, N = load_paired_data(args.cache, device)
    g1 = "g1_30"
    mapper = G1QposMapper(data[g1]["spec"])
    seqs = json.load(open(f"{args.cache}/sequences.json"))
    s = seqs[args.seq_idx]
    rows = torch.arange(s["start"], s["start"] + min(s["len"], args.max_frames), args.stride)
    print(f"sequence '{s['name']}' len={s['len']} rendering {len(rows)} frames")

    gap = args.gap
    gt_rot, gt_pos = gt_out(data, g1, rows)
    gt_ang, gt_quat, gt_res = mapper.angles_from_global(gt_rot)
    print(f"[GT hinge residual] mean={gt_res.mean():.3f} deg  max={gt_res.max():.3f} deg")

    if args.mode == "gtcheck":
        me, mx = numeric_check(mapper, gt_ang, gt_quat, gt_pos)
        print(f"[GT position check] mujoco-FK vs cache positions: mean={me:.2f}cm max={mx:.2f}cm")
        robots = [("gt_",)]
        capsules = [("cap", data[g1]["spec"], COL_GT)]
        robot_frames = [(gt_ang, gt_quat, np.array([0, -gap / 2, STAND_H]))]
        caps_frames = [(data[g1]["spec"], ardy_to_mj_pos(gt_pos) + np.array([0, gap / 2, STAND_H]))]
        tag = f"g1mesh_gtcheck_s{args.seq_idx}"
        print("panels: G1 mesh GT | capsule GT (same frames)")
    else:
        cfg = json.load(open(f"{args.run}/config.json"))
        model = build_model(cfg, device)
        model.load_state_dict(torch.load(f"{args.run}/model.pt", map_location=device), strict=False)
        model.eval()
        src = "soma77" if args.mode == "translate" else g1
        pr_rot, _ = model_out(model, data, src, g1, rows)
        pr_ang, pr_quat, pr_res = mapper.angles_from_global(pr_rot)
        print(f"[PRED hinge residual] mean={pr_res.mean():.2f} deg  max={pr_res.max():.2f} deg")
        robots = [("pred_",), ("gt_",)]
        robot_frames = [(pr_ang, pr_quat, np.array([0, -gap / 2, STAND_H])),
                        (gt_ang, gt_quat, np.array([0, gap / 2, STAND_H]))]
        capsules, caps_frames = [], []
        if args.mode == "translate":
            capsules = [("soma", data["soma77"]["spec"], COL_GT)]
            _, spos = gt_out(data, "soma77", rows)
            caps_frames = [(data["soma77"]["spec"],
                            ardy_to_mj_pos(spos) + np.array([0, -3 * gap / 2, STAND_H]))]
            print("panels: soma77 GT capsule | soma77->g1 PRED mesh (orange) | g1 GT mesh")
            tag = f"g1mesh_trans_s{args.seq_idx}"
        else:
            print("panels: g1->g1 RECON mesh (orange) | g1 GT mesh")
            tag = f"g1mesh_recon_s{args.seq_idx}"

    scene = build_scene(mapper.xml, robots, capsules)
    # tint the predicted robot orange
    for pre in [r[0] for r in robots if r[0] == "pred_"]:
        for gi in range(scene.ngeom):
            gname = mj.mj_id2name(scene, mj.mjtObj.mjOBJ_GEOM, gi)
            bname = mj.mj_id2name(scene, mj.mjtObj.mjOBJ_BODY, scene.geom_bodyid[gi])
            if bname and bname.startswith(pre) and scene.geom_group[gi] == 2:
                scene.geom_rgba[gi] = COL_PRED
                scene.geom_matid[gi] = -1
    mdata = mj.MjData(scene)
    renderer = mj.Renderer(scene, height=args.size[1], width=args.size[0])
    cam = mj.MjvCamera()
    ys = [float(rf[2][1]) for rf in robot_frames] + [float(cf[1][0, 0, 1]) for cf in caps_frames]
    cam.lookat[:] = [0, float(np.mean(ys)), 0.75]
    cam.distance = 2.2 + 0.9 * (len(robots) + len(capsules))
    cam.elevation = -12; cam.azimuth = args.azimuth

    frames = []
    for t in range(len(rows)):
        for (pre,), (ang, rq, rp) in zip(robots, robot_frames):
            set_robot(scene, mdata, pre, mapper, ang[t], rq[t], rp)
        for (tagc, cspec, _), (cs, pw) in zip(capsules, caps_frames):
            set_capsules(scene, mdata, tagc, cs, pw[t])
        mj.mj_forward(scene, mdata)
        renderer.update_scene(mdata, cam)
        frames.append(renderer.render().copy())
    import imageio.v2 as imageio
    gif = os.path.join(args.out, f"{tag}.gif")
    imageio.mimsave(gif, frames, duration=args.stride / 30.0, loop=0)
    if args.frame_png is not None:
        png = os.path.join(args.out, f"{tag}_f{args.frame_png}.png")
        imageio.imwrite(png, frames[int(args.frame_png)])
        print("PNG:", os.path.abspath(png))
    print("GIF:", os.path.abspath(gif))


if __name__ == "__main__":
    main()
