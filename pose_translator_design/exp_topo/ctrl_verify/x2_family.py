"""One clip per motion family: human (xwbc AMASS) -> Greenwich -> Agibot X2 (UNSEEN body, zero-shot).
Root position = human root trajectory scaled by reach ratio (X2 has no GT), root orientation = decoded pelvis
(upright families).  Metrics without GT + side-by-side-free video per family + beyondmimic-style CSV/npz."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, "/home/arenalabs/Desktop/AlphaMotion/src"); torch.set_grad_enabled(False); os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj, imageio.v2 as iio, cv2
from scipy.spatial.transform import Rotation as Rot
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.spatial import SkeletonSpec, build_global, fk_pos
from alphamotion.embodiment import registry
from alphamotion.engine import constraints as MP
from alphamotion.engine.nets.rotations import rot6d_to_matrix
from alphamotion.atlas.families import FAMILIES, family_of
C = "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full/"; OUT = "/var/tmp/alphamotion/ctrl_verify/x2/"; os.makedirs(OUT, exist_ok=True)
VID = '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/R4_VIZ/mp4/ctrl_verify/x2/'; os.makedirs(VID, exist_ok=True)
XML = "/var/tmp/alphamotion/robot_models/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]]); FPS = 30
PICK = {"walk": "ACCAD__Female1Walking_c3d__B3_-_walk1_stageii", "run": "ACCAD__Female1Running_c3d__C3_-_Run_stageii", "jump": "DFaust__50002__50002_one_leg_jump_stageii",
        "jacks": "DFaust__50002__50002_jumping_jacks_stageii", "dance": "KIT__572__dance_chacha01_stageii", "kick": "ACCAD__Male2MartialArtsKicks_c3d__G11-__roundhouse_leading_right_stageii",
        "punch": "KIT__291__punching03_stageii", "throw": "KIT__3__throw_left01_stageii", "sit": "ACCAD__Female1General_c3d__A7_-_crouch_stageii", "stand": "ACCAD__Female1General_c3d__A14_-_stand_to_skip_stageii"}
def load_spec(p):
    d = np.load(p, allow_pickle=True); return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], [str(n) for n in d["joint_names"]])
G = Greenwich.load(); hs = load_spec(C+"human_smpl_spec.npz"); hd = np.load(C+"human_smpl_dof.npz", allow_pickle=True)["dof"]
emb = registry.load_from_xml(XML, "agibot_x2"); xs, xd, xr, xq = emb.spec, emb.dof, emb.rest, emb.qnames
L = torch.load(C+"human_smpl_paired_local6d.pt", mmap=True); HR = torch.load(C+"human_smpl_root_world.pt", mmap=True); seqs = {s["name"]: s for s in json.load(open(C+"sequences.json"))}
sp = mj.MjSpec.from_file(XML); sp.worldbody.add_geom(name="_floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[30,30,0.1], rgba=[0.9,0.9,0.92,1]); sp.worldbody.add_light(name="_sun", pos=[0,-2,4], dir=[0,0.3,-1], diffuse=[0.6,0.6,0.6], castshadow=True)
model = sp.compile(); model.vis.global_.offwidth = 1280; model.vis.global_.offheight = 720; model.vis.headlight.ambient[:] = 0.4; data = mj.MjData(model)
assert model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE
hinge = list(range(1, model.njnt)); hnames = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in hinge]; hadr = np.array([model.jnt_qposadr[j] for j in hinge]); jrange = model.jnt_range[hinge]
foot_geoms = {s: [g for g in range(model.ngeom) if model.geom_contype[g] > 0 and s+"_ankle_roll" in mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])] for s in ["left","right"]}
print("X2 hinges", len(hnames), "foot geoms", {k: len(v) for k,v in foot_geoms.items()})
root_j = int(np.where(np.asarray(xs.parents) < 0)[0][0]); dof_t = torch.as_tensor(xd, dtype=torch.float64); rest_t = torch.as_tensor(xr, dtype=torch.float64)
# reach ratio for root trajectory scaling: X2 vs human (max root-relative joint distance at rest)
def rest_reach(spec):
    p = np.zeros((spec.J,3)); 
    for j in range(spec.J):
        if spec.parents[j] >= 0: p[j] = p[spec.parents[j]] + spec.rest_offsets[j]
    return np.linalg.norm(p, axis=1).max()
ratio = rest_reach(xs) / rest_reach(hs); print("reach ratio X2/human", round(ratio, 3))
def geom_low(g):
    R = data.geom_xmat[g].reshape(3,3); p = data.geom_xpos[g]
    if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]; a = model.mesh_vertadr[mid]; n = model.mesh_vertnum[mid]; return float((model.mesh_vert[a:a+n] @ R.T + p)[:,2].min())
    return float(p[2] - np.abs(R[2]).dot(model.geom_size[g]))
report = {}
for fam, name in PICK.items():
    s = seqs[name]; st, n = s["start"], min(s["len"], 240)
    l6 = L[st:st+n].float(); g6, gp, reach = build_global(l6, hs, "cpu")
    codes = G.encode(torch.cat([g6, gp/reach], -1).to(G.device), hs, hd)
    rot, _ = G.decode_full(codes, xs, xd); Rg = rot6d_to_matrix(rot).double().cpu()
    _, _, q = MP.project(Rg, xs, dof_t, rest=rest_t, method="global", lm_iters=20); q = q.detach().numpy()
    jp = {}
    for j, names3 in enumerate(xq):
        for k, nm in enumerate(names3):
            if nm: jp[nm] = q[:, j, k]
    missing = [h for h in hnames if h not in jp]
    quat = Rot.from_matrix(AX.T @ Rg[:, root_j].numpy() @ AX).as_quat()[:, [3,0,1,2]]
    root = (HR[st:st+n].numpy().astype(np.float64) - HR[st].numpy()) * ratio                    # human root deltas, scaled
    root_zup = root @ AX / 100.0; root_zup[:, 2] += 0.0
    qp = np.tile(model.qpos0, (n,1)); qp[:, :3] = root_zup; qp[:, 3:7] = quat
    for a, nm in zip(hadr, hnames):
        if nm in jp: qp[:, a] = jp[nm]
    # ground: put the lowest foot vertex over the clip on the plane
    lows = []; 
    for t in range(n):
        data.qpos[:] = qp[t]; mj.mj_kinematics(model, data); lows.append(min(geom_low(g) for gs in foot_geoms.values() for g in gs))
    lows = np.array(lows); qp[:, 2] -= lows.min()
    # metrics
    jpq = qp[:, hadr]; jv = np.gradient(jpq, 1/FPS, axis=0); over = np.maximum(jpq - jrange[:,1], 0) + np.maximum(jrange[:,0] - jpq, 0)
    zz = lows - lows.min(); fxy = []
    for t in range(n):
        data.qpos[:] = qp[t]; mj.mj_kinematics(model, data); fxy.append([np.mean([data.geom_xpos[g,:2] for g in foot_geoms[k]],0) for k in ["left","right"]])
    fxy = np.array(fxy)   # [T,2,2]
    m = {"frames": int(n), "missing_joints": missing, "limit_violation_frac": float((over>1e-4).mean()), "limit_excess_max_deg": float(np.degrees(over.max())),
         "joint_vel_p95": float(np.percentile(np.abs(jv),95)), "joint_vel_max": float(np.abs(jv).max()), "frames_with_spike_gt8": int((np.abs(jv)>8).any(1).sum()),
         "root_travel_m": float(np.linalg.norm(qp[-1,:2]-qp[0,:2])), "root_height_mean_m": float(qp[:,2].mean()), "foot_max_lift_cm": float(zz.max()*100),
         "lower_foot_speed_mean_cm_s": float(np.mean([np.linalg.norm(fxy[t+1, int(np.argmin(zz[t+1:t+2]))]-fxy[t, int(np.argmin(zz[t+1:t+2]))])*FPS for t in range(n-1)])*100)}
    report[fam] = m
    # csv (beyondmimic order = MJCF hinge order incl. head)
    np.savetxt(f"{OUT}{fam}__{name[:40]}__A_greenwich.csv", np.concatenate([qp[:, :3], qp[:, 3:7][:, [1,2,3,0]], qp[:, hadr]], 1), delimiter=",", fmt="%.6f")
    np.save(f"{OUT}{fam}__qposA.npy", qp)
    # video: lateral camera
    rend = mj.Renderer(model, height=560, width=640); cam = mj.MjvCamera(); trav = qp[-1,:2]-qp[0,:2]
    heading = np.degrees(np.arctan2(trav[1], trav[0])) if np.linalg.norm(trav) > 0.3 else 0.0
    cam.azimuth = heading + 90; cam.elevation = -12; cam.distance = 3.2; frames = []
    for t in range(n):
        data.qpos[:] = qp[t]; mj.mj_forward(model, data); cam.lookat[:] = [qp[t,0], qp[t,1], 0.9]; rend.update_scene(data, cam); f = rend.render().copy()
        cv2.putText(f, f"X2 zero-shot | {fam}: {name[:34]}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,128,0), 2); frames.append(f)
    del rend
    p = f"{VID}X2_{fam}.mp4"
    with iio.get_writer(p, format="FFMPEG", fps=FPS, codec="libx264", quality=7, pixelformat="yuv420p") as w:
        for f in frames: w.append_data(f)
    print(f"{fam:6s} {name[:38]:38s} lim {m['limit_violation_frac']:.3f}/{m['limit_excess_max_deg']:4.1f}deg  vmax {m['joint_vel_max']:5.1f} spikes {m['frames_with_spike_gt8']:3d}  travel {m['root_travel_m']:4.2f}m  lift {m['foot_max_lift_cm']:4.1f}cm  skate {m['lower_foot_speed_mean_cm_s']:5.1f}  missing {missing}")
json.dump(report, open(OUT+"report_x2.json","w"), indent=1)
