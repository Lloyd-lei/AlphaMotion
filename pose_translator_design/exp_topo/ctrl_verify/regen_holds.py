"""Regenerate the 8 holdout CSVs with contact-consistent z (same rule as x2_pool v2)."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, "/home/arenalabs/Desktop/AlphaMotion/src"); torch.set_grad_enabled(False); os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from contact_root import ContactRoot
from scipy.spatial.transform import Rotation as Rot
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.spatial import SkeletonSpec, build_global
from alphamotion.embodiment import registry
from alphamotion.engine import constraints as MP
from alphamotion.engine.nets.rotations import rot6d_to_matrix
C = "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full/"; OUT = "/var/tmp/alphamotion/ctrl_verify/x2_corpus/"
XML = "/var/tmp/alphamotion/robot_models/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]])
def load_spec(p):
    d = np.load(p, allow_pickle=True); return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], [str(n) for n in d["joint_names"]])
G = Greenwich.load(); hs = load_spec(C+"human_smpl_spec.npz"); hd = np.load(C+"human_smpl_dof.npz", allow_pickle=True)["dof"]
emb = registry.load_from_xml(XML, "agibot_x2"); xs, xd, xr, xq = emb.spec, emb.dof, emb.rest, emb.qnames
L = torch.load(C+"human_smpl_paired_local6d.pt", mmap=True); HR = torch.load(C+"human_smpl_root_world.pt", mmap=True); seqs = {s["name"]: s for s in json.load(open(C+"sequences.json"))}
model = mj.MjSpec.from_file(XML).compile(); data = mj.MjData(model); CR = ContactRoot(model)
hinge = list(range(1, model.njnt)); hnames = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in hinge]; hadr = np.array([model.jnt_qposadr[j] for j in hinge])
root_j = int(np.where(np.asarray(xs.parents) < 0)[0][0]); dof_t = torch.as_tensor(xd, dtype=torch.float64); rest_t = torch.as_tensor(xr, dtype=torch.float64)
def rest_reach(spec):
    p = np.zeros((spec.J,3))
    for j in range(spec.J):
        if spec.parents[j] >= 0: p[j] = p[spec.parents[j]] + spec.rest_offsets[j]
    return np.linalg.norm(p, axis=1).max()
ratio = rest_reach(xs) / rest_reach(hs); LEG = 0.782  # X2 hip->ankle / human hip->ankle
_mesh = {}; COLG = [g for g in range(model.ngeom) if model.geom_contype[g] > 0]
def lowz(row):
    data.qpos[:] = row; mj.mj_kinematics(model, data); lo = 1e9
    for g in COLG:
        Rm = data.geom_xmat[g].reshape(3,3); pm = data.geom_xpos[g]
        if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            if mid not in _mesh:
                a0 = model.mesh_vertadr[mid]; nv = model.mesh_vertnum[mid]; _mesh[mid] = model.mesh_vert[a0:a0+nv].copy()
            lo = min(lo, float((_mesh[mid] @ Rm.T + pm)[:, 2].min()))
        else: lo = min(lo, float(pm[2] - np.abs(Rm[2]).dot(model.geom_size[g])))
    return lo
def medfilt(x, k=5):
    pad = k//2; xp = np.pad(x, pad, mode="edge"); return np.array([np.median(xp[i:i+k]) for i in range(len(x))])
iToes = [hs.joint_names.index(n) for n in ["L_Toe","R_Toe","L_Ankle","R_Ankle"]]
man = json.load(open(OUT+"manifest.json"))
for e in man["hold"]:
    s = seqs[e["clip"]]; st, n = s["start"], min(s["len"], 240)
    l6 = L[st:st+n].float(); g6, gp, reach = build_global(l6, hs, "cpu")
    codes = G.encode(torch.cat([g6, gp/reach], -1).to(G.device), hs, hd)
    rot, _ = G.decode_full(codes, xs, xd); Rg = rot6d_to_matrix(rot).double().cpu()
    _, _, q = MP.project(Rg, xs, dof_t, rest=rest_t, method="global", lm_iters=10); q = q.detach().numpy()
    jp = {}
    for j, names3 in enumerate(xq):
        for k, nm in enumerate(names3):
            if nm: jp[nm] = q[:, j, k]
    quat = Rot.from_matrix(AX.T @ Rg[:, root_j].numpy() @ AX).as_quat()[:, [3,0,1,2]]
    root = (HR[st:st+n].numpy().astype(np.float64) - HR[st].numpy()) * ratio
    qp = np.tile(model.qpos0, (n,1)); qp[:, :3] = root @ AX / 100.0; qp[:, 3:7] = quat
    for a, nm in zip(hadr, hnames): qp[:, a] = jp.get(nm, 0.0)
    hw = HR[st:st+n].numpy()[:, 1][:, None] + gp[:, iToes, 1].numpy(); hmin = hw.min(1)
    clear = np.maximum(hmin - np.percentile(hmin, 10) - 3.0, 0.0) * ratio / 100.0
    hroot = HR[st:st+n].numpy().astype(np.float64); hfw = (hroot[:, None, :] + gp.numpy()[:, iToes, :]) @ AX / 100.0  # (T,4,3) Z-up m
    hfeet = np.stack([hfw[:, [0, 2]], hfw[:, [1, 3]]], 1)          # (T,2,2,3) [L,R] x [Toe,Ankle]
    stance = CR.human_stance(hfeet, 30.0)
    hv = np.vstack([[0, 0], np.diff((hroot - hroot[0]) @ AX / 100.0, axis=0)[:, :2] * 30.0]) * LEG
    qp, dg = CR.solve(qp, clear, hv, 30.0, stance=stance)
    dg["flight_frac"] = float(np.mean(stance < 0)); dg["expect_travel"] = float(np.linalg.norm(((hroot[-1]-hroot[0]) @ AX / 100.0)[:2]) * LEG)
    print("  feet-root:", {k: round(v, 3) for k, v in dg.items()}, "human-scaled travel %.2f" % np.linalg.norm(root[-1, :2] @ AX[:2, :2] / 100.0 if False else (root @ AX / 100.0)[-1, :2]))
    np.savetxt(f"{OUT}hold__{e['tag']}.csv", np.concatenate([qp[:, :3], qp[:, 3:7][:, [1,2,3,0]], qp[:, hadr]], 1), delimiter=",", fmt="%.6f")
    print("regen", e["tag"], "z range %.2f-%.2f" % (qp[:,2].min(), qp[:,2].max()), flush=True)
print("HOLDS DONE")
