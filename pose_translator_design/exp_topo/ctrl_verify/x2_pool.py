"""TOKEN POOL v1: all-AMASS windows -> X2 teacher qpos(30fps) + 3 token variants (fp16), sharded."""
import os, sys, json, numpy as np, torch, time
sys.path.insert(0, "/home/arenalabs/Desktop/AlphaMotion/src"); torch.set_grad_enabled(False); os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_root import ContactRoot
from scipy.spatial.transform import Rotation as Rot
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.spatial import SkeletonSpec, build_global
from alphamotion.embodiment import registry
from alphamotion.engine import constraints as MP
from alphamotion.engine.descriptor import geom_batch
from alphamotion.engine.nets.rotations import rot6d_to_matrix
C = "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full/"; OUT = "/home/arenalabs/alphamotion_artifacts/x2_pool_v3/"; os.makedirs(OUT, exist_ok=True)
XML = "/var/tmp/alphamotion/robot_models/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]]); WIN, STRIDE, N_MAX = 150, 300, 1500
def load_spec(p):
    d = np.load(p, allow_pickle=True); return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], [str(n) for n in d["joint_names"]])
G = Greenwich.load(); hs = load_spec(C+"human_smpl_spec.npz"); hd = np.load(C+"human_smpl_dof.npz", allow_pickle=True)["dof"]
emb = registry.load_from_xml(XML, "agibot_x2"); xs, xd, xr, xq = emb.spec, emb.dof, emb.rest, emb.qnames
L = torch.load(C+"human_smpl_paired_local6d.pt", mmap=True); HR = torch.load(C+"human_smpl_root_world.pt", mmap=True); seqs = json.load(open(C+"sequences.json"))
model = mj.MjSpec.from_file(XML).compile()
hinge = list(range(1, model.njnt)); hnames = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in hinge]; hadr = np.array([model.jnt_qposadr[j] for j in hinge])
root_j = int(np.where(np.asarray(xs.parents) < 0)[0][0]); dof_t = torch.as_tensor(xd, dtype=torch.float64); rest_t = torch.as_tensor(xr, dtype=torch.float64)
def rest_reach(spec):
    p = np.zeros((spec.J,3))
    for j in range(spec.J):
        if spec.parents[j] >= 0: p[j] = p[spec.parents[j]] + spec.rest_offsets[j]
    return np.linalg.norm(p, axis=1).max()
ratio = rest_reach(xs) / rest_reach(hs); LEG = 0.782  # X2 hip->ankle / human hip->ankle (feet define travel)
geom = G.geom(hs, hd)
# ---- contact-consistent root z machinery (owner rule 0820: feet-on-ground first) ----
data = mj.MjData(model); CR = ContactRoot(model)
_mesh_cache = {}
COLG = [g for g in range(model.ngeom) if model.geom_contype[g] > 0]
def lowest_vertex_z(qp_row):
    data.qpos[:] = qp_row; mj.mj_kinematics(model, data); lo = 1e9
    for g in COLG:
        Rm = data.geom_xmat[g].reshape(3,3); pm = data.geom_xpos[g]
        if model.geom_type[g] == mj.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            if mid not in _mesh_cache:
                a0 = model.mesh_vertadr[mid]; nv = model.mesh_vertnum[mid]; _mesh_cache[mid] = model.mesh_vert[a0:a0+nv].copy()
            lo = min(lo, float((_mesh_cache[mid] @ Rm.T + pm)[:, 2].min()))
        else:
            lo = min(lo, float(pm[2] - np.abs(Rm[2]).dot(model.geom_size[g])))
    return lo
def medfilt(x, k=5):
    pad = k//2; xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i+k]) for i in range(len(x))])
iToes = [hs.joint_names.index(n) for n in ["L_Toe","R_Toe","L_Ankle","R_Ankle"]]
names = hs.joint_names; SPARSE_KEEP = [names.index(n) for n in ["Pelvis","L_Wrist","R_Wrist","L_Ankle","R_Ankle"]]
# window list, skip holdout clips (same 8 as corpus manifest)
hold = {e["clip"] for e in json.load(open("/var/tmp/alphamotion/ctrl_verify/x2_corpus/manifest.json"))["hold"]}
wins = []
rng = np.random.default_rng(0)
for ci, s in enumerate(seqs):
    if s["name"] in hold or s["len"] < WIN: continue
    for st in range(0, s["len"] - WIN + 1, STRIDE): wins.append((s["name"], s["start"] + st))
rng.shuffle(wins); wins = wins[:N_MAX]
print("pool windows", len(wins), flush=True)
def feats(codes):
    q = (codes.float() - 4.0) / 3.5
    zp = G.model.fsqp.proj_out(q[:, :128]); zr = G.model.fsqr.proj_out(q[:, 128:])
    return torch.cat([zp.mean(1), zr.mean(1)], -1)
SH, t0 = 50, time.time(); man = []
DG = []
for i0 in range(0, len(wins), SH):
    shard = f"{OUT}shard_{i0//SH:04d}.npz"
    if os.path.exists(shard):
        man += list(np.load(shard, allow_pickle=True)["meta"]); continue
    Q, TF, ME = [], [], []
    for name, st in wins[i0:i0+SH]:
        l6 = L[st:st+WIN].float(); g6, gp, reach = build_global(l6, hs, "cpu")
        pose = torch.cat([g6, gp/reach], -1).to(G.device); gb = geom_batch(geom, WIN)
        v = []
        for mode in ["full", "rand", "sparse"]:
            if mode == "full": jm = None
            elif mode == "rand":
                jm = torch.rand(WIN, hs.J) < 0.4; jm[:, 0] = False
            else:
                jm = torch.ones(WIN, hs.J, dtype=torch.bool); jm[:, SPARSE_KEEP] = False
            fp, fr = G.model.encode_features(pose, gb, joint_mask=None if jm is None else jm.to(G.device))
            ip, ir = G.model.ints_from_features(fp, fr); codes = torch.cat([ip[0], ir[0]], 1).long()
            if mode == "full": codes_full = codes
            v.append(feats(codes).cpu().half().numpy())
        rot, _ = G.decode_full(codes_full, xs, xd); Rg = rot6d_to_matrix(rot).double().cpu()
        _, _, q = MP.project(Rg, xs, dof_t, rest=rest_t, method="global", lm_iters=10); q = q.detach().numpy()
        jp = {}
        for j, names3 in enumerate(xq):
            for k, nm in enumerate(names3):
                if nm: jp[nm] = q[:, j, k]
        quat = Rot.from_matrix(AX.T @ Rg[:, root_j].numpy() @ AX).as_quat()[:, [3,0,1,2]]
        root = (HR[st:st+WIN].numpy().astype(np.float64) - HR[st].numpy()) * ratio
        qp = np.tile(model.qpos0, (WIN,1)); qp[:, :3] = root @ AX / 100.0; qp[:, 3:7] = quat
        for a, nm in zip(hadr, hnames): qp[:, a] = jp.get(nm, 0.0)
        # feet-defined root (owner rule 0820): z = lowest sole planted, xy = stance foot pinned (stance from human data)
        hroot = HR[st:st+WIN].numpy().astype(np.float64)
        hw = hroot[:, 1][:, None] + gp[:, iToes, 1].numpy()
        hmin = hw.min(1); clear = np.maximum(hmin - np.percentile(hmin, 10) - 3.0, 0.0) * ratio / 100.0
        hfw = (hroot[:, None, :] + gp.numpy()[:, iToes, :]) @ AX / 100.0
        hfeet = np.stack([hfw[:, [0, 2]], hfw[:, [1, 3]]], 1)
        stance = CR.human_stance(hfeet, 30.0)
        hv = np.vstack([[0, 0], np.diff((hroot - hroot[0]) @ AX / 100.0, axis=0)[:, :2] * 30.0]) * LEG
        qp, dg = CR.solve(qp, clear, hv, 30.0, stance=stance)
        DG.append(dg)
        Q.append(qp.astype(np.float32)); TF.append(np.stack(v)); ME.append(f"{name}|{st}")
    np.savez(shard, qpos=np.stack(Q), tok=np.stack(TF), meta=np.array(ME, dtype=object))
    print("  skate p90 over shard: %.3f  flight frac %.2f" % (np.percentile([g["stance_speed_p90"] for g in DG], 90), np.mean([g.get("contact_frac",1) < 1 for g in DG])), flush=True); DG.clear()
    man += ME; el = time.time()-t0
    print(f"shard {i0//SH} done  {i0+len(Q)}/{len(wins)}  {el/60:.1f} min  eta {el/max(i0+SH,1)*(len(wins)-i0-SH)/60:.0f} min", flush=True)
json.dump(man, open(OUT+"pool_manifest.json","w"))
print("POOL DONE", len(man))
