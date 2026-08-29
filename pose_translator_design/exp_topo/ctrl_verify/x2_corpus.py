"""Build a GENERALIST X2 corpus: N AMASS windows (family-balanced) + holdout windows.
Per window: human -> Greenwich codes -> X2 decode -> global projection -> CSV (root = scaled human
trajectory + decoded root ori) ... same conventions as x2_family.py. Physics gate on each."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, "/home/arenalabs/Desktop/AlphaMotion/src"); torch.set_grad_enabled(False); os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco as mj
from scipy.spatial.transform import Rotation as Rot
from alphamotion.engine.greenwich import Greenwich
from alphamotion.engine.spatial import SkeletonSpec, build_global
from alphamotion.embodiment import registry
from alphamotion.engine import constraints as MP
from alphamotion.engine.nets.rotations import rot6d_to_matrix
from alphamotion.atlas.families import family_of
C = "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full/"; OUT = "/var/tmp/alphamotion/ctrl_verify/x2_corpus/"; os.makedirs(OUT, exist_ok=True)
XML = "/var/tmp/alphamotion/robot_models/agibot_x2_urdf/X2_URDF-v1.4.0/agibot_x2_ultra_mjcf.xml"
AX = np.array([[0., 1, 0], [0, 0, 1], [1, 0, 0]]); N_TRAIN, N_HOLD = 60, 8
def load_spec(p):
    d = np.load(p, allow_pickle=True); return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"], [str(n) for n in d["joint_names"]])
G = Greenwich.load(); hs = load_spec(C+"human_smpl_spec.npz"); hd = np.load(C+"human_smpl_dof.npz", allow_pickle=True)["dof"]
emb = registry.load_from_xml(XML, "agibot_x2"); xs, xd, xr, xq = emb.spec, emb.dof, emb.rest, emb.qnames
L = torch.load(C+"human_smpl_paired_local6d.pt", mmap=True); HR = torch.load(C+"human_smpl_root_world.pt", mmap=True)
seqs = json.load(open(C+"sequences.json"))
model = mj.MjModel.from_xml_path(XML); 
sp = mj.MjSpec.from_file(XML); model = sp.compile(); data = mj.MjData(model)
hinge = list(range(1, model.njnt)); hnames = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j) for j in hinge]; hadr = np.array([model.jnt_qposadr[j] for j in hinge]); jrange = model.jnt_range[hinge]
root_j = int(np.where(np.asarray(xs.parents) < 0)[0][0]); dof_t = torch.as_tensor(xd, dtype=torch.float64); rest_t = torch.as_tensor(xr, dtype=torch.float64)
def rest_reach(spec):
    p = np.zeros((spec.J,3))
    for j in range(spec.J):
        if spec.parents[j] >= 0: p[j] = p[spec.parents[j]] + spec.rest_offsets[j]
    return np.linalg.norm(p, axis=1).max()
ratio = rest_reach(xs) / rest_reach(hs)
# family-balanced pick, 90-240 frames, dedup by clip prefix; holdout = later windows never in train
rng = np.random.default_rng(42)
from collections import defaultdict
byf = defaultdict(list)
for x in seqs:
    if 90 <= x["len"] <= 240: byf[family_of(x["name"])].append(x)
for k in byf: rng.shuffle(byf[k])
fams = [f for f in byf if byf[f]]
picked, hold, used = [], [], set()
i = 0
while len(picked) < N_TRAIN or len(hold) < N_HOLD:
    f = fams[i % len(fams)]; i += 1
    while byf[f]:
        s = byf[f].pop()
        pre = s["name"][:25]
        if pre in used: continue
        used.add(pre)
        (picked if len(picked) < N_TRAIN else hold).append((f, s)); break
    if not any(byf[f] for f in fams): break
print("train", len(picked), "hold", len(hold))
man = {"train": [], "hold": []}
for split, lst in [("train", picked), ("hold", hold)]:
    for f, s in lst:
        st, n = s["start"], min(s["len"], 240)
        l6 = L[st:st+n].float(); g6, gp, reach = build_global(l6, hs, "cpu")
        codes = G.encode(torch.cat([g6, gp/reach], -1).to(G.device), hs, hd)
        rot, _ = G.decode_full(codes, xs, xd); Rg = rot6d_to_matrix(rot).double().cpu()
        _, _, q = MP.project(Rg, xs, dof_t, rest=rest_t, method="global", lm_iters=20); q = q.detach().numpy()
        jp = {}
        for j, names3 in enumerate(xq):
            for k, nm in enumerate(names3):
                if nm: jp[nm] = q[:, j, k]
        quat = Rot.from_matrix(AX.T @ Rg[:, root_j].numpy() @ AX).as_quat()[:, [3,0,1,2]]
        root = (HR[st:st+n].numpy().astype(np.float64) - HR[st].numpy()) * ratio
        qp = np.tile(model.qpos0, (n,1)); qp[:, :3] = root @ AX / 100.0; qp[:, 3:7] = quat
        for a, nm in zip(hadr, hnames): qp[:, a] = jp.get(nm, 0.0)
        jv = np.abs(np.gradient(qp[:, hadr], 1/30, axis=0))
        gate = {"vmax": float(jv.max()), "spike_frames": int((jv>8).any(1).sum()), "over_deg": float(np.degrees((np.maximum(qp[:,hadr]-jrange[:,1],0)+np.maximum(jrange[:,0]-qp[:,hadr],0)).max()))}
        tag = f"{f}__{s['name'][:36]}"
        np.savetxt(f"{OUT}{split}__{tag}.csv", np.concatenate([qp[:, :3], qp[:, 3:7][:, [1,2,3,0]], qp[:, hadr]], 1), delimiter=",", fmt="%.6f")
        man[split].append({"tag": tag, "clip": s["name"], "family": f, "frames": int(n), **gate})
        print(split, tag, gate, flush=True)
json.dump(man, open(OUT+"manifest.json","w"), indent=1)
