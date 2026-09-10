"""Rewrite AMASS human_smpl (SMPL frames) into the LAFAN human_smpl dialect:
R_L^j = R_A^j @ Q_j (body-fixed constant), local6d re-derived on LAFAN parents."""
import os, sys, numpy as np, torch, shutil, json, time
sys.path.insert(0, '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo')
from common import matrix_to_rot6d, SkeletonSpec
from train_global import fk_global
ROOT="/var/tmp/alphamotion/topo_exp"; dev="cuda"
SRC=f"{ROOT}/cache_amass_v3"; OUT=f"{ROOT}/cache_amass_lconv"
Qd=torch.load("/home/arenalabs/tmp/dialect/Q_final.pt"); Q=Qd["Q"].to(dev)
d=np.load(f"{ROOT}/cache_xwbc_amass_full/human_smpl_spec.npz",allow_pickle=True)
hsA=SkeletonSpec(str(d["name"]),d["parents"],d["rest_offsets"],list(d["joint_names"])); par=[int(p) for p in hsA.parents]
assert list(d["joint_names"])==Qd["joint_names"]
os.makedirs(OUT,exist_ok=True)
for f in os.listdir(SRC):
    if f.startswith("human_smpl_") and f.endswith((".pt",".npz")) and "dof" not in f and "root_world" not in f: continue
    dst=f"{OUT}/{f}"
    if not os.path.lexists(dst): os.symlink(os.path.realpath(f"{SRC}/{f}"),dst)
shutil.copy(f"{ROOT}/cache_xwbc_lafan/human_smpl_spec.npz",f"{OUT}/human_smpl_spec.npz")
l6=torch.load(f"{SRC}/human_smpl_paired_local6d.pt",mmap=True); N=l6.shape[0]
out=torch.empty(N,22,6,dtype=torch.float32); t0=time.time(); CH=65536
for i in range(0,N,CH):
    b=l6[i:i+CH].float().to(dev); RA,_=fk_global(b,hsA); RL=RA@Q[None]
    L=RL.clone()
    for j in range(1,22): L[:,j]=RL[:,par[j]].transpose(-1,-2)@RL[:,j]
    out[i:i+CH]=matrix_to_rot6d(L).cpu()
    if (i//CH)%12==0: print(f"{i}/{N} {time.time()-t0:.0f}s",flush=True)
torch.save(out,f"{OUT}/human_smpl_paired_local6d.pt")
json.dump({"src":SRC,"Q":"/home/arenalabs/tmp/dialect/Q_final.pt","spec":"lafan human_smpl","rows":N,
           "note":"AMASS human rewritten in LAFAN BVH frame dialect (robot-bridge fit + bone-direction snap)"},open(f"{OUT}/LCONV_meta.json","w"),indent=1)
print("DONE",N,f"{time.time()-t0:.0f}s")
