"""Estimate per-joint constant body-fixed rotation Q_j between the two human_smpl
dialects (AMASS SMPL-frames vs LAFAN BVH-frames) using the robot pairing as bridge:
  R_h^j(t) ~= R_r^b(t) @ C   (constant C per corpus)  =>  Q_j = C_A^T @ C_L,  R_L = R_A @ Q_j
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo')
from common import rot6d_to_matrix, SkeletonSpec
from train_global import fk_global
ROOT="/var/tmp/alphamotion/topo_exp"; dev="cuda"
ROBOT=os.environ.get("BRIDGE","unitree_g1_29dof"); NS=int(os.environ.get("NS","60000"))
def spec(c,n):
    d=np.load(f"{ROOT}/{c}/{n}_spec.npz",allow_pickle=True)
    return SkeletonSpec(str(d["name"]),d["parents"],d["rest_offsets"],list(d["joint_names"]))
def globals_(c,n,rows,sp):
    l6=torch.load(f"{ROOT}/{c}/{n}_paired_local6d.pt",mmap=True)[rows].float()
    Rs=[]
    for i in range(0,len(rows),8192):
        R,_=fk_global(l6[i:i+8192].to(dev),sp); Rs.append(R)
    return torch.cat(Rs)
def polar(M):
    U,S,Vt=torch.linalg.svd(M); R=U@Vt
    d=torch.det(R); U2=U.clone(); U2[...,:,2]*=torch.where(d<0,-1.0,1.0)[...,None]
    return U2@Vt
def geo(A,B):
    c=((A.transpose(-1,-2)@B).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
    return torch.rad2deg(torch.arccos(c.clamp(-1,1)))
g=torch.Generator().manual_seed(0)
out={}
for tag,c in [("L","cache_xwbc_lafan"),("A","cache_xwbc_amass_full")]:
    N=torch.load(f"{ROOT}/{c}/human_smpl_paired_local6d.pt",mmap=True).shape[0]
    rows=torch.randperm(N,generator=g)[:NS].sort().values
    hs=spec(c,"human_smpl"); rs=spec(c,ROBOT)
    Rh=globals_(c,"human_smpl",rows,hs); Rr=globals_(c,ROBOT,rows,rs)     # [n,22,3,3],[n,B,3,3]
    # fit C for every (j,b): M_jb = sum_t Rr_b^T Rh_j
    M=torch.einsum("tbxy,tjxz->jbyz",Rr,Rh)          # [22,B,3,3]
    C=polar(M)
    res=torch.stack([geo(Rh[:,j][:,None].expand(-1,Rr.shape[1],-1,-1), Rr@C[j][None]).mean(0) for j in range(22)])  # [22,B]
    out[tag]=dict(C=C.cpu(),res=res.cpu(),hs=hs,rs=rs)
    print(tag,"rows",N,"sampled",NS,"robot bodies",Rr.shape[1],flush=True)
hs=out["L"]["hs"]; rnames=out["L"]["rs"].joint_names; assert rnames==out["A"]["rs"].joint_names
Q=torch.zeros(22,3,3); tab=[]
for j in range(22):
    worst=torch.maximum(out["L"]["res"][j],out["A"]["res"][j]); b=int(worst.argmin())
    CL=out["L"]["C"][j,b]; CA=out["A"]["C"][j,b]; Q[j]=CA.T@CL
    ang=float(geo(torch.eye(3),Q[j]))
    tab.append((hs.joint_names[j],rnames[b],float(out["L"]["res"][j,b]),float(out["A"]["res"][j,b]),ang))
    print(f"{hs.joint_names[j]:11s} <- {rnames[b]:24s} resL={tab[-1][2]:5.1f}° resA={tab[-1][3]:5.1f}°  |Q|={ang:6.1f}°")
torch.save({"Q":Q,"table":tab,"robot":ROBOT,"joint_names":hs.joint_names},f"/home/arenalabs/tmp/dialect/Q_{ROBOT}.pt")
