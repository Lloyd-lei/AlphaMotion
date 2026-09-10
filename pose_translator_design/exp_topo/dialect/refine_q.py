"""Refine robot-fitted Q: (1) bootstrap two disjoint halves -> twist uncertainty;
(2) for real anatomical bones, snap the swing so the LAFAN-FK bone direction equals the
AMASS bone direction exactly (minimal rotation, keeps the fitted twist)."""
import os, sys, numpy as np, torch
sys.path.insert(0, '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo')
from common import SkeletonSpec
from train_global import fk_global
ROOT="/var/tmp/alphamotion/topo_exp"; dev="cuda"; ROBOT="unitree_g1_29dof"
def spec(c,n):
    d=np.load(f"{ROOT}/{c}/{n}_spec.npz",allow_pickle=True)
    return SkeletonSpec(str(d["name"]),d["parents"],d["rest_offsets"],list(d["joint_names"]))
def globals_(c,n,rows,sp):
    l6=torch.load(f"{ROOT}/{c}/{n}_paired_local6d.pt",mmap=True)[rows].float(); Rs=[]
    for i in range(0,len(rows),8192): Rs.append(fk_global(l6[i:i+8192].to(dev),sp)[0])
    return torch.cat(Rs)
def polar(M):
    U,S,Vt=torch.linalg.svd(M); d=torch.det(U@Vt); U=U.clone(); U[...,:,2]*=torch.where(d<0,-1.0,1.0)[...,None]; return U@Vt
def geo(A,B):
    c=((A.transpose(-1,-2)@B).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2; return torch.rad2deg(torch.arccos(c.clamp(-1,1)))
def rot_u_to_v(u,v):
    u=u/u.norm(); v=v/v.norm(); w=torch.cross(u,v,dim=0); s=w.norm(); c=(u*v).sum()
    if s<1e-9: return torch.eye(3,device=u.device)
    K=torch.tensor([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]],device=u.device)
    return torch.eye(3,device=u.device)+K+K@K*((1-c)/s**2)
fit=torch.load("/home/arenalabs/tmp/dialect/Q_unitree_g1_29dof.pt"); names=fit["joint_names"]; tab=fit["table"]
hsL=spec("cache_xwbc_lafan","human_smpl"); hsA=spec("cache_xwbc_amass_full","human_smpl"); par=[int(p) for p in hsL.parents]
rL=spec("cache_xwbc_lafan",ROBOT); rn=rL.joint_names
link=[rn.index(t[1]) for t in tab]
# ---- bootstrap
g=torch.Generator().manual_seed(7); Qh=[]
for tag,c in [("L","cache_xwbc_lafan"),("A","cache_xwbc_amass_full")]:
    N=torch.load(f"{ROOT}/{c}/human_smpl_paired_local6d.pt",mmap=True).shape[0]
    rows=torch.randperm(N,generator=g)[:120000].sort().values
    Rh=globals_(c,"human_smpl",rows,spec(c,"human_smpl")); Rr=globals_(c,ROBOT,rows,spec(c,ROBOT))
    Cs=[]
    for half in (slice(0,60000),slice(60000,120000)):
        M=torch.stack([torch.einsum("txy,txz->yz",Rr[half,link[j]],Rh[half,j]) for j in range(22)]); Cs.append(polar(M))
    Qh.append(Cs)
Q1=torch.stack([Qh[1][0][j].T@Qh[0][0][j] for j in range(22)]); Q2=torch.stack([Qh[1][1][j].T@Qh[0][1][j] for j in range(22)])
Qm=polar((Q1+Q2)/2)     # average of halves = full-fit estimate
boot=geo(Q1,Q2)
# ---- swing snap on anatomical bones
BONES={"L_Hip":"L_Knee","R_Hip":"R_Knee","L_Knee":"L_Ankle","R_Knee":"R_Ankle","L_Ankle":"L_Toe","R_Ankle":"R_Toe",
       "L_Shoulder":"L_Elbow","R_Shoulder":"R_Elbow","L_Elbow":"L_Wrist","R_Elbow":"R_Wrist"}
offL=torch.tensor(hsL.rest_offsets,device=dev,dtype=torch.float32); offA=torch.tensor(hsA.rest_offsets,device=dev,dtype=torch.float32)
Q=Qm.clone(); snap=torch.zeros(22)
for j,c in BONES.items():
    ji,ci=names.index(j),names.index(c); u=Q[ji]@offL[ci]; v=offA[ci]
    Rfix=rot_u_to_v(u,v); snap[ji]=geo(torch.eye(3,device=dev),Rfix).cpu(); Q[ji]=Rfix@Q[ji]
print(f"{'joint':11s} {'link':24s} boot(half-vs-half) swing-snap")
for j in range(22): print(f"{names[j]:11s} {rn[link[j]]:24s} {boot[j]:6.1f}°   {snap[j]:5.1f}°")
torch.save({"Q":Q.cpu(),"Q_unsnapped":Qm.cpu(),"boot_deg":boot.cpu(),"snap_deg":snap,"joint_names":names,"link":[rn[l] for l in link]},"/home/arenalabs/tmp/dialect/Q_final.pt")
