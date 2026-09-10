import os, sys, numpy as np, torch
sys.path.insert(0, '/home/arenalabs/Desktop/"be water, robot"/pose_translator_design/exp_topo')
from common import rot6d_to_matrix, matrix_to_rot6d, SkeletonSpec
from train_global import fk_global
ROOT="/var/tmp/alphamotion/topo_exp"; dev="cuda"
QF=os.environ.get("QF","/home/arenalabs/tmp/dialect/Q_unitree_g1_29dof.pt")
def spec(c,n):
    d=np.load(f"{ROOT}/{c}/{n}_spec.npz",allow_pickle=True)
    return SkeletonSpec(str(d["name"]),d["parents"],d["rest_offsets"],list(d["joint_names"]))
def to_local(R,par):
    L=R.clone()
    for j in range(1,R.shape[1]): L[:,j]=R[:,par[j]].transpose(-1,-2)@R[:,j]
    return L
def geo(A,B):
    c=((A.transpose(-1,-2)@B).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
    return torch.rad2deg(torch.arccos(c.clamp(-1,1)))
def chordal_mean(R):  # [n,3,3]
    U,S,Vt=torch.linalg.svd(R.mean(0)); M=U@Vt
    if torch.det(M)<0: U[:,2]*=-1; M=U@Vt
    return M
Q=torch.load(QF)["Q"].to(dev)
hsL=spec("cache_xwbc_lafan","human_smpl"); hsA=spec("cache_xwbc_amass_full","human_smpl"); par=[int(p) for p in hsL.parents]
g=torch.Generator().manual_seed(1)
NA=torch.load(f"{ROOT}/cache_xwbc_amass_full/human_smpl_paired_local6d.pt",mmap=True).shape[0]
rowsA=torch.randperm(NA,generator=g)[:40000].sort().values
l6A=torch.load(f"{ROOT}/cache_xwbc_amass_full/human_smpl_paired_local6d.pt",mmap=True)[rowsA].float().to(dev)
RA,PA=fk_global(l6A,hsA)
RL=RA@Q[None]                       # converted globals
LL=to_local(RL,par); l6conv=matrix_to_rot6d(LL)
RL2,PL=fk_global(l6conv,hsL)        # FK on LAFAN skeleton
# bone-direction agreement (physical truth, convention-free)
print("bone direction error after conversion (deg), per bone:")
for j in range(1,22):
    dA=torch.nn.functional.normalize(PA[:,j]-PA[:,par[j]],dim=-1); dL=torch.nn.functional.normalize(PL[:,j]-PL[:,par[j]],dim=-1)
    ang=torch.rad2deg(torch.arccos((dA*dL).sum(-1).clamp(-1,1)))
    print(f"  {hsL.joint_names[j]:11s} mean={ang.mean():5.1f} p95={ang.quantile(0.95):5.1f}")
# distribution check: mean local rotation, converted-AMASS vs native LAFAN, vs LAFAN half-vs-half floor
NL=torch.load(f"{ROOT}/cache_xwbc_lafan/human_smpl_paired_local6d.pt",mmap=True).shape[0]
rowsL=torch.randperm(NL,generator=g)[:80000]
l6L=torch.load(f"{ROOT}/cache_xwbc_lafan/human_smpl_paired_local6d.pt",mmap=True)[rowsL].float().to(dev)
Lnat=rot6d_to_matrix(l6L); Lconv=LL; Lraw=rot6d_to_matrix(l6A)
print("\nmean-local-rotation distance to native LAFAN (deg): raw AMASS | converted | LAFAN-vs-LAFAN floor")
for j in range(22):
    mN1=chordal_mean(Lnat[:40000,j]); mN2=chordal_mean(Lnat[40000:,j]); mC=chordal_mean(Lconv[:,j]); mR=chordal_mean(Lraw[:,j])
    print(f"  {hsL.joint_names[j]:11s} raw={float(geo(mN1,mR)):6.1f}  conv={float(geo(mN1,mC)):6.1f}  floor={float(geo(mN1,mN2)):5.1f}")
