"""I_roll eval: roll by B/2 inside ONE eval window (B=T) vs across a training-like batch."""
import sys, json, numpy as np, torch
from pathlib import Path
ROOT=Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
sys.path.insert(0,str(ROOT))
from am2exp_v1.common.paths import add_sealed_paths; add_sealed_paths()
import train_scaling as original
from am2exp_v1.model import geometry as G, relational as REL, ground
torch.set_num_threads(8)
cfg=json.loads((ROOT/'orchestration/configs/A_S_seed0.json').read_text())
bodies,_=original.load_scaling_bodies(ROOT,cfg,torch.device('cpu'))
by={b.name:b for b in bodies}
G.attach_dense_geometry([by['human_smpl']],torch.device('cpu'),grouping='g256')
h=by['human_smpl']; st=h._dense_surface_static
z=np.load('/var/tmp/alphamotion/am2exp/integration/e1/A0/sensor_on/human_smpl__unitree_g1_29dof__fallAndGetUp1_subject4@4136.npz')
x=torch.as_tensor(z['source_pose9']).float(); rh=torch.as_tensor(z['source_root_height_input_m']).float()
T=len(x); g=h.geom(T)
f0,v0=REL.relational_features(x,None,g,st,rh,roll=False)
f1,v1=REL.relational_features(x,None,g,st,rh,roll=True)
def near(f): return (f[...,2]>0)  # frac(h<3cm)>0 per group  [B,G]
def agree(a,b,valid): 
    a=near(a);b=near(b);m=valid
    return float(((a==b)&m).sum()/m.sum()), float(((a&b)&m).sum()/max(1,int((a&m).sum())))
print('window frames',T,'sensor-valid frames',int(torch.isfinite(rh).sum()))
print('eval roll (same window, +T/2 frames): agreement, recall of own near-ground groups =',agree(f0,f1,v0))
# training-like: 128 random train rows of human (different groups), with real sensor
rng=np.random.default_rng(0)
from pilot_data import sample_rows
rows=sample_rows(h,rng,128)
xb=h.tensor('pose9',rows,torch.device('cpu')).float()
rhb=torch.as_tensor(np.where(ground.row_ground_ok(h,rows,'cpu').numpy(),h.arrays['root_height_m'][rows],np.nan)).float()
gb=h.geom(128)
b0,w0=REL.relational_features(xb,None,gb,st,rhb,roll=False)
b1,w1=REL.relational_features(xb,None,gb,st,rhb,roll=True)
print('train roll (random rows across clips): agreement, recall =',agree(b0,b1,w0))
cos=lambda a,b: float(torch.nn.functional.cosine_similarity(a.flatten(1),b.flatten(1)).mean())
print('mean cosine own vs rolled features: eval window',cos(f0,f1),' train batch',cos(b0,b1))
