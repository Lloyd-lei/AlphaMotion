import sys; sys.path.insert(0,'.')
import numpy as np
from pcl_contact import sources as SR
d=SR.load_traj('lafan::fallAndGetUp1_subject4','human_smpl'); names=[str(x) for x in d['joint_names']]
jy=d['world_positions'][...,1]; W=slice(4136,4264)
for mode in (False,'rot_axis','rot_frame'):
    w=SR.world_points(d,'human_smpl',hand_fix=mode); y=w['X'][...,1]; reg=w['asset'].region.numpy()
    row=[]
    for s,r in (('L',7),('R',8)):
        j=names.index(s+'_Wrist'); hm=reg==r; pl=jy[:,j]<5
        row.append('%s planted n=%d hand-min minus wrist p5/50/95 %s'%(s,pl.sum(),np.percentile(y[pl][:,hm].min(1)-jy[pl,j],[5,50,95]).round(1)))
    print(mode,'| clip min p1 %.1f | window min %.1f (hand %.1f, non-hand %.1f) |'%(np.percentile(y.min(1),1),y[W].min(),y[W][:,np.isin(reg,[7,8])].min(),y[W][:,~np.isin(reg,[7,8])].min()),' ; '.join(row))
print('window joint min %.1f wrist min %.1f'%(jy[W].min(), jy[W][:,[names.index('L_Wrist'),names.index('R_Wrist')]].min()))
