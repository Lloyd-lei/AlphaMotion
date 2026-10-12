import numpy as np, sys
sys.path.insert(0,'.')
from pcl_contact import core as C
for b in ['unitree_g1_29dof','human_smpl']:
    z=np.load(f'out/labels_{b}__walk1_subject5_window.npz'); gn=list(z['group_names'])
    for k in ['cmp:L_foot','cmp:R_foot']:
        g=gn.index(k); c=z['contact'][:,g]; sp=z['slip_speed'][:,g]; ss=[]
        hist=np.zeros(10); cnt=np.zeros(10); vals=[[] for _ in range(10)]
        for a,e in C.segments(c):
            if a==0 or e==len(c): continue
            for t in range(a,e):
                ph=min(9,int(10*(t-a)/(e-a)))
                if np.isfinite(sp[t]): vals[ph].append(sp[t]); 
        print(b[:6],k,'stance-phase decile median slip cm/s:',[round(float(np.median(v)),1) if v else None for v in vals])
        print('            frac>4:',[round(float(np.mean(np.array(v)>4)),2) if v else None for v in vals])
