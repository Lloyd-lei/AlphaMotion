import numpy as np, sys
sys.path.insert(0,'.')
from pcl_contact import core as C
for b in ['unitree_g1_29dof','human_smpl']:
    z=np.load(f'out/labels_{b}__walk1_subject5_window.npz'); gn=list(z['group_names'])
    X=z['X'].astype(np.float32); d=z['d'].astype(np.float32); reg=z['region']
    V=C.velocity(C.smooth_positions(X,5,2),30.0)
    for k,r in [('cmp:R_foot',14)]:
        g=gn.index(k); c=z['contact'][:,g]; idx=np.flatnonzero(reg==r)
        res=[[] for _ in range(10)]
        for a,e in C.segments(c):
            if a==0 or e==len(c): continue
            for t in range(a,e):
                dg=d[t,idx]; sup=idx[dg<=dg.min()+0.5]
                ph=min(9,int(10*(t-a)/(e-a)))
                vt=np.linalg.norm(V[t,sup][:,[0,2]],axis=1); vy=V[t,sup][:,1]
                # position of support layer along foot long axis: fraction from heel(0) to toe(1)
                P=X[t,idx]; ax=np.linalg.svd(P[:,[0,2]]-P[:,[0,2]].mean(0))[2][0]
                pr=(X[t,idx][:,[0,2]]@ax); ps=(X[t,sup][:,[0,2]]@ax)
                fr=(np.median(ps)-pr.min())/(pr.max()-pr.min())
                res[ph].append((np.percentile(vt,25),np.median(vy),dg.min(),len(sup),fr))
        print(b, k)
        for ph in range(10):
            a=np.array(res[ph]); print('  decile',ph,'vt_p25 %.1f  vy %.1f  dmin %.2f  nsup %.0f  pos-along-foot %.2f (0/1=ends)'%tuple(np.median(a,0)))
