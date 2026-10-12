import sys, numpy as np, trimesh, mujoco
sys.path.insert(0, '/var/tmp/alphamotion/am2exp/review')
from rv_robot_fk import RB, S, AX, pose
for body in sys.argv[1:]:
    kin=RB.load_kinematics(body); A=S.load_surface(body); m=kin.model; d=mujoco.MjData(m)
    q=m.qpos0.copy(); R,P,_,_,_=kin.forward(q); d.qpos[:]=q; mujoco.mj_forward(m,d); root=d.xpos[kin.keep[0]].copy()
    p,_=pose(A,R,P); views=np.load(f'vis_{body}.npy'); occ=views==0
    pg=A.extra['point_geom']
    geoms,_,_=RB.visual_geoms(kin); inside=np.zeros(len(p),bool); nwt=0
    for g in geoms:
        V,F=RB.mesh_arrays(kin,g); Vw=((V@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g])-root)@AX.T*kin.unit
        mesh=trimesh.Trimesh(Vw,F,process=True)
        if not mesh.is_watertight: continue
        nwt+=1
        sel=np.flatnonzero(pg!=g)
        c=mesh.contains(p[sel]); 
        # exclude points within 2 mm of this mesh's surface
        _,dd,_=trimesh.proximity.closest_point(mesh,p[sel][c]) if c.any() else (None,np.array([]),None)
        idx=sel[c][dd>0.2] if c.any() else []
        inside[idx]=True
    print(body,'watertight geoms',nwt,'/',len(geoms),'points inside another geom',int(inside.sum()),'of which occluded',int((inside&occ).sum()),'occluded total',int(occ.sum()))
