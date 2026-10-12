import sys, re; sys.path.insert(0,'.'); import amctrl, mujoco, numpy as np
from amctrl.unitree_h2.h2_constants import get_spec, H2_ACTUATORS
def build(self_col):
    spec=get_spec(); spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10,10,0.1], contype=1, conaffinity=1)
    for b in spec.bodies:
        for g in b.geoms:
            if g.name.endswith('_collision'):
                g.contype = 1 if self_col else 0; g.conaffinity = 1
                if 'foot' in g.name: g.condim=3; g.priority=1; g.friction=[0.6,0.005,0.0001]
    for a in H2_ACTUATORS:
        for j in spec.joints:
            if any(re.fullmatch(p, j.name) for p in a.target_names_expr):
                act=spec.add_actuator(name='a_'+j.name, target=j.name, trntype=mujoco.mjtTrn.mjTRN_JOINT, ctrlrange=[-3.14,3.14], forcerange=[-a.effort_limit,a.effort_limit])
                act.gainprm[0]=a.stiffness; act.biasprm[1]=-a.stiffness; act.biasprm[2]=-a.damping; j.armature=a.armature
    m=spec.compile(); m.opt.timestep=0.005; return m
def box_min_z(m,d):
    zmin=1e9
    for g in range(m.ngeom):
        n=m.geom(g).name
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX and n.endswith('_collision'):
            R=d.geom_xmat[g].reshape(3,3); zmin=min(zmin, d.geom_xpos[g][2]-np.abs(R[2]).dot(m.geom_size[g]))
        elif 'prim' in n:   # cylinder: axis = local z; lowest point = centre - |axis_z|*half_len - radius*sqrt(1-axis_z^2)
            R=d.geom_xmat[g].reshape(3,3); az=abs(R[2,2]); r,h=m.geom_size[g][0],m.geom_size[g][1]
            zmin=min(zmin, d.geom_xpos[g][2]-az*h-r*np.sqrt(max(0.0,1-az*az)))
    return zmin
def clamp(m, rows, margin=0.005):
    d=mujoco.MjData(m); lift=np.zeros(len(rows))
    for t,r in enumerate(rows):
        d.qpos[:]=np.concatenate([r[:3], r[3:7][[3,0,1,2]], r[7:]]); mujoco.mj_kinematics(m,d); lift[t]=max(0.0, margin-box_min_z(m,d))
    return lift
def spawn_test(m, rows, t, hold_steps=40):
    d=mujoco.MjData(m); r=rows[t]; d.qpos[:]=np.concatenate([r[:3], r[3:7][[3,0,1,2]], r[7:]]); d.qvel[:]=0; d.ctrl[:]=r[7:]
    mujoco.mj_forward(m,d); p0=d.qpos[:3].copy(); h0=d.xpos[m.body('left_hand_link').id].copy(); vmax=0
    for _ in range(hold_steps):
        mujoco.mj_step(m,d); vmax=max(vmax, np.linalg.norm(d.qvel[:3]))
    return np.linalg.norm(d.qpos[:3]-p0)*100, np.linalg.norm(d.xpos[m.body('left_hand_link').id]-h0)*100, vmax
if __name__=='__main__':
    for name,csv in (('getup','/var/tmp/arena/h2_bm/csv/crutch_rise_up_R_001__A519.csv'),('sitfloor','/var/tmp/arena/h2_bm/csv/sit_croos_legged_stop_001__A024.csv')):
        rows=np.loadtxt(csv,delimiter=','); print('=====',name)
        for sc in (True, False):
            m=build(sc); lift=clamp(m, rows); rows_c=rows.copy(); rows_c[:,2]+=lift
            print(f'  self_collision={sc}: lift p50 {np.median(lift)*100:.1f} cm, max {lift.max()*100:.1f} cm, frames lifted {np.mean(lift>0):.2f}')
            for t in (0,30,90,150):
                a=spawn_test(m, rows, t); b=spawn_test(m, rows_c, t)
                print(f'    frame {t:3d}  raw: root moved {a[0]:5.1f} cm hand {a[1]:5.1f} cm vmax {a[2]:5.2f} m/s | clamped: root {b[0]:5.1f} cm hand {b[1]:5.1f} cm vmax {b[2]:5.2f} m/s')
