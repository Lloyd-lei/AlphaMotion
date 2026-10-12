"""Matched compiled MuJoCo meshes, rigid link transforms, no mj_step."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np

class MeshScene:
    def __init__(self,path,meta,z,view):
        self.m=mujoco.MjModel.from_binary_path(str(path));self.d=mujoco.MjData(self.m);mujoco.mj_forward(self.m,self.d)
        self.meta=meta['model'];self.z=z;self.view=view;self.A=np.array(self.meta['AX_zup_to_yup']);self.restR=np.array(self.meta['rest_body_rotations_world_zup']);self.ids=self.meta['body_ids'];self.parents=self.meta['parents']
        self.lookup={b:i for i,b in enumerate(self.ids)};self.r=mujoco.Renderer(self.m,height=352,width=640,max_geom=15000)
        self.m.light_active[:]=False
        self.m.vis.headlight.ambient[:]=[.25,.25,.25];self.m.vis.headlight.diffuse[:]=[.65,.65,.65]
        self.visual_group=1 if len(self.ids)>20 else 0
        self.opt=mujoco.MjvOption();self.opt.geomgroup[:]=1
        self.cam=mujoco.MjvCamera();self.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        allp=np.concatenate([z[k+'_position'] for k in ('reference','native','q0','q20','q100','default100')],0)/100@self.A
        lo=allp.min((0,1));hi=allp.max((0,1));self.cam.lookat[:]=(lo+hi)/2;self.cam.distance=max(float(np.max(hi-lo))*2.15,.18)
        self.cam.azimuth=135 if len(self.ids)<20 else 155;self.cam.elevation=-18 if len(self.ids)<20 else -32
        if len(self.ids)>20:
            self.cam.azimuth=65;self.cam.elevation=-40;self.cam.distance*=.53
            self.cam.lookat[:]=self.positions(z['reference_position'][:,2:]).mean((0,1))
        self.radius=max(float(np.max(hi-lo)),.10)
        self.full_camera=(self.cam.lookat.copy(),float(self.cam.distance))
        self.verify_q_transforms()

    def positions(self,p):return p/100@self.A
    def rotations(self,R):return self.A.T@R@self.A@self.restR

    def verify_q_transforms(self):
        errors=[]
        for stage in ('q0','q20','q100','default100'):
            P=self.positions(self.z[stage+'_position']);R=self.rotations(self.z[stage+'_rotation'])
            for t,q in enumerate(self.z[stage+'_native_q']):
                self.d.qpos[:]=self.m.qpos0;self.d.qpos[:len(q)]=q;mujoco.mj_forward(self.m,self.d)
                actual=self.d.xpos[self.ids]-self.d.xpos[self.ids[0]]
                err=max(float(np.max(np.abs(actual-P[t]))),float(np.max(np.abs(self.d.xmat[self.ids].reshape(-1,3,3)-R[t]))));errors.append(err)
        self.transform_error=max(errors)
        if self.transform_error>2e-5:raise ValueError('mesh/body canonical transform does not match independent mj_forward: '+str(self.transform_error))
        self.d.qpos[:]=self.m.qpos0;mujoco.mj_forward(self.m,self.d)

    def line(self,a,b,color,width=None):
        s=self.r.scene
        if np.linalg.norm(a-b)<1e-8:return
        g=s.geoms[s.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_CAPSULE,width or self.radius*.0025,a,b);s.ngeom+=1

    def render(self,stage,t,full=False):
        self.cam.lookat[:]=self.full_camera[0];self.cam.distance=self.full_camera[1]
        if self.view=='detail' and not full:
            ids=[i for i,n in enumerate(self.meta['joint_names']) if (i>=2 if len(self.ids)>20 else n in ('panda0_leftfinger','panda0_rightfinger','end_effector_site'))]
            self.cam.lookat[:]=self.positions(self.z[stage+'_position'][t,ids]).mean(0);self.cam.distance=.34 if len(self.ids)>20 else .42
        self.r.update_scene(self.d,camera=self.cam,scene_option=self.opt);s=self.r.scene
        P=self.positions(self.z[stage+'_position'][t]);R=self.rotations(self.z[stage+'_rotation'][t]);ref=self.positions(self.z['reference_position'][t]);keep=[]
        for k in range(s.ngeom):
            g=s.geoms[k]
            if g.objtype!=mujoco.mjtObj.mjOBJ_GEOM:continue
            gid=g.objid;b=int(self.m.geom_bodyid[gid])
            if b not in self.lookup:continue
            if self.view=='detail' and not full and self.lookup[b]<(2 if len(self.ids)>20 else 7):continue
            # Source visual geoms only when a body has a separate visual group.
            if self.m.geom_group[gid]!=self.visual_group:continue
            j=self.lookup[b];quat=np.empty(9);mujoco.mju_quat2Mat(quat,self.m.geom_quat[gid])
            g.pos[:]=P[j]+R[j]@self.m.geom_pos[gid];g.mat[:]=R[j]@quat.reshape(3,3)
            g.rgba[3]=1
            keep.append(k)
        for target,src in enumerate(keep):
            if target!=src:
                # MjvGeom is a copyable pybind struct via explicit exposed fields.
                x=s.geoms[src];y=s.geoms[target]
                for field in ('type','dataid','matid','objtype','objid','category','texcoord','segid','emission','specular','shininess','reflectance','camdist','modelrbound','transparent'):setattr(y,field,getattr(x,field))
                for field in ('size','pos','mat','rgba'):getattr(y,field)[:]=getattr(x,field)
        s.ngeom=len(keep)
        for j,p in enumerate(self.parents):
            if p>=0:
                self.line(ref[p],ref[j],[.40,.65,.80,.55]);self.line(P[p],P[j],[.04,.18,.21,.70],self.radius*.0016)
        ix=self.z['condition_joint_indices'][t]
        for c,j in enumerate(ix):
            target=self.positions(self.z['condition_position_cm'][t,c]);color=[.86,.15,.23,1] if c==0 else [.76,.31,.86,1]
            self.line(P[j],target,color,self.radius*.004)
            path=self.positions(self.z['condition_position_cm'][:,c]);actual=self.positions(self.z[stage+'_position'][:,j])
            for k in range(1,len(path)):self.line(path[k-1],path[k],[*color[:3],.35],self.radius*.0013)
            for k in range(1,t+1):self.line(actual[k-1],actual[k],[.05,.65,.50,.65],self.radius*.0015)
            for axis,cl in enumerate(([.9,.15,.2,1],[.15,.7,.2,1],[.2,.35,.9,1])):
                direction=(self.A.T@self.z['condition_global_rotation'][t,c])[:,axis]*self.radius*.055
                self.line(target,target+direction,cl,self.radius*.002)
        rgb=self.r.render().copy()
        self.r.enable_depth_rendering();depth=self.r.render();self.r.disable_depth_rendering()
        rgb[depth>=self.m.vis.map.zfar*self.m.stat.extent*.99]=[241,244,246]
        return rgb

    def close(self):self.r.close()
