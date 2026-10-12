"""Project every displayed detail mesh vertex for every source frame and stage."""
from pathlib import Path
import argparse,json,itertools,os
os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np
import mujoco
from contracts import load_bundle,STAGES
from scene import MeshScene
from prepare import write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();records=[]
    for file in sorted((a.root/'bundles').glob('*/*/manifest.json')):
        m,z,meta=load_bundle(file.parent);scene=MeshScene(file.parent.parent/'model.mjb',meta,z,'detail');scene.render('reference',0)
        cam=scene.r.scene.camera[0];forward=np.array(cam.forward);up=np.array(cam.up);right=np.cross(forward,up);near=cam.frustum_near;top=cam.frustum_top
        offset=(scene.r.scene.camera[0].pos+scene.r.scene.camera[1].pos)/2-scene.cam.lookat
        groups={}
        for gid in range(scene.m.ngeom):
            b=int(scene.m.geom_bodyid[gid]);j=scene.lookup.get(b,-1)
            if j<(2 if len(scene.ids)>20 else 7) or scene.m.geom_group[gid]!=scene.visual_group:continue
            # Finger ROI on Franka excludes the arm link and hand mount; source mesh finger bounds are exact.
            if len(scene.ids)<20 and scene.meta['joint_names'][j] not in ('panda0_leftfinger','panda0_rightfinger'):continue
            mid=scene.m.geom_dataid[gid]
            if scene.m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
                start=scene.m.mesh_vertadr[mid];vertices=scene.m.mesh_vert[start:start+scene.m.mesh_vertnum[mid]].copy()
            elif scene.m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_BOX:vertices=np.array(list(itertools.product([-1,1],repeat=3)))*scene.m.geom_size[gid]
            else:raise ValueError('unsupported ROI primitive')
            local=np.empty(9);mujoco.mju_quat2Mat(local,scene.m.geom_quat[gid]);vertices=vertices@local.reshape(3,3).T+scene.m.geom_pos[gid];groups.setdefault(j,[]).append(vertices)
        bad=[];maxxy=0.;maxdepth=0.
        ids=[i for i,n in enumerate(scene.meta['joint_names']) if (i>=2 if len(scene.ids)>20 else n in ('panda0_leftfinger','panda0_rightfinger','end_effector_site'))]
        for stage in STAGES:
            P=scene.positions(z[stage+'_position']);R=scene.rotations(z[stage+'_rotation'])
            for t in range(len(P)):
                eye=P[t,ids].mean(0)+offset;points=np.concatenate([v@R[t,j].T+P[t,j] for j,values in groups.items() for v in values]);v=points-eye;depth=v@forward
                xy=np.stack([(v@right)*near/depth/top/(640/352),(v@up)*near/depth/top],-1);extent=float(np.abs(xy).max());maxxy=max(maxxy,extent)
                if extent>=1 or depth.min()<near:bad.append({'stage':stage,'frame_id':int(z['frame_ids'][t]),'ndc_abs_max':extent})
        records.append({'body':m['body'],'run':m['run'],'all_source_frames':len(z['frame_ids']),'stages':list(STAGES),'roi':'all palm/finger visual meshes and thumb boxes' if len(scene.ids)>20 else 'both exact finger visual meshes','projected_vertex_abs_ndc_max':maxxy,'clipped_roi_frames':bad,'status':'passed' if not bad else 'failed'});scene.close()
    write(a.root/'all_frame_mesh_bounds_qa.json',{'schema':'mesh-projection-qa-v1','records':records,'all_passed':all(x['status']=='passed' for x in records)});print(json.dumps({'all_passed':all(x['status']=='passed' for x in records),'summaries':[{k:v for k,v in x.items() if k not in ('clipped_roi_frames','stages')} for x in records]}))

if __name__=='__main__':main()
