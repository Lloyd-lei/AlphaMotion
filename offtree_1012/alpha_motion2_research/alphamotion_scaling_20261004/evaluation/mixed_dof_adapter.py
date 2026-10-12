"""Hash-gated fixed-root hinge/slide tree from actual recorded-model metadata.

Raw state is [T,J,7]: own-joint rot6, then predicted slide coordinate in cm.
Actual q uses radians for hinges, metres for slides. No q or slide target is
accepted by project_free. Independent saved-MuJoCo FK validation is mandatory.
The earlier hinge-only adapter remains unchanged for its archived receipts.
"""
from pathlib import Path
import json, math
import numpy as np
import torch
from dof_protocol import VerifiedDOFAdapter
from hinge_dof_adapter import matrix_from_rot6, rot6_from_matrix, file_sha
import hinge_dof_adapter


class MixedJointTree:
    def __init__(self, metadata_path):
        self.metadata_path=Path(metadata_path);self.document=json.loads(self.metadata_path.read_text());m=self.document['model']
        self.parents=np.asarray(m['parents'],dtype=np.int64);self.names=tuple(m['joint_names']);J=len(self.names)
        if J<2 or self.parents.shape!=(J,) or self.parents[0]!=-1 or any(not 0<=p<j for j,p in enumerate(self.parents[1:],1)):raise ValueError('fixed-root parent-first tree required')
        if m.get('length_unit')!='m':raise ValueError('actual metadata lengths must be metres')
        self.offsets=torch.as_tensor(m['rest_offsets_m'],dtype=torch.float64)*100.
        if self.offsets.shape!=(J,3) or not torch.isfinite(self.offsets).all():raise ValueError('finite source offsets')
        entries=m['mechanical_joints'];by={}
        for e in entries:
            j=e['skeleton_index']
            if not 0<j<J or j in by or e['type'] not in ('hinge','slide') or e['origin_local_body_m']!=[0.,0.,0.]:raise ValueError('one known zero-origin hinge/slide per articulated non-root body required')
            if e['body_name']!=self.names[j]:raise ValueError('mechanical body identity mismatch')
            by[j]=e
        fixed=set()
        for node in m.get('fixed_body_nodes',[]):
            j=node['skeleton_index']
            if not 0<=j<J or j in fixed or node.get('body_name')!=self.names[j] or node.get('parent_skeleton_index')!=int(self.parents[j]) or not node.get('evidence'):raise ValueError('explicit actual fixed-node identity/evidence required')
            fixed.add(j)
        if 0 not in fixed or set(by)&fixed or set(by)|fixed!=set(range(J)):raise ValueError('all body mechanics must be explicitly mapped, including fixed root/tool')
        self.fixed=fixed
        ordered=sorted(entries,key=lambda e:e['qpos_address']);self.q_names=tuple(e['joint_name'] for e in ordered)
        self.q_addresses=[e['qpos_address'] for e in ordered];Q=len(ordered)
        if self.q_addresses!=list(range(Q)) or len(set(self.q_names))!=Q:raise ValueError('explicit contiguous actual qpos order required')
        self.body_slot={e['skeleton_index']:i for i,e in enumerate(ordered)}
        self.kinds=tuple(e['type'] for e in ordered);self.units=tuple('radian' if k=='hinge' else 'metre' for k in self.kinds)
        self.slide_bodies=tuple(e['skeleton_index'] for e in ordered if e['type']=='slide')
        self.axis=torch.as_tensor([e['axis_rest_canonical'] for e in ordered],dtype=torch.float64)
        if self.axis.shape!=(Q,3) or not torch.allclose(self.axis.norm(dim=-1),torch.ones(Q,dtype=torch.float64),atol=1e-8):raise ValueError('actual unit rest-canonical axes required')
        self.q0=torch.as_tensor([e['model_qpos0'] for e in ordered],dtype=torch.float64)
        self.lower=torch.as_tensor([e['range'][0] if e['limited'] else -math.inf for e in ordered],dtype=torch.float64)
        self.upper=torch.as_tensor([e['range'][1] if e['limited'] else math.inf for e in ordered],dtype=torch.float64)
        if not torch.isfinite(self.q0).all() or not torch.isfinite(self.lower).all() or not torch.isfinite(self.upper).all() or bool((self.lower>=self.upper).any()):raise ValueError('actual finite nonzero limit ranges required')
        for i,k in enumerate(self.kinds):
            if k=='hinge' and self.upper[i]-self.lower[i]>=2*math.pi:raise ValueError('hinge projection needs a range narrower than one full turn')
        if self.slide_bodies:
            contract=self.document.get('mixed_joint_contract',{})
            if not contract.get('coupling','').startswith('No MJCF mimic/equality constraint'):raise ValueError('source evidence of independent slide coordinates required; no invented coupling')
            if 'actual C=I' not in contract.get('prismatic_rotation',''):raise ValueError('validated fixed slide rotation inheritance required')

    def decode_q(self,q):
        if q.ndim!=2 or q.shape[1]!=len(self.q_names):raise ValueError('actual q[T,Q] order required')
        a=self.axis.to(q);theta=q-self.q0.to(q);z=torch.zeros_like(a[:,0]);B=len(q)
        K=torch.stack([z,-a[:,2],a[:,1],a[:,2],z,-a[:,0],-a[:,1],a[:,0],z],-1).reshape(-1,3,3)
        eye=torch.eye(3,device=q.device,dtype=q.dtype)
        local=eye+torch.sin(theta)[...,None,None]*K+(1-torch.cos(theta))[...,None,None]*(K@K)
        R=[eye.expand(B,3,3)];P=[q.new_zeros(B,3)];offsets=self.offsets.to(q)
        for j,p in enumerate(self.parents[1:],1):
            off=offsets[j].expand(B,3)
            if j in self.fixed:R.append(R[p])
            else:
                slot=self.body_slot[j]
                if self.kinds[slot]=='slide':
                    off=off+a[slot]*(theta[:,slot,None]*100.)
                    R.append(R[p])
                else:R.append(R[p]@local[:,slot])
            P.append(P[p]+torch.einsum('bij,bj->bi',R[p],off))
        return torch.stack(P,1),torch.stack(R,1)

    def pack_q(self,q):
        """Validation/observer conversion, never model initialization from GT q."""
        _,R=self.decode_q(q);slides=q.new_zeros(len(q),len(self.names))
        for j in self.slide_bodies:slides[:,j]=q[:,self.body_slot[j]]*100.
        return torch.cat([rot6_from_matrix(R),slides[...,None]],-1)

    def project_free(self,state):
        if state.ndim!=3 or state.shape[1:]!=(len(self.names),7):raise ValueError('raw mixed state must contain explicit predicted rot6 and slide cm: [T,J,7]')
        if not torch.isfinite(state).all():raise ValueError('nonfinite raw mixed state')
        R=matrix_from_rot6(state[...,:6]);values=[None]*len(self.q_names)
        for j,p in enumerate(self.parents[1:],1):
            if j in self.fixed:continue
            slot=self.body_slot[j]
            if self.kinds[slot]=='slide':values[slot]=state[:,j,6]/100.;continue
            M=R[:,p].transpose(-1,-2)@R[:,j];a=self.axis[slot].to(M)
            skew=torch.stack([M[:,2,1]-M[:,1,2],M[:,0,2]-M[:,2,0],M[:,1,0]-M[:,0,1]],-1)
            sine=(skew*a).sum(-1)/2
            cosine=(M.diagonal(dim1=-2,dim2=-1).sum(-1)-torch.einsum('i,bij,j->b',a,M,a))/2
            principal=torch.atan2(sine,cosine)+self.q0[slot].to(M)
            center=(self.lower[slot]+self.upper[slot]).to(M)/2
            values[slot]=principal+torch.round((center-principal)/(2*math.pi))*(2*math.pi)
        return torch.stack(values,-1)

    def aperture_cm(self,positions,rotations):
        if len(self.slide_bodies)!=2:raise ValueError('aperture requires exactly two source-declared slides')
        if not self.document.get('aperture_definition','').startswith('Signed projection of left-origin minus right-origin'):raise ValueError('no actual source aperture definition')
        left,right=self.slide_bodies
        declared=self.document.get('aperture_indices',{})
        if (declared.get('left_finger'),declared.get('right_finger'),declared.get('parent'))!=(left,right,int(self.parents[left])):raise ValueError('source aperture indices disagree with independent slide order')
        if self.parents[left]!=self.parents[right]:raise ValueError('aperture requires actual same-parent opposing slides')
        axis=self.axis[self.body_slot[left]].to(positions)
        world_axis=torch.einsum('bij,j->bi',rotations[:,self.parents[left]],axis)
        return ((positions[:,left]-positions[:,right])*world_axis).sum(-1)

    def verified_adapter(self,receipt):
        path=Path(receipt);v=json.loads(path.read_text())
        checks=(v.get('status')=='passed',v.get('metadata_sha256')==file_sha(self.metadata_path),v.get('adapter_sha256')==file_sha(__file__),
                v.get('rotation_helpers_sha256')==file_sha(hinge_dof_adapter.__file__),v.get('payload_sha256')==self.document.get('payload_sha256'),
                v.get('real_saved_qpos_frames')==self.document.get('frames'),v.get('real_saved_qpos_frames',0)>0)
        bounds={'max_fk_position_error_cm':1e-3,'max_global_rotation_matrix_error':1e-5,'hinge_roundtrip_max_error_rad':1e-6,'slide_roundtrip_max_error_m':1e-9,'finite_difference_vjp_max_error':1e-4}
        if self.slide_bodies:bounds['aperture_reference_max_error_cm']=1e-3
        if not all(checks) or any(not isinstance(v.get(k),(int,float)) or not math.isfinite(v.get(k,math.inf)) or v.get(k,math.inf)>limit for k,limit in bounds.items()):raise ValueError('actual mixed adapter validation absent/stale/failed')
        files=[self.metadata_path,path,Path(__file__),Path(hinge_dof_adapter.__file__)]
        return VerifiedDOFAdapter(self.q_names,self.units,self.lower,self.upper,self.project_free,self.decode_q,
            {str(p):file_sha(p) for p in files},
            'Actual fixed-root hinge/slide tree, explicit fixed tool, actual independent q order/axes/offsets/limits; positions cm; q radians/metres',
            'Hinges: closest axis rotation and actual branch; slides: model-predicted cm converted to metres; true limit clamp; no GT slide or inferred symmetry',
            default_q=torch.maximum(torch.minimum(self.q0,self.upper),self.lower),
            default_source='Actual model qpos0, clipped to actual per-coordinate limits. No contact/collision guarantee.')


def validate_saved_qpos(metadata_path,payload_path,output):
    tree=MixedJointTree(metadata_path);doc=tree.document
    if file_sha(payload_path)!=doc['payload_sha256']:raise ValueError('actual payload hash mismatch')
    with np.load(payload_path,allow_pickle=False) as z:
        raw=z['recorded_robot_qpos']
        if raw.ndim!=2 or raw.shape[1]!=len(tree.q_names) or len(raw)!=doc['frames']:raise ValueError('all recorded actual q coordinates and frames required')
        q=torch.tensor(raw,dtype=torch.float64);expected_position=z['position_native'];expected_rot6=z['global_rot6']
        aperture_reference=z['aperture_signed_m'] if 'aperture_signed_m' in z else None
    if tree.slide_bodies and aperture_reference is None:raise ValueError('actual two-finger aperture reference is required')
    pos=rot=hinge=slide=aperture=0.
    with torch.no_grad():
        for start in range(0,len(q),4096):
            sub=q[start:start+4096];P,R=tree.decode_q(sub)
            ep=torch.tensor(expected_position[start:start+4096],dtype=torch.float64)*100.
            er=matrix_from_rot6(torch.tensor(expected_rot6[start:start+4096],dtype=torch.float64))
            pos=max(pos,float((P-ep).abs().max()));rot=max(rot,float((R-er).abs().max()))
            delta=tree.project_free(tree.pack_q(sub))-sub
            for k,kind in enumerate(tree.kinds):
                if kind=='hinge':hinge=max(hinge,float(torch.atan2(torch.sin(delta[:,k]),torch.cos(delta[:,k])).abs().max()))
                else:slide=max(slide,float(delta[:,k].abs().max()))
            if aperture_reference is not None:
                ap=torch.tensor(aperture_reference[start:start+4096],dtype=P.dtype)*100.
                aperture=max(aperture,float((tree.aperture_cm(P,R)-ap).abs().max()))
    sample=q[[0,-1]].clone().requires_grad_(True);P,R=tree.decode_q(sample)
    gen=torch.Generator().manual_seed(21004);wp=torch.randn(P.shape,dtype=P.dtype,generator=gen);wr=torch.randn(R.shape,dtype=R.dtype,generator=gen)
    gradient=torch.autograd.grad((P*wp).sum()+(R*wr).sum(),sample)[0];numeric=torch.zeros_like(gradient);eps=1e-6
    with torch.no_grad():
        for b in range(len(sample)):
            for k in range(sample.shape[1]):
                lo=sample.detach().clone();hi=lo.clone();lo[b,k]-=eps;hi[b,k]+=eps
                pl,rl=tree.decode_q(lo);ph,rh=tree.decode_q(hi)
                numeric[b,k]=(((ph-pl)*wp).sum()+((rh-rl)*wr).sum())/(2*eps)
    ge=float((numeric-gradient).abs().max());violations=torch.relu(q-tree.upper)+torch.relu(tree.lower-q)
    passed=pos<=1e-3 and rot<=1e-5 and hinge<=1e-6 and slide<=1e-9 and ge<=1e-4 and aperture<=1e-3
    report={'status':'passed' if passed else 'failed','metadata_sha256':file_sha(metadata_path),'payload_sha256':file_sha(payload_path),
        'adapter_sha256':file_sha(__file__),'rotation_helpers_sha256':file_sha(hinge_dof_adapter.__file__),
        'real_saved_qpos_frames':len(q),'max_fk_position_error_cm':pos,'max_global_rotation_matrix_error':rot,
        'hinge_roundtrip_max_error_rad':hinge,'slide_roundtrip_max_error_m':slide,'aperture_reference_max_error_cm':aperture if aperture_reference is not None else None,
        'finite_difference_vjp_max_error':ge,'coordinate_names':list(tree.q_names),'coordinate_units':list(tree.units),
        'preserved_reference_limit_violations_per_coordinate':(violations>0).sum(0).tolist(),'reference_max_limit_overshoot_per_coordinate':violations.max(0).values.tolist(),
        'reference':'Hash-pinned saved recorded-qpos MuJoCo FK arrays. GT soft-limit excursions retained without clipping.',
        'device':'CPU','training':False,'optimizer_steps':0,'contact_collision_grasp_success':'unknown; not certified',
        'source_root':'fixed source root, no added free root DOF'}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not passed:raise ValueError(report)
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--metadata',required=True);p.add_argument('--payload',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(1);print(json.dumps(validate_saved_qpos(a.metadata,a.payload,a.output)))
