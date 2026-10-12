"""Exact fixed-root, one-zero-origin-hinge-per-link adapter from real model metadata.

This intentionally rejects ball/free/slide/multi-axis joints and nonzero pivots.
Global rotations use G_j = AX R_world,j(q) R_rest,j^T AX^T. Consequently
G_j = G_parent Rot(axis_rest_canonical,j, q_j-qpos0_j), and positions use
G_parent @ rest_offset. Verification against independently MuJoCo-generated
saved-qpos FK is mandatory before exposing a VerifiedDOFAdapter.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
import torch
from dof_protocol import VerifiedDOFAdapter


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix_from_rot6(x):
    a,b=x[...,:3],x[...,3:];a=torch.nn.functional.normalize(a,dim=-1)
    b=torch.nn.functional.normalize(b-(a*b).sum(-1,keepdim=True)*a,dim=-1)
    return torch.stack((a,b,torch.cross(a,b,dim=-1)),-1)


def rot6_from_matrix(x):return torch.cat((x[..., :,0],x[..., :,1]),-1)


class SingleHingeTree:
    def __init__(self,metadata_path):
        self.metadata_path=Path(metadata_path);self.document=json.loads(self.metadata_path.read_text());m=self.document['model']
        self.parents=np.asarray(m['parents'],dtype=np.int64);self.names=tuple(m['joint_names']);J=len(self.names)
        if J<2 or self.parents.shape!=(J,) or self.parents[0]!=-1 or any(not 0<=p<j for j,p in enumerate(self.parents[1:],1)):raise ValueError('fixed-root parent-first body tree')
        if m.get('length_unit')!='m':raise ValueError('metadata offsets must explicitly be metres')
        self.offsets=torch.as_tensor(m['rest_offsets_m'],dtype=torch.float64)*100.
        entries=m['mechanical_joints'];by={}
        for e in entries:
            j=e['skeleton_index']
            if j in by or j==0 or e['type']!='hinge' or e['origin_local_body_m']!=[0.,0.,0.]:raise ValueError('requires exactly one zero-origin hinge per non-root body, fixed root')
            if e['body_name']!=self.names[j]:raise ValueError('hinge/body identity mismatch')
            by[j]=e
        fixed_nodes=m.get('fixed_body_nodes',[])
        fixed=set()
        for node in fixed_nodes:
            j=node['skeleton_index']
            if node.get('body_name')!=self.names[j] or node.get('parent_skeleton_index')!=int(self.parents[j]) or not node.get('evidence'):raise ValueError('fixed-node source identity')
            fixed.add(j)
        if 0 not in fixed:raise ValueError('explicit actual fixed-root source evidence required')
        fixed.remove(0)
        if set(by)&fixed or set(by)|fixed!=set(range(1,J)):raise ValueError('missing actual hinge mapping or explicitly verified fixed-node declaration')
        self.fixed=fixed
        ordered=sorted(entries,key=lambda e:e['qpos_address']);self.q_names=tuple(e['joint_name'] for e in ordered)
        self.q_addresses=[e['qpos_address'] for e in ordered]
        Q=len(ordered)
        if self.q_addresses!=list(range(Q)) or len(set(self.q_names))!=Q:raise ValueError('only explicit contiguous robot qpos mapping supported')
        self.body_slot={e['skeleton_index']:i for i,e in enumerate(ordered)}
        self.axis=torch.as_tensor([e['axis_rest_canonical'] for e in ordered],dtype=torch.float64)
        if not torch.allclose(self.axis.norm(dim=-1),torch.ones(Q,dtype=torch.float64),atol=1e-8):raise ValueError('actual unit hinge axes required')
        self.q0=torch.as_tensor([e['model_qpos0'] for e in ordered],dtype=torch.float64)
        self.lower=torch.as_tensor([e['range'][0] if e['limited'] else -math.inf for e in ordered],dtype=torch.float64)
        self.upper=torch.as_tensor([e['range'][1] if e['limited'] else math.inf for e in ordered],dtype=torch.float64)
        if bool((self.lower>self.upper).any()):raise ValueError('reversed model limits')
        if bool(((self.upper-self.lower)>=2*math.pi).any()):raise ValueError('projection requires actual limited hinge intervals narrower than one full turn')

    def decode_q(self,q):
        if q.ndim!=2 or q.shape[1]!=len(self.q_names):raise ValueError('actual robot q[T,Q]')
        B=len(q);a=self.axis.to(q);theta=q-self.q0.to(q);zero=torch.zeros_like(a[:,0])
        K=torch.stack([zero,-a[:,2],a[:,1],a[:,2],zero,-a[:,0],-a[:,1],a[:,0],zero],-1).reshape(-1,3,3)
        eye=torch.eye(3,device=q.device,dtype=q.dtype)
        local=eye+torch.sin(theta)[...,None,None]*K+ (1-torch.cos(theta))[...,None,None]*(K@K)
        R=[eye.expand(B,3,3)];P=[q.new_zeros(B,3)];off=self.offsets.to(q)
        for j,p in enumerate(self.parents[1:],1):
            R.append(R[p] if j in self.fixed else R[p]@local[:,self.body_slot[j]])
            P.append(P[p]+torch.einsum('bij,j->bi',R[p],off[j]))
        return torch.stack(P,1),torch.stack(R,1)

    def project_free(self,global_rot6):
        if global_rot6.ndim!=3 or global_rot6.shape[1:]!=(len(self.names),6):raise ValueError('projector expects own-joint global/Tpose-relative rot6[T,J,6]')
        R=matrix_from_rot6(global_rot6);values=[None]*len(self.q_names)
        for j,p in enumerate(self.parents[1:],1):
            if j in self.fixed:continue
            slot=self.body_slot[j];M=R[:,p].transpose(-1,-2)@R[:,j];a=self.axis[slot].to(M)
            skew=torch.stack([M[:,2,1]-M[:,1,2],M[:,0,2]-M[:,2,0],M[:,1,0]-M[:,0,1]],-1)
            sin=(skew*a).sum(-1)/2
            cos=(M.diagonal(dim1=-2,dim2=-1).sum(-1)-torch.einsum('i,bij,j->b',a,M,a))/2
            principal=torch.atan2(sin,cos)+self.q0[slot].to(M)
            # Select the equivalent2pi branch closest to the actual legal interval.
            center=(self.lower[slot]+self.upper[slot]).to(M)/2
            turns=torch.round((center-principal)/(2*math.pi))
            values[slot]=principal+turns*(2*math.pi)
        return torch.stack(values,-1)

    def verified_adapter(self,validation_receipt):
        path=Path(validation_receipt);v=json.loads(path.read_text())
        if v.get('status')!='passed' or v.get('metadata_sha256')!=file_sha(self.metadata_path) or v.get('adapter_sha256')!=file_sha(__file__):raise ValueError('actual model adapter validation absent/stale')
        if v.get('payload_sha256')!=self.document.get('payload_sha256') or v.get('real_saved_qpos_frames')!=self.document.get('frames') or v.get('real_saved_qpos_frames',0)<1 or v.get('max_fk_position_error_cm',math.inf)>1e-3 or v.get('max_global_rotation_matrix_error',math.inf)>1e-5 or v.get('finite_difference_vjp_max_error',math.inf)>1e-4:raise ValueError('actual FK/gradient validation did not pass')
        return VerifiedDOFAdapter(self.q_names,('radian',)*len(self.q_names),self.lower,self.upper,self.project_free,self.decode_q,
            {str(self.metadata_path):file_sha(self.metadata_path),str(path):file_sha(path),str(Path(__file__)):file_sha(__file__)},
            'Actual model metadata: fixed source root; one hinge/zero pivot per articulated link and explicit fixed tool nodes; exact rest-frame rebase and limits; XYZ in centimetres',
            'Closest per-joint relative rotation about actual rest-canonical axis using atan2, then true joint-limit clamp; not a global IK optimum',
            default_q=torch.maximum(torch.minimum(self.q0,self.upper),self.lower),
            default_source='Actual source model.qpos0, clipped only to actual named-joint limits; collision/contact feasibility remains separately checked')


def validate_saved_qpos(metadata_path,payload_path,output):
    """CPU-only independent comparison to saved MuJoCo FK, plus finite-difference VJP."""
    tree=SingleHingeTree(metadata_path);doc=tree.document
    if file_sha(payload_path)!=doc['payload_sha256']:raise ValueError('real payload SHA mismatch')
    with np.load(payload_path,allow_pickle=False) as z:
        raw_q=z['recorded_robot_qpos'];extra=set(range(raw_q.shape[1]))-set(tree.q_addresses)
        allowed={e['qpos_address'] for e in doc.get('unsupported_prismatic_joints',[]) if e.get('retained_in_recorded_robot_qpos') and e.get('excluded_from_rotation_only_training_skeleton_and_surface_loss')}
        if extra!=allowed:raise ValueError('extra saved qpos components need explicit source-supported exclusion')
        q=torch.tensor(raw_q[:,tree.q_addresses],dtype=torch.float64)
        expected_position=z['position_native'];expected_rot6=z['global_rot6']
    pos=rot=roundtrip=0.
    with torch.no_grad():
        for start in range(0,len(q),4096):
            sub=q[start:start+4096];p,R=tree.decode_q(sub)
            ep=torch.tensor(expected_position[start:start+4096],dtype=torch.float64)*100.
            er=matrix_from_rot6(torch.tensor(expected_rot6[start:start+4096],dtype=torch.float64))
            pos=max(pos,float((p-ep).abs().max()));rot=max(rot,float((R-er).abs().max()))
            delta=tree.project_free(rot6_from_matrix(R))-sub
            roundtrip=max(roundtrip,float(torch.atan2(torch.sin(delta),torch.cos(delta)).abs().max()))
    sample=q[[0,-1]].clone().requires_grad_(True);p,R=tree.decode_q(sample)
    generator=torch.Generator().manual_seed(21004)
    wp=torch.randn(p.shape,dtype=p.dtype,generator=generator);wr=torch.randn(R.shape,dtype=R.dtype,generator=generator)
    scalar=(p*wp).sum()+(R*wr).sum();gradient=torch.autograd.grad(scalar,sample)[0]
    numeric=torch.zeros_like(gradient);eps=1e-6
    with torch.no_grad():
        for b in range(len(sample)):
            for j in range(sample.shape[1]):
                lo=sample.detach().clone();hi=lo.clone();lo[b,j]-=eps;hi[b,j]+=eps
                pl,rl=tree.decode_q(lo);ph,rh=tree.decode_q(hi)
                numeric[b,j]=(((ph-pl)*wp).sum()+((rh-rl)*wr).sum())/(2*eps)
    ge=float((numeric-gradient).abs().max())
    passed=pos<=1e-3 and rot<=1e-5 and ge<=1e-4 and roundtrip<=1e-6
    report={'status':'passed' if passed else 'failed','metadata_sha256':file_sha(metadata_path),'payload_sha256':file_sha(payload_path),
            'adapter_sha256':file_sha(__file__),'real_saved_qpos_frames':len(q),'max_fk_position_error_cm':pos,
            'max_global_rotation_matrix_error':rot,'free_rotation_q_roundtrip_max_error_rad':roundtrip,'finite_difference_vjp_max_error':ge,
            'reference':'independently recorded-qpos MuJoCo FK arrays in hash-pinned payload; actual axes/origin/limits metadata',
            'device':'CPU','optimizer_steps':0,'simulator_steps':0,'training':False,'root':tree.names[0]+' fixed; no fake global root DOF','explicit_excluded_qpos_addresses':sorted(extra),
            'contact':'unknown object/contact omitted; no grasp-success claim'}
    Path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if not passed:raise ValueError(report)
    return report


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--metadata',required=True);parser.add_argument('--payload',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();torch.set_num_threads(1);print(json.dumps(validate_saved_qpos(args.metadata,args.payload,args.output)))
