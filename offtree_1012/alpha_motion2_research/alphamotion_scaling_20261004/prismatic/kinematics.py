"""Shared A/B typed kinematics: free global revolute rotations + true slide q.

No inverse-kinematic/axis projection is claimed for revolute joints. Fixed and
prismatic rotations are derived, never independently predicted. Native lengths
are explicit and independent of the legacy descriptor's units.
"""
from pathlib import Path
from functools import wraps
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'base'/'legacy'))
import torch
from common import rot6d_to_matrix,matrix_to_rot6d

UNKNOWN,FIXED,REVOLUTE,PRISMATIC,ROOT=0,1,2,3,4


def geometry_float32(function):
    """Keep geometric matmul/einsum in float32 inside an outer AMP context.

    Geometric entry points take a tensor first and explicitly cast the operands
    they use. Disabling AMP alone does not promote an already-bfloat16 tensor.
    Neural projections/attention outside these entry points retain caller AMP.
    """
    first_name=function.__code__.co_varnames[0]
    @wraps(function)
    def wrapped(*args,**kwargs):
        first=args[0] if args else kwargs[first_name]
        with torch.autocast(device_type=first.device.type,enabled=False):
            return function(*args,**kwargs)
    return wrapped


def _field(g,key,shape,device,dtype,default=None):
    if key not in g:
        if default is None:raise ValueError('typed kinematics requires '+key)
        return torch.full(shape,default,device=device,dtype=dtype)
    value=torch.as_tensor(g[key],device=device,dtype=dtype)
    if value.shape==shape[1:]:value=value[None].expand(shape)
    if value.shape!=shape:raise ValueError('typed kinematic shape '+key+': '+str(tuple(value.shape)))
    return value


def joint_metadata(g,batch,joints,device,dtype=torch.float32):
    # The model may call this with an AMP-produced descriptor. Static rotation
    # orthogonality/determinant validation must never become bf16 matmul.
    with torch.autocast(device_type=torch.device(device).type,enabled=False):
        return _joint_metadata_float32(g,batch,joints,device)


def _joint_metadata_float32(g,batch,joints,device):
    dtype=torch.float32
    typ=_field(g,'joint_type',(batch,joints),device,torch.long,UNKNOWN)
    if bool(((typ<UNKNOWN)|(typ>ROOT)).any()):raise ValueError('unknown joint_type code')
    slide=typ==PRISMATIC;fixed=typ==FIXED;derived=slide|fixed
    active=bool(slide.any());rigid=bool(derived.any())
    axis=_field(g,'prismatic_axis',(batch,joints,3),device,dtype,0. if not active else None)
    lower=_field(g,'slide_lower',(batch,joints),device,dtype,0. if not active else None)
    upper=_field(g,'slide_upper',(batch,joints),device,dtype,0. if not active else None)
    reference=_field(g,'slide_reference',(batch,joints),device,dtype,0. if not active else None)
    if active:
        if not all(bool(torch.isfinite(v[slide]).all()) for v in (axis,lower,upper,reference)):raise ValueError('finite active slide axis/limits/reference required')
        if bool((upper[slide]<=lower[slide]).any()):raise ValueError('slide upper must exceed lower')
        if not torch.allclose(axis[slide].norm(dim=-1),torch.ones_like(lower[slide]),atol=2e-5):raise ValueError('unit canonical parent-rest slide axis required')
    # Invalid fields may contain NaN in a source payload: sanitize before arithmetic.
    axis=torch.where(slide[...,None],axis,torch.zeros_like(axis))
    lower=torch.where(slide,lower,torch.zeros_like(lower));upper=torch.where(slide,upper,torch.zeros_like(upper))
    reference=torch.where(slide,reference,torch.zeros_like(reference))
    if rigid:
        key='kinematic_fixed_rotation' if 'kinematic_fixed_rotation' in g else 'prismatic_fixed_rotation'
        rotation=_field(g,key,(batch,joints,3,3),device,dtype)
        selected=rotation[derived]
        if not torch.isfinite(selected).all() or not torch.allclose(selected.transpose(-1,-2)@selected,torch.eye(3,device=device,dtype=dtype).expand_as(selected),atol=3e-5) or not torch.allclose(torch.linalg.det(selected),torch.ones(len(selected),device=device,dtype=dtype),atol=3e-5):raise ValueError('proper parent-to-derived rotation required')
    else:rotation=torch.eye(3,device=device,dtype=dtype).expand(batch,joints,3,3)
    rotation=torch.where(derived[...,None,None],rotation,torch.eye(3,device=device,dtype=dtype))
    return dict(joint_type=typ,slide_valid=slide,derived_rotation=derived,axis=axis,
                lower=lower,upper=upper,reference=reference,fixed_rotation=rotation)


def parent_order(g,joints):
    if 'parents' not in g:raise ValueError('native FK requires parents')
    parent=torch.as_tensor(g['parents'])
    if parent.ndim==2:
        if not torch.equal(parent,parent[0].expand_as(parent)):raise ValueError('one parent topology per batch')
        parent=parent[0]
    ids=parent.tolist()
    if len(ids)!=joints or ids[0]!=-1 or any(not 0<=p<j for j,p in enumerate(ids[1:],1)):raise ValueError('parent-first kinematic tree required')
    return ids


@geometry_float32
def derive_joint_rotations(raw_rot6,g,metadata=None):
    raw_rot6=raw_rot6.float()
    b,j=raw_rot6.shape[:2];m=metadata or joint_metadata(g,b,j,raw_rot6.device)
    fixed_root=torch.as_tensor(g.get('root_fixed',False),device=raw_rot6.device,dtype=torch.bool)
    if fixed_root.ndim==0:fixed_root=fixed_root.expand(b)
    if fixed_root.shape!=(b,):raise ValueError('root_fixed scalar or [B] boolean required')
    if not bool(m['derived_rotation'].any()) and not bool(fixed_root.any()):return raw_rot6
    parent=parent_order(g,j)
    if bool(m['derived_rotation'][:,0].any()):raise ValueError('root cannot be fixed/prismatic in this free-root contract')
    raw=rot6d_to_matrix(raw_rot6.float());out=[]
    for index,p in enumerate(parent):
        value=raw[:,index]
        if index==0:value=torch.where(fixed_root[:,None,None],torch.eye(3,device=raw.device,dtype=raw.dtype),value)
        if p>=0:value=torch.where(m['derived_rotation'][:,index,None,None],out[p]@m['fixed_rotation'][:,index],value)
        out.append(value)
    return matrix_to_rot6d(torch.stack(out,1))


@geometry_float32
def decode_slide(logits,g,metadata=None):
    b,j=logits.shape;m=metadata or joint_metadata(g,b,j,logits.device)
    normalized=torch.where(m['slide_valid'],torch.sigmoid(logits.float()),torch.zeros_like(logits,dtype=torch.float32))
    native=torch.where(m['slide_valid'],m['lower']+normalized*(m['upper']-m['lower']),torch.zeros_like(normalized))
    return normalized,native,m['slide_valid']


def source_slide_values(values,g,batch,joints,device,hidden=None):
    """Visible source q only; hidden/missing non-slide values cannot leak or NaN."""
    m=joint_metadata(g,batch,joints,device)
    visible=m['slide_valid'] if hidden is None else m['slide_valid']&~hidden
    if values is None:
        if bool(visible.any()):raise ValueError('visible prismatic source requires source_slide_normalized')
        values=torch.zeros(batch,joints,device=device)
    values=torch.as_tensor(values,device=device,dtype=torch.float32)
    if values.shape!=(batch,joints):raise ValueError('source_slide_normalized shape')
    if not torch.isfinite(values[visible]).all():raise ValueError('visible source slide must be finite')
    # True source q may exceed limits slightly; preserve it for explicit data QC.
    values=torch.where(visible,values,torch.zeros_like(values))
    return values,visible,m


@geometry_float32
def mixed_forward_kinematics(rot6,g,slide_native=None):
    """P_j=P_parent+G_parent@(rest_offset+axis*(q-q_reference)).

    The axis must already be in the declared canonical rest parent frame. Rest
    offsets correspond to q_reference. No implicit mimic or gripper coupling.
    """
    b,j=rot6.shape[:2];m=joint_metadata(g,b,j,rot6.device)
    constrained=derive_joint_rotations(rot6,g,m);r=rot6d_to_matrix(constrained.float());parent=parent_order(g,j)
    off=_field(g,'kinematic_rest_off',(b,j,3),r.device,r.dtype)
    if not torch.isfinite(off).all():raise ValueError('finite native kinematic_rest_off required')
    if slide_native is None:
        if bool(m['slide_valid'].any()):raise ValueError('mixed FK requires explicit slide_native; no target q is inferred')
        q=torch.zeros(b,j,device=r.device,dtype=r.dtype)
    else:
        q=torch.as_tensor(slide_native,device=r.device,dtype=r.dtype)
        if q.shape!=(b,j) or not torch.isfinite(q[m['slide_valid']]).all():raise ValueError('finite active slide_native shape')
        q=torch.where(m['slide_valid'],q,torch.zeros_like(q))
    translated=off+m['axis']*(q-m['reference'])[...,None]
    pos=[]
    for index,p in enumerate(parent):pos.append(r.new_zeros(b,3) if p<0 else pos[p]+torch.einsum('bij,bj->bi',r[:,p],translated[:,index]))
    return {'rot6':constrained,'joint_rotation':r,'joint_position':torch.stack(pos,1),'slide_native':q,'slide_valid':m['slide_valid']}


@geometry_float32
def kinematic_predictions(raw_rot6,slide_logits,g):
    metadata=joint_metadata(g,*raw_rot6.shape[:2],raw_rot6.device)
    normalized,native,valid=decode_slide(slide_logits,g,metadata)
    result=mixed_forward_kinematics(raw_rot6,g,native)
    return {**result,'slide_normalized':normalized,'slide_valid':valid}
