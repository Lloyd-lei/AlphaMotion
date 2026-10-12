"""Differentiable global-rotation FK and verified mesh binding; no free surface output."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
_BASE=Path(__file__).resolve().parents[1]/'base'
sys.path.insert(0,str(_BASE/'legacy'))
sys.path.insert(0,str(_BASE.parent/'prismatic'))
import torch
from torch.nn import functional as F
from common import rot6d_to_matrix
from kinematics import mixed_forward_kinematics,geometry_float32
try:
    from .surface_contract import checked_surface
except ImportError:
    from surface_contract import checked_surface


@geometry_float32
def fk_global_positions(rot6,parents,rest_offsets):
    """Fixed-offset rotational tree only; outputs root-relative native body units."""
    r=rot6d_to_matrix(rot6.float());batch,joints=rot6.shape[:2]
    par=torch.as_tensor(parents)
    if par.ndim==2:
        if not torch.equal(par,par[0].expand_as(par)):raise ValueError('one skeleton per batch')
        par=par[0]
    ids=par.tolist()
    if len(ids)!=joints or ids[0]!=-1 or any(not 0<=p<j for j,p in enumerate(ids[1:],1)):raise ValueError('parent-first fixed rotational tree required')
    off=torch.as_tensor(rest_offsets,device=r.device,dtype=r.dtype)
    if off.ndim==2:off=off[None].expand(batch,-1,-1)
    if off.shape!=(batch,joints,3):raise ValueError('offset shape')
    positions=[]
    for j,p in enumerate(ids):positions.append(r.new_zeros(batch,3) if p<0 else positions[p]+torch.einsum('bij,bj->bi',r[:,p],off[:,j]))
    return torch.stack(positions,1)


@geometry_float32
def bound_surface(rot6,joint_positions,surface):
    """Rigid/LBS patch positions+normals in the joint FK frame, in native units.

    LBS is available only for verified bind weights/local frames. It is not an
    implementation of SMPL pose correctives and cannot create missing bindings.
    """
    r=rot6d_to_matrix(rot6.float());weights=surface['binding_weights'].float();joint_positions=joint_positions.float()
    local=surface['bind_local_xyz'].float();local_n=surface['bind_local_normal'].float()
    if joint_positions.shape!=(*rot6.shape[:2],3):raise ValueError('binding joint-position shape')
    moved=torch.einsum('bjrc,bpjc->bpjr',r,local)+joint_positions[:,None]
    normals=torch.einsum('bjrc,bpjc->bpjr',r,local_n)
    xyz=(weights[...,None]*moved).sum(2)
    normal=F.normalize((weights[...,None]*normals).sum(2),dim=-1)
    valid=surface['valid']
    return {'position':torch.where(valid[...,None],xyz,torch.zeros_like(xyz)),
            'normal':torch.where(valid[...,None],normal,torch.zeros_like(normal)),
            'valid':valid,'patch_ids':surface['patch_ids']}


@geometry_float32
def predict_bound_surface(rot6,g,slide_native=None):
    if 'kinematic_rest_off' not in g:raise ValueError('native-unit kinematic_rest_off required; descriptor rest_off is not a kinematic unit contract')
    fk=mixed_forward_kinematics(rot6,g,slide_native)
    pos=fk['joint_position'];rot6=fk['rot6']
    s=checked_surface(g,rot6.shape[0],rot6.shape[1])
    return {**fk,'joint_position':pos,'joint_rotation':rot6d_to_matrix(rot6.float()),
            'surface':None if s is None else bound_surface(rot6,pos,s)}


@geometry_float32
def surface_metrics(pred_rot6,target_rot6,g,*,pred_slide_native=None,target_slide_native=None,include_per_patch=True):
    """Identical fixed-mesh evaluation for ordinary A or surface B rot predictions.

    No model-specific dependency. Root-relative FK is used for both predictions.
    Position errors are reported separately in native units, /static reach, and
    meters only where declared metric. Missing surfaces return unsupported.
    """
    s=g.get('surface')
    if s is None:return {'supported':False,'reason':'no verified mesh binding','active_patches':0}
    if not s.get('_validated'):raise ValueError('validated surface required')
    pred=predict_bound_surface(pred_rot6,g,pred_slide_native)['surface'];target=predict_bound_surface(target_rot6,g,target_slide_native)['surface']
    valid=s['valid'];area=torch.where(valid,s['area'].float(),torch.zeros_like(s['area'],dtype=torch.float32))
    den=area.sum(1);count=valid.sum(1);frame_valid=den>0
    error=torch.linalg.vector_norm(pred['position']-target['position'],dim=-1)
    cosine=(pred['normal']*target['normal']).sum(-1).clamp(-1,1)
    def mean(x):
        value=(torch.where(valid,x,torch.zeros_like(x))*area).sum(1)/den.clamp_min(1e-12)
        return torch.where(frame_valid,value,torch.full_like(value,float('nan')))
    native=mean(error);normalized=native/s['pose_scale'].float()
    metric_valid=s['metric_valid']&frame_valid
    metric=torch.where(metric_valid,native*s['meters_per_unit'].float(),torch.full_like(native,float('nan')))
    result={'supported':True,'position_native_per_frame':native,'position_normalized_per_frame':normalized,
      'position_m_per_frame':metric,'normal_cosine_error_per_frame':mean(1-cosine),
      'normal_angle_rad_per_frame':mean(torch.acos(cosine)),
      'valid_patch_count_per_frame':count,'frame_valid':frame_valid,'metric_frame_valid':metric_valid,
      'active_patches':int(valid.sum().detach()),'patch_ids':s['patch_ids'],
      'normalization':{'pose_scale':s['pose_scale'],'geometry_scale':s['geometry_scale'],
       'area_sum_native2':den,'meters_per_unit':s['meters_per_unit'],'scale_policy':s['scale_policy'],
       'provenance':s.get('normalization_provenance')},
      'evidence':'fixed real mesh points bound to FK; common A/B evaluation; root-relative kinematic geometry'}
    if include_per_patch:result.update(position_native_per_patch=torch.where(valid,error,torch.full_like(error,float('nan'))),
      normal_cosine_error_per_patch=torch.where(valid,1-cosine,torch.full_like(error,float('nan'))),patch_valid=valid)
    return result


def _area_mean(values,mask,area):
    safe=torch.where(mask,values,torch.zeros_like(values))
    weights=torch.where(mask,area,torch.zeros_like(area));den=weights.sum(1)
    per=(safe*weights).sum(1)/den.clamp_min(1e-12)
    return per.sum()/(den>0).sum().clamp_min(1)


@geometry_float32
def surface_losses(pred_rot6,target_rot6,g,*,pred_slide_native=None,target_slide_native=None,root_height_m=None,root_height_valid=None,
                   contact_targets=None,contact_valid=None,contact_evidence_kind=None,
                   contact_partner_valid=None,contact_provenance=None,
                   contact_threshold_m=.02,contact_temperature_m=.0066666667,
                   w_position=1.,w_normal=.1,w_penetration=1.,w_contact=.1):
    """Patch kinematic reconstruction plus valid metric ground/contact geometry.

    Patch GT is actual mesh bound to the provided GT skeleton pose: geometry
    supervision, not independently captured surface/grasp/tactile evidence.
    Ground is the explicitly known Y=0 plane, after adding predicted root height
    in meters; root_height_valid must identify frames with a trusted ground datum.
    No pressure target is created. Unavailable ground/contact gives a graph-zero.
    """
    zero=pred_rot6.float().sum()*0
    s=g.get('surface')
    if s is None:
        return dict(total=zero,position=zero,normal=zero,penetration=zero,contact=zero,
                    diagnostics={'active_patches':0,'ground_frames':0,'contact_points':0,'evidence':'no verified mesh binding'})
    if not s.get('_validated'):raise ValueError('validated surface required')
    pred=predict_bound_surface(pred_rot6,g,pred_slide_native)['surface']
    with torch.no_grad():target=predict_bound_surface(target_rot6.detach(),g,None if target_slide_native is None else target_slide_native.detach())['surface']
    valid=s['valid'];area=s['area'].float();scale=s['pose_scale'].float()[:,None]
    lp=_area_mean(torch.linalg.vector_norm(pred['position']-target['position'],dim=-1)/scale,valid,area)
    ln=_area_mean(1-(pred['normal']*target['normal']).sum(-1).clamp(-1,1),valid,area)
    pen=zero;con=zero;ground_frames=0;contact_points=0
    if (root_height_m is None)!=(root_height_valid is None):raise ValueError('root height and trusted-ground mask must be supplied together')
    if root_height_m is not None:
        root_height_m=root_height_m.float()
        batch=len(pred_rot6)
        if root_height_m.shape!=(batch,) or root_height_valid.shape!=(batch,) or root_height_valid.dtype!=torch.bool:raise ValueError('root-height shape/mask')
        ground=root_height_valid&s['metric_valid']
        if not torch.isfinite(root_height_m[ground]).all():raise ValueError('valid metric root height must be finite')
        rh=torch.where(ground,root_height_m,torch.zeros_like(root_height_m))
        mpu=torch.where(s['metric_valid'],s['meters_per_unit'].float(),torch.ones_like(s['meters_per_unit'],dtype=torch.float32))
        height=pred['position'][...,1]*mpu[:,None]+rh[:,None]
        gm=valid&ground[:,None]
        # Scale-free penalty retains explicit metric conversion; no descriptor-derived units.
        metric_scale=(s['geometry_scale'].float()*mpu)[:,None]
        pen=_area_mean((F.relu(-height)/metric_scale).square(),gm,area)
        ground_frames=int(ground.sum().detach())
        if (contact_targets is None)!=(contact_valid is None):raise ValueError('contact target and validity must be supplied together')
        if contact_targets is not None:
            contact_targets=contact_targets.float()
            if contact_evidence_kind not in {'measured','simulated','cache_geometry_derived'}:raise ValueError('declare contact evidence provenance')
            if contact_targets.shape!=valid.shape or contact_valid.shape!=valid.shape or contact_valid.dtype!=torch.bool:raise ValueError('surface contact shape/mask')
            if contact_partner_valid is None:
                if bool((gm&contact_valid).any()):raise ValueError('active contact labels require verified source/partner validity')
                partner=torch.zeros_like(ground)
            else:
                if contact_partner_valid.shape!=(batch,) or contact_partner_valid.dtype!=torch.bool:raise ValueError('contact partner validity shape')
                partner=contact_partner_valid
            cv=gm&contact_valid&partner[:,None]
            if bool(cv.any()):
                if not isinstance(contact_provenance,dict) or contact_provenance.get('partner_kind')!='ground_plane' or len(contact_provenance.get('source_sha256',''))!=64:raise ValueError('contact source hash and ground partner provenance required')
            if not torch.isfinite(contact_targets[cv]).all() or bool(((contact_targets[cv]<0)|(contact_targets[cv]>1)).any()):raise ValueError('valid contact probabilities')
            if contact_threshold_m<=0 or contact_temperature_m<=0:raise ValueError('positive contact geometry thresholds')
            labels=torch.where(cv,contact_targets,torch.zeros_like(contact_targets))
            logits=(contact_threshold_m-height)/contact_temperature_m
            con=_area_mean(F.binary_cross_entropy_with_logits(logits,labels,reduction='none'),cv,area)
            contact_points=int(cv.sum().detach())
    elif contact_targets is not None or contact_valid is not None:raise ValueError('metric ground datum required for contact geometry')
    total=w_position*lp+w_normal*ln+w_penetration*pen+w_contact*con
    return dict(total=total,position=lp,normal=ln,penetration=pen,contact=con,
       diagnostics={'active_patches':int(valid.sum().detach()),'ground_frames':ground_frames,'contact_points':contact_points,
                    'binding_kind':s['binding_kind'],'evidence':'mesh bound to GT kinematics; not independent grasp or tactile measurement',
                    'tactile_valid':False,'contact_evidence_kind':contact_evidence_kind})
