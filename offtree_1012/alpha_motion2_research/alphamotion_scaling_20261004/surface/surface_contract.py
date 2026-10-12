"""Geometry-only stable patch contract; never estimates bindings/scales from heldout motion."""
from __future__ import annotations
import torch
from torch.nn import functional as F

TENSOR_FIELDS=('valid','rest_xyz','rest_normal','area','binding_weights','bind_local_xyz',
 'bind_local_normal','qwen_sem','semantic_valid','geometry_scale','pose_scale','metric_valid','meters_per_unit')


def prepare_static_surface(payload,joints,device='cpu',max_patches=96):
    """Validate one body and return [1,...] tensors, ready for expand_surface().

    Rigid convenience form: anchor_joint[P],local_xyz[P,3],local_normal[P,3].
    Dense/LBS form: binding_weights[P,J],bind_local_xyz/normal[P,J,3].
    Arrays have native body length units, an explicit Y-up canonical frame, and
    actual mesh face/barycentric IDs. Frozen qwen_sem contains1024D unit vectors.
    No input here is inferred from animation; absent mechanics/tactile stay absent.
    """
    if payload.get('schema_version')!='stable_surface_v1':raise ValueError('surface schema')
    if payload.get('canonical_up')!='Y':raise ValueError('surface requires explicit Y-up canonical frame')
    if payload.get('scale_policy') not in {'geometry_only','train_only'}:raise ValueError('surface scale cannot be fitted on heldout motion')
    if payload.get('heldout_body',False) and payload['scale_policy']!='geometry_only':raise ValueError('unseen-body descriptor scale must be geometry-only')
    if payload.get('binding_kind') not in {'rigid','lbs'}:raise ValueError('verified rigid or LBS binding required')
    if payload.get('joint_motion_kind','rotational_tree') not in {'rotational_tree','mixed_tree','rotational_prismatic_tree'}:raise ValueError('prismatic/q kinematics require verified mixed-tree metadata')
    if payload.get('evidence_kind') not in {'rigid_mjcf_mesh','verified_lbs_mesh','synthetic_test'}:raise ValueError('surface evidence kind')
    ids=list(payload.get('patch_ids',[]));p=len(ids)
    if not 0<p<=max_patches or len(set(ids))!=p:raise ValueError('stable unique patch IDs / budget')
    for kind,digest in payload.get('geometry_hashes',{}).items():
        if not isinstance(digest,str) or len(digest)!=64:raise ValueError('geometry hash '+kind)
    if not payload.get('geometry_hashes'):raise ValueError('geometry asset hashes required')
    def tensor(key,dtype=torch.float32):return torch.as_tensor(payload[key],device=device,dtype=dtype)
    valid=tensor('valid',torch.bool)
    if valid.shape!=(p,):raise ValueError('patch validity shape')
    def finite(key,shape):
        x=tensor(key)
        if x.shape!=shape or not torch.isfinite(x[valid]).all():raise ValueError('surface shape/finiteness '+key)
        return torch.where(valid.reshape((p,)+(1,)*(x.ndim-1)),x,torch.zeros_like(x))
    rest=finite('rest_xyz',(p,3));normal=finite('rest_normal',(p,3));area=finite('area',(p,))
    if bool((area[valid]<=0).any()):raise ValueError('positive mesh patch area required')
    if not torch.allclose(normal[valid].norm(dim=-1),torch.ones_like(area[valid]),atol=2e-4):raise ValueError('unit rest normals required')
    if 'binding_weights' in payload:
        weights=finite('binding_weights',(p,joints));local=finite('bind_local_xyz',(p,joints,3));ln=finite('bind_local_normal',(p,joints,3))
    else:
        anchor=tensor('anchor_joint',torch.long)
        if anchor.shape!=(p,) or bool(((anchor[valid]<0)|(anchor[valid]>=joints)).any()):raise ValueError('binding joint outside skeleton')
        anchor=torch.where(valid,anchor,torch.zeros_like(anchor))
        weights=F.one_hot(anchor,joints).float()*valid[:,None]
        local=weights[...,None]*finite('local_xyz',(p,3))[:,None,:]
        ln=weights[...,None]*finite('local_normal',(p,3))[:,None,:]
    if bool((weights<0).any()) or not torch.allclose(weights[valid].sum(-1),torch.ones_like(area[valid]),atol=1e-5):raise ValueError('binding weights must be nonnegative and sum to one')
    if payload['binding_kind']=='rigid' and bool(((weights>0).sum(-1)[valid]!=1).any()):raise ValueError('rigid patches must bind exactly one joint')
    active_influence=(weights>0)&valid[:,None]
    if not torch.allclose(ln[active_influence].norm(dim=-1),torch.ones_like(weights[active_influence]),atol=2e-4):raise ValueError('unit bind normals required')
    sem=tensor('qwen_sem');sv=tensor('semantic_valid',torch.bool)
    if sem.shape!=(p,1024) or sv.shape!=(p,) or bool((sv&~valid).any()):raise ValueError('frozen patch Qwen shape/mask')
    if not bool(torch.isfinite(sem[sv]).all()) or not torch.allclose(sem[sv].norm(dim=-1),torch.ones_like(area[sv]),atol=2e-4):raise ValueError('frozen patch Qwen normalization')
    sem=torch.where(sv[:,None],sem,torch.zeros_like(sem)).detach()
    geo=float(payload['geometry_scale']);pose=float(payload['pose_scale'])
    if not (0<geo<float('inf') and 0<pose<float('inf')):raise ValueError('positive finite explicit scales')
    metric=bool(payload.get('metric_valid',False));mpu=float(payload.get('meters_per_unit',float('nan')))
    if metric and not 0<mpu<float('inf'):raise ValueError('metric geometry needs declared conversion')
    if not metric:mpu=float('nan')
    out=dict(valid=valid,rest_xyz=rest,rest_normal=normal,area=area,binding_weights=weights,
       bind_local_xyz=local,bind_local_normal=ln,qwen_sem=sem,semantic_valid=sv,
       geometry_scale=torch.tensor(geo,device=device),pose_scale=torch.tensor(pose,device=device),
       metric_valid=torch.tensor(metric,device=device),meters_per_unit=torch.tensor(mpu,device=device))
    out={k:v[None] for k,v in out.items()}
    out.update(_validated=True,patch_ids=tuple(ids),binding_kind=payload['binding_kind'],
      scale_policy=payload['scale_policy'],evidence_kind=payload['evidence_kind'],
      geometry_hashes=dict(payload['geometry_hashes']),canonical_up='Y',
      tactile_valid=False,tactile_note='No pressure field is synthesized or supervised.',
      normalization_provenance=payload.get('normalization_provenance'),
      frame_provenance=payload.get('frame_provenance'))
    return out


def expand_surface(static,batch):
    if not static.get('_validated'):raise ValueError('prepare_static_surface first')
    return {k:(v.expand(batch,*v.shape[1:]) if k in TENSOR_FIELDS else v) for k,v in static.items()}


def scale_surface_units(static,factor,*,destination_unit):
    """Convert a validated binding's native length unit, preserving physical mesh.

    Example: factor=100,destination_unit='cm' for a source measured in meters.
    Caller must apply the same factor to kinematic_rest_off and skeleton pose_scale.
    Descriptor rest_off is separate. Normals, text, weights and stable IDs stay fixed.
    This is an explicit unit conversion; it never fits motion statistics.
    """
    import math
    factor=float(factor)
    if not static.get('_validated') or not math.isfinite(factor) or factor<=0:raise ValueError('validated surface and positive finite unit conversion required')
    if not isinstance(destination_unit,str) or not destination_unit:raise ValueError('explicit destination unit required')
    out={k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in static.items()}
    for k in ('rest_xyz','bind_local_xyz','geometry_scale','pose_scale'):out[k]=out[k]*factor
    out['area']=out['area']*factor**2
    out['meters_per_unit']=out['meters_per_unit']/factor
    out['unit_conversion']={'factor':factor,'destination_unit':destination_unit,'source_asset_sha256':static.get('asset_sha256'),'normalization_fit':False}
    return out


def checked_surface(g,batch,joints):
    s=g.get('surface')
    if s is None:return None
    if not s.get('_validated'):raise ValueError('surface must pass prepare_static_surface before forward')
    p=s['valid'].shape[1]
    if s['valid'].shape!=(batch,p) or s['binding_weights'].shape!=(batch,p,joints):raise ValueError('surface batch/skeleton mismatch')
    if s['qwen_sem'].requires_grad:raise ValueError('Qwen tower/cache must be frozen')
    if 'pose_scale' not in g or torch.as_tensor(g['pose_scale']).shape!=(batch,) or not torch.allclose(g['pose_scale'].to(s['pose_scale']),s['pose_scale'],atol=1e-6,rtol=1e-6):raise ValueError('surface and skeleton pose_scale must match explicitly')
    if g.get('joint_motion_kind','rotational_tree') not in {'rotational_tree','mixed_tree','rotational_prismatic_tree'}:raise ValueError('prismatic/q kinematics require verified mixed-tree metadata')
    return s


def load_surface_asset(path,qwen_vectors,joints,device='cpu',*,qwen_cache_sha256=None,allow_unencoded=False):
    """Load a derived patch JSON and attach root-owned frozen Qwen cache vectors.

    qwen_vectors maps stable SHA256 text keys from required_surface_texts.json to
    actual1024D normalized vectors. Missing vectors fail unless an explicit CPU
    geometry-only diagnostic requested allow_unencoded; never use that for B runs.
    """
    import json,hashlib
    from pathlib import Path
    raw=Path(path).read_bytes();payload=json.loads(raw)
    values=[];known=[]
    for key in payload['semantic_text_key']:
        if key in qwen_vectors:values.append(torch.as_tensor(qwen_vectors[key]).detach().cpu());known.append(True)
        elif allow_unencoded:values.append(torch.zeros(1024));known.append(False)
        else:raise ValueError('missing actual frozen patch text vector '+key)
    if any(known) and (not isinstance(qwen_cache_sha256,str) or len(qwen_cache_sha256)!=64):raise ValueError('actual frozen Qwen cache SHA256 required')
    payload['qwen_sem']=torch.stack(values);payload['semantic_valid']=known
    static=prepare_static_surface(payload,joints,device)
    static.update(asset_sha256=hashlib.sha256(raw).hexdigest(),qwen_cache_sha256=qwen_cache_sha256,
      semantic_evidence='actual frozen cache supplied by caller' if all(known) else 'geometry-only diagnostic: text vectors unavailable',
      binding_audit=payload['binding_audit'],kinematic_adapter=payload['kinematic_adapter'])
    return static
