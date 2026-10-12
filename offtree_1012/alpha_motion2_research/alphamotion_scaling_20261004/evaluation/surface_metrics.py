"""Compare A/B using the SAME verified mesh patch set bound through shared FK.

Mesh-bound GT is derived from skeleton kinematics, not an independent surface or
contact measurement. Mesh absence is NA; no synthetic patch ground truth added.
"""
import torch
from eval_engine import metric_summary


def evaluate_fixed_surface(predicted_rot6,reference_rot6,geom,*,bind_fn,expected_patch_sha256,predicted_slide_native=None,reference_slide_native=None):
    s=geom.get('surface')
    if s is None:return {'status':'NA','reason':'no verified mesh/binding for this body'}
    if not s.get('_validated') or len(expected_patch_sha256)!=64 or s.get('asset_sha256')!=expected_patch_sha256:raise ValueError('same verified fixed patch asset SHA required for A and B')
    with torch.no_grad():
        if predicted_slide_native is None and reference_slide_native is None:
            predicted=bind_fn(predicted_rot6,geom)['surface'];reference=bind_fn(reference_rot6,geom)['surface']
        else:
            if predicted_slide_native is None or reference_slide_native is None:raise ValueError('both predicted/reference slide states must be explicit')
            predicted=bind_fn(predicted_rot6,geom,slide_native=predicted_slide_native)['surface']
            reference=bind_fn(reference_rot6,geom,slide_native=reference_slide_native)['surface']
        if predicted is None or reference is None:raise ValueError('verified patch unexpectedly unavailable')
        if predicted['patch_ids']!=reference['patch_ids'] or predicted['patch_ids']!=s['patch_ids']:raise ValueError('patch identity changed')
        valid=s['valid'];area=s['area']
        if valid.ndim!=2 or not torch.equal(valid,valid[0].expand_as(valid)) or not torch.equal(area,area[0].expand_as(area)):raise ValueError('fixed patch set/area cannot change with model or frame')
        if not bool(valid.any()):return {'status':'NA','reason':'no valid verified mesh patches'}
        if not torch.equal(predicted['valid'],valid) or not torch.equal(reference['valid'],valid):raise ValueError('model-specific patch dropping prohibited')
        p=predicted['position'];t=reference['position'];pn=predicted['normal'];tn=reference['normal']
        distance=(p-t).norm(dim=-1);dot=(pn*tn).sum(-1).clamp(-1,1);angle=torch.acos(dot)*(180/torch.pi)
        metric=s['metric_valid']
        if not torch.equal(metric,metric[0].expand_as(metric)):raise ValueError('one declared unit system per body clip')
        unit='cm' if bool(metric[0]) else 'canonical_body_units'
        if bool(metric[0]):distance=distance*s['meters_per_unit'][:,None]*100.
        normed=(p-t).norm(dim=-1)/s['pose_scale'][:,None]
        weights=torch.where(valid,area,torch.zeros_like(area));den=weights.sum(1)
        def means(values):return (torch.where(valid,values,torch.zeros_like(values))*weights).sum(1)/den
        return {'status':'measured','patch_set_sha256':expected_patch_sha256,'patch_ids':list(s['patch_ids']),
                'valid_patches_per_frame':int(valid[0].sum()),'position_area_mean':metric_summary(means(distance).cpu().numpy(),unit),
                'position_over_reach_area_mean':metric_summary(means(normed).cpu().numpy(),'reach_fraction'),
                'normal_degrees_area_mean':metric_summary(means(angle).cpu().numpy(),'degree'),
                'evidence':'same fixed real mesh points bound to predicted and reference skeleton FK; no independent surface or tactile capture',
                'comparison_policy':'identical patch hash and geometry for A/B; no per-model resampling'}
