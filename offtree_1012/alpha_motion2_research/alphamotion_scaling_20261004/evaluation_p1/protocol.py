"""P1: fixed acceptance-scaled SE3 loss and actual joint-range coordinates."""
from pathlib import Path
import dataclasses, hashlib, math, sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'evaluation'))
from trajectory_refinement import RefinementConfig


def configuration():
    return RefinementConfig(steps=100,learning_rate=.01,position_scale=1.,
        position_weight=1.,rotation_weight=1./(2*(1-math.cos(math.radians(5)))),
        position_tolerance=1.,rotation_tolerance_deg=5.,
        state_representation='P1: free own-joint global rot6 plus slide cm; actual-q optimization uses declared joint-range coordinates')


def q_from_normalized(z,actual):
    center=(actual.lower+actual.upper)/2;half=(actual.upper-actual.lower)/2
    return center.to(z)+half.to(z)*z


def normalized_from_q(q,actual):
    center=(actual.lower+actual.upper)/2;half=(actual.upper-actual.lower)/2
    return (q-center.to(q))/half.to(q)


def normalize_adapter(actual):
    actual.validate()
    half=(actual.upper-actual.lower)/2
    if not bool(torch.isfinite(half).all() and (half>0).all()):
        raise ValueError('P1 requires actual finite nonzero joint ranges')
    sources=dict(actual.source_files_sha256)
    sources[str(Path(__file__))]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return dataclasses.replace(actual,
        coordinate_units=('dimensionless',)*len(actual.joint_names),
        lower=-torch.ones_like(actual.lower),upper=torch.ones_like(actual.upper),
        project_free=lambda state:normalized_from_q(actual.project_free(state),actual),
        decode_q=lambda z:actual.decode_q(q_from_normalized(z,actual)),
        default_q=normalized_from_q(actual.default_q,actual),source_files_sha256=sources,
        model_mapping_description=actual.model_mapping_description+'; P1 optimizer z linearly normalizes each actual q range to [-1,1]; decode restores native radians/metres.',
        projection_method=actual.projection_method+'; exact affine coordinate conversion; same physical hard bounds',
        default_source=actual.default_source+'; same physical q, represented in joint-range coordinates')
