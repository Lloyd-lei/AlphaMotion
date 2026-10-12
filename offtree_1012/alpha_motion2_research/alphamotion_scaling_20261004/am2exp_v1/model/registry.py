"""Arm -> model class registry and a seeded builder mirroring train_scaling.build_model."""
from __future__ import annotations
import random
import numpy as np
import torch
from ._sealed import ConditionalMegatron, SurfaceMegatron
from .slide_fix import SFMegatron, SFSurfaceMegatron
from .dense_surface import DenseSurfaceMegatron
from .relational import RelationalInputMegatron

MODEL_CLASSES = {
    'ConditionalMegatron': ConditionalMegatron,  # A (original)
    'SurfaceMegatron': SurfaceMegatron,  # B (original)
    'SFMegatron': SFMegatron,  # SF, SF_s1, SF_M, V1-new, G0/G1/G1_roll
    'SFSurfaceMegatron': SFSurfaceMegatron,
    'DenseSurfaceMegatron': DenseSurfaceMegatron,  # S, S_wrong (derangement in geometry), O, O_roll
    'RelationalInputMegatron': RelationalInputMegatron,  # I, I_roll (roll_relations=True)
}

# arm -> (class, model kwargs beyond config['model'], objective flags, geometry flags)
ARMS = {
    'A0': ('ConditionalMegatron', {}, {'slide_fix': False}, {}),
    'SF': ('SFMegatron', {}, {'slide_fix': True}, {}),
    'SF_s1': ('SFMegatron', {}, {'slide_fix': True}, {}),
    'SF_M': ('SFMegatron', {}, {'slide_fix': True}, {}),
    'S': ('DenseSurfaceMegatron', {}, {'slide_fix': True}, {'dense': True}),
    'S_wrong': ('DenseSurfaceMegatron', {}, {'slide_fix': True}, {'dense': True, 'derangement': 'default'}),
    'O': ('DenseSurfaceMegatron', {}, {'slide_fix': True, 'relational': True}, {'dense': True}),
    'O_roll': ('DenseSurfaceMegatron', {}, {'slide_fix': True, 'relational': True, 'relational_roll': True}, {'dense': True}),
    'I': ('RelationalInputMegatron', {}, {'slide_fix': True}, {'dense': True, 'root_height_sensor': True}),
    'I_roll': ('RelationalInputMegatron', {'roll_relations': True}, {'slide_fix': True}, {'dense': True, 'root_height_sensor': True}),
    'G0': ('SFMegatron', {}, {'slide_fix': True}, {}),
    'G1': ('SFMegatron', {}, {'slide_fix': True}, {}),
}


def model_kwargs(config, model_class, extra=None):
    kw = dict(config['model'])
    if model_class in ('SurfaceMegatron', 'SFSurfaceMegatron'):
        kw['max_patches'] = config['surface']['max_patches']
    kw.update(extra or {})
    return kw


def build_model(config, device, model_class='SFMegatron', extra=None):
    """Seeds torch/numpy/random with config['seed'] BEFORE construction (as train_scaling.build_model)."""
    torch.manual_seed(config['seed']); np.random.seed(config['seed']); random.seed(config['seed'])
    cls = MODEL_CLASSES[model_class]
    model = cls(**model_kwargs(config, model_class, extra)).to(device)
    if config.get('judge_body_order') is not None:
        model.set_judge_body_order(config['judge_body_order'])
    return model


def build_arm(config, device, arm):
    cls, extra, _, _ = ARMS[arm]
    return build_model(config, device, cls, extra)
