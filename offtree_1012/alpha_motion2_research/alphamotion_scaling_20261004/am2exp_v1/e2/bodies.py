"""Light loader for the two external bodies (adroit_pen, relay_franka_full_v2) exactly as
orchestration/scaling_data.load_scaling_bodies builds them, without loading the 16 XWBC/UniMate payloads.

Uses scaling_data.external_body + attach_geometry (sealed, read-only).  Surface patches are attached only
when the checkpoint's arm needs them (B arms); A/SF arms never read g['surface'].
"""
from __future__ import annotations
import json
from pathlib import Path

import torch

from ._env import ROOT, sealed


def load_config(run_or_config):
    p = Path(run_or_config)
    if p.is_dir():
        p = p / 'config.json'
    return json.loads(p.read_text())


def load_external_bodies(config, device='cpu', with_surface=False):
    sealed()
    import scaling_data
    from pilot_data import sha256, read_json
    qpath = ROOT / config.get('extended_qwen_cache', 'qwen/frozen_features.pt')
    if sha256(qpath) != config['extended_qwen_sha256']:
        raise ValueError('extended frozen Qwen cache changed')
    features = torch.load(qpath, map_location='cpu', weights_only=True)
    qwen = features['joint_names']
    out = {}
    for name in ['adroit_pen', 'relay_franka_full_v2']:
        b = scaling_data.external_body(ROOT, name, qwen, device)
        static = None
        if with_surface:
            from surface.surface_contract import prepare_static_surface
            path = ROOT / 'surface/patches' / f'{name}.json'
            p = read_json(path)
            if b.metadata.get('source_native_unit') == 'm':
                p = scaling_data.scaled_surface_payload(p, 100.)
            p['pose_scale'] = b.reach
            patch_qwen = features['surface_texts']
            p['qwen_sem'] = torch.stack([patch_qwen[k] for k in p['semantic_text_key']])
            p['semantic_valid'] = [True] * len(p['semantic_text_key'])
            static = prepare_static_surface(p, len(b.parents), device, max_patches=config['surface']['max_patches'])
        scaling_data.attach_geometry(b, static)
        out[name] = b
    return out


def load_model(checkpoint, config=None, model_class=None, device='cpu'):
    """Build + load a checkpoint (weights_only).  model_class: 'module:Class' (default by config group)."""
    sealed()
    import importlib
    import sys
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    cfg = config or payload.get('config')
    if cfg is None:
        cfg = load_config(Path(checkpoint).parent)
    if model_class is None:
        model_class = 'surface.surface_model:SurfaceMegatron' if cfg.get('group') == 'B' else 'pilot_model:ConditionalMegatron'
    mod, cls = model_class.split(':')
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    C = getattr(importlib.import_module(mod), cls)
    kw = {'max_patches': cfg['surface']['max_patches']} if cfg.get('group') == 'B' else {}
    model = C(**cfg['model'], **kw)
    model.load_state_dict(payload['model'], strict=True)
    model.to(device).eval()
    return model, cfg, int(payload.get('step', -1))
