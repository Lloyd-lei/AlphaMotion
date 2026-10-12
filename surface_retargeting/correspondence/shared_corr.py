"""Beta-zero (neutral SMPL-X, betas = 0) shared correspondence per robot, reused across all SMPL-X body shapes
(UMR's `correspondence.reuse_across_smplx_betas`, the mechanism UMR ships for HiPHI).  One build+train per robot
(~10 s CPU + ~20 s GPU) instead of one per (clip, robot); caches live under DATA/corr/<robot>/beta0 (on /home), never
in UMR/data or UMR/output (/var/tmp)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from surface_retargeting.paths import CORR, ROBOTS, UMR_ROOT

BETA0 = [0.0] * 10
BETA0_NAME = 'smplx_neutral_betas_b80de5d138'          # UMR: sha1(float32 zeros(10))[:10]


def paths(robot):
    d = CORR / robot / 'beta0'
    return dict(dir=d, dataset=d / 'correspondence_dataset.npz', train=d / 'train',
                slots=d / 'train' / 'correspondence_slots_final.npz', binding=d / 'smplx_surface_binding.npz',
                train_cfg=ROBOTS / robot / 'umr_shared_beta0.json')


def robot_cfg_for_training(robot):
    """Robot config (staged) + correspondence outputs pointing at the beta0 cache."""
    p = paths(robot)
    rcfg = json.load(open(ROBOTS / robot / 'umr_robot.json'))
    rcfg = copy.deepcopy(rcfg)
    rcfg['correspondence'].setdefault('dataset', {})['out'] = str(p['dataset'])
    rcfg['correspondence'].setdefault('train', {}).update(out_dir=str(p['train']), template_name='auto', device='auto')
    rcfg['correspondence']['smpl_name'] = 'auto'
    rcfg['smpl_template'] = dict(source='config', type='smplx', gender='neutral', use_betas=True, use_gender=True,
                                 name='auto', betas=BETA0)
    return rcfg


def runtime_correspondence_section(robot):
    """`correspondence` overrides for the HSI/HOI runtime config (merged over the robot config)."""
    p = paths(robot)
    return dict(reuse_across_smplx_betas=True, shared_config=str(p['train_cfg']), slots=str(p['slots']),
                slots_field='reconstructed_slots', smpl_name=BETA0_NAME,
                train=dict(out_dir=str(p['train'])), dataset=dict(out=str(p['dataset'])),
                shared_smplx_template=dict(name=BETA0_NAME, gender='neutral', betas=BETA0,
                                           model_dir=str(UMR_ROOT / 'smpl'), surface_binding=str(p['binding'])))
