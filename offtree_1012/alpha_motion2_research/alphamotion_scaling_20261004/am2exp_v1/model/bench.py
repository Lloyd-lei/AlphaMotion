"""CPU micro-benchmark (time + saved-activation memory) of one human->G1 training direction per arm.

Run from ROOT:
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" \
    timeout 1800 /home/arenalabs/miniconda3/envs/ardy/bin/python -m am2exp_v1.model.bench --batch 16
Writes OUT/model/bench_cpu_B<batch>.json.  Real dense assets are used when WP-ASSETS has written
them, otherwise schema-conformant synthetic capsule assets with the same P/G/K/S sizes.

GPU estimate: activation bytes scale ~linearly in batch; we report the ratio arm/A and scale the
measured A_S XWBC peak (2.38 GiB, logs/A_S_seed0_cuda_validation.json, B=128 bf16 incl. 9
directions run sequentially) by it.  CPU time ratios over-state GPU ratios (GPU steps are
launch-overhead bound: A_S 0.413 s vs B_S 0.581 s per XWBC step despite 3.7x encoder tokens).
"""
from __future__ import annotations
import argparse
import json
import statistics
import time
import numpy as np
import torch
from ._sealed import ROOT, OUT
from . import surface_api as sa
from .geometry import attach_dense_geometry, detach_dense_geometry
from .losses_relational import make_objective
from .registry import build_model
from .ground import source_root_height_sensor

A_S_XWBC_PEAK_GIB = 2.38
B_S_XWBC_PEAK_GIB = 6.81


def saved_bytes(fn):
    seen = {}

    def pack(t):
        try:
            st = t.untyped_storage(); key = (st.data_ptr(), st.nbytes())
        except Exception:
            key = (id(t), t.numel() * t.element_size())
        seen[key] = key[1]
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, int(sum(seen.values()))


def run(batch=16, reps=3, threads=8, names=('human_smpl', 'unitree_g1_29dof')):
    import train_scaling as original
    from pilot_data import make_inputs, sample_rows
    torch.set_num_threads(threads)
    dev = torch.device('cpu')
    cfg = json.loads((ROOT / 'orchestration/configs/A_S_seed0.json').read_text())
    bodies, audit = original.load_scaling_bodies(ROOT, cfg, dev)
    cfg['judge_body_order'] = audit['judge_body_order']
    by = {b.name: b for b in bodies}
    real = all(sa.real_asset_available(n) for n in names) and sa.real_module() is not None
    src, tgt = by[names[0]], by[names[1]]
    rng = np.random.default_rng(0); rows = sample_rows(src, rng, batch)
    inp, target = make_inputs(src, tgt, rows, rng, dev, 'full', contact_dropout=0.0)
    rh = source_root_height_sensor(src, rows, inp['source_contact'], dev)
    base = original.loss_function(cfg)
    results = {}
    arms = [('SF_warmup', 'SFMegatron', {}, 'g256', False),
            ('A', 'ConditionalMegatron', {}, 'g256', False), ('SF', 'SFMegatron', {}, 'g256', False),
            ('S_g256', 'DenseSurfaceMegatron', {}, 'g256', False),
            ('S_g256_joints_only', 'DenseSurfaceMegatron', {'dense_group_attention': 'joints_only'}, 'g256', False),
            ('S_g128', 'DenseSurfaceMegatron', {}, 'g128', False),
            ('I_g256', 'RelationalInputMegatron', {}, 'g256', False),
            ('O_g256', 'DenseSurfaceMegatron', {}, 'g256', True)]
    for label, cls, extra, grouping, relational in arms:
        detach_dense_geometry(bodies)
        if cls not in ('ConditionalMegatron', 'SFMegatron'):
            assets = None if real else {n: sa.build_synthetic_asset(n, by[n].parents, by[n].offsets, by[n].names) for n in names}
            attach_dense_geometry(bodies, dev, assets=assets, grouping=grouping)
        model = build_model(cfg, dev, cls, extra); model.train()
        obj = make_objective(base, slide_fix=cls != 'ConditionalMegatron', relational=relational)
        kw = {'source_root_height_m': rh} if cls == 'RelationalInputMegatron' else {}

        def step():
            out = model.forward_kinematic(**inp, source_geom=src.geom(batch), target_geom=tgt.geom(batch), **kw)
            loss, _ = obj(out, target, tgt, rows, inp['condition'], cfg['loss'])
            return loss
        times, fwd = [], []
        for r in range(reps + 1):
            model.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            if r == 0:
                loss, nbytes = saved_bytes(step)
            else:
                loss = step()
            t1 = time.perf_counter(); loss.backward(); t2 = time.perf_counter()
            if r:
                fwd.append(t1 - t0); times.append(t2 - t0)
        params = sum(p.numel() for p in model.parameters())
        results[label] = {'class': cls, 'extra': extra, 'grouping': grouping, 'params_total': params,
                          'cpu_fwd_s_median': statistics.median(fwd), 'cpu_fwd_bwd_s_median': statistics.median(times),
                          'saved_activation_bytes': nbytes}
        del model
    detach_dense_geometry(bodies)
    results.pop('SF_warmup')
    ref_t = results['SF']['cpu_fwd_bwd_s_median']; ref_m = results['A']['saved_activation_bytes']
    for label, r in results.items():
        r['time_ratio_vs_SF'] = r['cpu_fwd_bwd_s_median'] / ref_t
        r['activation_ratio_vs_A'] = r['saved_activation_bytes'] / ref_m
        r['est_gpu_xwbc_peak_gib_B128'] = A_S_XWBC_PEAK_GIB * r['activation_ratio_vs_A']
    report = {'batch': batch, 'threads': threads, 'direction': f'{names[0]}->{names[1]}', 'mode': 'full, contact on',
              'assets': 'real assets.surface_v2' if real else 'synthetic schema-conformant (P=4096,G=256,K=16,S=1024)',
              'reference_gpu': {'A_S_xwbc_peak_gib': A_S_XWBC_PEAK_GIB, 'B_S_xwbc_peak_gib': B_S_XWBC_PEAK_GIB,
                                'A_S_xwbc_step_s': 0.413, 'B_S_xwbc_step_s': 0.581},
              'arms': results}
    return report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch', type=int, default=16); ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--threads', type=int, default=8); args = ap.parse_args()
    rep = run(args.batch, args.reps, args.threads)
    out = OUT / 'model'; out.mkdir(parents=True, exist_ok=True)
    path = out / f'bench_cpu_B{args.batch}.json'
    path.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))
    print('wrote', path)


if __name__ == '__main__':
    main()
