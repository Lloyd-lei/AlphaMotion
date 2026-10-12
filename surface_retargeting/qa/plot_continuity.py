"""Temporal-continuity evidence for one or more parity runs: per frame root step (cm/frame at 30 fps) of the GPU UMR
solution and of the GPU strict output, the strict root offset from UMR, and the step excess (strict step - UMR step,
the quantity the continuity gate bounds), with flagged frames shaded.  Writes <out>/<clip>__<robot>_continuity.png
and prints/collects a JSON summary.
Usage: plot_continuity.py <out_dir> <clip>__<robot> [...]     (reads results/parity/<clip>__<robot>.series.npz)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('/var/tmp/alphamotion/umr_factory/umr_gpu')


def stats(name):
    S = np.load(OUT / 'results' / 'parity' / f'{name}.series.npz', allow_pickle=True)
    fps = float(S['fps']); s30 = fps / 30.0
    Qu = S['q_gpu_umr']; Qs = S['q_gpu_strict']; fl = S['flags']
    du = np.diff(Qu[:, :3], axis=0); ds = np.diff(Qs[:, :3], axis=0)
    step_u = np.linalg.norm(du, axis=1) * s30 * 100; step_s = np.linalg.norm(ds, axis=1) * s30 * 100
    exc = np.linalg.norm(ds - du, axis=1) * s30 * 100
    jexc = np.abs(np.diff(Qs[:, 7:], axis=0) - np.diff(Qu[:, 7:], axis=0)).max(1) * s30
    off = np.linalg.norm(Qs[:, :3] - Qu[:, :3], axis=1) * 100
    kept = (fl[1:] == 0) & (fl[:-1] == 0)
    r = dict(frames=int(len(fl)), fps=round(fps, 2), kept_frac=round(float(((fl & 255) == 0).mean()), 4),
             flags_hist={str(b): int(((fl & b) > 0).sum()) for b in (1, 2, 4, 8, 16, 32, 64, 128)},
             umr_root_step_cm=dict(p99=round(float(np.percentile(step_u, 99)), 2), max=round(float(step_u.max()), 2)),
             strict_root_step_cm=dict(p99=round(float(np.percentile(step_s, 99)), 2), max=round(float(step_s.max()), 2),
                                      max_kept=round(float(step_s[kept].max()), 2) if kept.any() else None,
                                      frames_gt_5cm=int((step_s > 5).sum()), frames_gt_5cm_kept=int((step_s[kept] > 5).sum())),
             root_excess_cm=dict(p50=round(float(np.median(exc)), 3), p99=round(float(np.percentile(exc, 99)), 3), max=round(float(exc.max()), 3),
                                 max_kept=round(float(exc[kept].max()), 3) if kept.any() else None,
                                 frames_gt_1cm=int((exc > 1).sum()), frames_gt_2cm=int((exc > 2).sum())),
             joint_excess_rad=dict(p99=round(float(np.percentile(jexc, 99)), 4), max=round(float(jexc.max()), 4),
                                   max_kept=round(float(jexc[kept].max()), 4) if kept.any() else None),
             root_offset_vs_umr_cm=dict(p50=round(float(np.median(off)), 2), p95=round(float(np.percentile(off, 95)), 2), max=round(float(off.max()), 2),
                                        max_change_per_frame=round(float(np.abs(np.diff(off)).max() * s30), 3)))
    return r, (step_u, step_s, exc, off, fl, fps)


def plot(name, data, outd):
    step_u, step_s, exc, off, fl, fps = data
    t = np.arange(len(fl)) / fps
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    for a in ax:
        bad = fl != 0
        if bad.any():
            a.fill_between(t, 0, 1, where=bad, transform=a.get_xaxis_transform(), color='#f4c7c3', alpha=0.6, lw=0, label='flagged (excluded)')
    ax[0].plot(t[1:], step_u, lw=0.8, color='#4a7bd0', label='GPU UMR raw')
    ax[0].plot(t[1:], step_s, lw=0.8, color='#222222', label='GPU strict')
    ax[0].set_ylabel('root step\ncm / frame @30fps'); ax[0].legend(loc='upper right')
    ax[1].plot(t[1:], exc, lw=0.8, color='#c03030'); ax[1].axhline(1.0, color='k', ls='--', lw=0.6)
    ax[1].set_ylabel('step excess vs UMR\ncm / frame @30fps')
    ax[2].plot(t, off, lw=0.8, color='#207040'); ax[2].set_ylabel('root offset\nvs UMR, cm'); ax[2].set_xlabel('time (s)')
    fig.suptitle(name); fig.tight_layout()
    fig.savefig(outd / f'{name}_continuity.png', dpi=90); plt.close(fig)


def main():
    outd = Path(sys.argv[1]); outd.mkdir(parents=True, exist_ok=True)
    allr = {}
    for name in sys.argv[2:]:
        r, data = stats(name)
        plot(name, data, outd)
        allr[name] = r
        print(name, json.dumps(r))
    json.dump(allr, open(outd / 'continuity_summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
