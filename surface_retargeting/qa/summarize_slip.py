"""Before/after table for the 10-11 slip rule (bit 256 follow_human_slip):  reference exports (results/export) vs a
re-run directory (default results/export_slip_v3).  Writes <newdir>/summary_slip.json and .md.

  kept before / after      frames with flags & 255 == 0 (before: flags == 0, bit 256 did not exist)
  bit 256 share            frames with a robot region legally following a human slide (kept, informational)
  bit 32 frames            robot slides while the human does not (or much faster)
  root step max            cm/frame scaled to 30 fps (all frames / kept frames)
  slip speed max           ground_slip_speed max (m/s): old gauge (forward median, all frames) / new ruler all / kept
"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import argparse
import json
from pathlib import Path

import numpy as np

OUT = Path(f'{SRCFG.work}/results')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--new', default='export_slip_v3'); ap.add_argument('--old', default='export')
    a = ap.parse_args()
    rows = []
    for fn in sorted((OUT / a.new).glob('*.npz')):
        fo = OUT / a.old / fn.name
        if not fo.exists():
            continue
        zn = np.load(fn, allow_pickle=True); zo = np.load(fo, allow_pickle=True)
        fl_n = zn['flags']; fl_o = zo['flags']; fps = float(zn['fps']); s30 = fps / 30.0
        keep_n = (fl_n & 255) == 0
        q = zn['qpos'].astype(np.float64)
        st = np.linalg.norm(np.diff(q[:, :3], axis=0), axis=1) * s30 * 100
        kk = keep_n[1:] & keep_n[:-1]
        sp_n = zn['ground_slip_speed'].astype(np.float32); sp_o = zo['ground_slip_speed'].astype(np.float32)
        spk = sp_n[keep_n]
        clip, robot = fn.stem.rsplit('__', 1)
        rows.append(dict(clip=clip, robot=robot, frames=len(fl_n), fps=round(fps, 1),
                         kept_before=round(float((fl_o == 0).mean()), 3), kept_after=round(float(keep_n.mean()), 3),
                         bit256_share=round(float(((fl_n & 256) > 0).mean()), 3),
                         bit256_kept_share=round(float((((fl_n & 256) > 0) & keep_n).mean()), 3),
                         bit32_frames_before=int(((fl_o & 32) > 0).sum()), bit32_frames_after=int(((fl_n & 32) > 0).sum()),
                         root_step_max_cm=round(float(st.max()), 2), root_step_kept_max_cm=round(float(st[kk].max()), 2) if kk.any() else None,
                         slip_speed_max_old=round(float(np.nanmax(sp_o)), 2),
                         slip_speed_max_new=round(float(np.nanmax(sp_n)), 2) if np.isfinite(sp_n).any() else None,
                         slip_speed_max_new_kept=round(float(np.nanmax(spk)), 3) if np.isfinite(spk).any() else None,
                         flags_hist_after={str(b): int(((fl_n & b) > 0).sum()) for b in (1, 2, 4, 8, 16, 32, 64, 128, 256)}))
    json.dump(rows, open(OUT / a.new / 'summary_slip.json', 'w'), indent=1)
    hd = ['clip', 'robot', 'frames', 'kept before', 'kept after', 'bit256 share (kept)', 'bit32 before -> after',
          'root step max cm/f@30 (kept)', 'slip speed max m/s old -> new (kept)']
    md = ['| ' + ' | '.join(hd) + ' |', '|' + '---|' * len(hd)]
    for r in rows:
        md.append(f"| {r['clip'][:28]} | {r['robot']} | {r['frames']} | {r['kept_before']} | {r['kept_after']} | "
                  f"{r['bit256_share']} ({r['bit256_kept_share']}) | {r['bit32_frames_before']} -> {r['bit32_frames_after']} | "
                  f"{r['root_step_max_cm']} ({r['root_step_kept_max_cm']}) | {r['slip_speed_max_old']} -> {r['slip_speed_max_new']} ({r['slip_speed_max_new_kept']}) |")
    (OUT / a.new / 'summary_slip.md').write_text('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
