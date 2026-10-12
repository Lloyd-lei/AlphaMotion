"""WP-INTEGRATION: write OUT/stage0_index.md, a path index of every Stage-0 artifact that exists on disk.

Only lists files (with size and mtime) and copies the headline gate verdicts from the owning packages' reports /
summaries; it computes no new numbers.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

from am2exp_v1.common.paths import CODE, OUT

INDEX = OUT / 'stage0_index.md'


def _f(p: Path):
    if not p.exists():
        return f'- MISSING `{p}`'
    if p.is_dir():
        n = sum(1 for _ in p.rglob('*') if _.is_file())
        return f'- `{p}/` ({n} files)'
    st = p.stat()
    size = st.st_size
    s = f'{size / 1e6:.1f} MB' if size > 1e6 else f'{size / 1e3:.0f} kB'
    t = datetime.datetime.fromtimestamp(st.st_mtime).strftime('%m-%d %H:%M')
    return f'- `{p}` ({s}, {t})'


def _glob(d: Path, pat):
    return sorted(d.glob(pat)) if d.exists() else []


def _json(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def write_index(path=INDEX):
    A = OUT / 'assets' / 'surface_v2'
    I = OUT / 'integration'
    L = []
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M %Z').strip()
    L += [f'# am2exp_v1 Stage-0 index', '', f'Generated {now} by `am2exp_v1/integration/index.py` (paths only; numbers are copied from the owning reports).', '']
    L += ['## Gate verdicts (from the owning reports, not recomputed)', '',
          '| gate | owner | verdict | source |', '|---|---|---|---|',
          '| Ruler gate, strict (>= 13/15 within 1 cm) | WP-JUDGE | **FAIL 6/15** (LAFAN only 11/15; flat-foot frames 13/15) | judge/stage0_report.md s.1 |',
          '| +-5 cm root shifts detected | WP-JUDGE | pass only under the rule revised after seeing the data (v0 rule failed) | judge/stage0_report.md s.1 |',
          '| Shuffled-source floor (>= 0.15 CSE worse than A0) | WP-JUDGE | **FAIL** (-0.006: A0 ignores the source for contact) | judge/stage0_report.md |',
          '| Headroom (numeric) | WP-JUDGE | pass (CSE_GT 0.774, FloatRate 0.50 vs GT 0) | judge/stage0_report.md |',
          '| Headroom (video, lloyd) | lloyd | videos now exist (section 4), **not yet seen by lloyd** | integration/videos/stage0 |',
          '| E2 G-0 (AN dev BA >= 0.75) | WP-E2 | pass on dev (0.765), **fragile**: test 0.539 | eval/e2/stage0_report.md |',
          '| E2 OOD (hammer/pen <= 2x) | WP-E2 | S size pass (1.60x / 1.61x), M size over (2.01x / 2.02x) | eval/e2/stage0_report.md |',
          '| Assets binding (< 0.5 mm exact-bound) | WP-ASSETS | pass; kuavo_s45 457 pts, unitree_h1_2 219 pts badly bound (masked downstream by integration) | assets/README.md |',
          '']
    L += ['## 1. Dense surface assets (WP-ASSETS)', '', _f(A / 'manifest.json'), _f(CODE / 'assets' / 'README.md')]
    L += ['', 'Per-body renders (front, side, AM2 rest, soles, pads):'] + [_f(p) for p in _glob(A / 'png', '*.png') if not p.name.startswith(('grid_', 'human_overlay'))]
    L += ['', 'Human overlay and 12-frame grids (fallAndGetUp1_subject4 [4136,4264)):'] + [_f(p) for p in _glob(A / 'png', 'grid_*.png')] + [_f(A / 'png' / 'human_overlay.png')]
    L += ['', 'Validation:'] + [_f(p) for p in _glob(A / 'validation', '*.json')]
    L += ['', '## 2. Judge J1 / E1 Stage-0 (WP-JUDGE)', '',
          _f(OUT / 'judge' / 'stage0_report.md'), _f(OUT / 'judge' / 'stage0_summary.json'),
          _f(OUT / 'judge' / 'heldout_root_bias_v1.json'), _f(OUT / 'judge' / 'ruler_v1'), _f(OUT / 'judge' / 'bindings_v1'),
          _f(OUT / 'eval' / 'cases_e1_v1.json'), _f(OUT / 'eval' / 'cases_e1_v1.sha256'), _f(OUT / 'eval' / 'cases_rootbias_v1.json')]
    L += [_f(p) for p in _glob(OUT / 'eval' / 'e1', '*') if p.is_dir()]
    L += ['', '## 3. E2 Stage-0 (WP-E2)', '', _f(OUT / 'eval' / 'e2' / 'stage0_report.md'), _f(OUT / 'eval' / 'e2' / 'stage0_summary.json'),
          _f(OUT / 'eval' / 'e2' / 'arms'), _f(OUT / 'eval' / 'e2' / 'labels'), _f(OUT / 'data' / 'e2' / 'd4rl')]
    L += ['', '## 4. Videos (mp4 + 12-frame full-duration grid each)', '']
    groups = [('Stage-0 [human | GMR GT | A0@3000], heldout GR3 and seen G1, fallAndGetUp1_s4 [4136,4264) (integration, from WP-JUDGE npz)', I / 'videos' / 'stage0'),
              ('Demos (WP-RENDER): GT demo and E2 [hand | AN | A0_S@3000]', OUT / 'videos' / 'demo'),
              ('Integration smoke [human | A0@3000 | S untrained] (pipeline check only; S is random init)', I / 'videos')]
    for title, d in groups:
        L += [f'**{title}**', '']
        for sub in sorted(x for x in d.glob('*') if x.is_dir() and x.name != 'stage0') if d.exists() else []:
            for name in ('triptych.mp4', 'grid12.png', 'manifest.json'):
                if (sub / name).exists():
                    L.append(_f(sub / name))
        L.append('')
    L += ['## 5. Model / ops probes', '', _f(OUT / 'model' / 'bench_cpu_B16.json'), _f(OUT / 'model' / 'gpu_timing_B16.json')]
    L += [_f(p) for p in _glob(OUT / 'runs' / '_smoke', '*')]
    L += ['', '## 6. Integration smoke (WP-INTEGRATION)', '', _f(I / 'dryrun_summary.json'), _f(I / 'dryrun7_b32'), _f(I / 'dryrun6_b32'), _f(I / 'e1_summary.json'),
          _f(I / 'render_summary.json'), _f(I / 'ckpt' / 'S_untrained' / 'checkpoint_step0000.pt'), _f(I / 'pytest_final.log'),
          _f(CODE / 'integration' / 'README.md')]
    d = _json(I / 'dryrun_summary.json') or {}
    if d:
        L += ['', '| arm | status | hashes checked / mismatches (5 UniMate steps) | independent re-check | RSS peak GiB |', '|---|---|---|---|---|']
        for arm, r in d.items():
            s = r.get('sampling') or {}
            L.append(f"| {arm} | {r.get('status')} | {s.get('steps_checked')} / {s.get('mismatches')} | "
                     f"{(r.get('independent_hash_check') or {}).get('all_ok')} | {r.get('rss_peak_gib')} |")
    L += ['', 'Not produced (by design in Stage-0): any training run beyond 12 GPU steps, GPU timing probe at batch 128, lloyd approvals.', '']
    Path(path).write_text('\n'.join(L))
    return path


if __name__ == '__main__':
    print(write_index())
