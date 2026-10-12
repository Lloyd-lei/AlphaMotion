"""Aggregate every OUT/<clip>/<robot>.qa.json into OUT/manifest.json (one row per clip x robot) and
OUT/qa_summary.json (dataset-level before/after penetration, flagged frames, slip, agreement)."""
import json, sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surface_retargeting.paths import OUT

rows = []
for f in sorted(glob.glob(str(OUT / '*' / '*.qa.json'))):
    q = json.load(open(f))
    pa, pb = q['penetration_after'], q['penetration_before']

    def g(d, k, f):
        return (d.get(k) or {}).get(f)
    feet = ['L_foot', 'R_foot']
    sb = {r: (q['ground_slip_before'].get(r) or {}).get('slip_frames') for r in feet}
    sa = {r: (q['ground_slip_after'].get(r) or {}).get('slip_frames') for r in feet}
    rows.append(dict(
        clip=q['clip'], robot=q['robot'], frames=q['frames'], fps=q['fps'], npz=q['npz'],
        floor_offset_removed_m=q['floor']['human_floor_m'],
        frames_flagged=q['frames_flagged_final'], flags_hist=q['flags_final_hist'],
        ground_pen_max_mm_before=g(pb, 'ground', 'max_mm'), ground_pen_max_mm_after=g(pa, 'ground', 'max_mm'),
        self_pen_max_mm_before=g(pb, 'self', 'max_mm'), self_pen_max_mm_after=g(pa, 'self', 'max_mm'),
        object_pen_max_mm_before=g(pb, 'object', 'max_mm'), object_pen_max_mm_after=g(pa, 'object', 'max_mm'),
        frames_needing_fix=q['projection']['frames_needing_fix'], ms_per_frame=round(q['projection']['ms_per_frame'], 2),
        foot_slip_frames_before=sb, foot_slip_frames_after=sa,
        foot_contact_agreement_after={r: q['ground_contact_agreement_human_vs_robot_after'].get(r) for r in feet},
        object_contact_frames=q.get('object_contact_frames')))
json.dump(rows, open(OUT / 'manifest.json', 'w'), indent=1)
tot = sum(r['frames'] for r in rows)
summ = dict(n_items=len(rows), total_frames=tot, frames_flagged=sum(r['frames_flagged'] for r in rows),
            items_with_flagged_frames=[(r['clip'], r['robot'], r['frames_flagged'], r['flags_hist']) for r in rows if r['frames_flagged']],
            worst_after_mm=dict(ground=max([r['ground_pen_max_mm_after'] or 0 for r in rows], default=None),
                                self=max([r['self_pen_max_mm_after'] or 0 for r in rows], default=None),
                                object=max([r['object_pen_max_mm_after'] or 0 for r in rows], default=None)),
            flag_bits={'1': 'ground pen > 0.5 mm', '2': 'object pen > 0.5 mm', '4': 'self pen > 0.5 mm',
                       '8': 'required ground contact not in [0,1] mm', '16': 'required object contact not in [0,1] mm'})
json.dump(summ, open(OUT / 'qa_summary.json', 'w'), indent=1)
print(json.dumps(summ, indent=1))
for r in rows:
    print(r['clip'], r['robot'], 'flagged', r['frames_flagged'], 'ground', r['ground_pen_max_mm_before'], '->', r['ground_pen_max_mm_after'],
          'self', r['self_pen_max_mm_before'], '->', r['self_pen_max_mm_after'], 'obj', r['object_pen_max_mm_before'], '->',
          r['object_pen_max_mm_after'], 'slipL/R', r['foot_slip_frames_before'], '->', r['foot_slip_frames_after'])
