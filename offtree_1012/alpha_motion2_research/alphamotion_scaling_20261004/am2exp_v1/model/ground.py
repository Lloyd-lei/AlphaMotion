"""Ground-label hygiene shared by the I-arm sensor and the O-arm objective.

* Excluded groups (clip level, all XWBC bodies): the 14 DFaust groups (human labels broken:
  root p50 0.35 m, knees 100% 'in contact', yet ground_valid=True) and the PosePrior clips whose
  human contact_trust == 0 (7 clips; the robots carry trust=1 for them, we still drop them).
  Group index == position in data_manifest_v2.json['xwbc']['groups'] (verified: split agrees).
* source_root_height_sensor(): the I-arm root-height input, tied to the existing contact
  sensor: valid iff root_height_valid & ground_valid & contact_trust & any finite source_contact
  & not excluded.  It consumes no RNG; the 0.3 contact dropout already decides finiteness.
"""
from __future__ import annotations
import json
import numpy as np
import torch
from ._sealed import ROOT

MANIFEST = ROOT / 'data' / 'data_manifest_v2.json'
_EXCLUDED = None


def excluded_ground_groups(human_body=None, manifest_path=MANIFEST):
    """frozenset of XWBC group indices excluded from every new ground/contact term or input."""
    global _EXCLUDED
    if _EXCLUDED is not None and human_body is None:
        return _EXCLUDED
    groups = json.loads(open(manifest_path).read())['xwbc']['groups']
    out = {i for i, g in enumerate(groups) if g['source_dataset'] == 'DFaust'}
    prior = [i for i, g in enumerate(groups) if g['source_dataset'] == 'PosePrior']
    if human_body is None:
        human_body = _load_human_trust()
    if human_body is None:  # conservative: cannot read trust -> drop every PosePrior clip
        out |= set(prior)
    else:
        grp, trust = human_body
        for i in prior:
            m = grp == i
            if m.any() and not bool(trust[m].any()): out.add(i)
    _EXCLUDED = frozenset(int(i) for i in out)
    return _EXCLUDED


def _load_human_trust():
    """(group_index, contact_trust) arrays of the sealed human payload, or None."""
    try:
        from pathlib import Path
        cfg = json.loads(open(ROOT / 'orchestration' / 'configs' / 'A_S_seed0.json').read())
        xd = Path(cfg['pilot_data_root']) / 'data_xwbc'
        xm = json.loads(open(xd / 'manifest.json').read())
        entry = next(e for e in xm['body_payloads'] if e.get('body') == 'human_smpl')
        with np.load(xd / entry['payload'], allow_pickle=False) as z:
            return z['group_index'].astype(np.int64), z['contact_trust'].astype(bool)
    except Exception:
        return None


def set_excluded_ground_groups(groups):
    global _EXCLUDED
    _EXCLUDED = frozenset(int(g) for g in groups)
    return _EXCLUDED


def excluded_ground_groups_from_bodies(bodies):
    """Preferred: compute from the already-loaded human Body (exact, no file parsing)."""
    human = next(b for b in bodies if b.name == 'human_smpl')
    return excluded_ground_groups((np.asarray(human.arrays['group'], np.int64), np.asarray(human.arrays['contact_trust'], bool)))


def row_ground_ok(body, rows, device, excluded=None):
    """bool [B]: trusted, valid, non-excluded ground rows of an XWBC body (target or source side)."""
    a = body.arrays
    rows = np.asarray(rows)
    if not body.metric or 'ground_valid' not in a:
        return torch.zeros(len(rows), dtype=torch.bool, device=device)
    ok = a['ground_valid'][rows] & a['root_height_valid'][rows] & a['contact_trust'][rows]
    ok = ok & np.isfinite(a['root_height_m'][rows])
    ex = excluded_ground_groups() if excluded is None else excluded
    if body.domain == 'xwbc' and ex:
        ok = ok & ~np.isin(a['group'][rows], np.fromiter(ex, np.int64))
    return torch.as_tensor(ok, device=device)


def source_root_height_sensor(src, rows, source_contact, device, excluded=None):
    """[B] float metres, NaN where the sensor is unavailable (I-arm encoder input)."""
    B = len(rows)
    nan = torch.full((B,), float('nan'), device=device)
    if source_contact is None or src.domain != 'xwbc':
        return nan
    ok = row_ground_ok(src, rows, device, excluded) & torch.isfinite(source_contact).any(1)
    rh = torch.as_tensor(src.arrays['root_height_m'][np.asarray(rows)], device=device, dtype=torch.float32)
    return torch.where(ok, rh, nan)
