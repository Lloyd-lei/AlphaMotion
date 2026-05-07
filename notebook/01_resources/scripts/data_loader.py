"""Data loading for NB1.

Stage 1 v1 protocol: **synthetic data is held out for TEST ONLY**.
  - Training:  AMASS canonical (subject-wise split)
  - Test sets: Kimodo OOD (balanced families) + AMASS heldout-labeled (realism check)
                + AMASS heldout-other (hardest, unseen real)

This intentionally throws away the 1599 Kimodo clips that splits.json puts in train,
because the Stage-1 publishable claim is "trained on real, generalizes to synthetic".
Stage 2 (caption encoder) likely relaxes this — see NB03.

Wraps MixedMotionDataset (in dataset.py — copied from experiment/stage1/data.py).
Exposes:
    load_mixed             — full MixedMotionDataset (read-only, cached)
    load_amass_canonical   — (train, heldout) over AMASS-only
    load_kimodo_ood        — Kimodo-only Subset (TEST ONLY, never trained on)
    load_amass_heldout_labeled — heldout ∩ labeled (Stage-1 realism eval)
    sample_batch           — small batch for forward shape trace
    per_source_breakdown   — {source: {total, subsets}}  for §2.1
    duration_breakdown     — {split: {clips, seconds, hours}}  for §2.1 (Q2)
    other_subset_breakdown — per-subject 'other' counts (§2.2.1.a)
    other_feature_matrix   — raw mid-frame joint features (§2.2.1.b PLACEHOLDER)
    kimodo_family_index    — family -> [indices] within Kimodo subset
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Subset

from dataset import MixedMotionDataset


# ---------- single load, cached at module scope ------------------------------

_DS_CACHE: Dict[str, MixedMotionDataset] = {}


def load_mixed(DATA: dict) -> MixedMotionDataset:
    """Load the full mixed dataset (AMASS + Kimodo + HY Motion). Cached."""
    npz_path = DATA["amass"]["npz"]
    if npz_path not in _DS_CACHE:
        _DS_CACHE[npz_path] = MixedMotionDataset(npz_path)
    return _DS_CACHE[npz_path]


# ---------- splits -----------------------------------------------------------

def load_amass_canonical(DATA: dict) -> Tuple[Subset, Subset]:
    """AMASS-only canonical subject split. Returns (train_subset, heldout_subset).

    Canonical = uses indices in splits.json's `train` / `held_out` lists.
    AMASS-only = filters out kimodo / hy_motion samples by source_name.
    """
    ds = load_mixed(DATA)
    splits = json.loads(Path(DATA["amass"]["heldout_subjects_file"]).read_text())

    amass_mask = (ds.source_name == "amass")
    amass_set  = set(np.where(amass_mask)[0].tolist())

    train_idx   = [i for i in splits["train"]    if i in amass_set]
    heldout_idx = [i for i in splits["held_out"] if i in amass_set]

    return Subset(ds, train_idx), Subset(ds, heldout_idx)


def load_kimodo_ood(DATA: dict) -> Subset:
    """Kimodo-only subset — Stage-1 TEST-ONLY OOD evaluation set.

    Never used for training (per Stage-1 v1 protocol, see module docstring).
    """
    ds = load_mixed(DATA)
    idx = np.where(ds.source_name == "kimodo")[0].tolist()
    return Subset(ds, idx)


def load_amass_heldout_labeled(DATA: dict) -> Subset:
    """Stage-1 realism eval set: AMASS heldout ∩ action_label != 'other'.

    Used by NB02 to report MPJPE on real motion with known categories.
    """
    ds = load_mixed(DATA)
    splits = json.loads(Path(DATA["amass"]["heldout_subjects_file"]).read_text())
    amass_set = set(np.where(ds.source_name == "amass")[0].tolist())
    idx = [i for i in splits["held_out"]
           if i in amass_set and str(ds.action_label[i]) != "other"]
    return Subset(ds, idx)


def load_amass_heldout_other(DATA: dict) -> Subset:
    """Hardest Stage-1 eval: AMASS heldout ∩ action_label == 'other'.

    Tests generalization to unseen real motion that the 12-family regex missed.
    """
    ds = load_mixed(DATA)
    splits = json.loads(Path(DATA["amass"]["heldout_subjects_file"]).read_text())
    amass_set = set(np.where(ds.source_name == "amass")[0].tolist())
    idx = [i for i in splits["held_out"]
           if i in amass_set and str(ds.action_label[i]) == "other"]
    return Subset(ds, idx)


# ---------- helpers used by NB1 ---------------------------------------------

def sample_batch(subset: Subset, n: int = 2) -> dict:
    """Stack n samples into batched tensors. Returns dict of [B, ...] tensors."""
    items = [subset[i] for i in range(min(n, len(subset)))]
    out = {}
    for k in items[0]:
        if isinstance(items[0][k], torch.Tensor):
            out[k] = torch.stack([s[k] for s in items])
        else:
            out[k] = torch.tensor([s[k] for s in items])
    return out


def kimodo_family_index(DATA: dict) -> Dict[str, List[int]]:
    """Map family -> list of indices within the Kimodo OOD subset."""
    ds = load_mixed(DATA)
    kimodo_idx = np.where(ds.source_name == "kimodo")[0]
    out: Dict[str, List[int]] = {}
    for local_i, global_i in enumerate(kimodo_idx):
        fam = str(ds.action_label[global_i])
        out.setdefault(fam, []).append(local_i)
    return out


# ---------- per-source breakdown (used by NB1 §2.1) -------------------------

def per_source_breakdown(DATA: dict) -> Dict[str, Dict[str, int]]:
    """Returns: {source_name: {'subsets': {subject_id: count}, 'total': int}}."""
    ds = load_mixed(DATA)
    out: Dict[str, Dict[str, int]] = {}
    for src in np.unique(ds.source_name):
        mask = (ds.source_name == src)
        subsets, counts = np.unique(ds.subset[mask], return_counts=True)
        out[str(src)] = {
            "total":   int(mask.sum()),
            "subsets": {str(s): int(c) for s, c in zip(subsets, counts)},
        }
    return out


# ---------- Q2: duration breakdown ------------------------------------------

# AMASS pipeline downsamples to 30 fps; build_mixed_dataset uses clip_len=90.
# So each clip is exactly 90 / 30 = 3.0 seconds.
FPS               = 30
FRAMES_PER_CLIP   = 90
SECONDS_PER_CLIP  = FRAMES_PER_CLIP / FPS


def duration_breakdown(DATA: dict, splits: List[Subset]) -> Dict[str, dict]:
    """Compute clip count + total wall-clock duration per split.

    splits: list of (label, Subset) pairs. Returns dict keyed by label.
    """
    out: Dict[str, dict] = {}
    for label, sub in splits:
        n = len(sub)
        secs = n * SECONDS_PER_CLIP
        out[label] = {
            "clips":   n,
            "seconds": round(secs, 1),
            "minutes": round(secs / 60, 2),
            "hours":   round(secs / 3600, 2),
        }
    return out


# ---------- "other" investigation helpers (NB1 §2.2.1) ----------------------

def other_subset_breakdown(DATA: dict, source: str = "amass") -> Dict[str, int]:
    """Per-subset count of clips labeled 'other' within one source."""
    ds = load_mixed(DATA)
    mask = (ds.source_name == source) & (ds.action_label == "other")
    out: Dict[str, int] = {}
    for s, c in zip(*np.unique(ds.subset[mask], return_counts=True)):
        out[str(s)] = int(c)
    return out


def other_feature_matrix(
    DATA: dict, source: str = "amass", *, frame_idx: int = 45,
):
    """Extract a flat per-clip feature for unsupervised clustering on 'other'.

    Feature = mid-frame joint positions, root-centered, flattened to [J*3].
    PLACEHOLDER — only one frame of joint position data, no rotation, no
    velocity, no temporal info. The real version (NB02 §5) uses trained
    pair-tensor encoder embeddings on the full clip.

    Returns:
      X            np.ndarray [N, J*3]
      sample_names np.ndarray [N,]
      subsets      np.ndarray [N,]
    """
    ds = load_mixed(DATA)
    mask = (ds.source_name == source) & (ds.action_label == "other")
    idx  = np.where(mask)[0]

    raw = ds.raw[idx, frame_idx]                          # [N, J, 3]
    raw = raw - raw[:, 0:1, :]                             # pelvis-centered
    X = raw.reshape(raw.shape[0], -1).astype(np.float32)   # [N, J*3]
    return X, ds.sample_name[idx], ds.subset[idx]
