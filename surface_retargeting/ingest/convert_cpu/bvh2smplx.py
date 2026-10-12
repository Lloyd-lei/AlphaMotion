"""Thin wrapper around hiphi2smplx (MIT, noitom-robotics/hiphi2smplx @65e5131).

Adds one thing: support for the older 51-joint Noitom BVH layout found in part of
OmniContact (Hips-Spine-Spine1-Neck-Head, no Spine2/3/4/Neck1).  For those files a
virtual joint `SpineMid` is inserted halfway along Spine->Spine1 with identity
local rotation, so the SMPL-X spine1/spine2/spine3 targets map to
Spine/SpineMid/Spine1.  55-joint files use the stock `hiphi` map unchanged.

Returns the hiphi2smplx MotionResult plus diagnostics.  Upstream code untouched.
"""
from __future__ import annotations
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml

import os
import sys

SRCFG.add_hiphi2smplx_to_path()

import numpy as np  # noqa: E402

import hiphi2smplx.pipeline as H  # noqa: E402
from hiphi2smplx import skeleton as S  # noqa: E402
from hiphi2smplx.beta_fitting import BetaFitConfig  # noqa: E402
from hiphi2smplx.bvh import load_bvh as _load_bvh  # noqa: E402
from hiphi2smplx.pipeline import ConversionConfig  # noqa: E402

MODEL = f'{SRCFG.smplx_neutral}'
BETA_DATA = f'{SRCFG.beta_fit}'

OMNI51_MAP = dict(S.HIPHI_MAP)
OMNI51_MAP.update({"spine": "Spine", "spine2": "SpineMid", "chest": "Spine1"})
S.SOURCE_SKELETON_MAPS["omni51"] = OMNI51_MAP


def _insert_spinemid(clip: S.MotionClip) -> S.MotionClip:
    names = list(clip.skeleton.joint_names)
    i_sp, i_s1 = names.index("Spine"), names.index("Spine1")
    assert clip.skeleton.parents[i_s1] == i_sp
    ins = i_sp + 1  # new joint index (directly after Spine)
    new_names = names[:ins] + ["SpineMid"] + names[ins:]
    old_parents = clip.skeleton.parents

    def remap(j):
        return j if j < ins else j + 1

    parents = [-1 if p < 0 else remap(int(p)) for p in old_parents]
    parents = parents[:ins] + [i_sp] + parents[ins:]
    parents[remap(i_s1)] = ins
    off = clip.skeleton.rest_offsets.copy()
    half = off[i_s1] * 0.5
    off = np.concatenate([off[:ins], half[None], off[ins:]], 0)
    off[remap(i_s1)] = half
    lr = clip.local_rotations
    lr = np.concatenate([lr[:, :ins], np.zeros_like(lr[:, :1]), lr[:, ins:]], 1)
    lt = clip.local_translations
    lt_half = lt[:, i_s1] * 0.5
    lt = np.concatenate([lt[:, :ins], lt_half[:, None], lt[:, ins:]], 1)
    lt[:, remap(i_s1)] = lt_half
    sk = S.Skeleton(tuple(new_names), np.asarray(parents), off)
    return S.MotionClip(sk, lr, clip.root_translations, clip.frame_time, lt)


def load_bvh_any(path, unit_scale=0.01):
    clip = _load_bvh(path, unit_scale=unit_scale)
    if "Spine4" not in clip.skeleton.joint_names and "Spine1" in clip.skeleton.joint_names:
        clip = _insert_spinemid(clip)
    return clip


H.load_bvh = load_bvh_any  # pipeline.fit_bvh resolves load_bvh from its module globals


def bvh_layout(path) -> str:
    with open(path, "r", errors="ignore") as f:
        head = []
        for line in f:
            if line.strip() == "MOTION":
                break
            head.append(line)
    names = [ln.split()[1] for ln in head if ln.strip().startswith(("ROOT", "JOINT"))]
    if "Spine4" in names:
        return "hiphi"
    return "omni51"


def convert(bvh_path, betas=None, beta_device="cpu", max_frames=None):
    layout = bvh_layout(bvh_path)
    cfg = ConversionConfig()
    bcfg = None
    if betas is None:
        bcfg = BetaFitConfig(device=beta_device, data_dir=BETA_DATA)
    res = H.fit_bvh(bvh_path, MODEL, betas, beta_fit_config=bcfg, config=cfg,
                    max_frames=max_frames, skeleton_map=layout)
    return res, layout


if __name__ == "__main__":
    import time
    t = time.time()
    r, lay = convert(sys.argv[1], max_frames=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    print(lay, r.poses.shape, "resid mean/p95/max cm", 100 * r.residuals.mean(),
          100 * np.percentile(r.residuals, 95), 100 * r.residuals.max(), f"{time.time()-t:.1f}s")
