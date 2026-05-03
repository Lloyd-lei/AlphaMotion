"""
Dataset for Stage 1: Kimodo-generated motion on SOMA-77 skeleton.

Loaded from a cached npz (produced by gen_kimodo_data.py) containing
    joints : [N, T, 77, 3] float32  — world-frame-relative joint positions

The masked motion modeling objective supports five mask modes:
    - "random"          i.i.d. bernoulli mask over (t, j) pairs
    - "joint"           mask entire joint trajectories
    - "time"            mask entire time slices
    - "keyframe"        keep sparse keyframes, mask the rest
    - "kinematic_chain" mask a joint AND all its downstream descendants in the
                        SOMA77 kinematic tree (simulates joint failure)

Also preserves the legacy synthetic synergy dataset for ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from soma_skeleton import JOINT_NAMES, build_descendants


# ----------------------------------------------------------------------------
# Legacy synthetic (kept for ablation; may be deleted once Kimodo path stable)
# ----------------------------------------------------------------------------


@dataclass
class SynergyConfig:
    K: int = 8
    T: int = 64
    J: int = 20
    C: int = 3
    N: int = 10_000
    noise_std: float = 0.02
    seed: int = 42
    temporal_width_min: int = 4
    temporal_width_max: int = 16
    spatial_sparsity_mean: float = 0.35
    spatial_sparsity_std: float = 0.10
    amplitude_alpha: float = 0.5


def make_ground_truth_synergies(cfg: SynergyConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    t_axis = np.arange(cfg.T)
    temporal = np.zeros((cfg.K, cfg.T), dtype=np.float32)
    for k in range(cfg.K):
        centre = rng.uniform(cfg.T * 0.15, cfg.T * 0.85)
        width = rng.uniform(cfg.temporal_width_min, cfg.temporal_width_max)
        bump = np.exp(-0.5 * ((t_axis - centre) / width) ** 2)
        bump /= (bump.sum() + 1e-8)
        temporal[k] = bump.astype(np.float32)
    spatial = np.zeros((cfg.K, cfg.J, cfg.C), dtype=np.float32)
    for k in range(cfg.K):
        sparsity = np.clip(
            rng.normal(cfg.spatial_sparsity_mean, cfg.spatial_sparsity_std),
            0.15, 0.75,
        )
        active = rng.random(cfg.J) < sparsity
        pattern = rng.standard_normal((cfg.J, cfg.C)).astype(np.float32)
        pattern = pattern * active[:, None]
        norm = np.linalg.norm(pattern) + 1e-8
        spatial[k] = pattern / norm
    return temporal, spatial


class SyntheticSynergyDataset(Dataset):
    def __init__(self, cfg: SynergyConfig):
        self.cfg = cfg
        self.temporal, self.spatial = make_ground_truth_synergies(cfg)
        self._sample_rng = np.random.default_rng(cfg.seed + 1)

    def __len__(self) -> int:
        return self.cfg.N

    def __getitem__(self, idx: int) -> torch.Tensor:
        w = self._sample_rng.dirichlet([self.cfg.amplitude_alpha] * self.cfg.K)
        amp = (w * self.cfg.K).astype(np.float32)
        motion = np.einsum(
            "k,kt,kjc->tjc", amp, self.temporal, self.spatial
        ).astype(np.float32)
        noise = self._sample_rng.normal(0, self.cfg.noise_std, motion.shape).astype(np.float32)
        return torch.from_numpy(motion + noise)


# ----------------------------------------------------------------------------
# Kimodo-generated dataset (the main one for Stage 1)
# ----------------------------------------------------------------------------


class MixedMotionDataset(Dataset):
    """Unified dataset covering AMASS + Kimodo + HY Motion (from
    `build_mixed_dataset.py`). Stores joints in world frame; we subtract the
    per-sample root-at-t0 on the fly so root-relative training behaves like
    the KimodoMotionDataset did.

    __getitem__ returns the same dict as KimodoMotionDataset so the training
    loop stays unchanged:
        {pos, rot6d, root, local_rot_mats}
    plus an additional `source_id` (int tensor scalar) for metadata.
    """

    def __init__(
        self,
        mixed_npz: str | Path,
        crop_frames: Optional[int] = None,
        normalise: bool = True,
    ):
        from soma_skeleton import (
            compute_bone_offsets, matrix_to_rot6d, build_parent_index,
        )
        mixed_npz = Path(mixed_npz)
        d = np.load(mixed_npz, allow_pickle=True)

        joints_world    = d["joints_world"].astype(np.float32)
        local_rot       = d["local_rot_mats"].astype(np.float32)
        global_rot      = d["global_rot_mats"].astype(np.float32)
        root_world      = d["root_positions"].astype(np.float32)
        self.source_id  = d["source_id"].astype(np.int32)
        self.source_name = d["source_name"]
        self.subset      = d["subset"]
        self.sample_name = d["sample_name"]
        self.action_label = d["action_label"] if "action_label" in d.files \
            else np.array(["other"] * len(self.source_id), dtype=object)

        if crop_frames is not None and crop_frames < joints_world.shape[1]:
            joints_world = joints_world[:, :crop_frames]
            local_rot    = local_rot[:, :crop_frames]
            global_rot   = global_rot[:, :crop_frames]
            root_world   = root_world[:, :crop_frames]

        # Per-sample root-at-t0 subtraction (mirrors KimodoMotionDataset)
        root_t0      = root_world[:, :1, :]             # [N, 1, 3]
        joints_rel   = joints_world - root_t0[:, :, None, :]
        root_rel     = root_world - root_t0

        self.raw = joints_rel
        self.local_rot_raw = local_rot
        self.root_raw = root_rel
        self.N, self.T, self.J, self.C = self.raw.shape

        # Bone offsets (one set, from world data — invariant per joint)
        self.bone_offsets = compute_bone_offsets(joints_world, global_rot)
        self.parents = np.array(build_parent_index(), dtype=np.int64)

        if normalise:
            flat = self.raw.reshape(-1, self.J, self.C)
            self.mean = flat.mean(axis=0, keepdims=True)
            self.std = flat.std(axis=0, keepdims=True) + 1e-6
            self.data = (self.raw - self.mean[None]) / self.std[None]

            self.root_mean = self.root_raw.reshape(-1, 3).mean(axis=0, keepdims=True)
            self.root_std = self.root_raw.reshape(-1, 3).std(axis=0, keepdims=True) + 1e-6
            self.root_data = (self.root_raw - self.root_mean[None]) / self.root_std[None]
        else:
            self.mean = np.zeros((1, self.J, self.C), dtype=np.float32)
            self.std = np.ones((1, self.J, self.C), dtype=np.float32)
            self.data = self.raw
            self.root_mean = np.zeros((1, 3), dtype=np.float32)
            self.root_std = np.ones((1, 3), dtype=np.float32)
            self.root_data = self.root_raw

        rot_t = torch.from_numpy(self.local_rot_raw).float()
        self.rot6d = matrix_to_rot6d(rot_t).numpy().astype(np.float32)

        self.pos_tensor = torch.from_numpy(self.data).float()
        self.rot_tensor = torch.from_numpy(self.rot6d).float()
        self.root_tensor = torch.from_numpy(self.root_data).float()
        self.rotmat_tensor = torch.from_numpy(self.local_rot_raw).float()

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        return {
            "pos":            self.pos_tensor[idx],
            "rot6d":          self.rot_tensor[idx],
            "root":           self.root_tensor[idx],
            "local_rot_mats": self.rotmat_tensor[idx],
            "source_id":      int(self.source_id[idx]),
        }


class KimodoMotionDataset(Dataset):
    """Wraps a cached npz of Kimodo-generated SOMA77 motions.

    Loads joint positions plus (optionally) local rotations and world-frame
    root positions so downstream training can predict the *full* SE(3) state of
    every joint — required to render through Kimodo's viser / SMPL mesh.

    Outputs from __getitem__ are torch tensors:
        {
          "pos":     [T, J, 3]   root-t0-relative joint positions (normalised)
          "rot6d":   [T, J, 6]   Zhou-et-al 6D local rotation
          "root":    [T, 3]      world-frame root (normalised)
          "local_rot_mats": [T, J, 3, 3]  (un-normalised) for FK supervision
        }
    """

    def __init__(
        self,
        cache_path: str | Path,
        crop_frames: Optional[int] = None,
        normalise: bool = True,
    ):
        from soma_skeleton import compute_bone_offsets, matrix_to_rot6d, build_parent_index

        cache_path = Path(cache_path)
        data = np.load(cache_path, allow_pickle=True)
        joints = data["joints"]                        # [N, T, J, 3] root-relative
        local_rot = data["local_rot_mats"]             # [N, T, J, 3, 3]
        root_pos = data["root_positions"]              # [N, T, 3]   root-relative
        joints_world = data["joints_world"]
        global_rot = data["global_rot_mats"]

        if crop_frames is not None and crop_frames < joints.shape[1]:
            joints = joints[:, :crop_frames, :, :]
            local_rot = local_rot[:, :crop_frames]
            root_pos = root_pos[:, :crop_frames]
            joints_world = joints_world[:, :crop_frames]
            global_rot = global_rot[:, :crop_frames]

        self.raw = joints.astype(np.float32)
        self.local_rot_raw = local_rot.astype(np.float32)           # [N, T, J, 3, 3]
        self.root_raw = root_pos.astype(np.float32)                 # [N, T, 3]
        self.N, self.T, self.J, self.C = self.raw.shape

        # Bone offsets computed from world-frame data (needed for FK)
        self.bone_offsets = compute_bone_offsets(joints_world, global_rot)   # [J, 3]
        self.parents = np.array(build_parent_index(), dtype=np.int64)         # [J]

        # Per-joint position normalisation (as before)
        if normalise:
            flat = self.raw.reshape(-1, self.J, self.C)
            self.mean = flat.mean(axis=0, keepdims=True)          # [1, J, C]
            self.std = flat.std(axis=0, keepdims=True) + 1e-6
            self.data = (self.raw - self.mean[None]) / self.std[None]

            # Root is in world units — normalise to unit scale
            self.root_mean = self.root_raw.reshape(-1, 3).mean(axis=0, keepdims=True)   # [1, 3]
            self.root_std = self.root_raw.reshape(-1, 3).std(axis=0, keepdims=True) + 1e-6
            self.root_data = (self.root_raw - self.root_mean[None]) / self.root_std[None]
        else:
            self.mean = np.zeros((1, self.J, self.C), dtype=np.float32)
            self.std = np.ones((1, self.J, self.C), dtype=np.float32)
            self.data = self.raw
            self.root_mean = np.zeros((1, 3), dtype=np.float32)
            self.root_std = np.ones((1, 3), dtype=np.float32)
            self.root_data = self.root_raw

        # Convert rotation matrices to 6D representation once
        rot_t = torch.from_numpy(self.local_rot_raw).float()
        rot6d_flat = matrix_to_rot6d(rot_t)   # [N, T, J, 6]
        self.rot6d = rot6d_flat.numpy().astype(np.float32)

        # Cache as tensors
        self.pos_tensor = torch.from_numpy(self.data).float()          # [N, T, J, 3]
        self.rot_tensor = torch.from_numpy(self.rot6d).float()          # [N, T, J, 6]
        self.root_tensor = torch.from_numpy(self.root_data).float()     # [N, T, 3]
        self.rotmat_tensor = torch.from_numpy(self.local_rot_raw).float()  # for FK supervision

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        return {
            "pos":            self.pos_tensor[idx],
            "rot6d":          self.rot_tensor[idx],
            "root":           self.root_tensor[idx],
            "local_rot_mats": self.rotmat_tensor[idx],
        }


# ----------------------------------------------------------------------------
# Masking utilities
# ----------------------------------------------------------------------------


_DESCENDANTS_CACHE: Optional[list[list[int]]] = None


def _get_descendants() -> list[list[int]]:
    global _DESCENDANTS_CACHE
    if _DESCENDANTS_CACHE is None:
        _DESCENDANTS_CACHE = build_descendants()
    return _DESCENDANTS_CACHE


def make_masked_batch(
    batch: torch.Tensor,
    mask_ratio: float = 0.25,
    mask_mode: str = "random",
    rng: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask a clean batch and return (masked_input, mask, target).

    batch : [B, T, J, C]
    mask  : [B, T, J] bool, True where masked (these are the positions whose
            target values the model must predict)
    """
    B, T, J, C = batch.shape
    if rng is None:
        rng = torch.Generator(device=batch.device).manual_seed(
            int(torch.randint(1 << 30, (1,)).item())
        )

    if mask_mode == "random":
        mask = torch.rand(B, T, J, device=batch.device, generator=rng) < mask_ratio

    elif mask_mode == "joint":
        joint_mask = torch.rand(B, J, device=batch.device, generator=rng) < mask_ratio
        mask = joint_mask.unsqueeze(1).expand(B, T, J)

    elif mask_mode == "time":
        time_mask = torch.rand(B, T, device=batch.device, generator=rng) < mask_ratio
        mask = time_mask.unsqueeze(-1).expand(B, T, J)

    elif mask_mode == "keyframe":
        # Keep a sparse set of keyframes (with prob `1 - mask_ratio`), mask the rest
        time_keep = torch.rand(B, T, device=batch.device, generator=rng) < (1 - mask_ratio)
        mask = (~time_keep).unsqueeze(-1).expand(B, T, J)

    elif mask_mode == "kinematic_chain":
        # For each sample pick a random "damaged" joint and mask its whole subtree
        # throughout all time steps.
        descendants = _get_descendants()
        if J != len(descendants):
            raise ValueError(
                f"kinematic_chain mask needs SOMA77 skeleton (J=77), got J={J}"
            )
        mask = torch.zeros(B, T, J, dtype=torch.bool, device=batch.device)
        # Choose a joint per batch element — exclude the root (index 0 / Hips)
        # because masking root subtree masks everything.
        choices = torch.randint(
            low=1, high=J, size=(B,), device=batch.device, generator=rng,
        )
        for b in range(B):
            joint_idx = int(choices[b].item())
            sub = descendants[joint_idx]
            mask[b, :, sub] = True

    else:
        raise ValueError(f"unknown mask_mode={mask_mode}")

    masked_input = batch.clone()
    masked_input[mask] = 0.0
    return masked_input, mask, batch


def mixed_mode_masking(
    batch: torch.Tensor,
    mask_ratio: float = 0.25,
    rng: torch.Generator | None = None,
    weights: dict[str, float] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pick a mask mode uniformly (or according to `weights`) per call.

    Used during training to expose the model to all mask modes.
    """
    modes = ["random", "joint", "time", "keyframe", "kinematic_chain"]
    if weights is None:
        probs = [1 / len(modes)] * len(modes)
    else:
        probs = [weights.get(m, 0.0) for m in modes]
        s = sum(probs)
        probs = [p / s for p in probs]
    pick = np.random.choice(modes, p=probs)
    return make_masked_batch(batch, mask_ratio=mask_ratio, mask_mode=pick, rng=rng)


if __name__ == "__main__":
    import sys
    cache = sys.argv[1] if len(sys.argv) > 1 else "runs/kimodo_cache.npz"
    ds = KimodoMotionDataset(cache)
    print(f"Dataset: {ds.N} samples, [T={ds.T}, J={ds.J}, C={ds.C}]")
    s = ds[0]
    print(f"Sample contents:")
    for k, v in s.items():
        print(f"  {k:<18} {tuple(v.shape)}  dtype={v.dtype}")
    print(f"Bone offsets: {ds.bone_offsets.shape}")

    # Collate example
    batch = torch.stack([ds[i]["pos"] for i in range(4)])
    for mode in ["random", "joint", "time", "keyframe", "kinematic_chain"]:
        masked, mask, _ = make_masked_batch(batch, mask_ratio=0.3, mask_mode=mode)
        frac = mask.float().mean().item()
        print(f"  mode={mode:<18} masked fraction: {frac:.3f}")
