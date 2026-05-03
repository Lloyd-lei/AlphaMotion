"""
Batch-generate motion data using Kimodo's Python API (not CLI, to avoid
re-loading the model per sample).

Output:
    runs/kimodo_cache.npz with keys:
        joints  : [N, T, J=77, 3] float32 — posed_joints (world coords)
        prompts : [N] object array of prompt strings

Usage:
    python gen_kimodo_data.py --num 200 --frames 90 --output runs/kimodo_cache.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# Diverse prompt templates covering locomotion / manipulation / balance / full-body
PROMPT_BANK = [
    # Locomotion
    "A person walks forward",
    "A person walks backward",
    "A person runs in a straight line",
    "A person jogs slowly",
    "A person walks while turning left",
    "A person walks while turning right",
    "A person sprints forward",
    "A person walks slowly with small steps",
    # Standing manipulation
    "A person picks up a cup with the right hand",
    "A person picks up a box with both hands",
    "A person opens a door with the right hand",
    "A person reaches up to a high shelf",
    "A person bends down to tie shoelaces",
    "A person waves hello with the right hand",
    "A person waves hello with both hands",
    "A person claps hands several times",
    "A person points forward with the right arm",
    # Sitting / kneeling
    "A person sits down on a chair",
    "A person stands up from a chair",
    "A person kneels on the right knee",
    "A person crosses legs while sitting",
    # Jumping / dynamic
    "A person jumps in place",
    "A person jumps forward",
    "A person hops on one foot",
    "A person does a small skip",
    # Whole-body coordinated
    "A person dances slowly",
    "A person stretches both arms above the head",
    "A person rotates the upper body",
    "A person twists the torso to the left",
    "A person twists the torso to the right",
    "A person bows forward",
    # Sports / martial arts
    "A person throws a ball with the right hand",
    "A person catches a ball",
    "A person swings a baseball bat",
    "A person performs a karate punch",
    "A person performs a karate kick",
    "A person performs a side kick",
    "A person performs a front kick",
    "A person blocks a punch with the left arm",
    # Daily activities
    "A person drinks from a cup",
    "A person types on a keyboard",
    "A person writes on a notepad",
    "A person folds clothes",
    "A person sweeps the floor",
    "A person carries a heavy bag",
    "A person climbs stairs",
    "A person descends stairs",
    "A person leans against a wall",
    "A person crawls on hands and knees",
    # Balance
    "A person stands on one foot",
    "A person walks on a narrow line",
    "A person balances with arms outstretched",
]


def make_prompt_list(n: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    if n <= len(PROMPT_BANK):
        # First n prompts deterministically, no repeats
        return PROMPT_BANK[:n]
    # Otherwise repeat with different seeds (Kimodo seed varies sample-level results)
    out = []
    while len(out) < n:
        idx = rng.permutation(len(PROMPT_BANK))
        for i in idx:
            out.append(PROMPT_BANK[i])
            if len(out) == n:
                break
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num", type=int, default=200, help="Number of samples to generate")
    p.add_argument("--frames", type=int, default=90, help="Frames per sample (3s at 30fps)")
    p.add_argument("--batch", type=int, default=4, help="Concurrent prompts per Kimodo call")
    p.add_argument("--steps", type=int, default=50, help="Diffusion denoising steps")
    p.add_argument("--model", default="Kimodo-SOMA-RP-v1")
    p.add_argument("--output", default="runs/kimodo_cache.npz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-postprocess", action="store_true",
                   help="Skip foot-skate cleanup (faster, uglier).")
    args = p.parse_args()

    torch.manual_seed(args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.model}...")
    from kimodo import load_model
    model = load_model(args.model, device="cuda", eval_mode=True)
    print("Model loaded.")

    prompts = make_prompt_list(args.num, seed=args.seed)
    print(f"Generating {len(prompts)} samples (frames={args.frames}, batch={args.batch})...")

    all_joints = []
    all_local_rot = []
    all_global_rot = []
    all_root_pos = []
    all_foot_contacts = []
    all_prompts = []
    t0 = time.time()
    pbar = tqdm(range(0, len(prompts), args.batch), desc="batch")
    for start in pbar:
        chunk = prompts[start:start + args.batch]
        out = model(
            prompts=chunk,
            num_frames=[args.frames] * len(chunk),
            num_denoising_steps=args.steps,
            return_numpy=True,
            post_processing=not args.no_postprocess,
            progress_bar=lambda x: x,
        )
        all_joints.append(out["posed_joints"])         # [B, T, 77, 3]
        all_local_rot.append(out["local_rot_mats"])    # [B, T, 77, 3, 3]
        all_global_rot.append(out["global_rot_mats"])  # [B, T, 77, 3, 3]
        all_root_pos.append(out["root_positions"])     # [B, T, 3]
        if "foot_contacts" in out:
            all_foot_contacts.append(out["foot_contacts"])
        all_prompts.extend(chunk)
        elapsed = time.time() - t0
        pbar.set_postfix(rate=f"{(start + args.batch) / elapsed:.1f} samp/s")

    joints_arr = np.concatenate(all_joints, axis=0).astype(np.float32)
    local_rot = np.concatenate(all_local_rot, axis=0).astype(np.float32)
    global_rot = np.concatenate(all_global_rot, axis=0).astype(np.float32)
    root_pos = np.concatenate(all_root_pos, axis=0).astype(np.float32)
    foot_contacts = (
        np.concatenate(all_foot_contacts, axis=0) if all_foot_contacts else None
    )
    prompts_arr = np.array(all_prompts, dtype=object)

    print(f"\nGenerated {joints_arr.shape[0]} samples in {time.time() - t0:.1f}s")
    print(f"  posed_joints    {joints_arr.shape}  {joints_arr.dtype}")
    print(f"  local_rot_mats  {local_rot.shape}   {local_rot.dtype}")
    print(f"  global_rot_mats {global_rot.shape}  {global_rot.dtype}")
    print(f"  root_positions  {root_pos.shape}")

    # Strip t=0 root translation from posed_joints (training target uses this)
    joints_relative = joints_arr - joints_arr[:, :1, :1, :]
    root_relative = root_pos - root_pos[:, :1, :]

    save_dict = dict(
        joints=joints_relative,
        joints_world=joints_arr,
        local_rot_mats=local_rot,
        global_rot_mats=global_rot,
        root_positions=root_relative,
        root_positions_world=root_pos,
        prompts=prompts_arr,
    )
    if foot_contacts is not None:
        save_dict["foot_contacts"] = foot_contacts
    np.savez(out_path, **save_dict)
    print(f"\nSaved to {out_path}")
    print(f"  joints range     [{joints_relative.min():.3f}, {joints_relative.max():.3f}]")
    print(f"  local_rot range  [{local_rot.min():.3f}, {local_rot.max():.3f}]")
    print(f"  local_rot is proper rotation? det ≈ 1 ?  "
          f"mean det = {np.linalg.det(local_rot.reshape(-1, 3, 3)).mean():.4f}")


if __name__ == "__main__":
    main()
