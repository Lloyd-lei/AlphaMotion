"""Two-stage teleop probe (lloyd 0917): BFM-Zero baseline tracks a reference that came from a HUMAN via Greenwich
(or the native retarget as control); z is causal: per-frame (window 1, the repo default) or a moving average over the
past --z-window frames.  Derived from stress_run.py (0831).

Same rollout as humanoidverse.tracking_inference, but it does NOT stop at
termination -- it keeps stepping so that fall-and-recovery can be observed --
and it logs full policy/reference state per step for offline robustness scoring.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import joblib, mediapy as media, numpy as np, torch
from torch.utils._pytree import tree_map

from humanoidverse.agents.load_utils import load_model_from_checkpoint_dir
from humanoidverse.mjlab_inference_utils import (
    MujocoQposRenderer, checkpoint_load_device, load_mjlab_env_cfg, render_policy_frame)
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.robot_spec import load_robot_training_spec, resolve_robot_config_path
from humanoidverse.tracking_inference import _control_to_qpos_order_indices, _expert_qpos_from_obs, _target_states_from_obs, _resize_nearest


@torch.no_grad()
def _tracking_z(model, obs, window=1):
    zb = model.backward_map(obs).clone()
    z = torch.empty_like(zb)
    for step in range(zb.shape[0]):
        z[step] = zb[max(0, step - window + 1):step + 1].mean(dim=0)     # causal: only past + current reference frames
    return model.project_z(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-folder", type=Path, required=True)
    ap.add_argument("--data-path", type=Path, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--motion-list", type=int, nargs="+", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--render-size", type=int, default=420)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--disable-dr", action="store_true")
    ap.add_argument("--disable-obs-noise", action="store_true")
    ap.add_argument("--z-window", type=int, default=1, help="causal moving-average window (frames) for z; 1 = repo default")
    ap.add_argument("--robot-config", default="configs/robots/g1_29dof.yaml")
    ap.add_argument("--no-tanh", action="store_true", help="bypass the repo tanh squash on the actor mean (probe for the released-checkpoint action-range mismatch)")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    robot_config = resolve_robot_config_path(a.robot_config)
    rt = load_robot_training_spec(robot_config)
    robot_xml = Path(rt.robot.xml_path).expanduser().resolve()
    num_dof = len(rt.robot.control_joint_names)
    dof_idx, _ = _control_to_qpos_order_indices(rt)

    model = load_model_from_checkpoint_dir(a.model_folder / "checkpoint", device=checkpoint_load_device(a.device))
    model.to(a.device); model.eval()
    if a.no_tanh:
        import torch.nn as _nn
        from humanoidverse.agents import nn_models as _nm
        from humanoidverse.agents.nn_models import TruncatedNormal as _TN
        def _fwd(self, obs, z, std):
            obs = self.input_filter(obs)
            e = torch.cat([self.embed_s(obs), self.embed_z(torch.cat([obs, z], dim=-1))], dim=-1)
            mu = self.policy(e)
            return _TN(mu, torch.ones_like(mu) * std, low=-5.0, high=5.0)
        _nm.ResidualActor.forward = _fwd
        _nm.Actor.forward = _fwd
        print("[INFO] tanh squash bypassed on actor mean")

    env_cfg, use_rh = load_mjlab_env_cfg(a.model_folder, data_path=a.data_path, robot_config=robot_config,
                                         device=a.device, headless=True, disable_dr=a.disable_dr,
                                         disable_obs_noise=a.disable_obs_noise, max_episode_length_s=10000.0)
    wrapped, _ = env_cfg.build(num_envs=1)
    env = wrapped._env
    env._motion_lib.load_all_motions()
    env.is_evaluating = True
    keys = [str(k) for k in env._motion_lib._motion_data_keys]
    renderer = MujocoQposRenderer(robot_xml, render_size=a.render_size, camera_distance=3.0,
                                  camera_azimuth=135.0, camera_elevation=-18.0, expected_qpos_size=7 + num_dof)
    body_names = list(rt.robot.body_names)
    try:
        for mid in a.motion_list:
            name = keys[mid]
            bobs, odict = get_backward_observation(env, mid, use_root_height_obs=use_rh)
            z = _tracking_z(model, tree_map(lambda x: x[1:].to(a.device) if hasattr(x, "to") else x, bobs), window=a.z_window)
            tgt = _target_states_from_obs(odict, device=a.device, num_dof=num_dof)
            obs, _ = wrapped.reset(to_numpy=False, target_states=tgt)
            T = int(z.shape[0])
            eq = _expert_qpos_from_obs(odict, num_dof=num_dof, dof_qpos_order_indices=dof_idx)
            log = {k: [] for k in ("bpos", "brot", "dpos", "dvel", "term", "trunc")}
            frames = []
            use_env_render = True
            print(f"[RUN] {a.tag} {name} (id={mid}) steps={T}", flush=True)
            for step in range(T):
                act = model.act(obs, z[step].unsqueeze(0), mean=True)
                obs, _r, term, trunc, _i = wrapped.step(act, to_numpy=False)
                sim = env.simulator
                log["bpos"].append(sim._rigid_body_pos.reshape(1, -1, 3)[0].detach().cpu().numpy())
                log["brot"].append(sim._rigid_body_rot.reshape(1, -1, 4)[0].detach().cpu().numpy())
                log["dpos"].append(sim.dof_state[..., 0][0].detach().cpu().numpy())
                log["dvel"].append(sim.dof_state[..., 1][0].detach().cpu().numpy())
                log["term"].append(bool(torch.as_tensor(term).any()))
                log["trunc"].append(bool(torch.as_tensor(trunc).any()))
                if not a.no_video:
                    pf, use_env_render = render_policy_frame(wrapped, renderer, use_env_render=use_env_render)
                    ef = renderer.render_qpos(eq[min(step + 1, len(eq) - 1)])
                    ef = _resize_nearest(ef, pf.shape[0], pf.shape[1])
                    frames.append(np.concatenate([ef, pf], axis=1))
                if (step + 1) % 200 == 0:
                    print(f"  {step+1}/{T}", flush=True)
            out = a.out_dir / f"{a.tag}__{name}"
            np.savez_compressed(str(out) + ".npz",
                                bpos=np.asarray(log["bpos"], np.float32), brot=np.asarray(log["brot"], np.float32),
                                dpos=np.asarray(log["dpos"], np.float32), dvel=np.asarray(log["dvel"], np.float32),
                                term=np.asarray(log["term"]), trunc=np.asarray(log["trunc"]),
                                ref_bpos=odict["ref_body_pos"].detach().cpu().numpy().astype(np.float32),
                                ref_brot=odict["ref_body_rots"].detach().cpu().numpy().astype(np.float32),
                                ref_dpos=odict["dof_pos"].detach().cpu().numpy().astype(np.float32),
                                body_names=np.array(body_names), motion=name, tag=a.tag)
            if not a.no_video:
                media.write_video(str(out) + ".mp4", frames, fps=a.fps)
            print(f"[OK] wrote {out}.npz" + ("" if a.no_video else f" and .mp4 ({len(frames)}f)"), flush=True)
    finally:
        renderer.close(); wrapped.close()


if __name__ == "__main__":
    main()
