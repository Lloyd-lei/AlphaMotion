"""Uniform-spawn zero-action probe (with AM_REF_RESIDUAL=1 action 0 holds the reference): which spawn frames die, and how?"""
import sys, os; sys.path.insert(0, '/home/arenalabs/alphamotion_ctrl'); import amctrl, torch, numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
task, motion, N = 'Mjlab-Tracking-Flat-Unitree-H2', sys.argv[1], 64
cfg = load_env_cfg(task, play=False); mc = cfg.commands['motion']; mc.motion_file = motion; mc.sampling_mode = 'uniform'; cfg.scene.num_envs = N
cfg.events.pop('push_robot', None); cfg.observations['actor'].enable_corruption = False
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0'); env.reset()
cmd = env.command_manager.get_term('motion'); names = list(cmd.cfg.body_names); tm = env.termination_manager
t0 = cmd.time_steps.clone().cpu().numpy(); died = np.full(N, -1); cause = ['']*N; worst = ['']*N; err0 = np.zeros(N)
for step in range(12):
    err = (cmd.body_pos_relative_w - cmd.robot_body_pos_w).norm(dim=-1).cpu().numpy()
    if step == 0: err0 = err.max(1)
    env.step(torch.zeros(N, env.action_manager.total_action_dim, device='cuda:0'))
    for i in range(N):
        if died[i] < 0:
            fired = [t for t in tm.active_terms if bool(tm.get_term(t)[i])]
            if fired: died[i] = step; cause[i] = fired[0]; worst[i] = names[int(err[i].argmax())]
alive = died < 0
print(f'spawn frames 0..{int(cmd.motion.time_step_total)}; {alive.sum()}/{N} alive after 12 steps; spawn err0 median {np.median(err0)*100:.1f} cm')
order = np.argsort(t0)
print('frame  died_step  cause  worst_body  err@spawn(cm)')
for i in order:
    if died[i] >= 0: print(f'{int(t0[i]):5d}  {died[i]:3d}  {cause[i]:12s} {worst[i]:24s} {err0[i]*100:5.1f}')
print('alive spawn frames:', sorted(int(t0[i]) for i in range(N) if alive[i]))
