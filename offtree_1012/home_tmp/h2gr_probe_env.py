"""Zero-action probe of the mjlab tracking env: where does the ee error come from at spawn?"""
import sys, os; sys.path.insert(0, '/home/arenalabs/alphamotion_ctrl'); import amctrl, torch, numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
task, motion = 'Mjlab-Tracking-Flat-Unitree-H2', sys.argv[1]
cfg = load_env_cfg(task, play=False); mc = cfg.commands['motion']; assert isinstance(mc, MotionCommandCfg)
mc.motion_file = motion; mc.sampling_mode = 'start'; cfg.scene.num_envs = 4; cfg.events.pop('push_robot', None)
cfg.observations['actor'].enable_corruption = False
env = ManagerBasedRlEnv(cfg=cfg, device='cuda:0'); env.reset()
cmd = env.command_manager.get_term('motion'); robot = env.scene['robot']
names = list(cmd.cfg.body_names); ee = cfg.terminations['ee_body_pos'].params['body_names']; thr = cfg.terminations['ee_body_pos'].params['threshold']
print('ee bodies', ee, 'thr', thr, 'ncon max', cfg.sim.nconmax)
for step in range(8):
    err = (cmd.body_pos_relative_w[0] - cmd.robot_body_pos_w[0]).norm(dim=-1).cpu().numpy()
    raw = (cmd.body_pos_w[0] - cmd.robot_body_pos_w[0]).norm(dim=-1).cpu().numpy()
    worst = int(err.argmax())
    rz = float(robot.data.root_link_pos_w[0, 2]); cz = float(cmd.body_pos_w[0, 0, 2])
    aq = cmd.anchor_quat_w[0].cpu().numpy().round(3); rq = cmd.robot_anchor_quat_w[0].cpu().numpy().round(3)
    ncon = int(env.sim.data.ncon[0]) if hasattr(env.sim.data, 'ncon') else -1
    print(f'step {step}: worst {names[worst]} rel_err {err[worst]*100:5.1f} cm (raw world err {raw[worst]*100:5.1f} cm) | robot root z {rz:.3f} ref pelvis z {cz:.3f} | ref anchor quat {aq} robot anchor quat {rq} | ncon {ncon}')
    _ret = env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device='cuda:0'))
    tm = env.termination_manager
    fired = [t for t in tm.active_terms if bool(tm.get_term(t)[0])]
    if fired: print('   TERMINATED by', fired)
