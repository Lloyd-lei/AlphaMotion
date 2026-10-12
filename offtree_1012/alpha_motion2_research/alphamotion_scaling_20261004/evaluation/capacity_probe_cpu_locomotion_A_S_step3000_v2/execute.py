from pathlib import Path
import subprocess,json,datetime,os
command=['/home/arenalabs/miniconda3/envs/ardy/bin/python', '-u', '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004/evaluation/run_locomotion_eval.py', '--root', '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004', '--run', '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004/runs/A_S_seed0', '--output', '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004/evaluation/capacity_probe_cpu_locomotion_A_S_step3000_v2', '--device', 'cpu', '--max-frames', '64', '--clips-per-split', '3', '--batch-size', '32', '--bootstrap-samples', '100', '--checkpoint', '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004/runs/A_S_seed0/checkpoint_step3000.pt']
out=Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004/evaluation/capacity_probe_cpu_locomotion_A_S_step3000_v2')
r=subprocess.run(command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
(out/'exit_status.json').write_text(json.dumps({'returncode':r.returncode,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_updates':0})+'\n')
raise SystemExit(r.returncode)
