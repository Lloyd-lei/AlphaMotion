"""ARDY G1 generation for 5 long prompts with stage timing (text encode / autoregressive generation), in-process.
Model loaded once. Per prompt: 1 warm-up + 3 timed runs (median); the seed-0 run is saved as npz (+ MuJoCo csv)."""
import os, sys, json, time, statistics, numpy as np, torch
sys.path.insert(0, os.getcwd())
from ardy.model import load_model
from ardy.model.registry import resolve_model_name
from ardy.motion_rep.tools import length_to_mask
from ardy.tools import seed_everything, to_numpy
from ardy.exports.mujoco import MujocoQposConverter
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
PROMPTS = {
 "n1_walkturn": "A person walks forward, turns around and walks back.",
 "n2_jacks": "A person does three jumping jacks.",
 "n3_squat": "A person squats down and stands up, twice.",
 "n4_sidestep": "A person side-steps to the left, then side-steps back to the right.",
 "n5_runstop": "A person runs forward and stops abruptly.",
}
_OLD = {
 'p1_box':   "A person walks forward three steps, stops, crouches down to pick up a heavy box with both hands, stands up slowly, turns around and walks back carrying the box.",
 'p2_hurdle':"A person jogs forward, jumps over a low obstacle, lands, stumbles slightly, regains balance and keeps walking.",
 'p3_wave':  "A person waves hello with the right hand, bows deeply, steps to the left, then does a full spin turn and stops.",
 'p4_kick':  "A person kicks a ball with the right foot, sidesteps to the right, crouches low, then stands up with both arms raised overhead.",
 'p5_sit':   "A person walks in a circle while looking around, sits down on the floor cross-legged, rests for a moment, then stands back up.",
}
DUR = 8.0; device = 'cuda:0'
ck = '/home/arenalabs/Desktop/"be water, robot"/experiment/ardy/checkpoints'
t0 = time.perf_counter(); name = resolve_model_name('g1', checkpoints_dir=ck); model = load_model(name, device=device, checkpoints_dir=ck); torch.cuda.synchronize(); load_s = time.perf_counter() - t0
fps = model.motion_rep.fps; patch = model.num_frames_per_token; H = model.gen_horizon_len
max_window = (int(10 * fps) // patch) * patch; hist = ((max_window - H) // patch) * patch
steps = int(model.diffusion.num_base_steps); nf = int(DUR * fps)
print(f'model {name} load {load_s:.1f}s fps {fps} token {patch} frames, horizon {H}, history {hist}, diffusion steps {steps}, frames {nf}', flush=True)
timing = {'gpu': os.popen('nvidia-smi --query-gpu=name --format=csv,noheader').read().strip(), 'model': name, 'model_load_s': load_s, 'fps': fps, 'horizon_frames': H, 'frames_per_token': patch, 'history_frames': hist, 'diffusion_steps': steps, 'duration_s': DUR, 'prompts': {}}
conv = MujocoQposConverter(model.skeleton)
def sync(): torch.cuda.synchronize()
for key, text in PROMPTS.items():
    lengths = torch.tensor([nf], device=device); pad = length_to_mask(lengths); head = torch.zeros(1, device=device)
    te, ge, saved = [], [], None
    for run in range(3):                                     # run 0 = warm-up
        seed_everything(0)
        with torch.no_grad():
            sync(); a = time.perf_counter(); text_feat, text_pad = model._encode_text([text]); sync(); b = time.perf_counter()
            motion = model([text], nf, num_denoising_steps=steps, pad_mask=pad, first_heading_angle=head, motion_mask=None, observed_motion=None, cfg_weight=(2.0, 2.0), crop_history_length=hist, text_feat=text_feat, text_pad_mask=text_pad)
            sync(); c = time.perf_counter()
            out = model.motion_rep.inverse(motion, is_normalized=True); sync(); d = time.perf_counter()
        if run == 0:
            continue
        te.append(b - a); ge.append(c - b)
        if run == 1:
            o = to_numpy(out); o = {k: (v[0] if hasattr(v, 'shape') and len(v.shape) > 0 and v.shape[0] == 1 else v) for k, v in o.items()}
            arrs = {k: np.asarray(v) for k, v in o.items()}; arrs['fps'] = np.asarray(fps); arrs['text'] = np.asarray(text); arrs['joint_names'] = np.asarray([str(n) for n in model.skeleton.joint_names]) if hasattr(model.skeleton, 'joint_names') else np.asarray([])
            np.savez(f'{OUT}/{key}.npz', **arrs); qpos = conv.dict_to_qpos(to_numpy(out) if False else out, device); conv.save_csv(qpos, f'{OUT}/{key}.csv'); inv_s = d - c
    n_win = int(np.ceil(nf / H))
    timing['prompts'][key] = {'text': text, 'frames': nf, 'windows': n_win, 'text_encode_ms': 1000 * statistics.median(te), 'generation_ms': 1000 * statistics.median(ge), 'generation_ms_per_window': 1000 * statistics.median(ge) / n_win,
                              'inverse_rep_ms': 1000 * inv_s, 'rtf_generation': statistics.median(ge) / DUR, 'rtf_text_plus_generation': (statistics.median(te) + statistics.median(ge)) / DUR, 'runs_ms': [1000 * x for x in ge]}
    print(key, json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in timing['prompts'][key].items() if k != 'text'}), flush=True)
json.dump(timing, open(f'{OUT}/ardy_timing.json', 'w'), indent=1); print('DONE')
