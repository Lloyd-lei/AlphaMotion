import sys, os, json, subprocess, numpy as np
S='/tmp/claude-1000/-home-arenalabs-Desktop--be-water--robot-/eff459e5-3969-4f6d-9ec8-eae021b0a07a/scratchpad'; T=f'{S}/alphamotion-locomotion-toolkit'; C=f'{S}/ardy_core'; O=f'{C}/sajio'
sys.path.insert(0, f'{T}/src')
from arena.eval.visual import resolve_xml
from alphamotion.engine.descriptor import build_from_mjcf
from greenwich_motion_sdk.render_review import render
H2_XML = '/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml'
meta = json.load(open('/var/tmp/arena/raw/hf_soma_h2/metadata/robot_h2.json')); hslot = {x['native_joint_name']: (x['node_index'], x['slot_index']) for x in meta['native_slots']}
ROBOTS = ['unitree_g1_29dof', 'unitree_h2', 'agibot_a3_t3d0', 'fourier_gr3', 'pnd_adam_lite']; PROMPTS = ['p1_box', 'p2_hurdle', 'p3_wave', 'p4_kick', 'p5_sit']
per_prompt = {p: [] for p in PROMPTS}
for name in ROBOTS:
    xml = H2_XML if name == 'unitree_h2' else resolve_xml(name); desc = f'{O}/desc/{name}.json'
    r = subprocess.run(['/home/arenalabs/miniconda3/bin/python3', f'{T}/toolkits/greenwich-soma-multibody/scripts/export_viewer_robot.py', '--repo', '/home/arenalabs/Desktop/arena-datagrid/packages/alphamotion', '--xml', xml, '--body', name, '--output', desc], capture_output=True, text=True)
    if r.returncode: print('descriptor FAILED', name, r.stderr[-600:]); continue
    pref = json.load(open(desc))['robot']
    spec_t, _, _, qn_t, _ = build_from_mjcf(xml, name); tslot = {n: (k, s) for k, names in enumerate(qn_t) for s, n in enumerate(names)}; rows = []
    for p in PROMPTS:
        z = np.load(f'{C}/runs/{p}/{name}.npz'); q, R, root, fps = z['q'], z['root_R'], z['root_world_cm'], float(z['fps']); Tn = len(q)
        src = {n: q[:, k, s] for n, (k, s) in hslot.items()} if name == 'unitree_h2' else {n: q[:, k, s] for k, names in enumerate(qn_t) for s, n in enumerate(names)}
        qt = np.zeros((Tn, spec_t.J, 3), np.float32)
        for n, (k, s) in tslot.items():
            if n in src: qt[:, k, s] = src[n]
        rot6d = np.concatenate([R[:, :, 0], R[:, :, 1]], 1).astype(np.float32); out = f'{O}/{p}__{name}.npz'
        np.savez(out, schema=np.array('greenwich.soma18.compact.v1'), stem=np.array(f'{p}__{name}'), fps=np.array(fps, np.float32), **{f'{pref}__q': qt, f'{pref}__root_rot6d': rot6d, f'{pref}__root_t_cm': root.astype(np.float32), f'{pref}__joint_names': np.asarray(list(spec_t.joint_names), dtype=str), f'{pref}__sole_height_cm': np.zeros((Tn, 2), np.float32), f'{pref}__model_contact': np.zeros((Tn, 2), bool)})
        rows.append({'archive': out, 'member': None, 'stem': f'{p}__{name}', 'review_category': p, 'robot_id': name})
    man = f'{O}/manifest_{name}.json'; json.dump({'clips': rows}, open(man, 'w'))
    try: res = render(man, desc, xml, f'{O}/out_{name}', '/usr/bin/ffmpeg', count=len(rows), height=720)
    except Exception as e: print('render FAILED', name, repr(e)[:400]); continue
    for item in res: per_prompt[item['review_category']].append(f'{O}/out_{name}/{item["video"]}')
    print('rendered', name, len(res), flush=True)
json.dump(per_prompt, open(f'{O}/per_prompt.json', 'w'), indent=1)
for p, vids in per_prompt.items():
    if len(vids) < 2: print('skip', p, vids); continue
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error'] + sum([['-i', v] for v in vids], []) + ['-filter_complex', ''.join(f'[{i}:v]' for i in range(len(vids))) + f'hstack=inputs={len(vids)},scale=3840:-2[v]', '-map', '[v]', '-c:v', 'libx264', '-crf', '22', '-pix_fmt', 'yuv420p', f'{O}/{p}__core_sajio_G1_H2_A3_GR3_Adam.mp4'], check=True); print('stacked', p, len(vids))
print('SAJIO DONE')
