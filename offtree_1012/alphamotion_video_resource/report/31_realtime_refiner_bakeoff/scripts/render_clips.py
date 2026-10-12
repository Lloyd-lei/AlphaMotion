"""Render candidate outputs with sajio's review renderer (descriptor + compact npz), then burn candidate/prompt captions with ffmpeg."""
import sys, os, json, subprocess, textwrap, numpy as np
from common import *
from greenwich_motion_sdk.render_review import render as sajio_render
OUT = f'{S}/bake/out'; VID = f'{S}/bake/vid'; os.makedirs(VID, exist_ok=True)
H2_XML = '/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml'
DESC = {'unitree_g1_29dof': f'{S}/ardy5/sajio/desc/unitree_g1_29dof.json', 'unitree_h2': f'{S}/ardy5/sajio/desc/unitree_h2.json'}
XML = {'unitree_g1_29dof': resolve_xml('unitree_g1_29dof'), 'unitree_h2': H2_XML}
PROMPT_TEXT = json.load(open(f'{OUT}/results_ardy_g1.json'))
CAND_DESC = {'R0': 'R0 baseline: greedy+LM25 projection, decoded root height, no correction', 'R1': 'R1 geometric snap (causal): planted sole on floor + xz foot latch, no optimization',
             'R2': 'R2 recursive LM (causal): warm start + AR(2) prior + foot pins, 2 iterations', 'R2h': 'R2h = R2 + hand pins (manipulation precision test)',
             'R3': 'R3 receding horizon: ARDY window, in-window plant segments pinned to median pose, 3 iterations', 'E': 'E Eric offline: 100 Adam steps over the whole clip (non-real-time)'}
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def esc(t): return t.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace(",", "\\,")
def main(sources, cands, packs):
    for pkname in packs:
        pk = pack(pkname); desc = json.load(open(DESC[pkname])); pref = desc['robot']; sx, _, _, qn_t, _ = build_from_mjcf(XML[pkname], pkname); tslot = {n: (k, s) for k, names in enumerate(qn_t) for s, n in enumerate(names)}
        rows = []
        for srcname in sources:
            for c in cands:
                f = f"{OUT}/npz/{srcname.replace('/', '__')}__{pkname}__{c}.npz"
                if not os.path.exists(f): print('missing', f); continue
                z = np.load(f); q, R, root, fps = z['q'], z['root_R'], z['root_cm'], float(z['fps']); Tn = len(q)
                qd = {n: q[:, k, s] for n, (k, s) in pk.slot.items()}; qt = np.zeros((Tn, sx.J, 3), np.float32)
                for n, (k, s) in tslot.items():
                    if n in qd: qt[:, k, s] = qd[n]
                stem = f"{srcname.split('/')[-1]}__{c}"; out = f'{VID}/{pkname}__{stem}.npz'
                np.savez(out, schema=np.array('greenwich.soma18.compact.v1'), stem=np.array(stem), fps=np.array(fps, np.float32), **{f'{pref}__q': qt, f'{pref}__root_rot6d': np.concatenate([R[:, :, 0], R[:, :, 1]], 1).astype(np.float32), f'{pref}__root_t_cm': root.astype(np.float32), f'{pref}__joint_names': np.asarray(list(sx.joint_names), dtype=str), f'{pref}__sole_height_cm': np.zeros((Tn, 2), np.float32), f'{pref}__model_contact': np.zeros((Tn, 2), bool)})
                rows.append({'archive': out, 'member': None, 'stem': stem, 'review_category': stem, 'src': srcname, 'cand': c})
        man = f'{VID}/manifest_{pkname}.json'; json.dump({'clips': rows}, open(man, 'w'))
        res = sajio_render(man, DESC[pkname], XML[pkname], f'{VID}/raw_{pkname}', '/usr/bin/ffmpeg', count=len(rows), height=720)
        for item in res:
            srcname, c = item['src'], item['cand']; key = srcname.split('/')[-1]; text = PROMPT_TEXT.get(srcname, {}).get('text', key)
            lines = textwrap.wrap(text, 70); draws = [f"drawtext=fontfile={FONT}:text='{esc(l)}':fontsize=24:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8:x=(w-text_w)/2:y=h-{40 + 32 * (len(lines) - 1 - i)}" for i, l in enumerate(lines)]
            head = f"drawtext=fontfile={FONT}:text='{esc(CAND_DESC[c])}':fontsize=21:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6:x=20:y=64"
            head2 = f"drawtext=fontfile={FONT}:text='{esc(srcname.split('/')[0] + ' -> Megatron -> ' + pkname + '   [' + key + ']')}':fontsize=21:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=6:x=20:y=96"
            dst = f"{VID}/{key}__{pkname}__{srcname.split('/')[0]}__{c}.mp4"
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', f"{VID}/raw_{pkname}/{item['video']}", '-vf', ','.join([head, head2] + draws), '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', dst], check=True); print('wrote', dst, flush=True)
if __name__ == '__main__':
    main(sys.argv[1].split(','), sys.argv[2].split(','), sys.argv[3].split(','))
