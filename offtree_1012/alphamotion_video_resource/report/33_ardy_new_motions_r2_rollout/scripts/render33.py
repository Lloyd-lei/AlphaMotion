"""Report 33 videos: sajio's review renderer per clip, then per prompt x robot ONE mp4 = [R0 raw | R2 refined] hstack (2560x720) with captions, plus a 12-frame grid."""
import os, sys, json, subprocess, textwrap, numpy as np
S = '/tmp/claude-1000/-home-arenalabs-Desktop--be-water--robot-/eff459e5-3969-4f6d-9ec8-eae021b0a07a/scratchpad'
sys.path.insert(0, f'{S}/alphamotion-locomotion-toolkit/src')
from alphamotion.engine.descriptor import build_from_mjcf
from arena.eval.visual import resolve_xml
from greenwich_motion_sdk.render_review import render as sajio_render
import imageio.v2 as iio
from PIL import Image
OUT = f'{S}/ardy33/out'; VID = f'{S}/ardy33/vid'; os.makedirs(VID, exist_ok=True)
H2_XML = '/var/tmp/alphamotion/robot_models/unitree_ros/robots/h2_description/unitree_h2_mjcf.xml'; H2_LOOP_XML = '/home/arenalabs/tmp/greenwich_workbench_0911/h2_assets/H2_loop.xml'
ROBOTS = ['unitree_g1_29dof', 'unitree_h2', 'agibot_a3_t3d0', 'fourier_gr3', 'pnd_adam_lite']
DESC = {n: f'{S}/ardy5/sajio/desc/{n}.json' for n in ROBOTS}
XML = {n: (H2_XML if n == 'unitree_h2' else resolve_xml(n)) for n in ROBOTS}
PACK_XML = {n: (H2_LOOP_XML if n == 'unitree_h2' else XML[n]) for n in ROBOTS}          # the skeleton the pipeline's q slots refer to
RES = json.load(open(f'{OUT}/results.json')); KEYS = list(RES.keys()); CAND = sys.argv[1] if len(sys.argv) > 1 else 'R2'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def esc(t): return t.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace(",", "\\,")
LABEL = {'R0': 'R0 raw decode (current path, no correction)', 'R2': 'R2 recursive refiner (causal, 2 LM steps/frame)', 'R2f': 'R2f recursive refiner + falling-edge latch release'}
for rn in ROBOTS:
    desc = json.load(open(DESC[rn])); pref = desc['robot']; sx, _, _, qn_t, _ = build_from_mjcf(XML[rn], rn); tslot = {n: (k, s) for k, names in enumerate(qn_t) for s, n in enumerate(names)}
    px, _, _, qn_p, _ = build_from_mjcf(PACK_XML[rn], rn); pslot = {n: (k, s) for k, names in enumerate(qn_p) for s, n in enumerate(names)}
    rows = []
    for key in KEYS:
        for c in ('R0', CAND):
            z = np.load(f'{OUT}/npz/{key}__{rn}__{c}.npz'); q, R, root, fps = z['q'], z['root_R'], z['root_cm'], float(z['fps']); Tn = len(q)
            qd = {n: q[:, k, s] for n, (k, s) in pslot.items()}; qt = np.zeros((Tn, sx.J, 3), np.float32)
            for n, (k, s) in tslot.items():
                if n in qd: qt[:, k, s] = qd[n]
            stem = f'{key}__{c}'; out = f'{VID}/{rn}__{stem}.npz'
            np.savez(out, schema=np.array('greenwich.soma18.compact.v1'), stem=np.array(stem), fps=np.array(fps, np.float32), **{f'{pref}__q': qt, f'{pref}__root_rot6d': np.concatenate([R[:, :, 0], R[:, :, 1]], 1).astype(np.float32), f'{pref}__root_t_cm': root.astype(np.float32), f'{pref}__joint_names': np.asarray(list(sx.joint_names), dtype=str), f'{pref}__sole_height_cm': np.zeros((Tn, 2), np.float32), f'{pref}__model_contact': np.zeros((Tn, 2), bool)})
            rows.append({'archive': out, 'member': None, 'stem': stem, 'review_category': stem, 'key': key, 'cand': c})
    man = f'{VID}/manifest_{rn}.json'; json.dump({'clips': rows}, open(man, 'w'))
    vj = f'{VID}/raw_{rn}/videos.json'
    if os.path.exists(vj) and len(json.load(open(vj))['clips']) == len(rows): res = json.load(open(vj))['clips']; print('reusing raw renders for', rn, flush=True)
    else: res = sajio_render(man, DESC[rn], XML[rn], f'{VID}/raw_{rn}', '/usr/bin/ffmpeg', count=len(rows), height=720)
    vid_of = {(it['key'], it['cand']): f"{VID}/raw_{rn}/{it['video']}" for it in res}
    for key in KEYS:
        text = RES[key]['text']; lines = textwrap.wrap(text, 90)
        draws = [f"drawtext=fontfile={FONT}:text='{esc(l)}':fontsize=28:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=8:x=(w-text_w)/2:y=h-{44 + 36 * (len(lines) - 1 - i)}" for i, l in enumerate(lines)]
        heads = [f"drawtext=fontfile={FONT}:text='{esc(LABEL['R0'] + '   |   ' + rn)}':fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6:x=20:y=64",
                 f"drawtext=fontfile={FONT}:text='{esc(LABEL[CAND] + '   |   ' + rn)}':fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6:x=1300:y=64"]
        dst = f'{VID}/{key}__{rn}__R0_vs_{CAND}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', vid_of[(key, 'R0')], '-i', vid_of[(key, CAND)], '-filter_complex', '[0:v][1:v]hstack=inputs=2,' + ','.join(heads + draws) + '[v]', '-map', '[v]', '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', dst], check=True)
        rd = iio.get_reader(dst); n = rd.count_frames(); idx = np.linspace(0, n - 1, 12).round().astype(int); sheet = Image.new('RGB', (1920, 540 * 12), (5, 5, 5))
        for r_, i in enumerate(idx): sheet.paste(Image.fromarray(rd.get_data(int(i))).resize((1920, 540)), (0, r_ * 540))
        rd.close(); sheet.save(f'{VID}/{key}__{rn}__R0_vs_{CAND}_grid12.jpg', quality=85); print('wrote', dst, flush=True)
