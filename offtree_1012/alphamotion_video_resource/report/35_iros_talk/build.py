"""Build the IROS deck as a native PPTX (16:9, 1920x1080 px grid), embedding the real mp4 clips."""
import os
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
S = '/tmp/claude-1000/-home-arenalabs-Desktop--be-water--robot-/eff459e5-3969-4f6d-9ec8-eae021b0a07a/scratchpad/iros_deck'
M, W = f'{S}/media', f'{S}/site/up'
COVER = '/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/pitch_deck_20260722/cover_robot.png'
OUT = f'{S}/pptx/AlphaMotion_IROS2026.pptx'
HEAD, BODY, MONO = 'Arial', 'Arial', 'Courier New'
BG, BG2, TXT, SOFT, MUTE, DIM, ORANGE = '0B0B0A', '151513', 'F2F0EA', 'D8D5CC', 'B9B6AD', '6E6C66', 'FF7A00'
px = lambda v: Emu(int(round(v * 6350)))
prs = Presentation(); prs.slide_width, prs.slide_height = px(1920), px(1080)
blank = prs.slide_layouts[6]

def slide(bg):
    s = prs.slides.add_slide(blank); f = s.background.fill; f.solid(); f.fore_color.rgb = RGBColor.from_string(bg); return s

def text(s, x, y, w, h, paras, size, color=TXT, font=BODY, bold=False, align=PP_ALIGN.LEFT, spacing=1.15, anchor=MSO_ANCHOR.TOP, caps=False, tracking=None):
    tb = s.shapes.add_textbox(px(x), px(y), px(w), px(h)); tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0; tf.vertical_anchor = anchor
    if isinstance(paras, str): paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = align; p.line_spacing = spacing
        runs = para if isinstance(para, list) else [(para, color)]
        for t, c in runs:
            r = p.add_run(); r.text = t.upper() if caps else t; f = r.font; f.size = Pt(size / 2); f.name = font; f.bold = bold; f.color.rgb = RGBColor.from_string(c)
            if tracking:
                r._r.get_or_add_rPr().set('spc', str(int(tracking * 100)))
    return tb

def rect(s, x, y, w, h, fill=None, line=None, lw=1, shape=MSO_SHAPE.RECTANGLE):
    sh = s.shapes.add_shape(shape, px(x), px(y), px(w), px(h))
    if fill: sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor.from_string(fill)
    else: sh.fill.background()
    if line: sh.line.color.rgb = RGBColor.from_string(line); sh.line.width = px(lw)
    else: sh.line.fill.background()
    sh.shadow.inherit = False
    return sh

def fit(path, x, y, w, h):
    iw, ih = Image.open(path).size; sc = min(w / iw, h / ih); fw, fh = iw * sc, ih * sc
    return x + (w - fw) / 2, y + (h - fh) / 2, fw, fh

def image(s, path, x, y, w, h, backdrop=None):
    if backdrop: rect(s, x, y, w, h, fill=backdrop)
    fx, fy, fw, fh = fit(path, x, y, w, h); return s.shapes.add_picture(path, px(fx), px(fy), px(fw), px(fh))

def video(s, stem, x, y, w, h, backdrop='000000'):
    mp4, jpg = f'{stem}.mp4', f'{stem}.jpg'
    rect(s, x, y, w, h, fill=backdrop)
    fx, fy, fw, fh = fit(jpg, x, y, w, h)
    return s.shapes.add_movie(mp4, px(fx), px(fy), px(fw), px(fh), poster_frame_image=jpg, mime_type='video/mp4')

def eyebrow(s, t, color=ORANGE): text(s, 128, 128, 1664, 36, t, 24, color, MONO, caps=True, tracking=4)
def title(s, t, y=176, h=80, size=64, color=TXT, w=1664): text(s, 128, y, w, h, t, size, color, HEAD, bold=True, spacing=1.05)
def footer(s, t, color=DIM): text(s, 128, 992, 1664, 36, t, 24, color)
def notes(s, t): s.notes_slide.notes_text_frame.text = t
def stat(s, x, y, w, big, small, color=TXT, size=72):
    text(s, x, y, w, size + 8, big, size, color, HEAD, bold=True, spacing=1.0); text(s, x, y + size + 14, w, 40, small, 26, MUTE)

# 1 cover
s = slide(BG)
im = Image.open(COVER); iw, ih = im.size; box_w, box_h = 820, 1080
pic = s.shapes.add_picture(COVER, px(1100), px(0), px(box_w), px(box_h))
ar, br = iw / ih, box_w / box_h
if ar > br: c = (1 - br / ar) / 2; pic.crop_left = pic.crop_right = c
else: c = (1 - ar / br) / 2; pic.crop_top = pic.crop_bottom = c
text(s, 128, 330, 900, 36, 'IROS 2026', 24, ORANGE, MONO, tracking=4)
text(s, 128, 380, 980, 220, 'A Cross-Embodiment Motion Tokenizer', 96, TXT, HEAD, bold=True, spacing=1.0)
text(s, 128, 612, 900, 60, 'Learning from joint-to-joint moving synergy', 40, 'C9C6BD')
rect(s, 128, 700, 120, 3, fill=ORANGE)
text(s, 128, 730, 900, 40, 'Lloyd Lei · Carnegie Mellon University', 28, '9C9A93')
notes(s, 'Open with the Thiel line about secrets, and thank the room for sharing theirs. Then the thesis in one line: robots are no longer starved for data; they are starved for data that works on their own body.')

# 2 gap
s = slide(BG); eyebrow(s, 'The bottleneck')
title(s, 'Motion data exists. Usable motion data for your robot does not.', h=170, size=72, w=1500)
cards = [('Tracking policies', 'One expert policy per motion. Strong, but it does not generalize across motions.', ORANGE),
         ('Generalists', 'SONIC shows the ceiling, on one body, with 700 h of mocap and about 9,000 GPU-hours.', '3A3934'),
         ('Planners and teleop', 'Built for walking: centimetre hands, window latency. Manipulation needs millimetres and a frame.', '3A3934')]
cw = (1664 - 64) / 3
for i, (h3, p, lc) in enumerate(cards):
    x = 128 + i * (cw + 32); rect(s, x, 430, cw, 3, fill=lc)
    text(s, x, 461, cw, 50, h3, 36, TXT, HEAD, bold=True); text(s, x, 525, cw, 180, p, 28, MUTE, spacing=1.35)
footer(s, 'Lloyd Lei · CMU · IROS 2026')
notes(s, 'Atlas and SONIC proved the control layer can be a network. BFM-Zero opened a new RL direction, but its representation is not shared with humans. Most commercial stacks are still per-motion tracking. The common answer when I asked why SONIC is hard to reproduce off G1: the data that works on a non-mainstream body is out of reach.')

# 3 surprise
s = slide(BG); eyebrow(s, 'What emerged'); title(s, 'One tokenizer. Any body. Zero-shot.')
video(s, f'{M}/c_8robots', 128, 288, 1110, 624)
text(s, 1300, 420, 492, 420, [[('A motion lifted from internet video, decoded on eight humanoids at once.', MUTE)], [('', MUTE)],
     [('No per-robot tuning. No robot-to-robot pairs in training.', MUTE)], [('', MUTE)], [('AgiBot X2 was never in the training set.', TXT)]], 28, spacing=1.35)
footer(s, 'Video → GenMo (SMPL) → tokenizer → Unitree G1, H1-2 · PND Adam · Booster T1 · Fourier N1, GR-3 · Kuavo S45 · AgiBot X2')
notes(s, 'In May we were training a motion representation, not a retargeter. As we scaled S, M, L, zero-shot cross-body mapping appeared, for locomotion and manipulation alike.')

# 4 how
s = slide(BG2); eyebrow(s, 'How it works'); title(s, 'The body is an input, not a hyperparameter')
steps = [('01 · Any source body', 'A named joint tree from its MJCF: axes, limits, rest frames, plus a language embedding of each joint name.', '33312C'),
         ('02 · Graph encoder', 'A transformer over joints with a learned hop-distance bias: joint-to-joint synergy along the kinematic tree.', '33312C'),
         ('03 · Discrete token', '132 slots per frame, finite scalar quantization. One token per frame, shared by every body.', ORANGE),
         ('04 · Any target body', "A decoder conditioned on the target's descriptor: joint rotations, root height, contacts.", '33312C')]
bw = (1664 - 3 * 64) / 4
for i, (h3, p, lc) in enumerate(steps):
    x = 128 + i * (bw + 64); rect(s, x, 320, bw, 400, fill='1E1D1A', line=lc, lw=1.5, shape=MSO_SHAPE.ROUNDED_RECTANGLE).adjustments[0] = 0.04
    text(s, x + 28, 352, bw - 56, 40, h3, 24, ORANGE, MONO); text(s, x + 28, 410, bw - 56, 290, p, 28, SOFT, spacing=1.35)
    if i < 3: rect(s, x + bw + 14, 505, 36, 30, fill='6E6C66', shape=MSO_SHAPE.RIGHT_ARROW)
text(s, 128, 770, 1600, 50, 'Trained on human mocap retargeted to 18 humanoids. Quadrupeds, arms and hands fit the same interface; not yet tested.', 28, '9C9A93')
footer(s, 'Lloyd Lei · CMU · IROS 2026')
notes(s, 'Joint names are language, so a new robot is described the way a person would describe it. Losses: rotation, forward kinematics, floor penetration, contact, re-encoding consistency, contrastive.')

# 5 invariance
s = slide(BG); eyebrow(s, 'Finding 1'); title(s, 'The token is embodiment-invariant, not motion-invariant', h=150)
for i, (big, small, col) in enumerate([('0.95', 'Token cosine after decoding onto another robot and re-encoding. Same as a same-body round trip.', ORANGE),
                                       ('0.2–0.5', 'Two performances of the same action from different sources. Barely above a one-second time shift.', DIM)]):
    x = 128 + i * 856; text(s, x, 380, 808, 150, big, 140, col, HEAD, bold=True, spacing=1.0); text(s, x, 555, 760, 130, small, 30, SOFT, spacing=1.35)
text(s, 128, 760, 1500, 50, 'It is a precise pose code that travels between bodies. Semantics live one layer above it.', 30, '9C9A93')
footer(s, 'Paired BONES-SEED clips, G1 ↔ H2 ↔ human, Megatron (M) tokenizer')
notes(s, 'This is why it works as retargeting infrastructure: the body changes, the code does not. It also tells us what is missing: a semantic layer for retrieval and planning.')

# 6 physics
s = slide(BG); eyebrow(s, 'Finding 2 · physics'); title(s, 'A Fourier GR3 learns to crawl from tokenizer data', h=150)
video(s, f'{M}/c_gr3_crawl', 128, 340, 1664, 376)
text(s, 128, 750, 808, 110, [[('Left: source. Middle: policy trained on tokenizer references, ', SOFT), ('4 of 4 clips pass', ORANGE), ('.', SOFT)]], 30, spacing=1.35)
text(s, 984, 750, 808, 110, 'Right: same policy on native GMR retargeting, 1 of 4 pass. Same controller, same budget.', 30, SOFT, spacing=1.35)
footer(s, 'mjlab (MuJoCo-Warp), BeyondMimic recipe, held-out AMASS hands-and-knees crawl · green ghost = reference')
notes(s, 'Ground contact is where retargeting usually breaks. Here the policy trained on tokenizer references clears every clip.')

# 7 x2
s = slide(BG); eyebrow(s, 'Finding 3 · zero-shot bodies'); title(s, 'A body it has never seen, kicking in simulation', h=150)
video(s, f'{M}/c_8robots_kick', 128, 340, 816, 459); video(s, f'{M}/c_x2_kick', 976, 340, 816, 459)
text(s, 128, 815, 816, 40, 'Kinematic reference, eight bodies', 28, MUTE); text(s, 976, 815, 816, 40, 'AgiBot X2 Ultra tracking it in mjlab', 28, MUTE)
footer(s, 'AgiBot X2 Ultra and A3 are not in the training set · green ghost = tokenizer reference')
notes(s, 'A new URDF is enough to get references a controller can learn from.')

# 8 manip
s = slide(BG); eyebrow(s, 'Finding 4 · manipulation'); title(s, 'Locomotion was the objective. Manipulation emerged.', h=150)
video(s, f'{W}/alphamotion-manipulation-eval', 128, 340, 330, 456); video(s, f'{W}/umi', 490, 340, 810, 456)
stat(s, 1332, 350, 460, '0.03 cm', 'hand SE(3) error', ORANGE, 88); stat(s, 1332, 530, 460, '0 cm', 'floor penetration, feet planted', TXT, 88)
text(s, 128, 825, 1600, 50, 'Left: manipulation evaluation. Right: UMI portable capture, hands only, completed into whole-body motion.', 28, '9C9A93')
footer(s, 'Sparse hand targets → tokenizer completes the body → causal constrained projection, 6–9 ms/frame')
notes(s, 'AlphaMotion began as a universal locomotion model; manipulation came out of the same representation. Egocentric video, UMI grippers and teleop logs only give hands. The tokenizer fills in the rest of the body, and the projection pins the hands exactly while the body moves less than 0.4 cm from the decoded pose.')

# 9 live
s = slide(BG); eyebrow(s, 'Finding 5 · any source, real time'); title(s, 'The open internet becomes robot motion')
video(s, f'{W}/zero-shot-open-world', 128, 288, 836, 470); video(s, f'{W}/hero-alphamotion-capability-grid-3x3', 988, 288, 418, 470)
text(s, 1430, 288, 362, 300, 'Internet and game video, egocentric capture, UMI and real-robot logs all enter the same token.', 28, SOFT, spacing=1.35)
for i, (big, small, col) in enumerate([('35×', 'real time, text planner', ORANGE), ('< 20 ms', 'encode + decode, 10 s clip', TXT), ('3.8 ms', 'per frame to joint angles', TXT), ('1 frame', 'streaming latency', TXT)]):
    stat(s, 128 + i * 428, 790, 392, big, small, col, 72)
footer(s, 'Source · clean · align · validate · train — one RTX PRO 6000')
notes(s, 'Every usable motion expands the corpus. With NVIDIA ARDY as a text planner, text to five robots runs at 35 times real time; projection went from 30 ms to 3.8 ms per frame with a warm-started single Levenberg–Marquardt step.')

# 10 tiangong
s = slide(BG); eyebrow(s, 'Finding 6 · zero-shot in physics'); title(s, 'Tiangong: seven actions, zero-shot motion transfer', h=150)
video(s, f'{W}/tiangong-seven-action-reel', 128, 340, 1000, 563); video(s, f'{W}/tiangong-jump', 1152, 340, 640, 270); video(s, f'{W}/tiangong-flip', 1152, 633, 640, 270)
text(s, 128, 922, 1600, 50, 'Turn and walk, dance, duck walk, squat, sit down, jump, flip. Silver = robot output, blue = reference motion.', 28, SOFT)
footer(s, 'Causal refiner on the zero-shot AgiBot A3: stance slip 391 → 26 cm/s, penetration 0, 6–9 ms/frame')
notes(s, 'The site labels this series zero-shot motion transfer: these actions were not in training. A causal refiner keeps feet planted in one frame of compute: warm start, AR(2) prior, contact-latched feet, no future frames.')

# 11 scaling (website figures and audit numbers only)
s = slide('F4F0E7'); eyebrow(s, 'Scaling', 'C25E00')
title(s, 'More data, bigger model: lower error on every body, including unseen ones', h=150, color='14130F')
image(s, f'{W}/scaling_cross.png', 128, 372, 600, 478); image(s, f'{W}/scaling_unseen.png', 752, 372, 600, 478)
text(s, 128, 868, 600, 40, [[('Cross-embodiment: ', '3D3B35'), ('13.5° → 4.8°', '14130F')]], 26)
text(s, 752, 868, 600, 40, [[('Unseen embodiment: ', '3D3B35'), ('15.8° → 6.4°', '14130F')]], 26)
for j, (big, small, col) in enumerate([('39 mm', 'median hand error, from 90 mm', 'C25E00'), ('4.5°', 'whole-body rotation error', '14130F'), ('0.93', 'contact F1, learned support', '14130F')]):
    text(s, 1392, 400 + j * 150, 400, 88, big, 80, col, HEAD, bold=True, spacing=1.0); text(s, 1392, 400 + j * 150 + 92, 400, 40, small, 26, '3D3B35')
footer(s, '600 held-out manipulation sequences · data 0.50M → 5.04M rows · model 3.7M → 57.7M parameters', '6D6A61')
notes(s, 'Error falls smoothly as both data and model grow, and the unseen-embodiment surface has the same shape as the trained one: generalization scales too. On the 600-sequence audit the latest large model cuts hand error from 90 to 39 millimetres and whole-body rotation error to 4.5 degrees, and it learns which limbs are supporting the body.')

# 12 benchmark
s = slide(BG); eyebrow(s, 'Paired benchmark'); title(s, '18 motions, one frozen controller')
video(s, f'{M}/c_g1_jump', 128, 300, 1000, 375)
stat(s, 1176, 330, 616, '≥ 96.9%', 'best success on all 18', ORANGE, 80); stat(s, 1176, 520, 616, '2.4×', 'smoother references (jerk)', TXT, 80)
text(s, 128, 720, 1600, 90, 'Jumps, hip-hop, crouch walks. Trainable under one frozen controller, with no per-robot tuning, statistically on par with hand-tuned retargeting.', 28, '9C9A93', spacing=1.35)
footer(s, 'BONES-SEED, BeyondMimic recipe, 4,096 environments, seed 20260919')
notes(s, 'A learned tokenizer with no per-robot tuning trains as well as hand-tuned retargeting, statistically, and its references are 2.4 times smoother.')

# 13 open data
s = slide(ORANGE); eyebrow(s, 'Open data', '14130F')
text(s, 128, 220, 1100, 210, '500,000 h', 200, '14130F', HEAD, bold=True, spacing=0.95)
text(s, 128, 450, 920, 110, 'of cleaned cross-embodiment motion, planned for open release.', 40, '14130F', spacing=1.25)
text(s, 128, 590, 920, 130, 'Free for academic and non-commercial use. Mapped to about 20 embodiments, including Unitree G1 and H2, AgiBot X2 and A3, Fourier GR3.', 30, '14130F', spacing=1.35)
text(s, 128, 800, 920, 50, 'Available today: 26,111 SOMA → Unitree H2 clips, 50.7 hours.', 30, '14130F')
image(s, f'{M}/qr_hf.png', 1292, 250, 440, 440)
text(s, 1252, 710, 520, 70, 'huggingface.co/datasets/sajio/alphamotion-soma-h2', 24, '14130F', MONO, align=PP_ALIGN.CENTER)
footer(s, 'Scan to download the first release', '3D2A10')
notes(s, 'This is the ask: use the data. If you bring a URDF, we can map the corpus onto it. The first release is the H2 set on Hugging Face.')

# 14 thanks
s = slide(BG); eyebrow(s, "Open problems · let's work on them together")
text(s, 128, 200, 1150, 280, ['•  A semantic layer above the pose token', '•  Ground and contact-rich motions: getting up, sitting', '•  The zero-shot skeleton tax on unseen human rigs',
                              '•  Quadrupeds, arms and dexterous hands', '•  A generalist controller that closes the loop'], 32, SOFT, spacing=1.45)
text(s, 128, 560, 1100, 90, "Let's turn the wheels together.", 72, TXT, HEAD, bold=True)
text(s, 128, 680, 900, 50, 'lloydl@andrew.cmu.edu', 32, '9C9A93')
image(s, f'{M}/qr_hf.png', 1432, 200, 360, 360)
text(s, 1432, 580, 360, 40, 'Open dataset', 24, '9C9A93', align=PP_ALIGN.CENTER)
footer(s, 'Lloyd Lei · CMU · IROS 2026')
notes(s, 'Close: the wheels to spin up the new era are in your hands.')

from lxml import etree
NS = 'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"'
def autoplay(sl):
    ids = [sh.shape_id for sh in sl.shapes if sh.shape_type is not None and sh.shape_type == 16]
    if not ids: return 0
    old = sl.element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}timing')
    if old is not None: sl.element.remove(old)
    n = [5]
    def nid():
        n[0] += 1; return n[0]
    calls = ''.join(f'<p:par><p:cTn id="{nid()}" presetID="1" presetClass="mediacall" presetSubtype="0" fill="hold" nodeType="withEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst><p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr><p:cTn id="{nid()}" dur="1" fill="hold"/><p:tgtEl><p:spTgt spid="{i}"/></p:tgtEl></p:cBhvr></p:cmd></p:childTnLst></p:cTn></p:par>' for i in ids)
    media = ''.join(f'<p:video><p:cMediaNode vol="0" mute="1"><p:cTn id="{nid()}" repeatCount="indefinite" fill="hold" display="0"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{i}"/></p:tgtEl></p:cMediaNode></p:video>' for i in ids)
    xml = (f'<p:timing {NS}><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>'
           f'<p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
           f'<p:par><p:cTn id="3" fill="hold"><p:stCondLst><p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond></p:stCondLst><p:childTnLst>'
           f'<p:par><p:cTn id="4" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>{calls}</p:childTnLst></p:cTn></p:par>'
           f'</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn>'
           f'<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq>'
           f'{media}</p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>')
    el = etree.fromstring(xml)
    ext = sl.element.find('{http://schemas.openxmlformats.org/presentationml/2006/main}extLst')
    if ext is not None: ext.addprevious(el)
    else: sl.element.append(el)
    return len(ids)
print('autoplay videos per slide:', [autoplay(sl) for sl in prs.slides])
prs.save(OUT); print('saved', OUT, os.path.getsize(OUT) // 1024, 'KB', len(prs.slides), 'slides')
