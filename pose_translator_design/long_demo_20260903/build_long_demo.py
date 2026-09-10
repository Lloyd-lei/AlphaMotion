#!/usr/bin/env python3
"""Build the evidence-first AlphaMotion capability reel.

The spatial scenes use FZ5C. Temporal scenes use the released Greenwich codec
with Equator because their discrete code contracts are intentionally separate.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WORKSPACE = Path('/home/arenalabs/Desktop/"be water, robot"')
MEDIA = Path('/home/arenalabs/Desktop/alphamotion video')
R4 = WORKSPACE / 'pose_translator_design' / 'R4_VIZ' / 'mp4'
OUT = MEDIA / '08_long_demo'
FONT_DIR = WORKSPACE / 'alpha_motion2' / 'runs' / 'synergy' / 'mechanistic_paper_20260721' / 'latex_assets'

W, H, FPS = 1920, 1080, 30
PAPAYA = (255, 128, 0)
CREAM = (247, 242, 232)
INK = (24, 22, 19)
WHITE = CREAM
MUTED = (103, 96, 85)
DIM = (143, 133, 118)
GRAPHITE = CREAM
PANEL = (234, 226, 211)
GREEN = (65, 214, 142)
BLUE = (91, 167, 255)


def font(size: int, medium: bool = False) -> ImageFont.FreeTypeFont:
    name = 'InterDisplay-Medium.ttf' if medium else 'InterDisplay-Regular.ttf'
    return ImageFont.truetype(str(FONT_DIR / name), size=size)


def fit_rgb(frame: np.ndarray, rect: tuple[int, int, int, int], cover: bool = False) -> np.ndarray:
    x, y, rw, rh = rect
    h, w = frame.shape[:2]
    scale = max(rw / w, rh / h) if cover else min(rw / w, rh / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((H, W, 3), GRAPHITE, dtype=np.uint8)
    if cover:
        sx = max(0, (nw - rw) // 2)
        sy = max(0, (nh - rh) // 2)
        canvas[y:y + rh, x:x + rw] = resized[sy:sy + rh, sx:sx + rw]
    else:
        dx = x + (rw - nw) // 2
        dy = y + (rh - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas


def paste_rgb(canvas: np.ndarray, frame: np.ndarray, rect: tuple[int, int, int, int], cover: bool = False) -> None:
    x, y, rw, rh = rect
    h, w = frame.shape[:2]
    scale = max(rw / w, rh / h) if cover else min(rw / w, rh / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    if cover:
        sx = max(0, (nw - rw) // 2)
        sy = max(0, (nh - rh) // 2)
        canvas[y:y + rh, x:x + rw] = resized[sy:sy + rh, sx:sx + rw]
    else:
        dx = x + (rw - nw) // 2
        dy = y + (rh - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = resized


def darken(frame: np.ndarray, amount: float) -> np.ndarray:
    return np.clip(frame.astype(np.float32) * amount, 0, 255).astype(np.uint8)


def fade(frame: np.ndarray, i: int, total: int, seconds: float = 0.24) -> np.ndarray:
    n = max(1, round(seconds * FPS))
    a = min(1.0, (i + 1) / n, (total - i) / n)
    if a >= 0.999:
        return frame
    bg = np.empty_like(frame)
    bg[:] = CREAM
    return np.clip(frame.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)


class ClipReader:
    def __init__(self, path: Path, start_s: float = 0.0, loop: bool = False):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f'cannot open {path}')
        self.src_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or FPS)
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.start = max(0, round(start_s * self.src_fps))
        self.loop = loop
        self.current = self.start - 1
        self.last: np.ndarray | None = None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)

    def at(self, seconds: float) -> np.ndarray:
        usable = max(1, self.count - self.start)
        offset = round(seconds * self.src_fps)
        target = self.start + (offset % usable if self.loop else min(offset, usable - 1))
        if target < self.current:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
            self.current = self.start - 1
            self.last = None
        while self.current < target:
            ok, bgr = self.cap.read()
            if not ok:
                break
            self.last = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.current += 1
        if self.last is None:
            raise RuntimeError(f'cannot decode {self.path}')
        return self.last

    def close(self) -> None:
        self.cap.release()


@dataclass(frozen=True)
class Evidence:
    capability: str
    status: str
    model: str
    source: str
    claim: str
    boundary: str


@dataclass(frozen=True)
class TimelineEntry:
    kind: str
    label: str
    start_s: float
    end_s: float
    sources: list[str]


def title_frames(path: Path, seconds: float) -> Iterator[np.ndarray]:
    total = round(seconds * FPS)
    for i in range(total):
        p = i / max(1, total - 1)
        canvas = Image.new('RGBA', (W, H), CREAM + (255,))
        draw = ImageDraw.Draw(canvas, 'RGBA')
        draw.rectangle((0, 0, 34, H), fill=PAPAYA + (255,))
        draw.rectangle((78, 152, 260, 164), fill=PAPAYA + (255,))
        draw.text((78, 210), 'ALPHAMOTION', font=font(26, True), fill=PAPAYA)
        draw.text((74, 380), 'ONE MOTION STATE.', font=font(82, True), fill=INK)
        draw.text((74, 484), 'ANY BODY.', font=font(104, True), fill=PAPAYA)
        draw.text((80, 644), 'SPATIAL COMPLETION  /  TEMPORAL BRIDGE  /  REAL-TIME CONTROL',
                  font=font(22, True), fill=MUTED)
        draw.rectangle((80, 1028, 80 + round(1760 * p), 1034), fill=PAPAYA + (255,))
        yield fade(np.asarray(canvas.convert('RGB')), i, total, 0.35)


def chapter_frames(number: str, title: str, subtitle: str, facts: list[str], seconds: float) -> Iterator[np.ndarray]:
    total = round(seconds * FPS)
    for i in range(total):
        p = i / max(1, total - 1)
        canvas = Image.new('RGBA', (W, H), CREAM + (255,))
        draw = ImageDraw.Draw(canvas, 'RGBA')
        draw.rectangle((0, 0, 34, H), fill=PAPAYA + (255,))
        draw.text((80, 96), f'CAPABILITY {number} / 08', font=font(18, True), fill=PAPAYA)
        draw.text((76, 250), title, font=font(76, True), fill=INK)
        draw.text((80, 370), subtitle, font=font(28), fill=MUTED)
        draw.line((80, 482, 80 + round(1520 * p), 482), fill=PAPAYA, width=5)
        for j, fact in enumerate(facts):
            y = 610 + j * 90
            draw.text((80, y), f'0{j + 1}', font=font(17, True), fill=PAPAYA)
            draw.text((140, y - 5), fact, font=font(28), fill=INK)
        yield fade(np.asarray(canvas.convert('RGB')), i, total, 0.28)


def clip_card_frames(title: str, subtitle: str, seconds: float = 1.25) -> Iterator[np.ndarray]:
    """Show clip context as a transition, never as an overlay on evidence footage."""
    total = round(seconds * FPS)
    for i in range(total):
        p = i / max(1, total - 1)
        canvas = Image.new('RGBA', (W, H), PAPAYA + (255,))
        draw = ImageDraw.Draw(canvas, 'RGBA')
        draw.rectangle((34, 34, W - 34, H - 34), fill=CREAM + (255,))
        draw.text((92, 368), title, font=font(58, True), fill=INK)
        draw.text((96, 475), subtitle, font=font(24), fill=MUTED)
        draw.rectangle((94, 588, 94 + round(820 * p), 594), fill=PAPAYA + (255,))
        yield fade(np.asarray(canvas.convert('RGB')), i, total, 0.16)


def video_frames(
    path: Path,
    seconds: float,
    section: str,
    count: str,
    title: str,
    subtitle: str,
    pipeline: str,
    status: str = 'VERIFIED',
    layout: str = 'standard',
    start_s: float = 0.0,
    loop: bool = False,
) -> Iterator[np.ndarray]:
    reader = ClipReader(path, start_s=start_s, loop=loop)
    total = round(seconds * FPS)
    try:
        yield from clip_card_frames(title, subtitle)
        for i in range(total):
            src = reader.at(i / FPS)
            frame = fit_rgb(src, (0, 0, W, H), cover=False)
            yield fade(frame, i, total)
    finally:
        reader.close()


def triptych_frames(
    paths: list[Path],
    labels: list[str],
    seconds: float,
    section: str,
    count: str,
    title: str,
    subtitle: str,
    pipeline: str,
    loop: bool = False,
) -> Iterator[np.ndarray]:
    readers = [ClipReader(p, loop=loop) for p in paths]
    total = round(seconds * FPS)
    rects = [(30, 80, 600, 920), (660, 80, 600, 920), (1290, 80, 600, 920)]
    try:
        yield from clip_card_frames(title, subtitle)
        for i in range(total):
            frame = np.full((H, W, 3), CREAM, dtype=np.uint8)
            for r, rect in zip(readers, rects):
                paste_rgb(frame, r.at(i / FPS), rect, cover=False)
            yield fade(frame, i, total)
    finally:
        for r in readers:
            r.close()


def token_zoom_frames(path: Path, seconds: float) -> Iterator[np.ndarray]:
    reader = ClipReader(path, loop=True)
    total = round(seconds * FPS)
    try:
        yield from clip_card_frames(
            'Discrete motion, decoded continuously.',
            'Finite scalar quantization exposes a compact per-frame motion state.',
        )
        for i in range(total):
            src = reader.at(i / FPS)
            frame = fit_rgb(src, (0, 0, W, H), cover=False)
            yield fade(frame, i, total)
    finally:
        reader.close()


def runtime_frames(seconds: float) -> Iterator[np.ndarray]:
    total = round(seconds * FPS)
    for i in range(total):
        p = i / max(1, total - 1)
        canvas = Image.new('RGBA', (W, H), CREAM + (255,))
        draw = ImageDraw.Draw(canvas, 'RGBA')
        draw.rectangle((0, 0, 34, H), fill=PAPAYA + (255,))
        draw.text((76, 112), 'CAPABILITY 08 / 08', font=font(18, True), fill=PAPAYA)
        draw.text((76, 190), 'REAL-TIME IS THE DEFAULT.', font=font(55, True), fill=INK)
        draw.text((76, 292), '2.0', font=font(178, True), fill=PAPAYA)
        draw.text((420, 405), 'ms', font=font(58, True), fill=INK)
        draw.text((82, 492), 'single-frame encode + decode', font=font(27), fill=MUTED)
        draw.line((80, 580, 1840, 580), fill=DIM, width=2)
        stats = [('~500', 'MODEL FPS'), ('1', 'FRAME CONTEXT'), ('NONE', 'SLIDING WINDOWS')]
        for j, (v, label) in enumerate(stats):
            x = 80 + j * 570
            draw.text((x, 658), v, font=font(74, True), fill=INK if j else PAPAYA)
            draw.text((x, 753), label, font=font(17, True), fill=MUTED)
        draw.text((80, 910), 'Measured: batch 1 / RTX PRO 6000 / model only', font=font(18, True), fill=INK)
        draw.text((80, 946), 'Excludes rendering, I/O and downstream control.', font=font(17), fill=MUTED)
        draw.rectangle((80, 1032, 80 + round(1760 * p), 1037), fill=PAPAYA)
        yield fade(np.asarray(canvas.convert('RGB')), i, total, 0.3)


def closing_frames(path: Path, seconds: float) -> Iterator[np.ndarray]:
    total = round(seconds * FPS)
    for i in range(total):
        p = i / max(1, total - 1)
        canvas = Image.new('RGBA', (W, H), CREAM + (255,))
        draw = ImageDraw.Draw(canvas, 'RGBA')
        draw.rectangle((0, 0, 34, H), fill=PAPAYA + (255,))
        draw.text((78, 340), 'A SHARED MOTION STATE', font=font(82, True), fill=INK)
        draw.text((78, 452), 'FROM DATA TO ANY BODY.', font=font(88, True), fill=PAPAYA)
        draw.text((82, 622), 'COMPLETE  /  BRIDGE  /  CONTROL', font=font(24, True), fill=MUTED)
        draw.rectangle((80, 1028, 80 + round(1760 * p), 1034), fill=PAPAYA + (255,))
        yield fade(np.asarray(canvas.convert('RGB')), i, total, 0.4)


def scene_plan() -> list[tuple[str, str, float, list[str], Iterable[np.ndarray]]]:
    sparse_hip = MEDIA / '01_neural_completion' / '6joints_hiphop_BONES.mp4'
    sparse_tennis = MEDIA / '01_neural_completion' / 'hands_masked_tennis_smash_yamaoka.mp4'
    sparse_walk = MEDIA / '01_neural_completion' / '6joints_walk_CMU77.mp4'
    kick = MEDIA / '02_amass_heldout' / 'SOMA_soma_subject2_kick_001_stageii_at34.mp4'
    jump = MEDIA / '02_amass_heldout' / 'SFU_0015_0015_JumpOverObstacle001_stagei_at77.mp4'
    pickup = MEDIA / '03_bones_seed' / 'item_pick_up_standing_R_001_A410_at23.mp4'
    fall = MEDIA / '04_ground_amass_heldout' / 'fall_and_getup_hamada.mp4'
    crouch = MEDIA / '02_amass_heldout' / 'ACCAD_Male2Walking_c3d_B24_walk_to_crouc_at16.mp4'
    se3 = R4 / 'se3_g1_gt' / 'bones_dancing_routine_1_001__A021_5cmd.mp4'
    se3_target = R4 / 'se3_x2' / 'synthetic_target_x2_3panel.mp4'
    retime = R4 / 'bridge_demo' / 'equator_retime_amass_300f_x2.mp4'
    composition = R4 / 'bridge_demo' / 'composition_walkwave_b5_cfg8_x2.mp4'
    speed = R4 / 'bridge_demo' / 'equator_speed_n30_60_120_x2.mp4'
    phone = [
        MEDIA / '07_video_to_robot_genmo' / 'walk_phone_0_source.mp4',
        MEDIA / '07_video_to_robot_genmo' / 'walk_phone_1_genmo_extract.mp4',
        MEDIA / '07_video_to_robot_genmo' / 'walk_phone_2_alphamotion_8robots.mp4',
    ]
    gta = [
        MEDIA / '07_video_to_robot_genmo' / 'gta_kick_0_source.mp4',
        MEDIA / '07_video_to_robot_genmo' / 'gta_kick_1_genmo_extract.mp4',
        MEDIA / '07_video_to_robot_genmo' / 'gta_kick_2_alphamotion_8robots.mp4',
    ]
    return [
        ('title', 'opening', 5.0, [str(kick)], title_frames(kick, 5.0)),
        ('chapter', 'native cross-embodiment', 2.6, [], chapter_frames(
            '01', 'ONE STATE. NINE BODIES.',
            'Native cross-embodiment decoding without pairwise retarget maps.',
            ['universal spatial motion codec', 'SMPL plus eight robot topologies', 'AgiBot X2 held out from training'], 2.6)),
        ('video', 'held-out kick', 7.0, [str(kick)], video_frames(
            kick, 7.0, 'cross-embodiment', '01 / 08', 'One motion state. Nine native decoders.',
            'Held-out AMASS kick transferred across radically different kinematic trees.',
            'FZ5C / SPATIAL', 'VERIFIED')),
        ('video', 'held-out jump', 7.0, [str(jump)], video_frames(
            jump, 7.0, 'cross-embodiment', '01 / 08', 'Topology changes. Motion identity remains.',
            'The same discrete state drives humanoids with different joint counts and proportions.',
            'FZ5C / SPATIAL', 'VERIFIED')),
        ('video', 'BONES pickup', 5.0, [str(pickup)], video_frames(
            pickup, 5.0, 'cross-embodiment', '01 / 08', 'A second motion source, the same latent contract.',
            'BONES-SEED BVH is mapped once to SMPL, then decoded natively to eight robots.',
            'FZ5C / SPATIAL', 'VERIFIED')),

        ('chapter', 'sparse-joint completion', 2.6, [], chapter_frames(
            '02', 'SIX JOINTS ARE ENOUGH.',
            'Observe only the body anchors. Infer the rest in one frame.',
            ['pelvis + head + hands + feet', '128 x 13 FSQ state', 'fisheye AR-glasses interface target'], 2.6)),
        ('video', 'six-joint hiphop', 5.0, [str(sparse_hip)], video_frames(
            sparse_hip, 5.0, 'sparse completion', '02 / 08', '6 observed joints -> a complete articulated body.',
            'The ghost shows the sparse input; every unobserved joint is inferred by the network.',
            'FZ5C / SPATIAL', 'VERIFIED')),
        ('video', 'masked-hand tennis', 7.0, [str(sparse_tennis)], video_frames(
            sparse_tennis, 7.0, 'sparse completion', '02 / 08', 'Hide the hands. Recover the whole-body action.',
            'Ablation isolates neural completion from direct joint copying.',
            'FZ5C / SPATIAL', 'VERIFIED')),

        ('chapter', 'contact state', 2.6, [], chapter_frames(
            '03', 'CONTACT TRAVELS WITH MOTION.',
            'Pose and support state share one spatial representation.',
            ['ground contact evaluated now', 'decoded contact checked against geometry', 'object + tactile contact is the next channel'], 2.6)),
        ('video', 'fall and get up', 5.0, [str(fall)], video_frames(
            fall, 5.0, 'contact', '03 / 08', 'From standing to ground contact and back.',
            'Orange: encoder contact input. Green: contact recovered from decoded geometry.',
            'FZ5C / SPATIAL', 'VERIFIED')),
        ('video', 'walk to crouch', 5.0, [str(crouch)], video_frames(
            crouch, 5.0, 'contact', '03 / 08', 'Support transitions remain visible across bodies.',
            'Current evidence covers ground contact; object and tactile contact remain an extension.',
            'FZ5C / SPATIAL', 'VERIFIED')),

        ('chapter', 'SE3 constraints', 2.6, [], chapter_frames(
            '04', 'COMMAND THE END EFFECTORS.',
            'SE(3) targets become a universal interface for UMI, mocap and teleoperation.',
            ['five commanded body parts', 'remaining body completed', 'joint limits enforced by feasibility projection'], 2.6)),
        ('video', 'five SE3 commands', 10.0, [str(se3)], video_frames(
            se3, 10.0, 'SE(3) control', '04 / 08', 'Five constraints. One coherent full-body pose.',
            'Gray is the external target; orange is model completion on held-out Unitree G1.',
            'GREENWICH / SPATIAL CONTROL', 'VERIFIED', layout='wide')),
        ('video', 'reachable and infeasible SE3', 6.0, [str(se3_target)], video_frames(
            se3_target, 6.0, 'SE(3) control', '04 / 08', 'Reachable targets land; impossible targets stay bounded.',
            'The hard-pin solver exposes feasibility instead of hiding unreachable commands.',
            'GREENWICH + FEASIBILITY', 'INTERFACE', layout='wide')),

        ('chapter', 'temporal bridge', 2.6, [], chapter_frames(
            '05', 'BRIDGE THE MISSING MOTION.',
            'Current state + target state + frame budget -> interior motion states.',
            ['discrete motion endpoints', 'bidirectional temporal bridge', '300-frame long-horizon completion'], 2.6)),
        ('video', '300-frame temporal bridge', 20.0, [str(retime)], video_frames(
            retime, 20.0, 'temporal bridge', '05 / 08', 'Sparse endpoints become a 300-frame trajectory.',
            'The bridge sees both boundaries and fills the interior; it is not causal forecasting.',
            'GREENWICH + EQUATOR', 'VERIFIED', layout='wide')),
        ('video', 'walk and wave composition', 4.5, [str(composition)], video_frames(
            composition, 4.5, 'temporal bridge', '05 / 08', 'Action components can be composed inside the bridge.',
            'Walk, wave and walk-plus-wave are decoded under matched endpoints.',
            'GREENWICH + EQUATOR', 'VERIFIED', layout='wide')),

        ('chapter', 'frame budget', 2.6, [], chapter_frames(
            '06', 'SAME PATH. DIFFERENT CLOCK.',
            'The requested frame count is part of the generation contract.',
            ['N = 30', 'N = 60', 'N = 120'], 2.6)),
        ('video', 'native speed control', 6.3, [str(speed)], video_frames(
            speed, 6.3, 'frame budget', '06 / 08', 'One motion code, three time budgets.',
            'Clock-conditioned decoding changes duration without re-authoring the trajectory.',
            'GREENWICH + EQUATOR', 'VERIFIED', layout='wide')),

        ('chapter', 'video ingestion', 2.6, [], chapter_frames(
            '07', 'THE INTERNET BECOMES MOTION DATA.',
            'Video is recovered as SMPL, quantized, then decoded to any registered body.',
            ['source video', '3D motion recovery', 'AlphaMotion tokens -> eight robots'], 2.6)),
        ('triptych', 'phone video pipeline', 12.5, [str(p) for p in phone], triptych_frames(
            phone, ['01 / SOURCE VIDEO', '02 / 3D MOTION', '03 / NATIVE BODIES'], 12.5,
            'video ingestion', '07 / 08', 'Pixels become a reusable motion state.',
            'A real monocular clip is recovered to SMPL and decoded across eight embodiments.',
            'GENMO -> FZ5C', False)),
        ('triptych', 'game video pipeline', 3.2, [str(p) for p in gta], triptych_frames(
            gta, ['01 / GAME VIDEO', '02 / 3D MOTION', '03 / NATIVE BODIES'], 3.2,
            'video ingestion', '07 / 08', 'Synthetic worlds feed the same pipeline.',
            'The extraction and AlphaMotion decode stages remain inspectable end to end.',
            'GENMO -> FZ5C', True)),

        ('chapter', 'motion tokens and runtime', 2.6, [], chapter_frames(
            '08', 'MOTION, TOKENIZED AT RUNTIME.',
            'A discrete spatial state for generation, search and control.',
            ['finite scalar quantization', 'single-frame context', 'warm resident models'], 2.6)),
        ('custom', 'token lattice zoom', 8.0, [str(sparse_walk)], token_zoom_frames(sparse_walk, 8.0)),
        ('custom', 'runtime benchmark', 6.0, [str(WORKSPACE / 'pose_translator_design' / 'exp_topo' / 'RESULTS.md')], runtime_frames(6.0)),
        ('closing', 'closing', 6.0, [str(kick)], closing_frames(kick, 6.0)),
    ]


def evidence_manifest() -> list[Evidence]:
    return [
        Evidence('native_cross_embodiment', 'verified', 'FZ5C spatial',
                 'held-out AMASS and BONES-SEED renders',
                 'One code is decoded by native body decoders; AgiBot X2 is held out.',
                 'Kinematic evidence, not physics-on execution.'),
        Evidence('se3_constraints', 'verified_plus_solver', 'Greenwich spatial',
                 'R4_VIZ SE3 G1 and X2 runs plus JSON metrics',
                 'Sparse commanded end effectors are completed into full-body states.',
                 'Sub-millimeter landing is from the feasibility projection, not raw soft output.'),
        Evidence('contact_state', 'verified_ground_only', 'FZ5C spatial',
                 'held-out fall/get-up and crouch renders',
                 'Ground contact enters the representation and decoded geometry is checked.',
                 'Object contact and tactile adaptation are roadmap extensions.'),
        Evidence('sparse_joint_completion', 'verified', 'FZ5C spatial',
                 '6-joint and masked-hand ablations',
                 'Hidden joints are inferred from sparse single-frame observations.',
                 'AR-glasses hardware itself was not evaluated in these clips.'),
        Evidence('sparse_keyframe_completion', 'verified_contract', 'Greenwich + Equator',
                 '300-frame bridge and composition renders',
                 'Interior states are generated from both endpoints and a frame budget.',
                 'This is bidirectional bridge completion, not future-only prediction.'),
        Evidence('frame_budget', 'verified_interface', 'Greenwich + Equator',
                 'N=30/60/120 matched-code render',
                 'Frame count controls decoded duration natively.',
                 'Historical 2x speed control is approximate, not perfectly invariant.'),
        Evidence('video_to_motion', 'verified_pipeline', 'GenMo/GEM-SMPL + FZ5C',
                 'phone and GTA three-stage renders',
                 'Monocular pixels become SMPL motion and then cross-body states.',
                 'Video recovery foot contact is imperfect on the current extractor.'),
        Evidence('runtime', 'measured_prior_spatial_codec', 'FSQ spatial codec',
                 'exp_topo/RESULTS.md',
                 '2.0 ms batch-1 model-only encode plus decode on RTX PRO 6000.',
                 'Excludes rendering, I/O, temporal generation and WBC.'),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / 'AlphaMotion_Stealth_Capability_Reel_20260903.mp4'
    mobile = OUT / 'AlphaMotion_Stealth_Capability_Reel_20260903_mobile.mp4'
    manifest_path = OUT / 'AlphaMotion_Stealth_Capability_Reel_20260903_sources.json'
    claim_path = OUT / 'AlphaMotion_Stealth_Capability_Reel_20260903_claims.json'
    scenes = scene_plan()
    command = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
        '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(output),
    ]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert proc.stdin is not None
    frame_count = 0
    timeline: list[TimelineEntry] = []
    try:
        for index, (kind, label, nominal_s, sources, frames) in enumerate(scenes, 1):
            start = frame_count / FPS
            local = 0
            for frame in frames:
                if frame.shape != (H, W, 3):
                    raise RuntimeError(f'{label}: unexpected frame shape {frame.shape}')
                proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                frame_count += 1
                local += 1
            end = frame_count / FPS
            timeline.append(TimelineEntry(kind, label, start, end, sources))
            print(f'[{index:02d}/{len(scenes):02d}] {label}: {local} frames ({end - start:.2f}s)', flush=True)
    finally:
        proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('ffmpeg encoder failed')

    subprocess.run([
        'ffmpeg', '-y', '-loglevel', 'error', '-i', str(output),
        '-vf', 'scale=1280:720:flags=lanczos', '-an', '-c:v', 'libx264', '-preset', 'medium',
        '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(mobile),
    ], check=True)

    manifest = {
        'protocol': 'alphamotion-stealth-capability-reel-v2',
        'created': '2026-09-03',
        'output': str(output),
        'mobile_output': str(mobile),
        'resolution': [W, H],
        'fps': FPS,
        'duration_s': frame_count / FPS,
        'frames': frame_count,
        'spatial_checkpoint': '/var/tmp/alphamotion/topo_exp/runs/FZ5c_f32_slipw2_hpair01_rp3_0902',
        'temporal_contract': 'Greenwich staged spatial codec + Equator A/B; never mixed with FZ5C FSQ codes',
        'timeline': [asdict(x) for x in timeline],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    claim_path.write_text(json.dumps([asdict(x) for x in evidence_manifest()], indent=2) + '\n')
    print(json.dumps({'output': str(output), 'mobile': str(mobile), 'duration_s': frame_count / FPS}, indent=2))


if __name__ == '__main__':
    main()
