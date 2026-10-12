"""PIL compositing helpers: fonts, text, min-height traces, 1920x1080 page."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ._env import FONT, FONT_BOLD

W, H = 1920, 1080
BG = (241, 244, 246)
INK = (21, 43, 56)
SUB = (55, 77, 94)
MUTED = (96, 112, 124)
GREEN = (38, 166, 76)
RED = (214, 48, 49)
GREY = (150, 156, 162)
BLUE = (40, 96, 170)

BODY_LABEL = {
    'human_smpl': '人体', 'unitree_g1_29dof': 'Unitree G1', 'unitree_g1_23dof': 'Unitree G1 (23 关节)',
    'unitree_h1': 'Unitree H1', 'unitree_h1_2': 'Unitree H1-2', 'unitree_h2': 'Unitree H2', 'unitree_r1': 'Unitree R1',
    'booster_k1': 'Booster K1', 'booster_t1': 'Booster T1', 'booster_t1_29dof': 'Booster T1 (29 关节)',
    'fourier_n1': 'Fourier N1', 'fourier_gr3': 'Fourier GR3', 'hightorque_hi': 'HighTorque Hi',
    'kuavo_s45': 'Kuavo S45', 'pnd_adam_lite': 'PNDbotics Adam Lite', 'tienkung': '天工 Tiangong',
    'adroit_pen': 'Adroit 灵巧手', 'relay_franka_full_v2': 'Franka 夹爪'}
HELDOUT = {'fourier_gr3', 'kuavo_s45', 'unitree_h2', 'unitree_r1'}


def body_label(body: str) -> str:
    s = BODY_LABEL.get(body, body)
    return s + '（训练没见过的机器人）' if body in HELDOUT else s


@lru_cache(maxsize=64)
def font(size: int, bold: bool = False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size, index=2)  # index 2 = Noto Sans CJK SC


def text(draw, xy, s, size=22, fill=INK, bold=False, anchor=None):
    draw.text(xy, s, font=font(size, bold), fill=fill, anchor=anchor)


def fit_text(draw, s, size, max_w, bold=False):
    """Largest font size <= size that fits max_w pixels."""
    while size > 10 and draw.textlength(s, font=font(size, bold)) > max_w:
        size -= 1
    return size


class Trace:
    """Static min-height trace (cm) with a moving cursor."""

    def __init__(self, w, h, series, colors, times, lo=None, hi=None, title='最低点离地高度（cm）', labels=None,
                 ground=True, bands=None, hlines=None, unit='cm', step=None):
        """ground=True: green touching band [-1, 2.5] and red penetration band below -1 (E1).
        bands: [(lo, hi, rgb)] shaded y ranges; hlines: [(value, rgb)] dashed reference lines (E2)."""
        self.unit = unit
        allv = np.concatenate([np.asarray(s, np.float64).ravel() for s in series]) if series else np.zeros(1)
        allv = allv[np.isfinite(allv)] if np.isfinite(allv).any() else np.zeros(1)
        if lo is None:
            lo = float(np.clip(np.floor(allv.min() / 5.0) * 5.0 - 5.0, -40.0, -10.0))
        if hi is None:
            hi = float(np.clip(np.ceil(allv.max() / 10.0) * 10.0 + 5.0, 30.0, 120.0))
        self.title = title
        self.w, self.h, self.lo, self.hi = w, h, lo, hi
        self.times = np.asarray(times, np.float64)
        self.series = [np.asarray(s, np.float64) for s in series]
        self.pad_l, self.pad_r, self.pad_t, self.pad_b = 46, 12, 34, 22
        img = Image.new('RGB', (w, h), (250, 251, 252))
        d = ImageDraw.Draw(img)
        d.rectangle((0, 0, w - 1, h - 1), outline=(215, 221, 226))
        text(d, (8, 2), title, 16, SUB)
        if labels:
            x = w - self.pad_r
            for lab, col in reversed(list(zip(labels, colors))):
                tw = d.textlength(lab, font=font(14))
                x -= tw
                text(d, (x, 4), lab, 14, col)
                x -= 18
                d.line((x + 2, 13, x + 14, 13), fill=col, width=3)
                x -= 8
        if ground:  # bands: touching (green) and penetrating (red)
            y_t, y_p = self.y(2.5), self.y(-1.0)
            d.rectangle((self.pad_l, y_t, w - self.pad_r, y_p), fill=(226, 244, 230))
            d.rectangle((self.pad_l, y_p, w - self.pad_r, self.y(lo)), fill=(250, 226, 226))
        for b0, b1, col in (bands or []):
            ya, yb = self.y(min(max(b1, lo), hi)), self.y(min(max(b0, lo), hi))
            d.rectangle((self.pad_l, ya, w - self.pad_r, yb), fill=col)
        step = step or (10 if hi - lo <= 60 else 20)
        for v in range(int(np.ceil(lo / step) * step), int(hi) + 1, step):
            if lo <= v <= hi:
                yy = self.y(v)
                d.line((self.pad_l, yy, w - self.pad_r, yy), fill=(205, 211, 216) if (v or not ground) else (120, 128, 136), width=1)
                if yy > self.pad_t - 2:
                    text(d, (self.pad_l - 6, yy), f'{v:d}', 13, MUTED, anchor='rm')
        for v, col in (hlines or []):
            if lo <= v <= hi:
                yy = self.y(v)
                xs = np.arange(self.pad_l, w - self.pad_r, 10)
                for x0 in xs:
                    d.line((x0, yy, min(x0 + 6, w - self.pad_r), yy), fill=col, width=2)
        for s, col in zip(self.series, colors):
            pts = [(self.x(i), self.y(float(np.clip(v, lo, hi)))) for i, v in enumerate(s) if np.isfinite(v)]
            if len(pts) > 1:
                d.line(pts, fill=col, width=2)
        self.base = img

    def x(self, i):
        n = max(len(self.times) - 1, 1)
        return self.pad_l + (self.w - self.pad_l - self.pad_r) * i / n

    def y(self, v):
        return self.pad_t + (self.h - self.pad_t - self.pad_b) * (self.hi - v) / (self.hi - self.lo)

    def frame(self, t, main_color=INK):
        img = self.base.copy()
        d = ImageDraw.Draw(img)
        x = self.x(t)
        d.line((x, self.pad_t, x, self.h - self.pad_b), fill=(60, 70, 80), width=1)
        if self.series:
            v = self.series[0][t]
            if np.isfinite(v):
                yy = self.y(float(np.clip(v, self.lo, self.hi)))
                d.ellipse((x - 4, yy - 4, x + 4, yy + 4), fill=main_color)
                lab = f'现在 {v:+.1f} {self.unit}' if self.unit == 'cm' else f'现在 {v:.0f} {self.unit}'
                tw = d.textlength(self.title, font=font(16))
                text(d, (8 + tw + 14, 3), lab, 15, main_color, bold=True)
        text(d, (self.pad_l, self.h - self.pad_b + 1), '0 s', 13, MUTED)
        text(d, (self.w - self.pad_r, self.h - self.pad_b + 1), f'{self.times[-1] - self.times[0]:.1f} s', 13, MUTED, anchor='ra')
        return img
