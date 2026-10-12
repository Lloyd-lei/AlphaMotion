"""12-frame contact sheet spanning the WHOLE duration: frame indices round(linspace(0, T-1, 12)), i.e. the first
and the last frame are always included.  From an mp4 (exact decoded-frame selection with ffmpeg select=eq(n,i))
or from in-memory frames.

  python -m render.grid12 VIDEO.mp4 [--out grid12.png] [--n 12] [--cols 4] [--title TEXT]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'render'

from . import layout as L  # noqa: E402


def grid_indices(T: int, n: int = 12) -> np.ndarray:
    if T <= 0:
        raise ValueError('empty video')
    return np.unique(np.round(np.linspace(0, T - 1, n)).astype(int)) if T < n else np.round(np.linspace(0, T - 1, n)).astype(int)


def probe(video) -> dict:
    d = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
                                            '-show_entries', 'stream=nb_read_frames,width,height', '-show_entries',
                                            'frame=best_effort_timestamp_time', '-of', 'json', str(video)]))
    s = d['streams'][0]
    pts = [float(f['best_effort_timestamp_time']) for f in d.get('frames', [])]
    return {'frames': int(s['nb_read_frames']), 'width': int(s['width']), 'height': int(s['height']), 'pts': pts}


def extract_frames(video, idx, tile_w=480, tile_h=270):
    """Decode exactly the frames `idx` (0-based decode order) -> list of PIL images (tile size)."""
    idx = [int(i) for i in idx]
    expr = '+'.join(f'eq(n\\,{i})' for i in idx)
    with tempfile.TemporaryDirectory(prefix='am2grid-') as td:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(video), '-vf', f"select='{expr}',scale={tile_w}:{tile_h}:flags=area",
                        '-vsync', '0', str(Path(td) / '%03d.png')], check=True, timeout=600)
        files = sorted(Path(td).glob('*.png'))
        if len(files) != len(idx):
            raise ValueError(f'expected {len(idx)} frames, decoded {len(files)}')
        return [Image.open(f).convert('RGB') for f in files]


def compose(tiles, idx, times, out_png, title='', cols=4, total=None):
    th, tw = tiles[0].height, tiles[0].width
    rows = int(np.ceil(len(tiles) / cols))
    head = 56
    img = Image.new('RGB', (cols * tw, head + rows * th), L.BG)
    d = ImageDraw.Draw(img)
    T = total if total is not None else (idx[-1] + 1)
    span = f'{times[idx[0]] - times[0]:.2f}–{times[idx[-1]] - times[0]:.2f} s' if times is not None else ''
    s = f'{title}   ·   12 帧均匀覆盖全程（第 1 帧到第 {T} 帧，{span}）' if title else f'12 帧均匀覆盖全程（第 1 帧到第 {T} 帧，{span}）'
    L.text(d, (16, 12), s, L.fit_text(d, s, 24, cols * tw - 32, True), L.INK, bold=True)
    for k, (tile, i) in enumerate(zip(tiles, idx)):
        x, y = (k % cols) * tw, head + (k // cols) * th
        img.paste(tile, (x, y))
        lab = f'第 {i + 1} 帧' + (f' · {times[i] - times[0]:.2f} s' if times is not None else '')
        bw = d.textlength(lab, font=L.font(18)) + 16
        d.rectangle((x + 4, y + 4, x + 4 + bw, y + 30), fill=(255, 255, 255))
        L.text(d, (x + 12, y + 5), lab, 18, L.INK)
        d.rectangle((x, y, x + tw - 1, y + th - 1), outline=(200, 206, 212))
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return {'indices': [int(i) for i in idx], 'size': img.size, 'out': str(out_png)}


def grid_from_video(video, out_png, times=None, n=12, cols=4, title=''):
    info = probe(video)
    T = info['frames']
    if times is not None and len(times) != T:
        raise ValueError(f'video has {T} frames, times has {len(times)}')
    if times is None:
        times = np.asarray(info['pts'], np.float64) if len(info['pts']) == T else None
    idx = grid_indices(T, n)
    tw = info['width'] // cols
    th = int(round(tw * info['height'] / info['width']))
    tiles = extract_frames(video, idx, tw, th)
    return compose(tiles, idx, times, out_png, title, cols, total=T)


def grid_from_frames(frames, out_png, times=None, n=12, cols=4, title=''):
    T = len(frames)
    idx = grid_indices(T, n)
    tiles = []
    for i in idx:
        im = frames[i] if isinstance(frames[i], Image.Image) else Image.fromarray(np.asarray(frames[i]))
        tw = im.width // cols
        tiles.append(im.resize((tw, int(round(tw * im.height / im.width))), Image.BILINEAR))
    return compose(tiles, idx, times, out_png, title, cols, total=T)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--out')
    ap.add_argument('--n', type=int, default=12)
    ap.add_argument('--cols', type=int, default=4)
    ap.add_argument('--title', default='')
    a = ap.parse_args(argv)
    out = a.out or str(Path(a.video).with_name(Path(a.video).stem + '_grid12.png'))
    print(json.dumps(grid_from_video(a.video, out, n=a.n, cols=a.cols, title=a.title), ensure_ascii=False))


if __name__ == '__main__':
    main()
