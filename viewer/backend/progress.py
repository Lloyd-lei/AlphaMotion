"""Read-only conversion-progress scanner (sources -> download -> SMPL-X -> robots).

Rules: only stat() and small header reads; never hash whole files, never expand
large archives, never write to the data root. A scan runs in a background
thread at most once per ``MIN_INTERVAL`` seconds; callers get the last result.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import re
import threading
import time
import zipfile

from . import npzhead
from .gpu_adapter import EXPORT_DIRS, PRODUCTION_SUBDIR, guess_source, invalid_engines, known_robots, split_export_name

MIN_INTERVAL = 300
GB = 1e9
# Known totals (hours): diversity/REPORT.md, 2026-10-07 inventory.
SOURCES = [
    dict(id='hiphi', name='HiPHI', known_h=617.5, orig_h=308.73, mirror_h=308.73, note='308.7 h 原片 + 308.7 h 镜像；32 个 tar 卷'),
    dict(id='bones_seed', name='BONES-SEED', known_h=144.22, orig_h=144.22, mirror_h=144.11, note='另有镜像 144.1 h；完整包未下载'),
    dict(id='grail', name='GRAIL', known_h=55.6, orig_h=55.6, note='有人体部分；1,527 段官方无人体（永远拿不到）'),
    dict(id='amass', name='AMASS', known_h=43.83, orig_h=43.83, note='本机只有 xwbc 派生数据'),
    dict(id='omnicontact', name='OmniContact', known_h=12.03, orig_h=12.03, note='690 段，已全部下载'),
    dict(id='lafan', name='LAFAN', known_h=4.6, orig_h=4.6, note='本机只有 1 段 UMR 样例'),
    dict(id='mixamo', name='Mixamo', known_h=2.41, orig_h=2.41, note='只有元数据和样例'),
]
PIPELINE_ZH = {'factory_v1': 'UMR CPU 工厂', 'gpu_export': 'UMR GPU 引擎（评测导出）', 'production_v1': 'UMR GPU 量产'}


def _tail(path, nbytes=262144):
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode('utf-8', 'replace').replace('\r', '\n')
    except OSError:
        return ''


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


class ProgressScanner:
    def __init__(self, data_root, cache_root, human_index):
        self.data_root = Path(data_root)
        self.cache_file = Path(cache_root) / 'progress_headers.json'
        self.human_index = human_index
        self._lock = threading.Lock()
        self._thread = None
        self._result = None
        self._started_at = None
        self._finished_at = 0.
        self._error = None
        self._prev_bytes = {}
        self._headers = _json(self.cache_file) or {}

    # ------------------------------------------------------------------ public
    def get(self, force=False):
        with self._lock:
            stale = time.time() - self._finished_at >= MIN_INTERVAL
            running = self._thread is not None and self._thread.is_alive()
            if (stale or (force and self._result is None)) and not running:
                self._thread = threading.Thread(target=self._run, name='progress-scan', daemon=True)
                self._started_at = time.time()
                self._thread.start()
                running = True
            result = dict(self._result) if self._result else None
            return {'result': result, 'scanning': running, 'scan_started_at': self._started_at,
                    'scan_finished_at': self._finished_at or None, 'error': self._error,
                    'min_interval_s': MIN_INTERVAL}

    def wait(self, timeout=600):
        t = self._thread
        if t is not None:
            t.join(timeout)
        return self.get()

    def _run(self):
        try:
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 15)
        except (OSError, AttributeError):
            pass
        try:
            r = self.scan()
            with self._lock:
                self._result = r; self._error = None
        except Exception as exc:  # report, never crash the service
            with self._lock:
                self._error = f'{type(exc).__name__}: {exc}'
        finally:
            with self._lock:
                self._finished_at = time.time()

    # ------------------------------------------------------------------ helpers
    def _header(self, path, fields=('qpos',)):
        """frames/fps of a robot npz from its header (cached by size+mtime)."""
        p = Path(path); st = p.stat(); stamp = [st.st_size, st.st_mtime_ns]
        old = self._headers.get(str(p))
        if old and old.get('stamp') == stamp and old.get('v') == 2:
            return old['info']
        info = {}
        try:
            with zipfile.ZipFile(p) as z:
                names = {i.filename[:-4] for i in z.infolist()}
                shape, _ = npzhead.member_header(p, 'qpos', z)
                info['frames'] = int(shape[0])
                if 'fps' in names:
                    info['fps'] = float(npzhead.small_member(p, 'fps', z))
                if 'provenance' in names:
                    prov = json.loads(str(npzhead.small_member(p, 'provenance', z)))
                    info['source_kind'] = prov.get('source_kind'); info['source_data'] = prov.get('source_data')
                    info['engine'] = prov.get('engine')
                if 'flags' in names:
                    fl = npzhead.small_member(p, 'flags', z)
                    info['flagged'] = int((fl != 0).sum())
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            info['error'] = str(exc)
        self._headers[str(p)] = {'stamp': stamp, 'info': info, 'v': 2}
        return info

    def _rate(self, key, nbytes):
        now = time.time(); prev = self._prev_bytes.get(key)
        self._prev_bytes[key] = (now, nbytes)
        if prev and now > prev[0] and nbytes >= prev[1]:
            return (nbytes - prev[1]) / (now - prev[0]) / 1e6
        return None

    # ------------------------------------------------------------------ download
    def _hiphi_download(self):
        root = self.data_root / 'raw' / 'hiphi'
        out = dict(kind='tar 卷')
        parts = {}
        man = root / 'data' / 'archive_manifest.csv'
        if man.exists():
            with open(man) as f:
                for r in csv.DictReader(f):
                    if r.get('kind') == 'data':
                        parts[r['archive_name']] = dict(part=f"{int(r['part_index']):04d}", bytes=int(r['compressed_bytes']),
                                                         pairs=int(r['pair_count']), hours=0.)
        # hours per part: motion_to_part (orig + mirror ids) joined with metadata durations
        dur = {}
        meta = root / 'metadata' / 'hiphi_metadata.csv'
        if meta.exists():
            with open(meta) as f:
                for r in csv.DictReader(f):
                    try:
                        dur[r['motion_id']] = float(r['duration_sec'])
                    except (KeyError, ValueError):
                        pass
        m2p = root / 'data' / 'motion_to_part.csv'
        if m2p.exists() and parts:
            with open(m2p) as f:
                for r in csv.DictReader(f):
                    p = parts.get(r['archive_name'])
                    if p is not None:
                        p['hours'] += (dur.get(r['motion_id'], 0.) + dur.get(r['mirrored_motion_id'], 0.)) / 3600
        status = _json(self.data_root / 'logs' / 'hiphi_streamc_status.json') or {}
        st_parts = {p['part']: p for p in status.get('parts', [])}
        extracted = {p.name.split('_')[-1] for p in (root / 'extracted').glob('part_*')} if (root / 'extracted').exists() else set()
        done_b = part_b = 0.; done_h = 0.; n_done = 0; rows = []
        for name, p in sorted(parts.items(), key=lambda kv: kv[1]['part']):
            tar = root / 'data' / name
            sp = st_parts.get(p['part'], {})
            state = sp.get('state') or ('extracted' if p['part'] in extracted else 'unknown')
            complete = p['part'] in extracted or state in ('ready', 'foreign', 'extracted', 'done') or (
                tar.exists() and not Path(str(tar) + '.aria2').exists() and tar.stat().st_size == p['bytes'])
            have = p['bytes'] if complete else 0.
            if not complete and sp.get('have_GB') is not None:
                have = float(sp['have_GB']) * GB
            elif not complete and Path(str(tar) + '.aria2').exists():
                state = state if state != 'unknown' else 'partial'
            if complete:
                n_done += 1; done_b += p['bytes']; done_h += p['hours']
            else:
                part_b += have
            rows.append(dict(part=p['part'], state=state, complete=complete, bytes=p['bytes'], have_bytes=have,
                             hours=round(p['hours'], 2), extracted=p['part'] in extracted))
        total_b = sum(p['bytes'] for p in parts.values())
        events = [l for l in _tail(self.data_root / 'logs' / 'hiphi_streamc.log', 65536).splitlines() if l.startswith('{')]
        last = None
        for l in reversed(events):
            try:
                last = json.loads(l); break
            except ValueError:
                continue
        out.update(total_bytes=total_b, done_bytes=done_b + part_b, complete_bytes=done_b, segments_total=len(parts),
                   segments_done=n_done, hours_done=round(done_h, 2), hours_total=round(sum(p['hours'] for p in parts.values()), 2),
                   current=status.get('current'), rate_MBps=status.get('our_rate_MBps_1m'),
                   rate_MBps_10m=status.get('our_rate_MBps_10m'), eta_h=status.get('eta_all_parts_h'),
                   remaining_GB=status.get('remaining_download_GB'), net_up=status.get('net_up'),
                   paused_reason=status.get('paused_reason'), status_time=status.get('t'),
                   deadline_in_h=status.get('deadline_in_h'), last_event=last, parts=rows,
                   source_files=['logs/hiphi_streamc_status.json', 'logs/hiphi_streamc.log', 'raw/hiphi/data/archive_manifest.csv',
                                 'raw/hiphi/metadata/hiphi_metadata.csv'])
        hold = list((root / 'hold').glob('*.tar.zst')) if (root / 'hold').exists() else []
        if hold:
            out['hold'] = [h.name for h in hold]
        return out

    def _grail_download(self):
        root = self.data_root / 'raw' / 'grail'
        tree = _json(root / '_tree.json') or {}
        total_b = have_b = 0; total_n = done_n = 0; recon_total = recon_done = 0
        for k, v in tree.items():
            if not isinstance(v, dict):
                continue
            for rel, size in v.get('files', []):
                total_b += size; total_n += 1
                fp = root / rel
                try:
                    s = fp.stat().st_size
                except OSError:
                    s = None
                if s is not None:
                    have_b += min(s, size)
                    if s >= size:
                        done_n += 1
                        if k.endswith('/recon'):
                            recon_done += 1
                if k.endswith('/recon'):
                    recon_total += 1
        log = _tail(self.data_root / 'logs' / 'grail' / 'dl_loop_restart_1007.log', 131072)
        rounds = re.findall(r'(\d\d:\d\d:\d\d) round (\d+): remaining files (\d+) ([\d.]+) GB', log)
        rates = re.findall(r'(\d\d:\d\d:\d\d) round (\d+) rc=(-?\d+) ([\d.]+) min, \+([\d.]+) GB \(([\d.]+) MB/s\)', log)
        man = _json(self.data_root / 'smplx' / 'grail' / 'manifest_summary.json') or {}
        out = dict(kind='文件', total_bytes=total_b, done_bytes=have_b, segments_total=total_n, segments_done=done_n,
                   recon_total=recon_total, recon_done=recon_done,
                   current=f'第 {rounds[-1][1]} 轮（{rounds[-1][0]}，剩 {rounds[-1][2]} 个文件 {rounds[-1][3]} GB）' if rounds else None,
                   rate_MBps=self._rate('grail', have_b),
                   last_round_rate_MBps=float(rates[-1][5]) if rates else None,
                   source_files=['raw/grail/_tree.json', 'logs/grail/dl_loop_restart_1007.log'])
        if man.get('total'):
            out['manifest'] = man['total']
        return out

    def _omni_download(self):
        root = self.data_root / 'raw' / 'omnicontact'
        man = root / 'metadata' / 'raw_mocap_manifest.csv'
        n = done = 0; nbytes = 0
        if man.exists():
            with open(man) as f:
                for r in csv.DictReader(f):
                    n += 1
                    b = root / r['bvh']
                    if b.exists():
                        done += 1; nbytes += b.stat().st_size
        return dict(kind='BVH 段', segments_total=n, segments_done=done, done_bytes=nbytes, total_bytes=None,
                    current='已完成' if n and done == n else None, source_files=['raw/omnicontact/metadata/raw_mocap_manifest.csv'])

    # ------------------------------------------------------------------ robots
    def _retargeted(self):
        robots_dir = self.data_root / 'robots'
        staging = _json(robots_dir / 'staging_summary.json') or {}
        robots = sorted(staging) if staging else sorted(p.name for p in robots_dir.iterdir() if p.is_dir())
        rows = []
        out = self.data_root / 'out'
        if out.exists():
            for p in sorted(out.glob('*/*.npz')):
                info = self._header(p)
                clip = p.parent.name
                src = guess_source(clip)
                if src == 'unknown' and info.get('source_data'):
                    src = guess_source(Path(info['source_data']).name)
                rows.append(dict(pipeline='factory_v1', source=src, clip=clip, robot=p.stem, path=str(p), **info))
        names = known_robots(self.data_root)
        for d in EXPORT_DIRS:
            if d.exists():
                for p in sorted(d.glob('*.npz')):
                    clip, robot = split_export_name(p.stem, names)
                    rows.append(dict(pipeline='gpu_export', source=guess_source(clip), clip=clip, robot=robot, path=str(p),
                                     **self._header(p)))
        prod = self.data_root / PRODUCTION_SUBDIR
        bad = invalid_engines(prod)
        if prod.exists():
            for p in sorted(prod.glob('*/*/*.npz')):
                if p.name.startswith('.') or p.parent.parent.name.startswith('_') or p.parent.name.startswith('_'):
                    continue
                rows.append(dict(pipeline='production_v1', source=p.parent.parent.name, clip=p.parent.name, robot=p.stem,
                                 path=str(p), **self._header(p)))
        matrix = {}
        for r in rows:
            r['invalid'] = r.get('engine') in bad
            if not r.get('frames') or not r.get('fps') or r['invalid']:
                continue
            h = r['frames'] / r['fps'] / 3600
            cell = matrix.setdefault(r['source'], {}).setdefault(r['robot'], {'hours': 0., 'clips': 0, 'by_pipeline': {}})
            cell['hours'] += h; cell['clips'] += 1
            bp = cell['by_pipeline'].setdefault(r['pipeline'], {'hours': 0., 'clips': 0})
            bp['hours'] += h; bp['clips'] += 1
        extra = sorted({r['robot'] for r in rows if r['robot'] and r['robot'] not in robots})
        return robots, extra, rows, matrix

    def _production_status(self):
        prod = self.data_root / PRODUCTION_SUBDIR
        if not prod.exists():
            return None
        jobs = 0
        jf = prod / '_jobs' / 'jobs.jsonl'
        if jf.exists():
            jobs = sum(1 for l in jf.read_text().splitlines() if l.strip())
        jd = prod / '_jobs'
        done = len(list(jd.glob('*.done'))) if jd.exists() else 0
        failed = len(list(jd.glob('*.failed*'))) if jd.exists() else 0
        log = [l for l in _tail(prod / '_logs' / 'runner.log', 16384).splitlines() if l.strip()]
        bad = invalid_engines(prod)
        return dict(root=str(prod), jobs=jobs, done=done, failed=failed, stop_file=(prod / '_STOP').exists(),
                    last_log=log[-3:], invalid_engines=bad)

    # ------------------------------------------------------------------ scan
    def scan(self):
        t0 = time.time()
        human = self.human_index.scan()
        smplx = {}
        for e in human:
            if e.get('error') or not e.get('duration_s'):
                continue
            s = smplx.setdefault(e['source'], {'hours': 0., 'clips': 0, 'variants': {}, 'clip_names': set()})
            v = s['variants'].setdefault(e['variant'], {'hours': 0., 'clips': 0})
            v['hours'] += e['duration_s'] / 3600; v['clips'] += 1
            if e['variant'] == 'gpu_30fps':
                continue  # master[::3] copy of the gpu variant: not new hours
            if e['clip'] in s['clip_names']:
                continue
            s['clip_names'].add(e['clip'])
            s['hours'] += e['duration_s'] / 3600; s['clips'] += 1
        for s in smplx.values():
            s.pop('clip_names')
        downloads = {}
        for sid, fn in (('hiphi', self._hiphi_download), ('grail', self._grail_download), ('omnicontact', self._omni_download)):
            try:
                downloads[sid] = fn()
            except (OSError, ValueError) as exc:
                downloads[sid] = {'error': str(exc)}
        robots, extra, rows, matrix = self._retargeted()
        production = self._production_status()
        funnel = []
        for s in SOURCES:
            sid = s['id']
            rt = matrix.get(sid, {})
            funnel.append(dict(s, download=downloads.get(sid) or {'note': '本机无下载任务'},
                               smplx=smplx.get(sid, {'hours': 0., 'clips': 0, 'variants': {}}),
                               retarget=dict(hours=sum(c['hours'] for c in rt.values()),
                                             clip_robot_pairs=sum(c['clips'] for c in rt.values()),
                                             robots=len(rt))))
        other = {k: v for k, v in matrix.items() if k not in {s['id'] for s in SOURCES}}
        for k in other:
            funnel.append(dict(id=k, name=k, known_h=None, note='未知来源（按片段名推断失败）', download={},
                               smplx={'hours': 0., 'clips': 0, 'variants': {}},
                               retarget=dict(hours=sum(c['hours'] for c in other[k].values()),
                                             clip_robot_pairs=sum(c['clips'] for c in other[k].values()),
                                             robots=len(other[k]))))
        try:
            tmp = self.cache_file.with_suffix('.tmp')
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self._headers))
            tmp.replace(self.cache_file)
        except OSError:
            pass
        return dict(generated_at=time.time(), scan_seconds=round(time.time() - t0, 2), funnel=funnel,
                    robots=robots, extra_robots=extra, matrix=matrix, pipelines=PIPELINE_ZH,
                    retarget_rows=[{k: r.get(k) for k in ('pipeline', 'source', 'clip', 'robot', 'frames', 'fps', 'flagged', 'error', 'invalid')}
                                   for r in rows],
                    invalid_files=sum(1 for r in rows if r.get('invalid')), production=production,
                    human_index=dict(entries=len(human), scan_seconds=self.human_index.scan_seconds),
                    notes=['已知总时长取自 diversity/REPORT.md（10-07 盘点）。',
                           'HiPHI 已下载小时 = 已完整落盘的 tar 卷内（原片+镜像）时长；SMPL-X 小时按片段去重，30 fps 变体（master[::3]）不重复计。',
                           '重定向矩阵按 片段×机器人 文件计时长；qa_v2/pilot 等窗口片段与完整片段可能重叠，按管线分开列出。',
                           'GRAIL 的 SMPL-X 小时来自转换清单 manifest.jsonl（status=ok）。'])
