"""Read-only, bounded CPU factory_npz_v1 source adapter.

No imports from the factory pipeline (some helpers write beside source assets).
All expanded NPY members live in the application's content-addressed cache.
"""
from __future__ import annotations
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml

from collections import OrderedDict
from pathlib import Path
from functools import wraps
import fcntl
import hashlib
import json
import os
import pickle
import re
import shutil
import threading
import uuid
import zipfile
import numpy as np
import mujoco
from . import geometry


class AdapterError(ValueError):
    code = 'data_error'


class UnsupportedFormat(AdapterError):
    code = 'unsupported_format'


class SnapshotChanged(AdapterError):
    code = 'stale_snapshot'


def fingerprint(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def sha256_file(path):
    before = fingerprint(path); h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    if fingerprint(path) != before:
        raise SnapshotChanged('文件读取期间发生变化，请刷新后重新审阅')
    return h.hexdigest()


def locked(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class NpyArchive:
    """Expand ZIP members in <=1 MiB streaming blocks, then memory-map on demand."""
    MAX_EXPANDED = 2 * 1024**3
    MAX_CACHE = 4 * 1024**3

    def __init__(self, path, cache_root, expected_sha=None):
        self.path = Path(path).resolve()
        self.stamp = fingerprint(path)
        self.sha = sha256_file(path)
        if expected_sha is not None and self.sha != expected_sha:
            raise SnapshotChanged('npz 哈希已改变；当前审阅快照过期，不能自动重绑')
        self.directory = Path(cache_root) / 'npz' / self.sha
        self._arrays = {}
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        with open(self.directory.parent / f'{self.sha}.lock', 'a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not (self.directory / 'complete.json').exists():
                tmp = self.directory.parent / f'.{self.sha}.{uuid.uuid4().hex}'
                tmp.mkdir()
                try:
                    with zipfile.ZipFile(self.path) as z:
                        infos = z.infolist()
                        total = sum(i.file_size for i in infos)
                        if total > self.MAX_EXPANDED:
                            raise AdapterError('单个 npz 解压缓存超过 2 GiB 安全上限')
                        if shutil.disk_usage(tmp).free < total + 256*1024**2:
                            raise AdapterError('应用缓存磁盘余量不足')
                        cached_bytes = sum(p.stat().st_size for p in self.directory.parent.glob('*/*.npy'))
                        if cached_bytes + total > self.MAX_CACHE:
                            raise AdapterError('应用解压缓存已达 4 GiB 上限；停止服务后清理项目 cache 再继续，不删除源数据')
                        names = []
                        for info in infos:
                            name = info.filename
                            if not re.fullmatch(r'[A-Za-z0-9_]+\.npy', name) or name in names:
                                raise AdapterError('不支持的 npz 成员名或重复成员')
                            names.append(name)
                            with z.open(info) as src, open(tmp / name, 'wb') as dest:
                                shutil.copyfileobj(src, dest, length=1024*1024)
                            # Correspondence archives contain unrelated object metadata.
                            # Preserve its bytes, but never unpickle it or mmap it. Numeric
                            # members used by this reader remain separately mmap-backed.
                            with open(tmp / name, 'rb') as f:
                                version = np.lib.format.read_magic(f)
                                _, _, dtype = np.lib.format._read_array_header(f, version)
                            if not dtype.hasobject:
                                probe = np.load(tmp / name, mmap_mode='r', allow_pickle=False)
                                del probe
                    self.check()
                    (tmp / 'complete.json').write_text(json.dumps({'sha256': self.sha, 'members': names, 'expanded_bytes': total}))
                    os.rename(tmp, self.directory)
                finally:
                    if tmp.exists():
                        shutil.rmtree(tmp)
        self.files = [p.stem for p in self.directory.glob('*.npy')]

    def check(self):
        try:
            current = fingerprint(self.path)
        except OSError as e:
            raise SnapshotChanged(f'快照源文件已不可读：{self.path.name}') from e
        if current != self.stamp:
            raise SnapshotChanged(f'源文件发生变化，审阅快照已过期：{self.path.name}')

    def __contains__(self, name):
        return name in self.files

    def __getitem__(self, name):
        if name not in self.files:
            raise KeyError(name)
        if name not in self._arrays:
            self._arrays[name] = np.load(self.directory / f'{name}.npy', mmap_mode='r', allow_pickle=False)
        return self._arrays[name]

    def scalar(self, name, default=None):
        if name not in self:
            return default
        return np.asarray(self[name]).item()


REGION_KEYS = ('ground_contact', 'ground_clearance', 'ground_clearance_visual', 'ground_slip',
               'ground_slip_speed', 'ground_contact_required', 'human_ground_contact',
               'human_ground_slip', 'human_ground_clearance', 'object_contact', 'object_clearance',
               'object_clearance_visual', 'object_slip', 'object_slip_speed', 'object_contact_required',
               'human_object_contact', 'human_object_slip', 'human_object_clearance')
FRAME_KEYS = ('flags', 'flags_initial', 'pen_ground_m', 'pen_object_m', 'pen_self_m',
              'ground_min_visual_m', 'ground_min_collision_m', 'object_min_visual_m', 'object_min_collision_m')


def finite_max(array, default=0.):
    a = np.asarray(array); finite = a[np.isfinite(a)]
    return float(finite.max()) if len(finite) else default


def contact_consistency(archive, surface):
    """Report, rather than repair or assert away, inconsistent packed/region labels."""
    reg = np.asarray(surface['region']); n = len(reg); result = {}
    for prefix in ('ground', 'object'):
        pk, rk = f'{prefix}_point_contact', f'{prefix}_contact'
        if pk not in archive or rk not in archive:
            continue
        packed, labels = archive[pk], archive[rk]
        mismatches = counts = labeled_without_point = points_without_label = 0
        first = []
        for lo in range(0, len(packed), 256):
            bits = np.unpackbits(np.asarray(packed[lo:lo+256]), axis=1, bitorder='big')[:, :n].astype(bool)
            counts += int(bits.sum())
            for r in range(labels.shape[1]):
                hit = bits[:, reg == r].any(axis=1)
                target = np.asarray(labels[lo:lo+256, r], dtype=bool)
                bad = hit != target
                mismatches += int(bad.sum())
                labeled_without_point += int((target & ~hit).sum())
                points_without_label += int((hit & ~target).sum())
                for idx in np.flatnonzero(bad)[:max(0, 8-len(first))]:
                    first.append([lo+int(idx), r])
        result[prefix] = dict(point_contacts=counts, mismatch_region_frames=mismatches,
                              labeled_without_point=labeled_without_point, points_without_label=points_without_label,
                              first_mismatches=first, bitorder='big', rule='any packed point in region == saved region contact')
    return result


class FactoryNpzAdapter:
    format = 'factory_npz_v1'

    def open(self, service, key, expected_sha=None):
        return service._open_factory_snapshot(key, expected_sha)


class UnimplementedAdapter:
    def __init__(self, format):
        self.format = format

    def open(self, service=None, key=None, expected_sha=None):
        raise UnsupportedFormat(f'数据格式 {self.format} 尚未实现；不会猜测字段或套用 factory_npz_v1')


class FactoryService:
    def __init__(self, data_root, cache_root):
        self.data_root = Path(data_root).resolve()
        self.cache_root = Path(cache_root).resolve()
        if self.cache_root == self.data_root or self.data_root in self.cache_root.parents:
            raise ValueError('缓存必须在独立项目目录，不得写入数据目录')
        self.cache_root.mkdir(parents=True, exist_ok=True)
        from .gpu_adapter import GpuNpzAdapter  # viewer lives in gpu_adapter.GpuService
        self.adapters = {'factory_npz_v1': FactoryNpzAdapter(),
                         'gpu_npz_v2': GpuNpzAdapter(),
                         'packed_v1': UnimplementedAdapter('packed_v1')}
        self._lock = threading.RLock()
        self._items, self._snapshots = {}, OrderedDict()
        self._hash_cache = {}
        self._models = OrderedDict()
        self.max_chunk = 64

    def _sha(self, path):
        stamp = fingerprint(path); key = str(Path(path).resolve())
        old = self._hash_cache.get(key)
        if old is None or old[0] != stamp:
            old = (stamp, sha256_file(path)); self._hash_cache[key] = old
        return old[1]

    def _scan(self):
        rows, warnings, paths = {}, [], set()
        queue = self.data_root / 'audit/queue.jsonl'
        if queue.exists():
            for i, line in enumerate(queue.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    if not isinstance(item.get('item_id'), str) or not isinstance(item.get('npz'), str):
                        raise ValueError('缺少 item_id/npz')
                    item = dict(item, enqueued=True, queued=True)
                    rows[item['item_id']] = item
                    paths.add(str(Path(item['npz']).resolve()))
                except (ValueError, TypeError) as e:
                    warnings.append(f'queue.jsonl 第 {i} 行无效：{e}')
        for p in sorted((self.data_root / 'out').glob('*/*.npz')):
            if str(p.resolve()) in paths:
                continue
            clip, robot = p.parent.name, p.stem
            item_id = f'scan/{clip}::{robot}'
            rows[item_id] = dict(item_id=item_id, clip_id=clip, robot=robot, npz=str(p),
                                 format='factory_npz_v1', enqueued=False, queued=False,
                                 priority=0., sample_kind='scan', spec_version='DATA_SPEC_v2')
        self._items = {}
        for row in rows.values():
            row['key'] = hashlib.sha256(row['item_id'].encode()).hexdigest()[:24]
            self._items[row['key']] = row
        return warnings

    def _item(self, key):
        self._scan()
        if key not in self._items:
            raise KeyError('找不到审阅条目')
        item = self._items[key]
        fmt = item.get('format')
        if fmt != 'factory_npz_v1':
            raise UnsupportedFormat(f'数据格式 {fmt!r} 未实现；不会猜测格式')
        p = Path(item['npz']).resolve()
        if self.data_root / 'out' not in p.parents or p.suffix != '.npz':
            raise AdapterError('factory_npz_v1 的 npz 必须位于只读 out 目录')
        return item

    def _bind(self, snap, path):
        path = Path(path).resolve()
        allowed = (self.data_root, Path(f'{SRCFG.factory_root}/umr').resolve())
        if not any(path == root or root in path.parents for root in allowed):
            raise AdapterError('关联资产路径不在本任务只读数据源 allowlist 内')
        stamp = fingerprint(path)
        old = snap['companions'].get(str(path))
        if old is not None and old != stamp:
            raise SnapshotChanged('静态资产或人体源文件改变，请刷新后重新审阅')
        snap['companions'][str(path)] = stamp
        return path

    def _check(self, snap):
        snap['archive'].check()
        for path, stamp in snap['companions'].items():
            try:
                current = fingerprint(path)
            except OSError as e:
                raise SnapshotChanged('静态资产或人体源文件已不可读，当前快照已过期') from e
            if current != stamp:
                raise SnapshotChanged('静态资产或人体源文件改变，当前快照已过期')

    def _model(self, snap, xml):
        deps = geometry.xml_dependencies(xml)
        for p in deps:
            self._bind(snap, p)
        signature = tuple((str(p), fingerprint(p)) for p in deps)
        if signature not in self._models:
            self._models[signature] = mujoco.MjModel.from_xml_path(str(xml))
            while len(self._models) > 3:
                self._models.popitem(last=False)
        self._models.move_to_end(signature)
        return self._models[signature]

    def _snapshot(self, key, expected_sha=None):
        self._scan()
        if key not in self._items:
            raise KeyError('找不到审阅条目')
        format = self._items[key].get('format')
        adapter = self.adapters.get(format)
        if adapter is None:
            raise UnsupportedFormat(f'数据格式 {format!r} 未实现；不会猜测格式')
        return adapter.open(self, key, expected_sha)

    def _open_factory_snapshot(self, key, expected_sha=None):
        item = self._item(key); path = Path(item['npz'])
        try:
            actual = self._sha(path)
        except OSError as e:
            if expected_sha:
                raise SnapshotChanged('审阅快照的 npz 已不可读') from e
            raise
        if expected_sha is not None and expected_sha != actual:
            raise SnapshotChanged('npz 哈希已改变；当前审阅快照过期，必须重新审阅')
        ident = (key, actual)
        if ident in self._snapshots:
            snap = self._snapshots[ident]; self._check(snap)
            self._snapshots.move_to_end(ident)
            return snap
        a = NpyArchive(path, self.cache_root, actual)
        required = ('qpos', 'qpos_umr', 'fps', 'robot_xml', 'regions', 'ground_contact',
                    'ground_point_contact', 'floor_offset_removed', 'provenance')
        if any(k not in a for k in required):
            raise AdapterError('缺少必要字段：' + ', '.join(k for k in required if k not in a))
        snap = {'archive': a, 'companions': {}, 'item': dict(item), 'sha': actual}
        xml = Path(a.scalar('robot_xml')).resolve()
        if self.data_root / 'robots' not in xml.parents:
            raise AdapterError('robot_xml 必须来自只读 robots 资产目录')
        m = self._model(snap, xml)
        surface_path = self._bind(snap, xml.parent / 'surface.npz')
        surf = NpyArchive(surface_path, self.cache_root)
        snap['surface'] = surf
        q = a['qpos']; t = len(q); n = len(surf['points_local'])
        if q.ndim != 2 or q.shape[1] != m.nq or a['qpos_umr'].shape != q.shape:
            raise AdapterError('qpos/原始 qpos 维度与源 MJCF 不一致')
        fps = float(a.scalar('fps'))
        if not np.isfinite(fps) or fps <= 0 or t == 0:
            raise AdapterError('帧数或 fps 无效')
        for field in REGION_KEYS + FRAME_KEYS + ('ground_point_contact', 'object_point_contact', 'object_pos', 'object_rot'):
            if field in a and (a[field].ndim == 0 or len(a[field]) != t):
                raise AdapterError(f'{field} 帧数与 qpos 不一致')
        if surf['geom'].min() < 0 or surf['geom'].max() >= m.ngeom:
            raise AdapterError('surface geom 索引超出源 MJCF 范围')
        if surf['region'].min() < 0 or surf['region'].max() >= len(a['regions']):
            raise AdapterError('surface region 索引越界')
        for name in ('ground_point_contact', 'object_point_contact'):
            if name in a and a[name].shape != (t, (n+7)//8):
                raise AdapterError(f'{name} 打包点数与 surface 不一致')
        prov = json.loads(str(a.scalar('provenance')))
        hp = Path(prov['human_slots']).resolve() if prov.get('human_slots') else None
        if hp is not None and hp.exists():
            self._bind(snap, hp)
        else:
            hp = None
        consistency = contact_consistency(a, surf)
        warnings = ['标签为文件原值；界面现算几何量只用于核对，不修改源标签。',
                    '机器人 qpos / qpos_umr / 物体位姿已校准地面；仅人体 source_slots 减一次 floor_offset_removed。']
        if abs(fps-30.) > .01:
            warnings.append(f'文件 fps={fps:.7f}，与 DATA_SPEC_v2 的 30 Hz 不同；按实际 fps 播放。')
        cfg = prov.get('label_cfg', {})
        if cfg.get('slip_speed', 0.01) != 0.01:
            warnings.append(f'源标签打滑阈值 {cfg.get("slip_speed")} m/s；规范阈值 0.01 m/s，不能等同。')
        if cfg.get('enter', .001) > .001 or cfg.get('leave', .002) > .002:
            warnings.append('源接触滞回阈值为 1/2 cm，规范为 1/2 mm；区域标签与点标签分别展示。')
        for prefix, c in consistency.items():
            if c['mismatch_region_frames']:
                warnings.append(f'{prefix} 点标签与区域标签存在 {c["mismatch_region_frames"]} 个区域帧不一致（未修正）。')
        if hp is None:
            warnings.append('provenance.human_slots 缺失或文件不存在，无法展示人体点云。')
        if item.get('npz_sha256') and item['npz_sha256'] != actual:
            warnings.append('入队哈希与当前 npz 不一致；队列版本已过期。')
        source = item.get('source')
        if not source:
            source = 'omnicontact' if 'omnicontact' in prov.get('source_data', '') else ('lafan_sample' if 'lafan' in prov.get('source_data', '') else prov.get('source_kind', 'unknown'))
        auto = dict(item.get('auto_qa') or {})
        for axis in ('ground', 'object', 'self'):
            f = f'pen_{axis}_m'
            if f in a:
                auto.setdefault(f'max_pen_{axis}_mm', finite_max(a[f])*1000)
        flagged = int(np.count_nonzero(a['flags'])) if 'flags' in a else 0
        auto.setdefault('flagged_frames', flagged)
        auto.setdefault('excluded_frames', None)  # flags do not establish actual training exclusion
        auto.setdefault('contact_mismatch_region_frames', sum(x['mismatch_region_frames'] for x in consistency.values()))
        speeds = np.concatenate([np.asarray(a[x]).reshape(-1) for x in ('ground_slip_speed', 'object_slip_speed') if x in a])
        finite = speeds[np.isfinite(speeds)]
        auto.setdefault('slip_p95_cm_s', float(np.percentile(finite, 95))*100 if len(finite) else None)
        qa = path.with_suffix('.qa.json')
        qa_source = {}
        if qa.exists():
            self._bind(snap, qa)
            try:
                qa_source = json.loads(qa.read_text())
            except ValueError:
                warnings.append('同名 qa.json 无法解析。')
        if str(a.scalar('object_name', '')).lower().startswith('box') and qa_source.get('object', {}).get('kind') == 'sphere':
            warnings.append('源 QA 将 Box 物体分类为 sphere；界面仍渲染真实源网格，现算几何独立于源分类。')
        detail = dict(item, sha256=actual, npz_sha256=actual, queue_npz_sha256=item.get('npz_sha256'),
                      source=source, frames=t, fps=fps, duration_s=t/fps, regions=a['regions'].tolist(),
                      joint_names=a['joint_names'].tolist() if 'joint_names' in a else [],
                      fields=a.files, auto_qa=auto, contact_consistency=consistency, warnings=warnings,
                      floor_height=float(a.scalar('floor_height', 0)), floor_offset_removed=float(a.scalar('floor_offset_removed')),
                      human_scale=float(a.scalar('human_scale', 1)), has_human=hp is not None,
                      has_object='object_pos' in a, object_name=a.scalar('object_name'),
                      provenance=prov, source_qa=qa_source,
                      coordinate_system='z-up，米；校准后地面 z=0；机器人/物体不再次减 floor',
                      source_snapshot={'npz_sha256': actual, 'surface_sha256': surf.sha})
        snap.update(detail=detail, xml=xml, human_path=hp)
        self._check(snap)
        self._snapshots[ident] = snap
        while len(self._snapshots) > 8:
            self._snapshots.popitem(last=False)
        return snap

    @locked
    def catalog(self):
        warnings = self._scan(); keys = list(self._items); items = []
        for key in keys:
            row = dict(self._items[key])
            try:
                d = self._snapshot(key)['detail']
                row.update({k: d[k] for k in ('sha256', 'npz_sha256', 'queue_npz_sha256', 'source', 'frames', 'fps',
                                              'duration_s', 'auto_qa', 'warnings', 'has_object', 'object_name')})
                row['action'] = '踢球' if 'kick' in row.get('clip_id', '') else ('舞蹈' if 'dance' in row.get('clip_id', '') else '搬运/其它')
            except (ValueError, OSError, KeyError) as e:
                row.update(error=str(e), warnings=[str(e)], supported=False)
            items.append(row)
        items.sort(key=lambda x: (-float(x.get('priority', 0)), x['item_id']))
        return {'items': items, 'warnings': warnings, 'formats': {k: k == 'factory_npz_v1' for k in self.adapters}}

    @locked
    def detail(self, key, expected_sha=None):
        snap = self._snapshot(key, expected_sha)
        return dict(snap['detail'])

    def _human_region(self, snap):
        """Reproduce the documented nearest correspondence-face mapping read-only."""
        if 'human_region' in snap:
            return snap['human_region']
        p = snap['detail']['provenance']; r = snap['item']['robot']
        log = Path(p['umr_out']).with_suffix('.umr.log')
        ds_path = sl_path = None
        if log.exists():
            self._bind(snap, log)
            s = log.read_text(errors='replace')
            ds = re.findall(r'correspondence dataset: (\S+\.npz)', s)
            sl = re.findall(r'correspondence slots: (\S+\.npz)', s)
            if ds and sl:
                ds_path, sl_path = Path(ds[-1]), Path(sl[-1])
            elif 'shared correspondence mode' in s:
                # This exact upstream log identifies beta0's shared slot file.
                # Do not infer shared mode solely from a convenient file existing.
                sl = re.findall(r'--slots (\S+/beta0/train/correspondence_slots_final\.npz)', s)
                if sl:
                    sl_path = Path(sl[-1]); ds_path = sl_path.parent.parent / 'correspondence_dataset.npz'
        if ds_path is None:
            tag = hashlib.sha1(f'{p["source_kind"]}:{p["source_data"]}:{p["seq_key"]}'.encode()).hexdigest()[:10] if p['source_kind'] == 'hoi' else 'smplx'
            folder = self.data_root / 'corr' / r / tag
            ds_path, sl_path = folder / 'correspondence_dataset.npz', folder / 'train/correspondence_slots_final.npz'
        segpath = Path(f'{SRCFG.umr_root}/assets/smplx_parts_segm.pkl')
        for path in (ds_path, sl_path, segpath):
            self._bind(snap, path)
        ds = NpyArchive(ds_path, self.cache_root); sl = NpyArchive(sl_path, self.cache_root)
        from scipy.spatial import cKDTree
        target = np.asarray(ds['points'][0]); slots = np.asarray(sl['reconstructed_slots'][0])
        if len(slots) != len(snap['surface']['points_local']):
            raise AdapterError('人体 correspondence 槽位数量无法验证')
        if not np.allclose(sl['target_points'][0], target, atol=1e-5):
            raise AdapterError('人体 correspondence target_points 校验失败')
        _, nearest = cKDTree(target).query(slots)
        with open(segpath, 'rb') as f:
            seg = pickle.load(f, encoding='latin1')['segm']
        parts = seg[np.asarray(ds['sample_face_ids_0'])[nearest]]
        mapping = {1:1,2:2,4:3,5:4,7:5,10:5,8:6,11:6,16:7,17:8,18:9,19:10,20:11,21:12}
        mapping.update({i:11 for i in range(25,40)}); mapping.update({i:12 for i in range(40,55)})
        snap['human_region'] = np.array([mapping.get(int(x), 0) for x in parts], np.uint8)
        return snap['human_region']

    def _object(self, snap):
        if 'object' in snap:
            return snap['object']
        a = snap['archive']
        if 'object_xml' not in a:
            snap['object'] = ([], {}, None)
            return snap['object']
        xml = Path(a.scalar('object_xml'))
        for dep in geometry.xml_dependencies(xml):
            self._bind(snap, dep)
        model = geometry.load_object_model(xml, a.scalar('human_scale', 1.))
        geoms, arrays = geometry.static_scene(model, prefix='object', object_model=True)
        gauge = geometry.object_diagnostic(geoms, arrays)
        snap['object'] = (geoms, arrays, gauge)
        return snap['object']

    @locked
    def scene(self, key, expected_sha):
        if not expected_sha:
            raise SnapshotChanged('必须提供审阅时的 npz sha256')
        snap = self._snapshot(key, expected_sha); a = snap['archive']; s = snap['surface']
        model = self._model(snap, snap['xml'])
        geoms, arrays = geometry.static_scene(model)
        arrays.update(surface_points=np.asarray(s['points_local'], np.float32),
                      surface_normals=np.asarray(s['normals_local'], np.float32),
                      surface_geom=np.asarray(s['geom'], np.uint32), surface_region=np.asarray(s['region'], np.uint8))
        warnings = list(snap['detail']['warnings'])
        try:
            arrays['human_region'] = self._human_region(snap)
        except (ValueError, OSError, KeyError) as e:
            warnings.append(f'人体点区域映射不可验证：{e}；显示原人体点云和原区域曲线，不猜测点区域。')
        og, oa, gauge = self._object(snap); arrays.update(oa)
        for field in REGION_KEYS + FRAME_KEYS:
            if field in a:
                arrays['timeline_' + field] = np.asarray(a[field])
        meta = dict(snap['detail'], geoms=geoms, object_geoms=og,
                    body_parents=model.body_parentid.tolist(), joint_body=model.jnt_bodyid.tolist(),
                    ngeom=model.ngeom, nbody=model.nbody, njnt=model.njnt,
                    surface_point_count=len(s['points_local']), bitorder='big', warnings=warnings,
                    object_scale_applied=float(a.scalar('human_scale', 1.)),
                    object_distance_description=gauge['description'] if gauge else None,
                    human_region_available='human_region' in arrays,
                    source_geometry='compiled MJCF geom-local mesh; geom pose includes compiler transform')
        self._check(snap)
        return meta, arrays

    @locked
    def chunk(self, key, start, count, mode, expected_sha):
        if not expected_sha:
            raise SnapshotChanged('必须提供审阅时的 npz sha256')
        snap = self._snapshot(key, expected_sha); a = snap['archive']; s = snap['surface']; d = snap['detail']
        if mode not in ('raw', 'corrected'):
            raise AdapterError('mode 仅支持 raw 或 corrected')
        if not isinstance(start, int) or not isinstance(count, int) or start < 0 or start >= d['frames'] or count < 1 or count > self.max_chunk:
            raise AdapterError('帧范围无效；每块必须为 1–64 帧')
        stop = min(start+count, d['frames']); count = stop-start
        model = self._model(snap, snap['xml']); q = a['qpos_umr' if mode == 'raw' else 'qpos']
        # One neighbouring frame on either side supplies meaningful UI velocity.
        lo, hi = max(0,start-1), min(d['frames'],stop+1)
        posed = geometry.forward(model, q[lo:hi])
        world = geometry.world_points(posed, s['points_local'], np.asarray(s['geom'], dtype=int))
        local = slice(start-lo, stop-lo)
        arrays = {k: v[local] for k, v in posed.items()}
        arrays['root_pos'] = np.asarray(q[start:stop,:3], np.float32)
        arrays['frame_indices'] = np.arange(start, stop, dtype=np.uint32)
        velocity = np.gradient(world, 1/d['fps'], axis=0) if len(world)>1 else np.zeros_like(world)
        for f in REGION_KEYS + FRAME_KEYS + ('ground_point_contact', 'object_point_contact', 'object_pos', 'object_rot'):
            if f in a:
                arrays[f] = np.asarray(a[f][start:stop])
        _, _, gauge = self._object(snap)
        object_velocity = None
        if 'object_pos' in arrays:
            dist = geometry.object_distance(world[local], arrays['object_pos'], arrays['object_rot'], gauge)
            if dist is not None:
                arrays['ui_object_distance'] = dist
            local_object = np.einsum('cpi,cij->cpj', world-np.asarray(a['object_pos'][lo:hi])[:,None,:], np.asarray(a['object_rot'][lo:hi]))
            ov = np.gradient(local_object, 1/d['fps'], axis=0) if len(local_object)>1 else np.zeros_like(local_object)
            object_velocity = np.asarray(ov[local], np.float32)
        # Arrows exist only on source-labeled slipping contact points. Sending
        # dense velocity for every ordinary point doubled the transfer size.
        arrow_frame, arrow_point, arrow_kind, arrow_velocity = [], [], [], []
        region_ids = np.asarray(s['region'], dtype=int)
        for kind, prefix, v in ((0, 'ground', velocity[local]), (1, 'object', object_velocity)):
            pk, sk = prefix+'_point_contact', prefix+'_slip'
            if v is None or pk not in arrays or sk not in arrays:
                continue
            contact = np.unpackbits(arrays[pk], axis=1, bitorder='big')[:, :len(region_ids)].astype(bool)
            slipping = np.asarray(arrays[sk], bool)[:, region_ids]
            cf, cp = np.nonzero(contact & slipping)
            arrow_frame.append(cf.astype(np.uint32)); arrow_point.append(cp.astype(np.uint32))
            arrow_kind.append(np.full(len(cf), kind, dtype=np.uint8))
            arrow_velocity.append(np.asarray(v[cf, cp], np.float32))
        arrays['ui_arrow_frame'] = np.concatenate(arrow_frame) if arrow_frame else np.empty(0, np.uint32)
        arrays['ui_arrow_point'] = np.concatenate(arrow_point) if arrow_point else np.empty(0, np.uint32)
        arrays['ui_arrow_kind'] = np.concatenate(arrow_kind) if arrow_kind else np.empty(0, np.uint8)
        arrays['ui_arrow_velocity'] = np.concatenate(arrow_velocity) if arrow_velocity else np.empty((0,3), np.float32)
        warnings = list(d['warnings'])
        if mode == 'raw':
            warnings.append('正在展示 UMR 原始骨架；文件接触/间隙/打滑/flags 仍属于严格修正后姿态，不能当作原始标签。')
        if snap['human_path'] is not None:
            if 'human_archive' not in snap:
                snap['human_archive'] = NpyArchive(snap['human_path'], self.cache_root)
            human = snap['human_archive']['source_slots']
            if len(human) != d['frames']:
                # Explicit upstream frame_ids only; never assume stride or silently truncate.
                raw_path = Path(d['provenance']['umr_out']); self._bind(snap, raw_path)
                raw = NpyArchive(raw_path, self.cache_root)
                if 'frame_ids' not in raw or len(raw['frame_ids']) != d['frames']:
                    raise AdapterError('人体与机器人帧数不同且无可验证 frame_ids')
                ids = np.asarray(raw['frame_ids'][start:stop], int)
                if ids.min()<0 or ids.max()>=len(human):
                    raise AdapterError('人体 frame_ids 越界')
                points = np.array(human[ids], dtype=np.float32, copy=True)
            else:
                points = np.array(human[start:stop], dtype=np.float32, copy=True)
            points[:,:,2] -= np.float32(d['floor_offset_removed'])
            arrays['human_points'] = points
        meta = dict(key=key, item_id=d['item_id'], sha256=d['sha256'], npz_sha256=d['sha256'],
                    start=start, count=count, frames=d['frames'], fps=d['fps'], mode=mode,
                    warnings=warnings, bitorder='big', labels_pose='corrected', human_floor_shift_applied=True,
                    object_distance_description=gauge['description'] if gauge else None,
                    diagnostic_velocity='界面现算：原位置有限差分，未平滑；物体速度另为物体局部坐标系。',
                    arrow_contract={'frame':'块内帧索引', 'point':'surface 点索引',
                                    'kind':{'0':'地面，世界坐标速度','1':'物体，物体局部坐标速度'},
                                    'selection':'源 point_contact && 源 region slip；不重写标签'},
                    coordinate_system=d['coordinate_system'])
        self._check(snap)
        return meta, arrays
