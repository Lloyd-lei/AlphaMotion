"""Read-only gpu_npz_v2 adapter (umr_gpu export, DATA_SPEC_v2 layout).

Field semantics follow /home/arenalabs/umr_gpu/umr_gpu/export.py and strict.py
(read, never imported):

* ``qpos`` [T,nq] floor-relative (calibrated ground z=0), ``floor_offset_removed``
  is the raw-world floor; ``object_pose`` [T,7] floor-relative pos + quat wxyz.
* ``contact_ptr/idx/partner``: per-frame CSR of dense surface points with
  clearance in [-0.5, 1] mm; partner 0 = ground, 1 = object.
* ``near_ptr/idx/partner/near_dist_q``: points within 2 cm, d = q*0.1 mm - 0.5 mm.
* region arrays [T,13]; ``flags`` bitmask (strict.py FLAG_*); ``qc_json``.
* Surface ``points_local`` are geom-local (strict.RobotPoints applies geom_quat/
  geom_pos), so world = geom_xmat @ p + geom_xpos, as the factory adapter does.

Human source: the UMR capture ``problems/<clip>/<robot>.stock.cpu.npz`` member
``src`` (the exact 4096 human slots fed to the solver, raw world, scaled),
memory-mapped in place (ZIP_STORED) and shifted by ``floor_offset_removed``.
Object mesh: the capture's ``objpen_mjb`` object geoms (already scaled).
"""
from __future__ import annotations
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml

from collections import OrderedDict
import hashlib
import json
import re
from pathlib import Path
import threading
import zipfile

import mujoco
import numpy as np

from . import geometry, npzhead
from .adapters import AdapterError, NpyArchive, SnapshotChanged, UnsupportedFormat, fingerprint
from .human_smplx import reconstruct_slots

GPU_ROOT = Path(f'{SRCFG.work}')
EXPORT_DIRS = (GPU_ROOT / 'results' / 'export',)
PROBLEMS = GPU_ROOT / 'problems'
PRODUCTION_SUBDIR = Path('production') / 'v1'
FORMAT = 'gpu_npz_v2'
REGIONS = ['trunk', 'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot',
           'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm', 'L_hand', 'R_hand']
# strict.py: FLAG_GROUND_PEN, FLAG_OBJ_PEN, FLAG_SELF_PEN, FLAG_GROUND_CONTACT, FLAG_OBJ_CONTACT = 1, 2, 4, 8, 16;
# FLAG_SLIP = 32; FLAG_LIFT = 64
FLAG_BITS = [
    {'bit': 1, 'name': 'FLAG_GROUND_PEN', 'zh': '地面穿透 > 0.5 mm'},
    {'bit': 2, 'name': 'FLAG_OBJ_PEN', 'zh': '物体穿透 > 0.5 mm'},
    {'bit': 4, 'name': 'FLAG_SELF_PEN', 'zh': '自身穿透（GPU 估计 < +0.5 mm）'},
    {'bit': 8, 'name': 'FLAG_GROUND_CONTACT', 'zh': '要求的地面接触不实（不在 [-0.5,1] mm）'},
    {'bit': 16, 'name': 'FLAG_OBJ_CONTACT', 'zh': '要求的物体接触不实'},
    {'bit': 32, 'name': 'FLAG_SLIP', 'zh': '打滑：锁定接触点偏离 > 0.15 mm 或朝向 > 1.5 mrad'},
    {'bit': 64, 'name': 'FLAG_LIFT', 'zh': '抬离不一致：人体离地 > 2 cm 而机器人 < 5 mm'},
]
Q_STEP_MM, Q_OFF_MM = 0.1, 0.5
REQUIRED = ('qpos', 'fps', 'contact_ptr', 'contact_idx', 'contact_partner', 'near_ptr', 'near_idx',
            'near_partner', 'near_dist_q', 'ground_contact', 'flags', 'qc_json', 'floor_offset_removed')
TIMELINE = ('ground_contact', 'human_ground_contact', 'ground_slip', 'ground_pen_mm', 'ground_clearance_mm',
            'human_ground_clearance_mm', 'object_contact', 'human_object_contact', 'object_pen_mm',
            'object_clearance_mm', 'human_object_clearance_mm', 'ground_slip_speed')
CLIP_SOURCE = (('case', 'omnicontact'), ('omni', 'omnicontact'), ('soccer', 'omnicontact'), ('box', 'omnicontact'),
               ('hiphi', 'hiphi'), ('getup', 'hiphi'), ('lafan', 'lafan'), ('grail', 'grail'),
               ('amass', 'amass'), ('bones', 'bones_seed'), ('mixamo', 'mixamo'))


def guess_source(clip):
    c = clip.lower()
    for k, v in CLIP_SOURCE:
        if c.startswith(k) or f'__{k}' in c or c.startswith(f'qa_v2__{k}'):
            return v
    return 'unknown'


def quat_wxyz_to_mat(q):
    q = np.asarray(q, np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - w * z); R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y); R[..., 2, 1] = 2 * (y * z + w * x); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def known_robots(data_root):
    names = set()
    for root in (GPU_ROOT / 'robots', Path(data_root) / 'robots'):
        if root.exists():
            names.update(p.name for p in root.iterdir() if p.is_dir())
    return sorted(names, key=len, reverse=True)


def split_export_name(stem, robots):
    for r in robots:
        if stem.endswith('__' + r):
            return stem[:-len(r) - 2], r
    if '__' in stem:
        clip, robot = stem.rsplit('__', 1)
        return clip, robot
    return stem, None


def invalid_engines(production_root):
    """Engine ids that the pipeline itself marked invalid under production/v1/_invalid (read-only)."""
    d = Path(production_root) / '_invalid'
    out = {}
    if d.exists():
        for f in d.iterdir():
            for m in re.findall(r'(umr_gpu-[0-9a-f]{6,})', f.name):
                out.setdefault(m, str(d / 'README.txt'))
        readme = d / 'README.txt'
        if readme.exists():
            txt = readme.read_text(errors='replace')
            for m in re.findall(r'(umr_gpu-[0-9a-f]{6,})', txt):
                out[m] = txt.strip().splitlines()[0][:300] if txt.strip() else str(readme)
    return out


def object_from_mjb(mjb_path, geom_ids):
    """Object geoms (body frame) + mesh arrays from a compiled UMR objpen model."""
    m = mujoco.MjModel.from_binary_path(str(mjb_path))
    geoms, arrays = [], {}
    for g in np.asarray(geom_ids, int).reshape(-1):
        t = geometry.GEOM_NAMES.get(int(m.geom_type[g]))
        if t is None:
            raise AdapterError(f'物体 geom 类型不支持：{int(m.geom_type[g])}')
        mesh_key = None
        if t == 'mesh':
            mi = int(m.geom_dataid[g]); mesh_key = f'object_mesh_{mi}'
            va, vn = int(m.mesh_vertadr[mi]), int(m.mesh_vertnum[mi])
            fa, fn = int(m.mesh_faceadr[mi]), int(m.mesh_facenum[mi])
            arrays[mesh_key + '_vertices'] = np.asarray(m.mesh_vert[va:va + vn], np.float32).copy()
            arrays[mesh_key + '_faces'] = np.asarray(m.mesh_face[fa:fa + fn], np.uint32).copy()
        rot = np.zeros(9); mujoco.mju_quat2Mat(rot, m.geom_quat[g])
        rgba = m.geom_rgba[g].tolist()
        geoms.append(dict(id=int(g), name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f'geom_{g}',
                          type=t, mesh=mesh_key, size=m.geom_size[g].tolist(),
                          rgba=[.93, .55, .18, 1.] if rgba[3] == 0 else rgba,
                          local_pos=m.geom_pos[g].tolist(), local_rot=rot.reshape(3, 3).tolist()))
    return geoms, arrays


class GpuService:
    """Catalog + snapshot reader for gpu_npz_v2 files; all compute bounded and CPU-only."""

    def __init__(self, data_root, cache_root, export_dirs=EXPORT_DIRS, problems=PROBLEMS,
                 extra_roots=(Path(f'{SRCFG.factory_root}'),)):
        self.data_root = Path(data_root).resolve()
        self.cache_root = Path(cache_root).resolve()
        if self.cache_root == self.data_root or self.data_root in self.cache_root.parents:
            raise ValueError('缓存必须在独立项目目录，不得写入数据目录')
        self.export_dirs = tuple(Path(d) for d in export_dirs)
        self.production = self.data_root / PRODUCTION_SUBDIR
        self.problems = Path(problems)
        self.allowed = (self.data_root,) + tuple(Path(r).resolve() for r in extra_roots)
        self._lock = threading.RLock()
        self._items = {}
        self._snaps = OrderedDict()
        self._models = OrderedDict()
        self._hash = {}
        self._head_cache = {}
        self.max_chunk = 64
        self.source_resolver = None  # optional: seq_key -> SMPL-X index entries
        self.smplx_model = None      # optional: () -> human_smplx.SmplxModel (production human rebuild)

    # ------------------------------------------------------------------ scan
    def _allowed(self, p):
        p = Path(p).resolve()
        if not any(p == r or r in p.parents for r in self.allowed):
            raise AdapterError(f'路径不在只读 allowlist 内：{p}')
        return p

    def scan(self):
        robots = known_robots(self.data_root)
        rows = {}
        for d in self.export_dirs:
            if not d.exists():
                continue
            for p in sorted(d.glob('*.npz')):
                clip, robot = split_export_name(p.stem, robots)
                rows[str(p)] = dict(clip_id=clip, robot=robot, npz=str(p), pipeline='gpu_export',
                                    source=guess_source(clip), manifest=None)
        if self.production.exists():
            for p in sorted(self.production.glob('*/*/*.npz')):
                if p.name.startswith('.') or p.parent.parent.name.startswith('_') or p.parent.name.startswith('_'):
                    continue  # _assets/_jobs/_logs and atomic-write temp files are not outputs
                source, clip, robot = p.parent.parent.name, p.parent.name, p.stem
                man = p.parent / 'manifest.json'
                rows[str(p)] = dict(clip_id=clip, robot=robot, npz=str(p), pipeline='production_v1', source=source,
                                    manifest=str(man) if man.exists() else None)
        bad = invalid_engines(self.production)
        items = {}
        for path, row in rows.items():
            row['item_id'] = f'gpu/{row["pipeline"]}/{row["clip_id"]}::{row["robot"]}'
            row['key'] = 'g' + hashlib.sha256(row['item_id'].encode()).hexdigest()[:23]
            row['format'] = FORMAT
            try:
                st = Path(path).stat(); row['bytes'] = st.st_size; row['mtime'] = st.st_mtime
                stamp = (st.st_size, st.st_mtime_ns)
                old = self._head_cache.get(path)
                if old and old[0] == stamp:
                    row.update(old[1]); items[row['key']] = row
                    continue
                with zipfile.ZipFile(path) as z:
                    shape, _ = npzhead.member_header(path, 'qpos', z)
                    row['frames'] = int(shape[0])
                    row['fps'] = float(npzhead.small_member(path, 'fps', z))
                    row['duration_s'] = row['frames'] / row['fps']
                    qc = json.loads(str(npzhead.small_member(path, 'qc_json', z)))
                    row['qc'] = qc
                    names = {i.filename[:-4] for i in z.infolist()}
                    row['has_object'] = 'object_pose' in names
                    row['engine'] = None
                    if 'provenance' in names:
                        try:
                            row['engine'] = json.loads(str(npzhead.small_member(path, 'provenance', z))).get('engine')
                        except ValueError:
                            pass
                self._head_cache[path] = (stamp, {k: row[k] for k in ('frames', 'fps', 'duration_s', 'qc', 'has_object', 'engine')})
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                row['error'] = str(exc)
            items[row['key']] = row
        for row in items.values():
            if row.get('engine') in bad:
                row['invalid'] = bad[row['engine']]
        with self._lock:
            self._items = items
        return items

    def catalog(self, q='', source='', pipeline='', offset=0, limit=200):
        items = self.scan()
        clips = {}
        for row in items.values():
            clips.setdefault((row['pipeline'], row['clip_id']), []).append(row)
        facets = {}
        for (pl, clip), rows in clips.items():
            f = facets.setdefault(f"{rows[0]['source']}|{pl}", {'source': rows[0]['source'], 'pipeline': pl, 'clips': 0, 'files': 0})
            f['clips'] += 1; f['files'] += len(rows)
        ql = q.lower()
        keys = sorted(k for k, rows in clips.items() if (not source or rows[0]['source'] == source)
                      and (not pipeline or k[0] == pipeline) and (not ql or ql in k[1].lower()))
        total = len(keys)
        out = []
        for key in keys[offset:offset + limit]:
            clip, rows = key[1], clips[key]
            rows.sort(key=lambda r: r['robot'] or '')
            out.append(dict(clip_id=clip, pipeline=key[0], clip_key=f'{key[0]}|{clip}', source=rows[0]['source'], frames=rows[0].get('frames'),
                            fps=rows[0].get('fps'), duration_s=rows[0].get('duration_s'),
                            has_object=any(r.get('has_object') for r in rows),
                            human_capture=self._problem(clip, rows[0]['robot']) is not None,
                            invalid=any(r.get('invalid') for r in rows),
                            robots=[{k: r.get(k) for k in ('key', 'robot', 'pipeline', 'frames', 'fps', 'qc', 'error', 'npz', 'engine', 'invalid')}
                                    for r in rows]))
        return {'clips': out, 'total': total, 'offset': offset, 'limit': limit, 'facets': list(facets.values()),
                'format': FORMAT, 'export_dirs': [str(d) for d in self.export_dirs],
                'production_root': str(self.production), 'production_exists': self.production.exists(),
                'flag_bits': FLAG_BITS}

    def _problem(self, clip, robot):
        if not robot:
            return None
        p = self.problems / clip / f'{robot}.stock.cpu.npz'
        return p if p.exists() else None

    # ------------------------------------------------------------------ snapshot
    def _sha(self, path):
        stamp = fingerprint(path); key = str(path)
        old = self._hash.get(key)
        if old is None or old[0] != stamp:
            from .adapters import sha256_file
            old = (stamp, sha256_file(path)); self._hash[key] = old
        return old[1]

    def _bind(self, snap, path):
        path = self._allowed(path)
        stamp = fingerprint(path)
        old = snap['companions'].get(str(path))
        if old is not None and old != stamp:
            raise SnapshotChanged('关联资产（MJCF/表面/捕获）已改变，请刷新')
        snap['companions'][str(path)] = stamp
        return path

    def _check(self, snap):
        snap['archive'].check()
        for path, stamp in snap['companions'].items():
            try:
                cur = fingerprint(path)
            except OSError as exc:
                raise SnapshotChanged('关联资产已不可读') from exc
            if cur != stamp:
                raise SnapshotChanged('关联资产已改变，当前快照过期')

    def _model(self, snap, xml):
        deps = geometry.xml_dependencies(xml)
        for p in deps:
            self._bind(snap, p)
        sig = tuple((str(p), fingerprint(p)) for p in deps)
        if sig not in self._models:
            self._models[sig] = mujoco.MjModel.from_xml_path(str(xml))
            while len(self._models) > 6:
                self._models.popitem(last=False)
        self._models.move_to_end(sig)
        return self._models[sig]

    def _open(self, key, expected_sha=None):
        if key not in self._items:
            self.scan()
        if key not in self._items:
            raise KeyError('找不到 GPU 输出条目')
        item = self._items[key]
        path = self._allowed(item['npz'])
        sha = self._sha(path)
        if expected_sha is not None and expected_sha != sha:
            raise SnapshotChanged('npz 哈希已改变；当前快照过期')
        ident = (key, sha)
        if ident in self._snaps:
            snap = self._snaps[ident]; self._check(snap); self._snaps.move_to_end(ident)
            return snap
        a = NpyArchive(path, self.cache_root, sha)
        missing = [k for k in REQUIRED if k not in a]
        if missing:
            raise AdapterError('gpu_npz_v2 缺少字段：' + ', '.join(missing))
        snap = {'archive': a, 'companions': {}, 'item': dict(item), 'sha': sha}
        prov = {}
        if 'provenance' in a:
            try:
                prov = json.loads(str(a.scalar('provenance')))
            except ValueError:
                prov = {}
        xml_s = a.scalar('robot_xml') if 'robot_xml' in a else prov.get('robot_xml')
        surf_s = a.scalar('surface_asset') if 'surface_asset' in a else prov.get('surface_asset')
        if not xml_s or not surf_s:
            raise AdapterError('gpu_npz_v2 缺少 robot_xml / surface_asset（顶层与 provenance 都没有）')
        xml = self._bind(snap, xml_s)
        model = self._model(snap, xml)
        surf_path = self._bind(snap, surf_s)
        surf = NpyArchive(surf_path, self.cache_root)
        q = a['qpos']; T = len(q); n = len(surf['points_local'])
        if q.ndim != 2 or q.shape[1] != model.nq:
            raise AdapterError(f'qpos 维度 {q.shape} 与 MJCF nq={model.nq} 不一致')
        fps = float(a.scalar('fps'))
        if not np.isfinite(fps) or fps <= 0 or T == 0:
            raise AdapterError('帧数或 fps 无效')
        for k in ('contact_ptr', 'near_ptr'):
            if len(a[k]) != T + 1:
                raise AdapterError(f'{k} 长度应为 T+1')
        for k in TIMELINE + ('flags', 'object_pose'):
            if k in a and len(a[k]) != T:
                raise AdapterError(f'{k} 帧数与 qpos 不一致')
        if int(np.asarray(surf['geom']).max()) >= model.ngeom:
            raise AdapterError('surface geom 索引超出 MJCF')
        if len(a['contact_idx']) and int(np.max(a['contact_idx'])) >= n:
            raise AdapterError('contact_idx 超出表面点数')
        if len(a['near_idx']) and int(np.max(a['near_idx'])) >= n:
            raise AdapterError('near_idx 超出表面点数')
        regions = [str(x) for x in np.asarray(surf['regions']).tolist()] if 'regions' in surf else REGIONS
        qc = json.loads(str(a.scalar('qc_json')))
        flags = np.asarray(a['flags'])
        warnings = ([f'⚠ 此文件的引擎版本 {item.get("engine")} 已被流水线自己标记为作废（production/v1/_invalid）：{item["invalid"]}']
                    if item.get('invalid') else []) + ['标签与 qc 为文件原值；界面只做显示与反量化，不改写源数据。',
                    'qpos 与 object_pose 已是地面系（z=0 为校准地面），前端不再减地面。']
        if abs(fps - 30.) > .01:
            warnings.append(f'文件 fps={fps:.5f}，与 DATA_SPEC_v2 的 30 Hz 不同；按实际 fps 播放。')
        if 'object_slip' not in a and 'object_pose' in a:
            warnings.append('export.py 文档列出 object_slip，但文件没有该字段（代码只写 ground_slip）。')
        if 'human_ground_contact' in a:
            warnings.append('人体区域接触标签来自 umr_gpu 人体标注（labels.CFG：进入 < 1 cm / 离开 > 2 cm 滞回），'
                            '与机器人区域标签的 1/2 mm 尺子不同；界面人体逐点接触为现算。')
        # Region label vs point consistency (reported, not repaired)
        ptr = np.asarray(a['contact_ptr']); idx = np.asarray(a['contact_idx']).astype(np.int64)
        part = np.asarray(a['contact_partner']); reg = np.asarray(surf['region']).astype(np.int64)
        fr = np.repeat(np.arange(T), np.diff(ptr))
        hit = np.zeros((T, len(regions)), bool)
        g = part == 0
        hit[fr[g], reg[idx[g]]] = True
        lab = np.asarray(a['ground_contact'], bool)
        consistency = {'ground': dict(labeled_without_point=int((lab & ~hit).sum()),
                                      points_without_label=int((hit & ~lab).sum()),
                                      rule='区域标签（1/2 mm 滞回、≥3 帧）对比该区域是否有 [-0.5,1] mm 接触点；滞回与最短段规则会造成合理差异')}
        if 'object_contact' in a:
            hit_o = np.zeros_like(hit); o = part == 1
            hit_o[fr[o], reg[idx[o]]] = True
            lo = np.asarray(a['object_contact'], bool)
            consistency['object'] = dict(labeled_without_point=int((lo & ~hit_o).sum()),
                                         points_without_label=int((hit_o & ~lo).sum()))
        root = np.asarray(a['qpos'][:, :3], np.float64)
        jump = np.zeros(T, np.float32)
        if T > 1:
            jump[1:] = np.linalg.norm(np.diff(root, axis=0), axis=1) * 100
        n_jump = int((jump > 10).sum())
        root_motion = dict(max_cm_per_frame=float(jump.max()), p99_cm_per_frame=float(np.percentile(jump, 99)),
                           frames_over_10cm=n_jump, frames_over_10cm_unflagged=int(((jump > 10) & (flags == 0)).sum()),
                           rule='界面现算：根位置相邻帧位移（cm/帧）；人体全程移动通常 ≤ 2–5 cm/帧')
        if n_jump:
            warnings.append(f'根位置逐帧跳变（界面现算）：{n_jump} 帧 > 10 cm/帧，最大 {jump.max():.1f} cm/帧'
                            f'（{int(((jump > 10) & (flags == 0)).sum())} 帧 flags=0 未被标记）。')
        speed = np.asarray(a['ground_slip_speed'], np.float32) if 'ground_slip_speed' in a else None
        if speed is not None and np.isfinite(speed).any() and float(np.nanmax(speed)) > 1.0:
            warnings.append(f'ground_slip_speed 最大 {float(np.nanmax(speed)):.2f} m/s（> 1 m/s，疑似分块边界或跳变，未修改）。')
        problem = self._problem(item['clip_id'], item['robot'])
        human = None
        if problem is not None:
            self._bind(snap, problem)
            try:
                shape, _ = npzhead.member_header(problem, 'src')
                if shape[0] != T or shape[1] != n:
                    raise AdapterError(f'捕获 src 形状 {shape} 与输出 T={T}/点数={n} 不一致')
                with zipfile.ZipFile(problem) as z:
                    part_ids = npzhead.small_member(problem, 'part_ids', z)
                    args = json.loads(str(npzhead.small_member(problem, 'args_json', z)))
                    names = {i.filename[:-4] for i in z.infolist()}
                    og = npzhead.small_member(problem, 'objpen_object_geoms', z) if 'objpen_object_geoms' in names else None
                    mjb = str(npzhead.small_member(problem, 'objpen_mjb', z)) if 'objpen_mjb' in names else None
                from .human_smplx import part_to_region
                human = dict(path=str(problem), region=part_to_region(part_ids), args=args, object_geoms=og, mjb=mjb)
            except (OSError, ValueError, KeyError) as exc:
                warnings.append(f'人体捕获不可用：{exc}')
                human = None
        elif prov.get('source_dir') and prov.get('static_prep') and 'frame_ids_in_source' in a and 'human_scale' in a:
            try:
                static = self._bind(snap, prov['static_prep'])
                src_dir = Path(prov['source_dir'])
                motion_files = []
                from .human_smplx import SlotBinding, load_motion_dir
                binding = SlotBinding(static)
                motion = load_motion_dir(src_dir)
                for f in motion['files']:
                    self._bind(snap, f)
                fids = np.asarray(a['frame_ids_in_source'], np.int64)
                if len(fids) != T or fids.max() >= len(motion['poses']):
                    raise AdapterError('frame_ids_in_source 与源动作帧数不一致')
                if len(binding.face_ids) != n:
                    raise AdapterError('静态准备的人体槽位数与机器人表面点数不同')
                human = dict(mode='smplx', binding=binding, motion=motion, frame_ids=fids,
                             scale=float(a.scalar('human_scale')), region=binding.region, path=str(static),
                             args={'data': str(src_dir), 'seq_key': prov.get('clip_id')}, object_geoms=None, mjb=None)
                warnings.append('量产文件没有捕获的人体槽位：人体源面板由后端 CPU 按 provenance（源目录 + frame_ids_in_source + human_scale'
                                ' + 静态准备的槽位绑定）现算 SMPL-X 重建，并与文件 human_ground_clearance_mm 逐块核对。')
            except (OSError, ValueError, KeyError) as exc:
                warnings.append(f'人体源重建不可用：{exc}')
                human = None
        else:
            warnings.append('没有对应的 UMR 捕获（problems/<clip>/<robot>.stock.cpu.npz）：人体源面板与物体网格不可显示。')
        source_ref = None
        if human:
            args = human['args']
            data = Path(args.get('data', ''))
            seq = args.get('seq_key')
            cand = data / f'{seq}.npz' if data.is_dir() and seq and (data / f'{seq}.npz').exists() else data
            source_ref = dict(data=str(data), seq_key=seq, resolved=str(cand) if cand.exists() else None,
                              smpl_scale_robot_height=args.get('robot_height'), stride=args.get('stride'))
            if self.source_resolver is not None and seq:
                try:
                    source_ref['factory_matches'] = [dict(id=m['id'], path=m['path'], source=m['source'], variant=m['variant'])
                                                     for m in self.source_resolver(seq, str(cand))]
                except Exception as exc:  # resolver is advisory only
                    source_ref['factory_matches_error'] = str(exc)
        detail = dict(item, key=key, sha256=sha, npz_sha256=sha, format=FORMAT, frames=T, fps=fps, duration_s=T / fps,
                      regions=regions, qc=qc, flag_bits=FLAG_BITS,
                      flagged_frames=int(np.count_nonzero(flags)),
                      flag_counts={str(b['bit']): int(((flags & b['bit']) > 0).sum()) for b in FLAG_BITS},
                      contact_points_total=int(len(idx)), near_points_total=int(len(a['near_idx'])),
                      floor_offset_removed=float(a.scalar('floor_offset_removed')),
                      floor_height=float(a.scalar('floor_height', 0.)), fields=sorted(a.files),
                      has_object='object_pose' in a, has_human=human is not None, source_ref=source_ref,
                      consistency=consistency, root_motion=root_motion, warnings=warnings, robot_xml=str(xml),
                      surface_asset=str(surf_path),
                      near_dist_rule=f'd_mm = q × {Q_STEP_MM} − {Q_OFF_MM}（export.py Q_STEP/Q_OFF）；q ≤ 205 ⇒ d < 20 mm',
                      contact_rule='contact_idx：离地/离物体 [-0.5, 1] mm 的稠密点；partner 0=地面 1=物体',
                      coordinate_system='z-up，米；校准地面 z=0')
        snap.update(detail=detail, xml=xml, surface=surf, human=human, model_nq=model.nq, root_jump_cm=jump)
        self._check(snap)
        self._snaps[ident] = snap
        while len(self._snaps) > 12:
            self._snaps.popitem(last=False)
        return snap

    def _locked(fn):
        def call(self, *a, **k):
            with self._lock:
                return fn(self, *a, **k)
        call.__name__ = fn.__name__
        return call

    @_locked
    def detail(self, key, expected_sha=None):
        return dict(self._open(key, expected_sha)['detail'])

    @_locked
    def scene(self, key, expected_sha):
        if not expected_sha:
            raise SnapshotChanged('必须提供 npz sha256')
        snap = self._open(key, expected_sha); a = snap['archive']; s = snap['surface']
        model = self._model(snap, snap['xml'])
        geoms, arrays = geometry.static_scene(model)
        arrays.update(surface_points=np.asarray(s['points_local'], np.float32),
                      surface_normals=np.asarray(s['normals_local'], np.float32),
                      surface_geom=np.asarray(s['geom'], np.uint32),
                      surface_region=np.asarray(s['region'], np.uint8))
        for k in TIMELINE:
            if k in a:
                v = np.asarray(a[k])
                arrays['timeline_' + k] = v.astype(np.uint8) if v.dtype == bool else v.astype(np.float32)
        arrays['timeline_flags'] = np.asarray(a['flags'], np.int32)
        arrays['timeline_root_jump_cm'] = snap['root_jump_cm']
        object_geoms = []
        h = snap['human']
        warnings = list(snap['detail']['warnings'])
        if h is not None:
            arrays['human_region'] = h['region']
            if 'object_pose' in a and h.get('mjb') and h.get('object_geoms') is not None:
                try:
                    mjb = self._bind(snap, h['mjb'])
                    object_geoms, oa = object_from_mjb(mjb, h['object_geoms'])
                    arrays.update(oa)
                except (OSError, ValueError) as exc:
                    warnings.append(f'物体网格不可读：{exc}')
        if not object_geoms and 'object_pose' in a and 'object_xml' in a:
            try:
                oxml = self._bind(snap, a.scalar('object_xml'))
                for dep in geometry.xml_dependencies(oxml):
                    self._bind(snap, dep)
                om = geometry.load_object_model(oxml, float(a.scalar('object_scale', 1.)))
                object_geoms, oa = geometry.static_scene(om, prefix='object', object_model=True)
                arrays.update(oa)
            except (OSError, ValueError) as exc:
                warnings.append(f'物体网格不可读：{exc}')
        if not object_geoms and 'object_pose' in a:
            warnings.append('有 object_pose 但找不到物体网格来源；只能显示物体位姿。')
        meta = dict(snap['detail'], geoms=geoms, object_geoms=object_geoms,
                    body_parents=model.body_parentid.tolist(), nbody=model.nbody, ngeom=model.ngeom,
                    body_names=[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or f'body_{i}' for i in range(model.nbody)],
                    surface_point_count=len(s['points_local']), human_point_count=len(h['region']) if h else 0,
                    warnings=warnings, source_geometry='compiled MJCF geom-local mesh; FK on CPU (MuJoCo)')
        self._check(snap)
        return meta, arrays

    @_locked
    def chunk(self, key, start, count, expected_sha, human=True):
        if not expected_sha:
            raise SnapshotChanged('必须提供 npz sha256')
        snap = self._open(key, expected_sha); a = snap['archive']; d = snap['detail']
        T = d['frames']
        if not isinstance(start, int) or not isinstance(count, int) or start < 0 or start >= T or count < 1 or count > self.max_chunk:
            raise AdapterError('帧范围无效；每块 1–64 帧')
        stop = min(T, start + count)
        model = self._model(snap, snap['xml'])
        posed = geometry.forward(model, a['qpos'][start:stop])
        arrays = {'geom_pos': posed['geom_pos'], 'geom_rot': posed['geom_rot'], 'body_pos': posed['body_pos'],
                  'frame_indices': np.arange(start, stop, dtype=np.uint32),
                  'root_pos': np.asarray(a['qpos'][start:stop, :3], np.float32),
                  'flags': np.asarray(a['flags'][start:stop], np.int32)}
        for pref in ('contact', 'near'):
            ptr = np.asarray(a[pref + '_ptr'][start:stop + 1], np.int64)
            lo, hi = int(ptr[0]), int(ptr[-1])
            arrays[pref + '_ptr'] = (ptr - lo).astype(np.uint32)
            arrays[pref + '_idx'] = np.asarray(a[pref + '_idx'][lo:hi], np.uint32)
            arrays[pref + '_partner'] = np.asarray(a[pref + '_partner'][lo:hi], np.uint8)
            if pref == 'near':
                arrays['near_dist_q'] = np.asarray(a['near_dist_q'][lo:hi], np.uint8)
        if 'object_pose' in a:
            op = np.asarray(a['object_pose'][start:stop], np.float64)
            arrays['object_pos'] = op[:, :3].astype(np.float32)
            arrays['object_rot'] = quat_wxyz_to_mat(op[:, 3:7]).astype(np.float32)
        h = snap['human']
        human_check = None
        if h is not None and human:
            if h.get('mode') == 'smplx':
                if self.smplx_model is None:
                    raise AdapterError('SMPL-X 模型未接入，不能重建量产人体源')
                pts = reconstruct_slots(self.smplx_model(), h['binding'], h['motion'], h['frame_ids'][start:stop],
                                        h['scale'], d['floor_offset_removed']).astype(np.float32)
            else:
                src = npzhead.stored_memmap(h['path'], 'src')
                pts = np.array(src[start:stop], np.float32, copy=True)
                pts[..., 2] -= np.float32(d['floor_offset_removed'])
            arrays['human_points'] = pts
            if 'human_ground_clearance_mm' in a:
                reg = np.asarray(h['region'], np.int64)
                zr = np.full((stop - start, len(d['regions'])), np.inf)
                for r in range(len(d['regions'])):
                    k = reg == r
                    if k.any():
                        zr[:, r] = pts[:, k, 2].min(1) * 1000
                ref = np.asarray(a['human_ground_clearance_mm'][start:stop], np.float64)
                near = np.isfinite(zr) & np.isfinite(ref) & (np.abs(ref) < 100)
                human_check = float(np.abs(zr - ref)[near].max()) if near.any() else None
        meta = dict(key=key, sha256=d['sha256'], start=start, count=stop - start, frames=T, fps=d['fps'],
                    near_dist_rule=d['near_dist_rule'], contact_rule=d['contact_rule'],
                    human_floor_shift_applied=h is not None and human, coordinate_system=d['coordinate_system'],
                    human_mode=(h.get('mode', 'capture') if h else None),
                    human_check_max_abs_mm=human_check,
                    human_check_rule='人体点逐区最低点高度 vs 文件 human_ground_clearance_mm（|值|<100 mm 的区，float16）')
        self._check(snap)
        return meta, arrays


class GpuNpzAdapter:
    """FactoryService hook for queue items in gpu_npz_v2: viewing lives in the GPU page."""
    format = FORMAT

    def open(self, service=None, key=None, expected_sha=None):
        raise UnsupportedFormat('gpu_npz_v2 已实现只读查看器：请在「GPU 输出」页查看；审阅页判决暂只支持 factory_npz_v1')
