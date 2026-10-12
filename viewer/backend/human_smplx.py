"""Read-only SMPL-X clip index + CPU (numpy) SMPL-X forward for the raw-human browser.

* Index: header-only reads (poses shape, fps); results cached in the project
  cache keyed by (path, size, mtime_ns). Never expands whole archives.
* Forward: numpy LBS with the same conventions UMR uses (smplx.SMPLX with
  use_pca=False, flat_hand_mean=True, num_betas=10; hands/jaw/eyes zeroed as
  ``zero_source_finger_pose``), y-up sources rotated with UMR's
  Y_UP_TO_Z_UP_MATRIX. Chunks are capped at 64 frames.
* Surface points: the 4096 human slots of the UMR correspondence (face id +
  barycentric weights on the SMPL-X template) read from a captured UMR problem,
  i.e. the exact sampling rule of umr_gpu's human slots; region = part_ids via
  the smplx_55_to_19_upper_arm_v1 -> 13-region table (umr_gpu/labels.py).
* Floor: calibrated per clip with umr_gpu's ``floor_from_points`` rule on the
  foot-region slots (computed here, labelled as UI-computed).
"""
from __future__ import annotations

from collections import OrderedDict
import csv
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import zipfile

import numpy as np

from . import npzhead

DATA_ROOT = Path('/home/arenalabs/umr_factory_data')
SMPLX_MODEL = Path('/var/tmp/alphamotion/umr_factory/umr/UMR/smpl/SMPLX_NEUTRAL.npz')
PROBLEMS = Path('/var/tmp/alphamotion/umr_factory/umr_gpu/problems')
SLOT_PROBLEM_PREFERENCE = ('lafan1_smplx__dance1_subject2/unitree_g1_29dof.stock.cpu.npz',
                           'hiphi_getup__getup0004_neutral_30fps/unitree_g1_29dof.stock.cpu.npz')
REGIONS = ['trunk', 'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot',
           'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm', 'L_hand', 'R_hand']
RID = {r: i for i, r in enumerate(REGIONS)}
# umr_gpu/labels.py SEG19_REGION (UMR body-segment schema smplx_55_to_19_upper_arm_v1)
SEG19_REGION = {1: 'R_hand', 2: 'R_thigh', 3: 'L_upper_arm', 4: 'trunk', 5: 'L_shin', 6: 'L_foot', 7: 'trunk',
                8: 'R_foot', 9: 'R_upper_arm', 10: 'L_hand', 11: 'R_shin', 12: 'L_forearm', 13: 'R_forearm',
                14: 'L_thigh', 15: 'trunk', 16: 'trunk', 17: 'trunk', 18: 'L_upper_arm', 19: 'R_upper_arm'}
Y_UP_TO_Z_UP = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], np.float64)
BODY_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
MAX_CHUNK = 64
INDEX_VERSION = 3


def part_to_region(part_ids):
    return np.array([RID[SEG19_REGION.get(int(p), 'trunk')] for p in part_ids], np.uint8)


def _json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _scalar(v):
    a = np.asarray(v)
    return a.item() if a.ndim == 0 else a.reshape(-1)[0].item()


# --------------------------------------------------------------------------- index
class SmplxIndex:
    """All SMPL-X clips on disk (HiPHI CPU/GPU/30 fps, OmniContact) + GRAIL manifest rows."""

    def __init__(self, data_root=DATA_ROOT, cache_root=None):
        self.data_root = Path(data_root)
        self.cache_file = Path(cache_root) / 'smplx_index.json' if cache_root else None
        self._lock = threading.Lock()
        self._scan_lock = threading.Lock()
        self._cache = {}
        if self.cache_file and self.cache_file.exists():
            try:
                saved = json.loads(self.cache_file.read_text())
                if saved.get('version') == INDEX_VERSION:
                    self._cache = saved.get('files', {})
            except ValueError:
                self._cache = {}
        self.entries = []
        self.by_id = {}
        self.scanned_at = None
        self.scan_seconds = None

    def _stamp(self, p):
        s = p.stat()
        return [s.st_size, s.st_mtime_ns]

    def _npz_entry(self, p, source, variant, root):
        key = str(p); stamp = self._stamp(p)
        old = self._cache.get(key)
        if old and old.get('stamp') == stamp:
            return old['entry']
        rel = p.relative_to(root)
        parts = rel.parts  # data/<Frame>/<lu>/<motion>/motion_actor_smplx.npz
        clip = parts[-2] if len(parts) >= 2 else p.stem
        category = '/'.join(parts[1:3]) if len(parts) >= 5 and parts[0] == 'data' else '/'.join(parts[:-2])
        e = dict(source=source, variant=variant, category=category, clip=clip, path=str(p), format='smplx_npz')
        try:
            with zipfile.ZipFile(p) as z:
                names = {i.filename[:-4] for i in z.infolist()}
                shape, _ = npzhead.member_header(p, 'poses', z)
                fps_name = 'fps' if 'fps' in names else 'mocap_framerate'
                fps = float(_scalar(npzhead.small_member(p, fps_name, z))) if fps_name in names else None
                gender = str(_scalar(npzhead.small_member(p, 'gender', z))) if 'gender' in names else None
                up = str(_scalar(npzhead.small_member(p, 'output_up', z))) if 'output_up' in names else (
                    str(_scalar(npzhead.small_member(p, 'coordinate_up', z))) if 'coordinate_up' in names else 'z')
            e.update(frames=int(shape[0]), pose_dim=int(np.prod(shape[1:])), fps=fps, gender=gender, up=up,
                     has_residual_series='per_frame_residuals' in names)
            if fps:
                e['duration_s'] = shape[0] / fps
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            e['error'] = f'头信息不可读：{exc}'
        qa = _json(p.parent / 'qa.json') or {}
        r = qa.get('ik_residual_cm') or {}
        e.update(residual_cm_mean=r.get('mean'), residual_cm_p95=r.get('p95'), residual_cm_max=r.get('max'),
                 is_hoi=qa.get('is_hoi'), mirrored=qa.get('mirrored'), actor=qa.get('actor_id'),
                 objects=[o.get('object_id') for o in qa.get('objects', []) if isinstance(o, dict)],
                 converter_problems=qa.get('problems'))
        self._cache[key] = {'stamp': stamp, 'entry': e}
        return e

    def _omni_entry(self, d):
        poses = d / 'poses.npy'
        key = str(poses); stamp = self._stamp(poses)
        old = self._cache.get(key)
        if old and old.get('stamp') == stamp:
            return old['entry']
        rel = d.relative_to(self.data_root / 'smplx' / 'omnicontact')
        e = dict(source='omnicontact', variant='cpu', category='/'.join(rel.parts[:2]), clip=rel.parts[-1],
                 path=str(d), format='smplx_npy_dir')
        try:
            with open(poses, 'rb') as f:
                shape, _, _ = npzhead._read_header(f)
            fps = float(_scalar(np.load(d / 'mocap_framerate.npy', allow_pickle=False)))
            gender = str(_scalar(np.load(d / 'gender.npy', allow_pickle=False))) if (d / 'gender.npy').exists() else None
            up = str(_scalar(np.load(d / 'output_up.npy', allow_pickle=False))) if (d / 'output_up.npy').exists() else 'z'
            e.update(frames=int(shape[0]), pose_dim=int(np.prod(shape[1:])), fps=fps, gender=gender, up=up,
                     duration_s=shape[0] / fps, has_residual_series=(d / 'per_frame_residuals.npy').exists())
        except (OSError, ValueError) as exc:
            e['error'] = f'头信息不可读：{exc}'
        qa = _json(d / 'qa.json') or {}
        r = qa.get('ik_residual_cm') or {}
        e.update(residual_cm_mean=r.get('mean'), residual_cm_p95=r.get('p95'), residual_cm_max=r.get('max'),
                 is_hoi=True, mirrored=False, actor=None,
                 objects=[o.get('object_id') for o in qa.get('objects', []) if isinstance(o, dict)],
                 converter_problems=qa.get('problems'))
        self._cache[key] = {'stamp': stamp, 'entry': e}
        return e

    def _grail_entries(self):
        man = self.data_root / 'smplx' / 'grail' / 'manifest.jsonl'
        out = []
        if not man.exists():
            return out
        key = str(man); stamp = self._stamp(man)
        old = self._cache.get(key)
        if old and old.get('stamp') == stamp:
            return old['entries']
        for line in man.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get('status') != 'ok':
                continue
            qa = r.get('qa') or {}
            out.append(dict(source='grail', variant='cpu', category=r.get('category'), clip=r.get('clip'),
                            path=(r.get('paths') or {}).get('recon'), format='grail_joblib_pkl',
                            frames=r.get('frames'), fps=r.get('fps'), duration_s=r.get('seconds'),
                            residual_cm_mean=None, residual_cm_p95=None, src_valid_frac=r.get('src_valid_frac'),
                            src_pen_gt10mm_frac=qa.get('src_pen_gt10mm_frac'), flags_text=r.get('flags'),
                            is_hoi=True, objects=[(r.get('object') or {}).get('name')], gender='neutral', up='z',
                            viewer_unsupported='GRAIL recon 为 joblib pickle；后端按安全规则不反序列化 pickle，暂只列清单'))
        self._cache[key] = {'stamp': stamp, 'entries': out}
        return out

    def scan(self):
        with self._scan_lock:
            return self._scan()

    def ensure(self, max_age=300):
        if self.scanned_at is None or time.time() - self.scanned_at > max_age:
            self.scan()

    def _scan(self):
        t0 = time.time()
        entries = []
        roots = [(self.data_root / 'smplx' / 'hiphi', 'hiphi', 'cpu'),
                 (self.data_root / 'smplx_gpu' / 'hiphi', 'hiphi', 'gpu'),
                 (self.data_root / 'smplx_gpu' / 'hiphi_30fps', 'hiphi', 'gpu_30fps')]
        seen = set()
        for root, source, variant in roots:
            if not root.exists():
                continue
            for p in sorted(root.glob('data/*/*/*/motion_actor_smplx.npz')):
                seen.add(str(p))
                entries.append(self._npz_entry(p, source, variant, root))
        omni = self.data_root / 'smplx' / 'omnicontact'
        if omni.exists():
            for d in sorted(omni.glob('*/*/*')):
                if (d / 'poses.npy').exists():
                    seen.add(str(d / 'poses.npy'))
                    entries.append(self._omni_entry(d))
        entries.extend(self._grail_entries())
        for e in entries:
            e['id'] = hashlib.sha1(f"{e['source']}:{e['variant']}:{e['path']}:{e['clip']}".encode()).hexdigest()[:16]
            e.setdefault('viewer_unsupported', e.get('error'))
        # HiPHI 30 fps files are master[::3] variants; mark for de-duplicated hour totals.
        with self._lock:
            self.entries = entries
            self.by_id = {e['id']: e for e in entries}
            self.scanned_at = time.time()
            self.scan_seconds = round(time.time() - t0, 2)
        if self.cache_file:
            keep = {k: v for k, v in self._cache.items() if k in seen or 'entries' in v}
            tmp = self.cache_file.with_suffix('.tmp')
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps({'version': INDEX_VERSION, 'files': keep}))
            tmp.replace(self.cache_file)
        return entries

    def get(self, clip_id):
        with self._lock:
            if clip_id not in self.by_id:
                raise KeyError('找不到人体片段')
            return dict(self.by_id[clip_id])

    def find_by_clip_name(self, name, path=None):
        with self._lock:
            return [dict(e) for e in self.entries if e.get('clip') == name or (path and e.get('path') == path)]


# --------------------------------------------------------------------------- model
def rodrigues(aa):
    """aa [...,3] -> rotation matrices [...,3,3] (float64)."""
    aa = np.asarray(aa, np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    small = theta < 1e-8
    k = np.where(small, 0., aa / np.where(small, 1., theta))
    c = np.cos(theta)[..., None]; s = np.sin(theta)[..., None]
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] = k[..., 1]
    K[..., 1, 0] = k[..., 2]; K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] = k[..., 0]
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + s * K + (1 - c) * (K @ K)


class SmplxModel:
    """Numpy SMPL-X (neutral) restricted to body joints 0..21 (others identity)."""

    def __init__(self, path=SMPLX_MODEL):
        path = Path(path)
        with zipfile.ZipFile(path) as z:
            ld = lambda n: npzhead.small_member(path, n, z, max_bytes=512 * 1024 * 1024)
            self.v_template = ld('v_template').astype(np.float64)
            self.shapedirs = ld('shapedirs')[:, :, :10].astype(np.float64)
            # Only joints 1..21 can rotate (jaw/eyes/hands zero) -> first 21*9 pose features.
            self.posedirs = np.ascontiguousarray(ld('posedirs')[:, :, :189], np.float32)
            self.J_regressor = ld('J_regressor').astype(np.float64)
            self.weights = ld('weights').astype(np.float64)
            kin = ld('kintree_table')
            self.faces = ld('f').astype(np.int64)
        parents = kin[0].astype(np.int64); parents[0] = -1
        self.parents = parents
        assert list(parents[:22]) == BODY_PARENTS, 'SMPL-X kintree mismatch'
        # Joints 22..54 (jaw, eyes, fingers) never rotate here, so their skinning
        # weight folds into the nearest body ancestor: exact, and 55 -> 22 joints.
        w22 = self.weights[:, :22].copy()
        for i in range(22, len(parents)):
            a = i
            while a >= 22:
                a = int(parents[a])
            w22[:, a] += self.weights[:, i]
        self.weights22 = w22
        self.path = str(path)

    def shaped(self, betas):
        b = np.zeros(10); bb = np.asarray(betas, np.float64).reshape(-1)[:10]; b[:len(bb)] = bb
        v = self.v_template + np.einsum('vck,k->vc', self.shapedirs, b)
        return v, self.J_regressor @ v

    def forward(self, v_shaped, J, poses66, transl, vert_idx):
        """poses66 [C,66] axis-angle, transl [C,3] -> verts [C,len(vert_idx),3], joints [C,22,3]."""
        C = len(poses66)
        R = rodrigues(np.asarray(poses66, np.float64).reshape(C, 22, 3))           # [C,22,3,3]
        pf = (R[:, 1:] - np.eye(3)).reshape(C, 189).astype(np.float32)
        pd = self.posedirs[vert_idx].reshape(-1, 189)                              # [n*3,189]
        vp = v_shaped[vert_idx][None] + (pf @ pd.T).reshape(C, len(vert_idx), 3)
        G = np.zeros((C, 22, 4, 4)); G[:, :, 3, 3] = 1
        G[:, 0, :3, :3] = R[:, 0]; G[:, 0, :3, 3] = J[0]
        for i in range(1, 22):
            p = BODY_PARENTS[i]
            local = np.zeros((C, 4, 4)); local[:, :3, :3] = R[:, i]; local[:, :3, 3] = J[i] - J[p]; local[:, 3, 3] = 1
            G[:, i] = G[:, p] @ local
        joints = G[:, :, :3, 3].copy()
        A = G.copy()
        A[:, :, :3, 3] -= np.einsum('cjab,jb->cja', G[:, :, :3, :3], J[:22])
        T = np.matmul(self.weights22[vert_idx], A[:, :, :3, :].reshape(C, 22, 12)).reshape(C, len(vert_idx), 3, 4)
        verts = np.einsum('cvab,cvb->cva', T[..., :3], vp) + T[..., 3]
        t = np.asarray(transl, np.float64)[:, None, :]
        return verts + t, joints + t


class HumanSlots:
    """UMR human slot binding (4096 points) from a captured problem file."""

    def __init__(self, problems=PROBLEMS):
        last = None
        for rel in SLOT_PROBLEM_PREFERENCE:
            p = Path(problems) / rel
            if p.exists():
                break
        else:
            cands = sorted(Path(problems).glob('*/*.stock.cpu.npz'))
            if not cands:
                raise FileNotFoundError('找不到 UMR 捕获的人体槽位（problems/*/*.stock.cpu.npz）')
            p = cands[0]
        with zipfile.ZipFile(p) as z:
            self.face_ids = npzhead.small_member(p, 'x_bind_face_ids', z).astype(np.int64)
            self.bary = npzhead.small_member(p, 'x_bind_bary', z).astype(np.float64)
            self.part_ids = npzhead.small_member(p, 'part_ids', z).astype(np.int64)
        self.region = part_to_region(self.part_ids)
        self.source = str(p)
        self.robot = p.name.split('.')[0]


# --------------------------------------------------------------------------- clip service
def floor_from_points(X, reg, fps, static_speed=0.05, low_tol=0.005):
    """umr_gpu/labels.py floor_from_points (feet = L_foot, R_foot), re-implemented read-only."""
    zmin_all = X[..., 2].min(1)
    samples = []
    for r in (RID['L_foot'], RID['R_foot']):
        idx = np.flatnonzero(reg == r)
        if not len(idx):
            continue
        V = np.gradient(X[:, idx], 1.0 / fps, axis=0) if len(X) > 1 else np.zeros_like(X[:, idx])
        sp = np.median(np.linalg.norm(V, axis=-1), axis=1)
        fmin = X[:, idx, 2].min(1)
        ok = (sp < static_speed) & (fmin <= zmin_all + low_tol)
        samples.append(fmin[ok])
    s = np.concatenate(samples) if samples else np.zeros(0)
    if len(s) >= 10:
        return float(np.median(s)), int(len(s)), '静止支撑帧最低脚点中位数（umr_gpu floor_from_points 规则）'
    allfoot = np.concatenate([X[:, np.flatnonzero(reg == r), 2].min(1) for r in (RID['L_foot'], RID['R_foot'])])
    return float(np.percentile(allfoot, 5)), int(len(s)), '退化：最低脚点 5 分位（静止帧不足 10）'


class HumanService:
    def __init__(self, data_root=DATA_ROOT, cache_root=None):
        self.data_root = Path(data_root)
        self.index = SmplxIndex(data_root, cache_root)
        self._model = None
        self._slots = None
        self._clips = OrderedDict()
        self._lock = threading.RLock()

    @property
    def model(self):
        if self._model is None:
            self._model = SmplxModel()
        return self._model

    @property
    def slots(self):
        if self._slots is None:
            self._slots = HumanSlots()
        return self._slots

    def _motion(self, e):
        p = Path(e['path'])
        if e['format'] == 'smplx_npz':
            with zipfile.ZipFile(p) as z:
                names = {i.filename[:-4] for i in z.infolist()}
                poses = npzhead.small_member(p, 'poses', z, max_bytes=256 * 1024 * 1024)
                trans = npzhead.small_member(p, 'trans' if 'trans' in names else 'transl', z, max_bytes=64 * 1024 * 1024)
                betas = npzhead.small_member(p, 'betas', z)
                res = npzhead.small_member(p, 'per_frame_residuals', z) if 'per_frame_residuals' in names else None
        elif e['format'] == 'smplx_npy_dir':
            poses = np.load(p / 'poses.npy', mmap_mode='r', allow_pickle=False)
            tp = p / 'transl.npy' if (p / 'transl.npy').exists() else p / 'trans.npy'
            trans = np.load(tp, mmap_mode='r', allow_pickle=False)
            betas = np.load(p / 'betas.npy', allow_pickle=False)
            rp = p / 'per_frame_residuals.npy'
            res = np.load(rp, allow_pickle=False) if rp.exists() else None
        else:
            raise ValueError(e.get('viewer_unsupported') or f'格式 {e["format"]} 不支持')
        poses = np.asarray(poses).reshape(len(poses), -1)
        if poses.shape[1] < 66:
            raise ValueError('poses 维度不足 66')
        if len(trans) != len(poses):
            raise ValueError('poses/trans 帧数不一致')
        return poses, np.asarray(trans).reshape(-1, 3), betas, res

    def clip(self, clip_id):
        with self._lock:
            if clip_id in self._clips:
                self._clips.move_to_end(clip_id)
                return self._clips[clip_id]
            e = self.index.get(clip_id)
            if e.get('viewer_unsupported'):
                raise ValueError(e['viewer_unsupported'])
            stamp = Path(e['path']).stat().st_mtime_ns
            poses, trans, betas, res = self._motion(e)
            m = self.model; sl = self.slots
            v_shaped, J = m.shaped(betas)
            faces = m.faces[sl.face_ids]                                   # [N,3]
            vert_idx, inv = np.unique(faces.reshape(-1), return_inverse=True)
            inv = inv.reshape(faces.shape)
            up = Y_UP_TO_Z_UP if str(e.get('up', 'z')).lower() == 'y' else np.eye(3)
            fps = float(e['fps'])
            c = dict(entry=e, poses=poses, trans=trans, betas=np.asarray(betas), v_shaped=v_shaped, J=J,
                     vert_idx=vert_idx, inv=inv, up=up, fps=fps, frames=len(poses), stamp=stamp,
                     residuals=None if res is None else np.asarray(res, np.float32))
            # Floor (UI-computed): foot slots only, strided to ~30 fps.
            stride = max(1, int(round(fps / 30.)))
            foot = np.flatnonzero(np.isin(sl.region, [RID['L_foot'], RID['R_foot']]))
            fidx = vert_idx  # same binding; restrict LBS to foot faces' vertices
            ff = faces[foot]; fv, finv = np.unique(ff.reshape(-1), return_inverse=True); finv = finv.reshape(ff.shape)
            X = []
            frames = np.arange(0, len(poses), stride)
            for s in range(0, len(frames), 256):
                ids = frames[s:s + 256]
                verts, _ = m.forward(v_shaped, J, poses[ids, :66], trans[ids], fv)
                verts = verts @ up.T
                X.append(np.einsum('nk,cnkd->cnd', sl.bary[foot], verts[:, finv]))
            X = np.concatenate(X)
            reg = sl.region[foot]
            floor, n_static, how = floor_from_points(X, reg, fps / stride)
            c.update(floor=floor, floor_samples=n_static, floor_method=how, floor_stride=stride)
            del fidx
            self._clips[clip_id] = c
            while len(self._clips) > 3:
                self._clips.popitem(last=False)
            return c

    LIST_KEYS = ('id', 'source', 'variant', 'category', 'clip', 'frames', 'fps', 'duration_s', 'residual_cm_mean',
                 'residual_cm_p95', 'residual_cm_max', 'src_valid_frac', 'is_hoi', 'mirrored', 'gender', 'objects',
                 'viewer_unsupported', 'format')

    def catalog(self, source='', variant='', category='', q='', sort='source', offset=0, limit=200):
        self.index.ensure()
        with self.index._lock:
            entries = list(self.index.entries)
        facets = {}
        for e in entries:
            f = facets.setdefault(e['source'], {'clips': 0, 'hours': 0., 'variants': {}, 'categories': {}})
            if e['variant'] != 'gpu_30fps':  # 30 fps files are master[::3] copies: not new clips/hours
                f['clips'] += 1; f['hours'] += (e.get('duration_s') or 0) / 3600
            f['variants'][e['variant']] = f['variants'].get(e['variant'], 0) + 1
            top = (e.get('category') or '').split('/')[0]
            f['categories'][top] = f['categories'].get(top, 0) + 1
        ql = q.lower()
        rows = [e for e in entries if (not source or e['source'] == source) and (not variant or e['variant'] == variant)
                and (not category or (e.get('category') or '').startswith(category))
                and (not ql or ql in (e.get('clip') or '').lower() or ql in (e.get('category') or '').lower())]
        keyf = {'source': lambda e: (e['source'], e.get('category') or '', e.get('clip') or ''),
                'duration': lambda e: -(e.get('duration_s') or 0),
                'residual': lambda e: -(e.get('residual_cm_mean') or -1)}.get(sort)
        if keyf:
            rows.sort(key=keyf)
        total = len(rows)
        page = [{k: e.get(k) for k in self.LIST_KEYS} for e in rows[offset:offset + limit]]
        return dict(total=total, offset=offset, limit=limit, items=page, facets=facets,
                    scanned_at=self.index.scanned_at, scan_seconds=self.index.scan_seconds)

    def detail(self, clip_id):
        c = self.clip(clip_id)
        e = c['entry']
        res = c['residuals']
        return dict(e, frames=c['frames'], fps=c['fps'], floor_m_raw=c['floor'], floor_samples=c['floor_samples'],
                    floor_method=c['floor_method'], floor_stride=c['floor_stride'],
                    slot_source=self.slots.source, slot_robot=self.slots.robot, point_count=len(self.slots.face_ids),
                    regions=REGIONS, body_parents=BODY_PARENTS,
                    residual_series=bool(res is not None),
                    notes=['人体表面点 = UMR 对应关系的 4096 个人体槽位（与 umr_gpu 人体槽位同一采样规则，绑定取自 '
                           + self.slots.robot + ' 的 UMR 捕获）。',
                           'SMPL-X 前向为后端 CPU numpy 现算（手指/下巴/眼睛置零，与 UMR zero_source_finger_pose 一致）。',
                           '地面按 umr_gpu floor_from_points 规则现算并减去（校准地面 z=0）；逐点接触/距离为界面现算，不是文件标签。'])

    def scene(self, clip_id):
        c = self.clip(clip_id)
        arrays = {'human_region': self.slots.region}
        if c['residuals'] is not None:
            arrays['residuals'] = c['residuals']
        meta = self.detail(clip_id)
        meta['key'] = clip_id
        return meta, arrays

    def chunk(self, clip_id, start, count):
        c = self.clip(clip_id)
        T = c['frames']
        if not isinstance(start, int) or start < 0 or start >= T or count < 1 or count > MAX_CHUNK:
            raise ValueError('帧范围无效；每块 1–64 帧')
        stop = min(T, start + count)
        if Path(c['entry']['path']).stat().st_mtime_ns != c['stamp']:
            raise ValueError('人体源文件已变化，请刷新')
        m = self.model; sl = self.slots
        verts, joints = m.forward(c['v_shaped'], c['J'], c['poses'][start:stop, :66], c['trans'][start:stop], c['vert_idx'])
        up = c['up']
        verts = verts @ up.T; joints = joints @ up.T
        verts[..., 2] -= c['floor']; joints[..., 2] -= c['floor']
        tri = verts[:, c['inv']]                                            # [C,N,3,3]
        pts = np.einsum('nk,cnkd->cnd', sl.bary, tri)
        nrm = np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
        z = pts[..., 2]
        near_f, near_p = np.nonzero(z < 0.02)
        arrays = {'human_points': pts.astype(np.float32), 'joints': joints.astype(np.float32),
                  'frame_indices': np.arange(start, stop, dtype=np.uint32),
                  'near_frame': near_f.astype(np.uint32), 'near_point': near_p.astype(np.uint32),
                  'near_normal': nrm[near_f, near_p].astype(np.float32)}
        meta = dict(key=clip_id, start=start, count=stop - start, frames=T, fps=c['fps'],
                    floor_m_raw=c['floor'], coordinate_system='z-up，米；界面现算地面 z=0',
                    contact_rule='界面现算：点高 [0,1] mm 为接触（规格），< -0.5 mm 为穿透；距离热力 0–20 mm')
        return meta, arrays


# --------------------------------------------------------------------------- production human reconstruction
class SlotBinding:
    """4096 human slots (face id + barycentric) from any UMR capture/static-prep npz."""

    def __init__(self, capture_path):
        p = Path(capture_path)
        with zipfile.ZipFile(p) as z:
            self.face_ids = npzhead.small_member(p, 'x_bind_face_ids', z).astype(np.int64)
            self.bary = npzhead.small_member(p, 'x_bind_bary', z).astype(np.float64)
            self.part_ids = npzhead.small_member(p, 'part_ids', z).astype(np.int64)
        self.region = part_to_region(self.part_ids)
        self.path = str(p)


def load_motion_dir(source_dir):
    """SMPL-X motion from a source clip directory (flat npy layout or motion_actor_smplx.npz)."""
    d = Path(source_dir)
    if (d / 'poses.npy').exists():
        poses = np.load(d / 'poses.npy', mmap_mode='r', allow_pickle=False)
        tp = d / 'transl.npy' if (d / 'transl.npy').exists() else d / 'trans.npy'
        trans = np.load(tp, mmap_mode='r', allow_pickle=False)
        betas = np.load(d / 'betas.npy', allow_pickle=False)
        up = str(_scalar(np.load(d / 'output_up.npy', allow_pickle=False))) if (d / 'output_up.npy').exists() else 'z'
        files = [d / 'poses.npy', tp]
    else:
        cands = [d / 'motion_actor_smplx.npz'] if (d / 'motion_actor_smplx.npz').exists() else sorted(d.glob('*.npz'))
        if not cands:
            raise FileNotFoundError(f'源目录里没有 SMPL-X 动作：{d}')
        p = cands[0]
        with zipfile.ZipFile(p) as z:
            names = {i.filename[:-4] for i in z.infolist()}
            poses = npzhead.small_member(p, 'poses', z, max_bytes=256 * 1024 * 1024)
            trans = npzhead.small_member(p, 'trans' if 'trans' in names else 'transl', z, max_bytes=64 * 1024 * 1024)
            betas = npzhead.small_member(p, 'betas', z)
            up = str(_scalar(npzhead.small_member(p, 'output_up', z))) if 'output_up' in names else 'z'
        files = [p]
    poses = np.asarray(poses).reshape(len(poses), -1)
    return dict(poses=poses, trans=np.asarray(trans).reshape(-1, 3), betas=np.asarray(betas).reshape(-1),
                up=Y_UP_TO_Z_UP if up.lower() == 'y' else np.eye(3), files=[str(f) for f in files])


def reconstruct_slots(model, binding, motion, frame_ids, scale, floor):
    """UMR source slots as umr_gpu SourcePrep builds them: SMPL-X -> z-up -> x scale -> barycentric; minus floor."""
    if 'v_shaped' not in motion:
        motion['v_shaped'], motion['J'] = model.shaped(motion['betas'])
        faces = model.faces[binding.face_ids]
        vi, inv = np.unique(faces.reshape(-1), return_inverse=True)
        motion['vi'], motion['inv'] = vi, inv.reshape(faces.shape)
    ids = np.asarray(frame_ids, np.int64)
    verts, _ = model.forward(motion['v_shaped'], motion['J'], np.asarray(motion['poses'][ids, :66]),
                             np.asarray(motion['trans'][ids]), motion['vi'])
    verts = (verts @ motion['up'].T) * float(scale)
    pts = np.einsum('nk,cnkd->cnd', binding.bary, verts[:, motion['inv']])
    pts[..., 2] -= float(floor)
    return pts
