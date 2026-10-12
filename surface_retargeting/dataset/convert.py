"""Factory npz -> UMRS shards, incremental (safe to re-run / run in --watch mode while the factory writes).

    python -m umr_dataset.convert --root /home/arenalabs/umr_dataset/data [--src .../out] [--tier core|standard|full]
                                  [--fps 30] [--workers 8] [--settle 120] [--watch --interval 300 --lifetime 43200]

* An (clip, robot) npz is (re)ingested when its (size, mtime_ns) differs from the catalog's live entry AND its
  sha256 differs; files younger than --settle seconds are skipped (the factory writes npz in place).
* New entries go to new immutable shards (one per (robot, has_object) per run, split at --max-rows); a re-ingested
  (clip, robot) supersedes the old entry (catalog keeps the newest; `compact` reclaims dead rows).
* Human SMPL-X is written once per clip (first robot that arrives fixes the clip's stride / frame grid).
* Strictness survives quantisation: `decode_guard` sets flag bit 64 on any frame whose DECODED pose could break
  pen <= 0.5 mm or the [0, 1] mm contact band (static displacement bound + exact FK re-check of the < 1 % borderline
  frames); the loader only serves frames with flags == 0.
* Shard file names are unique per run and never overwritten; `compact` merges small shards (watch: --compact-every).
* Stored frame grid: source frame f is kept iff f % stride == 0, stride = floor(fps_src / fps_target) (>= 1), so the
  stored rate is >= target (90 -> 30, 60 -> 30, 120 -> 30, 50 -> 50, 24 -> 24). Row k of every body of a clip is
  source frame k * stride  ==> robots and human are aligned by k.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from . import schema as S
from .container import write_shard, file_sha256
from . import catalog as CAT
from . import sources as SRC

DEFAULT_SRC = Path('/home/arenalabs/umr_factory_data/out')
DEFAULT_ROOT = Path('/home/arenalabs/umr_dataset/data')

ROBOT_KEYS_CORE = ['qpos', 'fps', 'floor_offset_removed', 'human_scale', 'flags', 'provenance', 'joint_names', 'regions',
                   'ground_contact', 'ground_slip', 'human_ground_contact', 'human_ground_slip']
OBJ_KEYS_CORE = ['object_contact', 'object_slip', 'human_object_contact', 'human_object_slip', 'object_pos', 'object_rot',
                 'object_name', 'object_xml']


def choose_stride(fps: float, target: float) -> int:
    if not target or target <= 0:
        return 1
    return max(1, int(np.floor(fps / target + 1e-3)))


def _runs(valid: np.ndarray):
    """bool [n] -> list of [a, b) runs of True."""
    v = np.concatenate([[False], valid, [False]]).astype(np.int8)
    d = np.diff(v)
    return np.stack([np.flatnonzero(d == 1), np.flatnonzero(d == -1)], 1).tolist()


# ------------------------------------------------------------------------------------------------ robot worker
def process_robot(job):
    """Load one factory npz -> per-frame encodings at the stored grid (joints left float; quantised by the parent)."""
    path, clip, robot, tier, fps_target, size, mtime_ns, spec_root = job
    try:
        sha = file_sha256(path)
        with np.load(path, allow_pickle=False) as z:
            has_obj = 'object_contact' in z.files
            prov = json.loads(str(z['provenance']))
            qpos = np.asarray(z['qpos'])
            T = len(qpos)
            fps = float(z['fps'])
            # source frame ids (UMR frame_ids), contiguous 0..T-1 unless UMR subsampled
            fid = None
            uo = prov.get('umr_out')
            if uo and Path(uo).is_file():
                try:
                    with np.load(uo, allow_pickle=False) as u:
                        if 'frame_ids' in u.files and len(u['frame_ids']) == T:
                            fid = np.asarray(u['frame_ids'], np.int64)
                except Exception:
                    fid = None
            if fid is None:
                fid = np.arange(T, dtype=np.int64)
            src_step = int(np.median(np.diff(fid))) if T > 1 else 1
            fps_src = fps * src_step
            stride = choose_stride(fps_src, fps_target)
            keep = (fid % stride) == 0
            idx = np.flatnonzero(keep)
            k = fid[idx] // stride
            flags = np.asarray(z['flags']).astype(np.int64)
            if (flags >= S.FLAG_DROPPED).any():
                raise ValueError('factory flag bits collide with FLAG_DROPPED')
            dropped = np.zeros(int(fid.max() // stride) + 2, np.int64)
            nk = ~keep
            np.bitwise_or.at(dropped, fid[nk] // stride, flags[nk])
            f_out = (flags[idx] | np.where(dropped[k] != 0, S.FLAG_DROPPED, 0)).astype(np.uint8)
            guard = dict(root_quat_f=qpos[idx, 3:7].astype(np.float32), pen_self=np.asarray(z['pen_self_m'])[idx].astype(np.float32),
                         gclr=np.asarray(z['ground_clearance'])[idx].astype(np.float32),
                         # factory gauge version: 'union' (visual points U collision geoms, >= 2026-10-04 22:47) keeps
                         # the visual-only part separately; 'visual' (older outputs): the clearance IS the visual gauge
                         gclr_vis=np.asarray(z['ground_clearance_visual' if 'ground_clearance_visual' in z.files
                                               else 'ground_clearance'])[idx].astype(np.float32),
                         gauge='union' if 'ground_clearance_visual' in z.files else 'visual',
                         pen_ground=np.asarray(z['pen_ground_m'])[idx].astype(np.float32),
                         pen_tol=float(prov.get('pen_tol_m', 5e-4)), contact_tol=float(prov.get('contact_tol_m', 1e-3)))
            # strict contact label = the factory's enforced + verified set (`*_contact_required`); in factory code since
            # 2026-10-04 `*_contact` is identical to it (checked per entry, mismatches counted in meta)
            gcm = np.asarray(z['ground_contact_required'] if 'ground_contact_required' in z.files else z['ground_contact'], bool)
            req_mismatch = dict(ground=int((gcm != np.asarray(z['ground_contact'], bool)).sum()))
            if has_obj:
                ocm = np.asarray(z['object_contact_required'] if 'object_contact_required' in z.files else z['object_contact'], bool)
                req_mismatch['object'] = int((ocm != np.asarray(z['object_contact'], bool)).sum())
            if has_obj:
                guard.update(pen_obj=np.asarray(z['pen_object_m'])[idx].astype(np.float32),
                             oclr=np.asarray(z['object_clearance'])[idx].astype(np.float32))
            rec = dict(root_pos=qpos[idx, :3].astype(np.float32), root_q=S.enc_quat(qpos[idx, 3:7]),
                       joints_f=qpos[idx, 7:].astype(np.float32), flags=f_out,
                       gc=S.pack_regions(gcm[idx]), gs=S.pack_regions(z['ground_slip'][idx]),
                       hgc=S.pack_regions(z['human_ground_contact'][idx]), hgs=S.pack_regions(z['human_ground_slip'][idx]))
            extra = {}
            if has_obj:
                rec.update(oc=S.pack_regions(ocm[idx]), os=S.pack_regions(z['object_slip'][idx]),
                           hoc=S.pack_regions(z['human_object_contact'][idx]), hos=S.pack_regions(z['human_object_slip'][idx]))
            if tier in ('standard', 'full'):
                rec.update(gclr=S.enc_sdist(z['ground_clearance'][idx]), hgclr=S.enc_sdist(z['human_ground_clearance'][idx]),
                           gspd=S.enc_speed(z['ground_slip_speed'][idx]))
                if has_obj:
                    rec.update(oclr=S.enc_sdist(z['object_clearance'][idx]), ospd=S.enc_speed(z['object_slip_speed'][idx]))
            if tier == 'full':
                rec.update(gclr_f=z['ground_clearance'][idx].astype(np.float16),
                           hgclr_f=z['human_ground_clearance'][idx].astype(np.float16),
                           gspd_f=z['ground_slip_speed'][idx].astype(np.float16),
                           pen=np.stack([z['pen_ground_m'][idx], z['pen_self_m'][idx],
                                         z['pen_object_m'][idx] if has_obj else np.zeros(len(idx), np.float32)], 1).astype(np.float16))
                extra['ground_point_contact'] = np.asarray(z['ground_point_contact'][idx], np.uint8)
                if has_obj:
                    rec.update(oclr_f=z['object_clearance'][idx].astype(np.float16), ospd_f=z['object_slip_speed'][idx].astype(np.float16))
                    extra['object_point_contact'] = np.asarray(z['object_point_contact'][idx], np.uint8)
            scale = float(z['human_scale']); floor = float(z['floor_offset_removed'])
            obj = None
            if has_obj:
                op = np.asarray(z['object_pos'], np.float64)[idx]
                orot = np.asarray(z['object_rot'], np.float64)[idx]
                if op.ndim == 2:
                    op, orot = op[:, None], orot[:, None]
                ph = (op + np.array([0.0, 0.0, floor])) / scale                   # human metric Z-up frame, floor z=0
                qh = S.rotmat_to_quat(orot)
                names = np.atleast_1d(np.asarray(z['object_name'])).astype(str).tolist()
                xmls = np.atleast_1d(np.asarray(z['object_xml'])).astype(str).tolist()
                obj = dict(pos=ph.astype(np.float32), quat=qh.astype(np.float32), names=names, xmls=xmls)
            jn = np.asarray(z['joint_names']).astype(str).tolist()
            regions = np.asarray(z['regions']).astype(str).tolist()
        if regions != S.REGIONS:
            raise ValueError(f'regions mismatch {regions}')
        spec_p = Path(spec_root) / 'robots' / robot / 'spec.json'
        if spec_p.exists() and json.load(open(spec_p))['joint_names'] != jn[1:]:
            raise ValueError('joint names differ from the robot spec (re-staged MJCF?)')
        # contiguous segments of k (normally one)
        brk = np.flatnonzero(np.diff(k) != 1) + 1
        segs = np.split(np.arange(len(k)), brk)
        flags_hist = {str(b): int(((f_out & b) > 0).sum()) for b in (1, 2, 4, 8, 16, 128)}
        meta = dict(clip=clip, body=robot, has_object=has_obj, fps_src=fps_src, stride=stride, fps=fps_src / stride,
                    scale=scale, floor_offset=floor, n_src=int(T),
                    src=dict(path=str(path), size=size, mtime_ns=mtime_ns, sha256=sha),
                    prov={k_: prov.get(k_) for k_ in ('source_kind', 'source_data', 'seq_key', 'umr_commit', 'umr_out',
                                                     'created', 'label_cfg', 'pen_tol_m', 'contact_tol_m', 'staged_mjcf')},
                    flags_hist=flags_hist, n_flagged=int((f_out != 0).sum()), contact_vs_required_mismatch=req_mismatch,
                    gauge=guard['gauge'])
        return dict(ok=True, meta=meta, prov=prov, rec=rec, extra=extra, k=k, segs=segs, obj=obj, guard=guard, idx=idx)
    except Exception as e:
        return dict(ok=False, path=str(path), clip=clip, body=robot, error=repr(e), tb=traceback.format_exc(limit=3))


# ------------------------------------------------------------------------------------------------ human worker
def process_human(job):
    clip, prov, stride, fps_src, obj, scale, floor, assets_dir = job
    try:
        h = SRC.load_human(prov)
        P = np.asarray(h['poses'], np.float32)
        T = len(P)
        if abs(h['fps'] - fps_src) > 0.01 * max(fps_src, 1.0):
            raise ValueError(f"source fps {h['fps']} != robot-derived {fps_src}")
        kk = np.arange(0, T, stride)
        P = P[kk]
        if P.shape[1] < 66:
            raise ValueError(f'pose width {P.shape[1]}')
        body = P[:, :66].reshape(-1, 22, 3)
        ex = np.zeros((len(P), 33, 3), np.float32)
        if P.shape[1] >= 165:
            ex = P[:, 66:165].reshape(-1, 33, 3)
        has_extra = bool(np.abs(ex).max() > 0)
        rec = dict(transl=np.asarray(h['trans'], np.float32)[kk], body_aa=body)
        if has_extra:
            rec['extra_aa'] = ex
        nobj = 0
        objmeta = None
        if obj is not None:
            nobj = obj['pos'].shape[1]
            pos = np.full((len(kk), nobj, 3), np.nan, np.float32); quat = np.full((len(kk), nobj, 4), np.nan, np.float32)
            kr = obj['k']
            ok = kr < len(kk)
            pos[kr[ok]] = obj['pos'][ok]; quat[kr[ok]] = obj['quat'][ok]
            rec.update(obj_pos=pos, obj_quat=quat)
            objmeta = dict(names=obj['names'], xmls=obj['xmls'], from_robot=obj['robot'],
                           assets=[SRC.copy_object_assets(x, assets_dir) for x in obj['xmls'] if x],
                           frame='human metric Z-up (UMR retarget frame / scale), floor z=0; robot frame = scale*p - floor_offset*z')
        meta = dict(clip=clip, body='human', n_src=int(T), stride=stride, fps_src=float(h['fps']), fps=float(h['fps']) / stride,
                    betas=np.asarray(h['betas'], np.float32).tolist(), gender=h['gender'], up=h['up'],
                    human_scale_src=float(h.get('human_scale', 1.0)), layout=h['layout'], file=h['file'],
                    human_floor_z=float(floor / scale), floor_from_robot=obj['robot'] if obj else None,
                    has_extra=has_extra, n_obj=nobj, objects=objmeta, metadata=h.get('metadata'),
                    dataset=SRC.dataset_tag(prov), file_sha256=file_sha256(h['file']) if Path(h['file']).is_file() else None)
        return dict(ok=True, meta=meta, rec=rec, has_extra=has_extra, nobj=nobj)
    except Exception as e:
        return dict(ok=False, clip=clip, body='human', error=repr(e), tb=traceback.format_exc(limit=3))


# ------------------------------------------------------------------------------------------------ decode guard
_FK = {}


def decode_guard(root: Path, robot: str, results: list, spec: dict, device='auto', batch=8192, eps=5e-6):
    """Flag (bit 64) frames whose DECODED pose may break a strict tolerance (pen <= 0.5 mm ground / object / self,
    required contact regions within [-0.5, 1] mm of ground / object).

    Stage 1 (numpy, every frame): a configuration-independent bound D of the decode displacement
    (kinematics.static_disp_bound: per region D_r for the region gauges, D for any point / geom) proves most frames
    safe from the factory's own source gauges: ground clearance - D_r >= -0.5 mm, contact regions in
    [-0.5 mm + D_r, 1 mm - D_r], pen_self + 2D <= 0.5 mm, pen_object + D <= 0.5 mm, object contact band +- D_r.
    Stage 2 (torch FK, only the remaining still-valid frames, typically < 1 %): the visual ground gauge (dense points U
    hull vertices) is re-measured exactly on the decoded pose; the collision-geom part of the factory's union gauge
    (factory >= 2026-10-04 22:47) and self / object use the per-frame bound d_t = max displacement of every gauge point
    and every geom bounding sphere between source and decoded pose.
    eps (5 um) absorbs float32 storage of the factory's float64 poses.  Mutates rec['flags']; returns stats."""
    from .kinematics import static_disp_bound
    bnd = static_disp_bound(robot, root)
    D, Dr = float(bnd['all_m']), np.asarray(bnd['region_m'])
    lo, step = np.asarray(spec['q_lo']), np.asarray(spec['q_step'])
    if not results:
        return dict(newly_flagged=0, candidates=0, frames=0, fk_seconds=0.0, static_bound_mm=D * 1e3)
    cands = []
    for r in results:
        rec, g = r['rec'], r['guard']
        PEN, CT = g['pen_tol'], g['contact_tol']
        clr = g['gclr']
        gc = S.unpack_regions(rec['gc'])
        with np.errstate(invalid='ignore'):
            low_ok = (np.where(np.isfinite(clr), clr - Dr, np.inf) >= -PEN + eps).all(1)
            band_ok = ~(gc & ~((clr - Dr >= -PEN + eps) & (clr + Dr <= CT - eps))).any(1)
            ok = low_ok & band_ok & (g['pen_self'] + 2 * D <= PEN - eps) & (g['pen_ground'] + D <= PEN - eps)
            if 'oc' in rec:
                o, oc = g['oclr'], S.unpack_regions(rec['oc'])
                ok &= g['pen_obj'] + D <= PEN - eps
                ok &= ~(oc & ~((o - Dr >= -PEN + eps) & (o + Dr <= CT - eps))).any(1)
        cands.append(np.flatnonzero(~ok & (rec['flags'] == 0)))
    N = int(sum(len(c) for c in cands))
    nflag, t_fk = 0, 0.0
    if N:
        import torch
        from .kinematics import TorchFK
        dev = 'cuda' if device in ('cuda', 'auto') and torch.cuda.is_available() else 'cpu'   # CPU FK: ~1k frames/s under load
        dev = torch.device(dev)
        key = (robot, str(dev))
        if key not in _FK:
            if dev.type == 'cuda' and not any(k[1].startswith('cuda') for k in _FK):
                di = dev.index if dev.index is not None else torch.cuda.current_device()
                tot = torch.cuda.get_device_properties(di).total_memory
                torch.cuda.set_per_process_memory_fraction(min(1.0, 16e9 / tot), di)
            _FK[key] = TorchFK.load(root / 'robots' / robot / 'kin.npz', device=dev, dtype=torch.float32)
        fk = _FK[key]
        rbound = np.asarray(bnd['geom_rbound'])
        t1 = time.time()
        qs = np.concatenate([np.concatenate([r['rec']['root_pos'][c], r['guard']['root_quat_f'][c], r['rec']['joints_f'][c]], 1)
                             for r, c in zip(results, cands)]).astype(np.float64)
        qd = np.concatenate([np.concatenate([r['rec']['root_pos'][c], S.dec_quat(r['rec']['root_q'][c]),
                                             S.dec_joints(S.enc_joints(r['rec']['joints_f'][c], lo, step), lo, step)], 1)
                             for r, c in zip(results, cands)]).astype(np.float64)
        off = qs[:, :3].copy(); off[:, 2] = 0                      # remove root xy: float32 FK stays sub-micron
        off = np.pad(off, ((0, 0), (0, qs.shape[1] - 3)))
        Dt = np.zeros(N); C = np.zeros((N, S.NREG))
        with torch.no_grad():
            for s0 in range(0, N, batch):
                s1 = min(N, s0 + batch)
                A = torch.from_numpy((qs[s0:s1] - off[s0:s1]).astype(np.float32)).to(dev)
                B = torch.from_numpy((qd[s0:s1] - off[s0:s1]).astype(np.float32)).to(dev)
                gxa, gRa = fk.geoms(A)
                gxb, gRb = fk.geoms(B)
                d1 = fk.disp_bound(gxa, gRa, gxb, gRb)                       # gauge points
                d2 = fk.disp_bound_geoms(gxa, gRa, gxb, gRb, rbound)         # every geom bounding sphere
                Dt[s0:s1] = torch.maximum(d1, d2).double().cpu().numpy()
                C[s0:s1] = fk.clearance_from_geoms(gxb, gRb).double().cpu().numpy()
        t_fk = time.time() - t1
        if dev.type == 'cuda':
            torch.cuda.empty_cache()                       # shared card: give the cached blocks back between runs
        o0 = 0
        for r, c in zip(results, cands):
            n = len(c)
            if not n:
                continue
            rec, g = r['rec'], r['guard']
            PEN, CT = g['pen_tol'], g['contact_tol']
            d, cv = Dt[o0:o0 + n], C[o0:o0 + n]; o0 += n
            if g['gauge'] == 'union':
                # decoded union clearance: visual part exact (FK), collision part bounded from the source union value
                # (coll_src >= union_src; coll_src == union_src where union < visual) and the per-frame displacement d
                us, vs = g['gclr'][c].astype(np.float64), g['gclr_vis'][c].astype(np.float64)
                lower = np.minimum(cv, us - d[:, None])
                upper = np.minimum(cv, np.where(us < vs - 1e-9, us + d[:, None], np.inf))
            else:
                lower = upper = cv
            low = np.where(np.isfinite(lower), lower, np.inf).min(1)
            bad = low < -PEN + eps
            bad |= (S.unpack_regions(rec['gc'][c]) & ~((lower >= -PEN + eps) & (upper <= CT - eps))).any(1)
            bad |= g['pen_self'][c] + 2 * d > PEN - eps
            if 'oc' in rec:
                o = g['oclr'][c]
                bad |= g['pen_obj'][c] + d > PEN - eps
                bad |= (S.unpack_regions(rec['oc'][c]) & ((o - d[:, None] < -PEN + eps) | (o + d[:, None] > CT - eps))).any(1)
            fl = rec['flags']
            fl[c[bad]] |= S.FLAG_DECODE
            r['_newly'] = int(bad.sum())
    for r, c in zip(results, cands):
        rec = r['rec']
        nb = r.pop('_newly', 0)
        nflag += nb
        r['meta']['decode_guard'] = dict(static_bound_mm=D * 1e3, candidates=int(len(c)), newly_flagged=nb)
        r['meta']['n_flagged'] = int((rec['flags'] != 0).sum())
        r['meta']['flags_hist'][str(S.FLAG_DECODE)] = int(((rec['flags'] & S.FLAG_DECODE) > 0).sum())
    return dict(newly_flagged=nflag, candidates=N, frames=int(sum(len(r['rec']['flags']) for r in results)),
                fk_seconds=t_fk, static_bound_mm=D * 1e3)


# ------------------------------------------------------------------------------------------------ shard assembly
def _batch_id():
    import uuid
    return time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + f'-{os.getpid()}-{uuid.uuid4().hex[:6]}'


def write_robot_shards(root: Path, robot: str, has_obj: bool, results: list, tier: str, spec: dict, batch: str,
                       max_rows: int) -> list:
    lo = np.asarray(spec['q_lo']); step = np.asarray(spec['q_step'])
    dt = S.robot_dtype(spec['nj'], tier, has_obj)
    out, part, cur, rows = [], 0, [], 0
    groups = []
    for r in results:
        n = len(r['k'])
        if cur and rows + n > max_rows:
            groups.append(cur); cur, rows = [], 0
        cur.append(r); rows += n
    if cur:
        groups.append(cur)
    for g in groups:
        N = sum(len(r['k']) for r in g)
        arr = np.zeros(N, dt)
        extras = {}
        entries, row = [], 0
        for r in g:
            rec, n = r['rec'], len(r['k'])
            sl = slice(row, row + n)

            arr['joints'][sl] = S.enc_joints(rec['joints_f'], lo, step)
            for f in dt.names:
                if f != 'joints':
                    arr[f][sl] = rec[f]
            for name, a in r['extra'].items():
                extras.setdefault(name, []).append(a)
            for seg in r['segs']:
                a, b = int(seg[0]), int(seg[-1]) + 1
                e = dict(r['meta'])
                sr = r['idx'][a:b]                      # rows of the factory npz (UMR output frames) behind k0..k0+n
                st_ = int(sr[1] - sr[0]) if len(sr) > 1 else 1
                e.update(row0=row + a, n=b - a, k0=int(r['k'][a]), valid_runs=_runs(rec['flags'][a:b] == 0),
                         src_row0=int(sr[0]), src_row_step=st_, ingested_at=time.time())
                if len(sr) > 1 and not (np.diff(sr) == st_).all():
                    e['src_rows'] = sr.tolist()
                entries.append(e)
            row += n
        blobs = {'frames': arr}
        for name, lst in extras.items():
            blobs[name] = np.concatenate(lst, 0)
        meta = dict(schema_version=S.SCHEMA_VERSION, robot=robot, tier=tier, has_object=has_obj, nj=spec['nj'],
                    joint_names=spec['joint_names'], q_lo=spec['q_lo'], q_step=spec['q_step'], spec_kin_sha256=spec['kin_sha256'],
                    mjcf_sha256=spec['mjcf_sha256'], regions=S.REGIONS, quat_codec='smallest-three u64: 3x20-bit in [-1/sqrt2,1/sqrt2], bits60-61 index, bit62 sign',
                    sdist=dict(D0=S.SD_D0, K=S.SD_K, missing=S.SD_MISSING), speed=dict(V0=S.SP_V0, K=S.SP_K, missing=S.SP_MISSING),
                    flag_bits=S.FLAG_BITS, entries=entries, created=time.time(), batch=batch)
        name = f"{robot}-{'hoi' if has_obj else 'body'}-{batch}-{part:03d}.umrs"
        info = write_shard(root / 'robots' / robot / name, blobs, meta, kind='robot')
        info['rel'] = str(Path('robots') / robot / name)
        out.append(info); part += 1
    return out


def write_human_shards(root: Path, results: list, batch: str, max_rows: int) -> list:
    out = []
    keyed = {}
    for r in results:
        keyed.setdefault((r['has_extra'], r['nobj']), []).append(r)
    part = 0
    for (has_extra, nobj), lst in sorted(keyed.items()):
        dt = S.human_dtype(has_extra, nobj)
        groups, cur, rows = [], [], 0
        for r in lst:
            n = len(r['rec']['transl'])
            if cur and rows + n > max_rows:
                groups.append(cur); cur, rows = [], 0
            cur.append(r); rows += n
        if cur:
            groups.append(cur)
        for g in groups:
            N = sum(len(r['rec']['transl']) for r in g)
            body_scale = S.aa_scale(np.concatenate([r['rec']['body_aa'].ravel() for r in g]))
            ex_scale = S.aa_scale(np.concatenate([r['rec']['extra_aa'].ravel() for r in g])) if has_extra else 0.0
            arr = np.zeros(N, dt)
            entries, row = [], 0
            for r in g:
                rec = r['rec']; n = len(rec['transl'])
                sl = slice(row, row + n)
                arr['transl'][sl] = rec['transl']
                arr['body_aa'][sl] = S.enc_aa(rec['body_aa'], body_scale)
                if has_extra:
                    arr['extra_aa'][sl] = S.enc_aa(rec['extra_aa'], ex_scale)
                if nobj:
                    arr['obj_pos'][sl] = rec['obj_pos']; arr['obj_quat'][sl] = rec['obj_quat']
                e = dict(r['meta']); e.update(row0=row, n=n, k0=0, valid_runs=[[0, n]], ingested_at=time.time())
                entries.append(e)
                row += n
            meta = dict(schema_version=S.SCHEMA_VERSION, body='human', has_extra=has_extra, n_obj=nobj,
                        body_aa_scale=body_scale, extra_aa_scale=ex_scale, entries=entries, created=time.time(), batch=batch,
                        smplx_layout='poses[0:3]=global_orient, [3:66]=body(21), [66:69]=jaw, [69:75]=eyes, [75:165]=hands(15+15)')
            name = f'human-{batch}-{part:03d}.umrs'
            info = write_shard(root / 'human' / name, {'frames': arr}, meta, kind='human')
            info['rel'] = str(Path('human') / name)
            out.append(info); part += 1
    return out


# ------------------------------------------------------------------------------------------------ driver
def scan_src(src: Path):
    for cd in sorted(p for p in src.iterdir() if p.is_dir() and p.name not in ('tests',)):
        for f in sorted(cd.glob('*.npz')):
            st = f.stat()
            yield cd.name, f.stem, f, st.st_size, st.st_mtime_ns, st.st_mtime


def _gauge_of(path) -> str:
    """'union' = factory gauge visual points U collision geoms (outputs >= 2026-10-04 22:47), else 'visual'."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return 'union' if 'ground_clearance_visual' in z.files else 'visual'
    except Exception:
        return 'unreadable'


def ingest(root: Path, src: Path, tier='standard', fps=30.0, workers=8, settle=120.0, max_rows=2_000_000,
           robots=None, guard_device='auto', max_batch_files=1500, min_gauge='union', log=print) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    lockf = open(root / '.convert.lock', 'a+')
    fcntl.flock(lockf, fcntl.LOCK_EX)
    try:
        cat = CAT.load_live_entries(root)                       # {(clip, body): entry meta}
        specs = {}
        jobs, skipped_young, unchanged, stale = [], 0, 0, 0
        now = time.time()
        for clip, robot, f, size, mtime_ns, mtime in scan_src(src):
            if robots and robot not in robots:
                continue
            e = cat.get((clip, robot))
            if e is not None and e['src']['size'] == size and e['src']['mtime_ns'] == mtime_ns:
                unchanged += 1; continue
            if now - mtime < settle:
                skipped_young += 1; continue
            if e is not None and e['src'].get('sha256') == file_sha256(f):
                unchanged += 1; continue
            if min_gauge == 'union' and _gauge_of(f) != 'union':
                stale += 1; continue                    # made before the factory's visual+collision gauge: not strict
            spec_p = root / 'robots' / robot / 'spec.json'
            if not spec_p.exists():
                from .kinematics import build_spec
                build_spec(robot, root)
            specs[robot] = json.load(open(spec_p))
            jobs.append((str(f), clip, robot, tier, fps, size, mtime_ns, str(root)))
        log(f'[ingest] scan: {len(jobs)} new/changed, {unchanged} unchanged, {skipped_young} younger than {settle}s, '
            f'{stale} skipped (gauge older than {min_gauge!r})')
        if not jobs:
            return dict(new=0, seconds=time.time() - t0)
        ctx = mp.get_context('fork')
        human_live = {c: e for (c, b), e in cat.items() if b == 'human'}
        grid = {c: e['stride'] for c, e in human_live.items()}
        shards, tm = [], dict(robot_pool=0.0, human_pool=0.0, guard=0.0, write=0.0)
        n_kept = n_bad = n_hok = n_hbad = nfr = 0
        gstats = {}
        with ctx.Pool(max(1, min(workers, len(jobs))), maxtasksperchild=256) as pool:
            for c0 in range(0, len(jobs), max_batch_files):
                chunk = jobs[c0:c0 + max_batch_files]
                batch = _batch_id()
                t1 = time.time()
                res = pool.map(process_robot, chunk, chunksize=1)
                tm['robot_pool'] += time.time() - t1
                ok = [r for r in res if r['ok']]
                bad = [r for r in res if not r['ok']]
                for x in bad:
                    log(f"[ingest] FAIL {x['clip']} {x['body']}: {x['error']}")
                # clip stride consistency: a clip's grid is fixed by its human entry (or the first robot seen)
                hjobs, kept = {}, []
                for r in ok:
                    m = r['meta']; c = m['clip']
                    if c in grid and grid[c] != m['stride']:
                        log(f"[ingest] FAIL {c} {m['body']}: stride {m['stride']} != clip grid {grid[c]}"); continue
                    grid.setdefault(c, m['stride'])
                    kept.append(r)
                    if c not in human_live and (c not in hjobs or (r['obj'] is not None and hjobs[c][4] is None)):
                        obj = None
                        if r['obj'] is not None:
                            obj = dict(r['obj']); obj['k'] = r['k']; obj['robot'] = m['body']
                        hjobs[c] = (c, r['prov'], m['stride'], m['fps_src'], obj, m['scale'], m['floor_offset'],
                                    root / 'objects')
                t1 = time.time()
                hres = pool.map(process_human, list(hjobs.values()), chunksize=1) if hjobs else []
                tm['human_pool'] += time.time() - t1
                hok = [h for h in hres if h['ok']]
                for h in hres:
                    if not h['ok']:
                        log(f"[ingest] FAIL human {h['clip']}: {h['error']}")
                for h in hok:
                    human_live[h['meta']['clip']] = h['meta']
                # robot entries whose clip has no human (failed) are still written: robot<->robot pairs stay usable
                by = {}
                for r in kept:
                    by.setdefault((r['meta']['body'], r['meta']['has_object']), []).append(r)
                for (robot, ho), lst in sorted(by.items()):
                    if guard_device != 'off':
                        t1 = time.time()
                        g = decode_guard(root, robot, lst, specs[robot], guard_device)
                        tm['guard'] += time.time() - t1
                        k_ = f'{robot}/{int(ho)}'
                        gs = gstats.setdefault(k_, dict(frames=0, candidates=0, newly_flagged=0, bound_mm=g['static_bound_mm']))
                        for f_ in ('frames', 'candidates', 'newly_flagged'):
                            gs[f_] += g[f_]
                    t1 = time.time()
                    shards += write_robot_shards(root, robot, ho, lst, tier, specs[robot], batch, max_rows)
                    tm['write'] += time.time() - t1
                t1 = time.time()
                if hok:
                    shards += write_human_shards(root, hok, batch, max_rows)
                tm['write'] += time.time() - t1
                CAT.register_shards(root, [x for x in shards if 'registered' not in x])
                for x in shards:
                    x['registered'] = True
                n_kept += len(kept); n_bad += len(bad); n_hok += len(hok); n_hbad += len(hres) - len(hok)
                nfr += sum(len(r['k']) for r in kept)
                del res, ok, kept, hres, hok, by
        if gstats:
            log('[ingest] decode guard (frames, FK-checked candidates, newly flagged, static bound mm): ' +
                json.dumps({k: (v['frames'], v['candidates'], v['newly_flagged'], round(v['bound_mm'], 3)) for k, v in gstats.items()}))
        t1 = time.time()
        CAT.build(root, log=log)
        tm['catalog'] = time.time() - t1
        dt = time.time() - t0
        log(f'[ingest] wrote {len(shards)} shards, {n_kept} robot entries ({nfr} frames), {n_hok} human clips, '
            f'{n_bad} robot failures, {n_hbad} human failures in {dt:.1f}s ' +
            json.dumps({k: round(v, 1) for k, v in tm.items()}))
        return dict(new=n_kept, human=n_hok, failures=n_bad + n_hbad, frames=nfr, seconds=dt, timing=tm,
                    shards=[x['rel'] for x in shards])
    finally:
        fcntl.flock(lockf, fcntl.LOCK_UN)
        lockf.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    ap.add_argument('--src', type=Path, default=DEFAULT_SRC)
    ap.add_argument('--tier', default='standard', choices=S.TIERS)
    ap.add_argument('--fps', type=float, default=30.0, help='target stored rate (integer-stride decimation, >= target)')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--settle', type=float, default=120.0)
    ap.add_argument('--max-rows', type=int, default=2_000_000)
    ap.add_argument('--robots', default='', help='comma list (default all)')
    ap.add_argument('--guard-device', default='auto', help="decode guard FK device: auto | cuda | cpu | off")
    ap.add_argument('--max-batch-files', type=int, default=1500, help='npz per in-memory batch (bounds RAM)')
    ap.add_argument('--min-gauge', default='union', choices=('union', 'visual'),
                    help="union: only ingest outputs verified with the factory's visual+collision gauge (default)")
    ap.add_argument('--list-ingested', action='store_true',
                    help='print source npz whose current (size, mtime) [and sha256 with --verify-sha] equal the live '
                         'catalog entry, i.e. safe for the factory owner to prune; no conversion')
    ap.add_argument('--verify-sha', action='store_true')
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--interval', type=float, default=300.0)
    ap.add_argument('--lifetime', type=float, default=12 * 3600.0, help='watch mode hard stop (s)')
    ap.add_argument('--compact-every', type=int, default=12, help='watch mode: run compaction every N cycles (0 = never)')
    ap.add_argument('--grace-hours', type=float, default=6.0, help='compaction: keep replaced shards this long')
    a = ap.parse_args(argv)
    robots = set(x for x in a.robots.split(',') if x) or None
    if a.list_ingested:
        live = CAT.load_live_entries(a.root)
        n = nb = 0
        for clip, robot, f, size, mtime_ns, _ in scan_src(a.src):
            e = live.get((clip, robot))
            ok = e is not None and e['src']['size'] == size and e['src']['mtime_ns'] == mtime_ns
            if ok and a.verify_sha:
                ok = file_sha256(f) == e['src']['sha256']
            if ok:
                print(f'{f}\t{size}\t{e["src"]["sha256"]}'); n += 1; nb += size
        print(f'# {n} source files ({nb / 1e9:.2f} GB) are ingested and unchanged', file=sys.stderr)
        return
    t_end = time.time() + a.lifetime
    cycle = 0
    while True:
        r = ingest(a.root, a.src, a.tier, a.fps, a.workers, a.settle, a.max_rows, robots, a.guard_device,
                   a.max_batch_files, a.min_gauge)
        print(time.strftime('%H:%M:%S'), json.dumps({k: v for k, v in r.items() if k != 'shards'}), flush=True)
        cycle += 1
        if a.watch and a.compact_every and cycle % a.compact_every == 0:
            from .compact import compact
            compact(a.root, target_rows=a.max_rows, grace_hours=a.grace_hours)
        if not a.watch or time.time() + a.interval > t_end:
            break
        time.sleep(a.interval)


if __name__ == '__main__':
    main()
