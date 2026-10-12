"""Paired-window loader (numpy + torch only; no mujoco needed).

    from surface_retargeting.dataset.dataset import PairedWindows, collate
    ds = PairedWindows('/home/arenalabs/umr_dataset/data', window=64, pair_mode='human2robot')
    dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=8, collate_fn=collate, persistent_workers=True,
                                     pin_memory=True, prefetch_factor=4)
    for batch in dl: batch['slot0'] (source), batch['slot1'] (target) ...

A sample = one clip, one window of `window` stored frames [k, k+window) on the clip grid, and `n_bodies` bodies
(slot 0 = source, slot 1 = target in pair modes).  Every frame of the window is strictly valid (flags == 0: factory
flags, decode guard and dropped-intermediate bits all clear) for every robot slot.  Clips are drawn proportional to
their frame span; windows uniformly over the valid intersection of the chosen bodies.  Sampling is a pure function of
(seed, epoch, index).  A whole batch is gathered and decoded vectorised in the worker (`__getitems__`), so `collate`
only passes the ready dict through.

Batch layout (B samples, W frames), per slot j -> batch[f'slot{j}']:
  is_human [B] bool, body [B] (catalog body index, 0 = human)
  robot samples: qpos [B,W,Qmax] f32 (MuJoCo qpos: root xyz, root wxyz, joints; zero padded), q_mask [B,Qmax],
     scale [B], floor_offset [B] (robot frame = scale * human Z-up metric frame - floor_offset * z),
     contact / slip / human_contact / human_slip [B,W,13] bool (ground), object_* [B,W,13] (+ object_valid [B]),
     flags [B,W] u8 (all zero by construction), optional clearance/human_clearance/slip_speed [B,W,13] f32 ('clearance'),
     object_pos [B,W,K,3] / object_quat [B,W,K,4] in this robot's frame (NaN when no object)
  human samples: SMPL-X transl [B,W,3], global_orient [B,W,3], body_pose [B,W,21,3], jaw/leye/reye [B,W,3],
     left_hand_pose/right_hand_pose [B,W,15,3], betas [B,16], num_betas [B], up_y [B] (source up axis is +Y),
     gender [B] (0 neutral 1 male 2 female), floor_z [B] (floor height in the Z-up metric frame)
batch: clip [B], k [B], fps [B], bodies [B,n_slots]; object_pos_human / object_quat [B,W,K,*] (human Z-up frame).
"""
from __future__ import annotations

import bisect
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np

from . import schema as S
from .catalog import Catalog
from .container import Shard

PAIR_MODES = ('human2robot', 'robot2human', 'robot2robot', 'any', 'multi')
LABEL_PAIRS = [('gc', 'contact'), ('gs', 'slip'), ('hgc', 'human_contact'), ('hgs', 'human_slip')]
OBJ_PAIRS = [('oc', 'object_contact'), ('os', 'object_slip'), ('hoc', 'human_object_contact'), ('hos', 'human_object_slip')]
FLOAT_FIELDS = ('clearance', 'human_clearance', 'slip_speed', 'object_clearance', 'object_slip_speed',
                'clearance_f', 'human_clearance_f', 'slip_speed_f', 'pen')


def decode_robot(fr: np.ndarray, meta: dict, fields=('labels',)) -> dict:
    """Structured robot records [...] -> decoded arrays [...] (vectorised over any leading shape)."""
    names = fr.dtype.names
    nj = meta['nj']
    q = np.empty(fr.shape + (7 + nj,), np.float32)
    q[..., :3] = fr['root_pos']
    q[..., 3:7] = S.dec_quat(fr['root_q'])
    q[..., 7:] = (np.asarray(meta['q_lo'], np.float64) + fr['joints'] * np.asarray(meta['q_step'], np.float64))
    out = dict(qpos=q, flags=np.asarray(fr['flags']))
    for f, n in LABEL_PAIRS:
        out[n] = S.unpack_regions(fr[f])
    if 'oc' in names:
        for f, n in OBJ_PAIRS:
            out[n] = S.unpack_regions(fr[f])
    if 'clearance' in fields and 'gclr' in names:
        out.update(clearance=S.dec_sdist(fr['gclr']), human_clearance=S.dec_sdist(fr['hgclr']), slip_speed=S.dec_speed(fr['gspd']))
        if 'oclr' in names:
            out.update(object_clearance=S.dec_sdist(fr['oclr']), object_slip_speed=S.dec_speed(fr['ospd']))
    if 'exact' in fields and 'gclr_f' in names:
        out.update(clearance_f=fr['gclr_f'].astype(np.float32), human_clearance_f=fr['hgclr_f'].astype(np.float32),
                   slip_speed_f=fr['gspd_f'].astype(np.float32), pen=fr['pen'].astype(np.float32))
    return out


def decode_human(fr: np.ndarray, meta: dict) -> dict:
    aa = S.dec_aa(fr['body_aa'], meta['body_aa_scale'])
    out = dict(transl=np.asarray(fr['transl'], np.float32), global_orient=aa[..., 0, :], body_pose=aa[..., 1:22, :])
    if 'extra_aa' in fr.dtype.names:
        ex = S.dec_aa(fr['extra_aa'], meta['extra_aa_scale'])
    else:
        ex = np.zeros(fr.shape + (33, 3), np.float32)
    out.update(jaw_pose=ex[..., 0, :], leye_pose=ex[..., 1, :], reye_pose=ex[..., 2, :], left_hand_pose=ex[..., 3:18, :],
               right_hand_pose=ex[..., 18:33, :])
    if 'obj_pos' in fr.dtype.names:
        out.update(object_pos=np.asarray(fr['obj_pos'], np.float32), object_quat=np.asarray(fr['obj_quat'], np.float32))
    return out


class Reader:
    """Random access to decoded windows of any live (clip, body) entry."""

    def __init__(self, root):
        self.root = Path(root)
        self.cat = Catalog(self.root)
        self._shards = {}
        cat = self.cat
        self.seg = {}                                     # (clip, body) -> [(k0, n, shard, row0, entry)]
        for e in range(len(cat.e_clip)):
            self.seg.setdefault((int(cat.e_clip[e]), int(cat.e_body[e])), []).append(
                (int(cat.e_k0[e]), int(cat.e_n[e]), int(cat.e_shard[e]), int(cat.e_row0[e]), e))

    def shard(self, i) -> Shard:
        s = self._shards.get(i)
        if s is None:
            s = Shard(self.root / self.cat.shards[i])
            self._shards[i] = s
        return s

    def __getstate__(self):
        return dict(root=self.root, cat=self.cat, seg=self.seg)

    def __setstate__(self, st):
        self.root, self.cat, self.seg, self._shards = st['root'], st['cat'], st['seg'], {}

    def entries(self, c, b):
        return [s[4] for s in self.seg.get((c, b), [])]

    def entry_meta(self, c, b, k=None) -> dict:
        """Full metadata (provenance, source path, src_row mapping, ...) of the segment of (clip c, body b) holding
        stored frame k (first segment if k is None).  Reads the shard header once (cached by sha256)."""
        from .catalog import _shard_entries, load_manifest
        segs = self.seg[(c, b)]
        k0, n, sh, r0, e = next((x for x in segs if k is None or x[0] <= k < x[0] + x[1]), segs[0])
        rel = self.cat.shards[sh]
        if not hasattr(self, '_man'):
            self._man = load_manifest(self.root)
        ents = _shard_entries(self.root, rel, self._man['shards'][rel]['sha256'])['entries']
        return next(x for x in ents if x['clip'] == self.cat.clips[c] and x['body'] == self.cat.bodies[b] and x['row0'] == r0)

    def locate_rows(self, c, b, k, w):
        """(shard index, first row, entry) of window [k, k+w) of body b in clip c (inside one segment)."""
        for k0, n, sh, r0, e in self.seg.get((c, b), ()):
            if k0 <= k and k + w <= k0 + n:
                return sh, r0 + (k - k0), e
        raise KeyError((c, b, k, w))

    def robot(self, c, b, k, w, fields=('labels',)):
        sh, r0, e = self.locate_rows(c, b, k, w)
        s = self.shard(sh)
        out = decode_robot(s.blob('frames')[r0:r0 + w], s.meta, fields)
        if 'points' in fields and s.has('ground_point_contact'):
            out['point_contact'] = np.unpackbits(s.blob('ground_point_contact')[r0:r0 + w], axis=1).astype(bool)
            if s.has('object_point_contact'):
                out['object_point_contact'] = np.unpackbits(s.blob('object_point_contact')[r0:r0 + w], axis=1).astype(bool)
        return out

    def human(self, c, k, w):
        sh, r0, e = self.locate_rows(c, 0, k, w)
        s = self.shard(sh)
        return decode_human(s.blob('frames')[r0:r0 + w], s.meta)


class PairedWindows:
    """torch map-style Dataset of paired windows; batches are produced by __getitems__ (vectorised)."""

    def __init__(self, root, window=64, pair_mode='human2robot', n_bodies=2, robots=None, source_robots=None,
                 target_robots=None, split='train', val_permille=50, datasets=None, samples_per_epoch=1_000_000,
                 seed=0, fields=('labels',), with_objects=True, min_fps=None, max_fps=None, batch_same_bodies=False,
                 cache_size=65536):
        assert pair_mode in PAIR_MODES, pair_mode
        self.rd = Reader(root)
        cat = self.rd.cat
        self.window, self.mode, self.seed, self.epoch = int(window), pair_mode, int(seed), 0
        self.n_bodies = 2 if pair_mode != 'multi' else int(n_bodies)
        self.fields, self.with_objects, self.n = tuple(fields), with_objects, int(samples_per_epoch)
        self.batch_same_bodies = batch_same_bodies
        bi = cat.body_index
        allr = [b for b in cat.bodies if b != 'human']
        self.robots = set(bi[b] for b in (robots or allr) if b in bi)
        self.src_robots = set(bi[b] for b in (source_robots or [cat.bodies[i] for i in self.robots]) if b in bi)
        self.tgt_robots = set(bi[b] for b in (target_robots or [cat.bodies[i] for i in self.robots]) if b in bi)
        ds_names = cat.meta['datasets']
        clips, weights, avail = [], [], []
        for c in range(len(cat.clips)):
            if split == 'train' and cat.c_hash[c] < val_permille:
                continue
            if split == 'val' and cat.c_hash[c] >= val_permille:
                continue
            if datasets and ds_names[cat.c_dataset[c]] not in datasets:
                continue
            fps = float(cat.c_fps[c])
            if (min_fps and fps < min_fps) or (max_fps and fps > max_fps):
                continue
            bodies = {}
            for e in cat.entries_of_clip(c):
                b = int(cat.e_body[e])
                if b != 0 and b not in self.robots:
                    continue
                bodies.setdefault(b, []).extend(tuple(x) for x in cat.runs_of(e).tolist())
            if pair_mode in ('human2robot', 'robot2human') and 0 not in bodies:
                continue
            nrob = len([b for b in bodies if b != 0])
            if nrob < (2 if pair_mode == 'robot2robot' else 1) or len(bodies) < self.n_bodies:
                continue
            span = max(b for r in bodies.values() for (_, b) in r) - min(a for r in bodies.values() for (a, _) in r)
            if span < self.window:
                continue
            clips.append(c); weights.append(span); avail.append({b: sorted(r) for b, r in bodies.items()})
        if not clips:
            raise ValueError('no eligible clips for this selection')
        self.clips, self.avail = clips, avail
        self.cum = np.cumsum(np.asarray(weights, np.float64)); self.total = float(self.cum[-1])
        self._icache = OrderedDict(); self._cache_size = cache_size
        self.nj = {b: cat.meta['robots'][cat.bodies[b]]['nj'] for b in range(1, len(cat.bodies))}
        hum_e = {c: segs[0][4] for (c, b), segs in self.rd.seg.items() if b == 0}
        self.clip_has_obj = {c: bool(cat.e_obj[e]) for c, e in hum_e.items()}

    def __len__(self):
        return self.n

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    # --------------------------------------------------------------------------------------- sampling
    def _rng(self, idx):
        return random.Random((self.seed * 1_000_003 + self.epoch) * 4_294_967_311 + int(idx))

    def _pick_bodies(self, rng, have):
        rob = sorted(b for b in have if b != 0)
        m = self.mode
        if m == 'human2robot':
            t = [b for b in rob if b in self.tgt_robots]
            return (0, rng.choice(t)) if t else None
        if m == 'robot2human':
            s = [b for b in rob if b in self.src_robots]
            return (rng.choice(s), 0) if s else None
        if m == 'robot2robot':
            s = [b for b in rob if b in self.src_robots]
            if not s:
                return None
            a = rng.choice(s)
            t = [b for b in rob if b in self.tgt_robots and b != a]
            return (a, rng.choice(t)) if t else None
        pool = ([0] if 0 in have else []) + rob
        if len(pool) < self.n_bodies:
            return None
        return tuple(rng.sample(pool, self.n_bodies))

    def _intersection(self, ci, bodies):
        key = (ci, tuple(sorted(bodies)))
        r = self._icache.get(key)
        if r is not None:
            self._icache.move_to_end(key)
            return r
        av = self.avail[ci]
        cur = av[bodies[0]]
        for b in bodies[1:]:
            cur = _intersect(cur, av[b])
        W = self.window
        ivs = [(a, b) for a, b in cur if b - a >= W]
        cum = np.cumsum([b - a - W + 1 for a, b in ivs]) if ivs else np.zeros(0)
        r = (ivs, cum)
        self._icache[key] = r
        if len(self._icache) > self._cache_size:
            self._icache.popitem(last=False)
        return r

    def locate(self, idx, bodies=None):
        """(catalog clip index, bodies tuple, k) for sample idx (deterministic)."""
        rng = self._rng(idx)
        for _ in range(64):
            ci = min(bisect.bisect_right(self.cum, rng.random() * self.total), len(self.clips) - 1)
            have = self.avail[ci]
            if bodies is not None:
                bs = bodies if all(b in have for b in bodies) else None
            else:
                bs = self._pick_bodies(rng, have)
            if bs is None:
                continue
            ivs, cum = self._intersection(ci, bs)
            if not ivs:
                continue
            u = rng.randrange(int(cum[-1]))
            j = int(np.searchsorted(cum, u, side='right'))
            a, _ = ivs[j]
            k = a + u - (int(cum[j - 1]) if j else 0)
            return self.clips[ci], tuple(bs), k
        raise RuntimeError(f'could not place sample {idx} (bodies={bodies})')

    # --------------------------------------------------------------------------------------- batched fetch
    def _gather(self, locs, idx, body_of, ar):
        """Group samples idx by shard -> yields (shard, sample indices, structured [n,W] records, entries)."""
        groups = {}
        rd, W = self.rd, self.window
        for i in idx:
            c, _, k = locs[i]
            sh, r0, e = rd.locate_rows(c, int(body_of[i]), k, W)
            groups.setdefault(sh, []).append((i, r0, e))
        for sh, lst in groups.items():
            s = rd.shard(sh)
            ii = np.array([x[0] for x in lst]); rows = np.array([x[1] for x in lst])[:, None] + ar
            yield s, ii, s.blob('frames')[rows], np.array([x[2] for x in lst])

    def fetch_batch(self, locs):
        rd, cat, W = self.rd, self.rd.cat, self.window
        B = len(locs); ns = len(locs[0][1])
        ar = np.arange(W)
        cs = np.array([l[0] for l in locs], np.int64)
        out = dict(clip=cs, k=np.array([l[2] for l in locs], np.int64), fps=cat.c_fps[cs].astype(np.float32),
                   bodies=np.array([l[1] for l in locs], np.int16))
        hum_obj = {}                                            # sample -> (pos, quat) in the human Z-up frame
        for j in range(ns):
            bj = np.array([l[1][j] for l in locs])
            d = dict(is_human=bj == 0, body=bj.astype(np.int16))
            rob = np.flatnonzero(bj != 0)
            if len(rob):
                Q = 7 + max(self.nj[int(b)] for b in bj[rob])
                qp = np.zeros((B, W, Q), np.float32); qm = np.zeros((B, Q), bool)
                lab = {n: np.zeros((B, W, S.NREG), bool) for _, n in LABEL_PAIRS + OBJ_PAIRS}
                flags = np.zeros((B, W), np.uint8)
                scale = np.ones(B, np.float32); floor = np.zeros(B, np.float32); objv = np.zeros(B, bool)
                extra = {}
                for s, ii, fr, es in self._gather(locs, rob, bj, ar):
                    dec = decode_robot(fr, s.meta, self.fields)
                    nq = dec['qpos'].shape[-1]
                    qp[ii, :, :nq] = dec['qpos']; qm[ii, :nq] = True; flags[ii] = dec['flags']
                    scale[ii] = cat.e_scale[es]; floor[ii] = cat.e_floor[es]
                    for _, n in LABEL_PAIRS:
                        lab[n][ii] = dec[n]
                    if 'object_contact' in dec:
                        for _, n in OBJ_PAIRS:
                            lab[n][ii] = dec[n]
                        objv[ii] = True
                    for n in FLOAT_FIELDS:
                        if n in dec:
                            if n not in extra:
                                extra[n] = np.full((B,) + dec[n].shape[1:], np.inf if 'clearance' in n else np.nan, np.float32)
                            extra[n][ii] = dec[n]
                d.update(qpos=qp, q_mask=qm, flags=flags, scale=scale, floor_offset=floor, object_valid=objv, **lab, **extra)
            hm = np.flatnonzero(bj == 0)
            if len(hm):
                hd = None
                for s, ii, fr, es in self._gather(locs, hm, bj, ar):
                    dec = decode_human(fr, s.meta)
                    if hd is None:
                        hd = {n: np.zeros((B,) + v.shape[1:], np.float32) for n, v in dec.items() if not n.startswith('object')}
                    for n in hd:
                        hd[n][ii] = dec[n]
                    if 'object_pos' in dec:
                        for t, i in enumerate(ii):
                            hum_obj[int(i)] = (dec['object_pos'][t], dec['object_quat'][t])
                h = bj == 0
                hd.update(betas=np.where(h[:, None], cat.c_betas[cs], 0).astype(np.float32),
                          num_betas=np.where(h, cat.c_nbetas[cs], 0).astype(np.int16),
                          up_y=h & (cat.c_up[cs] == 1), gender=np.where(h, cat.c_gender[cs], 0).astype(np.int8),
                          floor_z=np.where(h, cat.c_floor_h[cs], 0).astype(np.float32))
                d.update(hd)
            out[f'slot{j}'] = d
        if self.with_objects:
            need = [i for i in range(B) if i not in hum_obj and self.clip_has_obj.get(int(cs[i]), False)]
            if need:
                for s, ii, fr, es in self._gather(locs, need, np.zeros(B, np.int64), ar):
                    dec = decode_human(fr, s.meta)
                    for t, i in enumerate(ii):
                        hum_obj[int(i)] = (dec['object_pos'][t], dec['object_quat'][t])
            if hum_obj:
                K = max(v[0].shape[1] for v in hum_obj.values())
                op = np.full((B, W, K, 3), np.nan, np.float32); oq = np.full((B, W, K, 4), np.nan, np.float32)
                for i, (p, q) in hum_obj.items():
                    op[i, :, :p.shape[1]] = p; oq[i, :, :q.shape[1]] = q
                out['object_pos_human'] = op; out['object_quat'] = oq
                for j in range(ns):
                    d = out[f'slot{j}']
                    if 'scale' in d:
                        p = op * d['scale'][:, None, None, None]
                        p[..., 2] -= d['floor_offset'][:, None, None]
                        d['object_pos'] = p; d['object_quat'] = oq
        return _to_torch(out)

    def __getitem__(self, idx):
        return self.fetch_batch([self.locate(idx)])

    def __getitems__(self, idxs):
        if not self.batch_same_bodies:
            return self.fetch_batch([self.locate(i) for i in idxs])
        first = self.locate(idxs[0])
        return self.fetch_batch([first] + [self.locate(i, bodies=first[1]) for i in idxs[1:]])


def _to_torch(d):
    import torch
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _to_torch(v)
        elif isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(np.ascontiguousarray(v))
        else:
            out[k] = v
    return out


def _intersect(A, B):
    out, i, j = [], 0, 0
    while i < len(A) and j < len(B):
        a = max(A[i][0], B[j][0]); b = min(A[i][1], B[j][1])
        if a < b:
            out.append((a, b))
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return out


def collate(batch):
    """__getitems__ already returns a collated batch (dict of tensors); pass it through."""
    if isinstance(batch, dict):
        return batch
    if isinstance(batch, (list, tuple)) and len(batch) == 1 and isinstance(batch[0], dict):
        return batch[0]
    raise TypeError('use DataLoader(dataset, batch_size=B, collate_fn=collate): batches come from __getitems__')


def clip_meta(root) -> dict:
    """clip name -> human / provenance metadata (betas, gender, up axis, floor, text annotation, objects)."""
    out = {}
    with open(Path(root) / 'clips.jsonl') as f:
        for line in f:
            d = json.loads(line)
            out[d['clip']] = d
    return out
