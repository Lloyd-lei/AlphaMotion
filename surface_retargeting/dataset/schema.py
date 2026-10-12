"""Record layouts, tiers and codecs of the UMR paired dataset (numpy-only; torch decode lives in dataset.py).

Per-frame records are array-of-structs (one contiguous record per frame) so a training window is one contiguous
read per body.  Everything is little endian.

Robot record (per robot shard, stored frame rate, see README):
  core      root_pos f4[3] (metres, exact copy) | root_q u8 (wxyz, smallest-three 3x20 bit, sign kept) |
            joints u2[nj] (lo + code*step, per-shard per-joint lo/step) | flags u1 |
            gc, gs, hgc, hgs u2  (13-bit region masks: robot ground contact / slip, human ground contact / slip)
            + HOI shards: oc, os, hoc, hos u2 (object contact / slip, human object contact / slip)
  standard  core + gclr i1[13], hgclr i1[13] (signed-log clearance), gspd u1[13] (log slip speed)
            + HOI: oclr i1[13], hoclr? (not produced by the factory) , ospd u1[13]
  full      standard + gclr_f f2[13], hgclr_f f2[13], gspd_f f2[13], pen f2[3] (+ HOI oclr_f, ospd_f)
            + separate blobs ground_point_contact u1[512], object_point_contact u1[512] (packbits of 4096 points)

Human record (per human shard, same stored-frame grid as the robots of the clip):
  transl f4[3] (SMPL-X trans, source frame and up axis) | body_aa i2[22,3] (global_orient + 21 body joints,
  axis-angle, symmetric per-shard scale) | optional extra_aa i2[33,3] (jaw, leye, reye, 15+15 hand joints) |
  HOI shards: obj_pos f4[K,3], obj_quat f4[K,4] (object pose in the human metric Z-up frame, floor at z=0)
"""
from __future__ import annotations

import numpy as np

REGIONS = ['trunk', 'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot',
           'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm', 'L_hand', 'R_hand']
NREG = len(REGIONS)
FLAG_BITS = {1: 'ground pen > 0.5 mm', 2: 'object pen > 0.5 mm', 4: 'self pen > 0.5 mm',
             8: 'required ground contact not in [0,1] mm', 16: 'required object contact not in [0,1] mm',
             64: 'decode guard: the DECODED pose may exceed a strict tolerance (pen 0.5 mm / contact band)',
             128: 'a dropped intermediate source frame (decimation) carried a flag'}
FLAG_DECODE = 64
FLAG_DROPPED = 128
TIERS = ('core', 'standard', 'full')
SCHEMA_VERSION = 1

# ------------------------------------------------------------------------------------------------ bit masks
_W = (1 << np.arange(NREG)).astype(np.uint16)


def pack_regions(b: np.ndarray) -> np.ndarray:
    """bool [T,13] -> uint16 [T]."""
    b = np.asarray(b, bool)
    assert b.shape[-1] == NREG, b.shape
    return (b.astype(np.uint16) * _W).sum(-1).astype(np.uint16)


def unpack_regions(m: np.ndarray) -> np.ndarray:
    """uint16 [...] -> bool [...,13]."""
    return (np.asarray(m, np.uint16)[..., None] & _W) != 0


# ------------------------------------------------------------------------------------------------ quaternions
# "smallest three" in one uint64, sign preserved: bits [0,20) [20,40) [40,60) = the three non-largest components
# (in order, each in [-1/sqrt2, 1/sqrt2], 20-bit uniform), [60,62) = index of the largest-|.| component,
# bit 62 = that component is negative.  Max component error 6.7e-7 -> rotation error ~1.5e-6 rad (int16 wxyz: 5.5e-5).
_Q3_BITS = 20
_Q3_MAX = (1 << _Q3_BITS) - 1
_Q3_R = 1.0 / np.sqrt(2.0)


def enc_quat(q):
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    sh = q.shape[:-1]
    q = q.reshape(-1, 4)
    i = np.abs(q).argmax(1)
    neg = q[np.arange(len(q)), i] < 0
    mask = np.ones_like(q, bool); mask[np.arange(len(q)), i] = False
    rest = q[mask].reshape(-1, 3)
    c = np.round((np.clip(rest, -_Q3_R, _Q3_R) + _Q3_R) / (2 * _Q3_R) * _Q3_MAX).astype(np.uint64)
    code = c[:, 0] | (c[:, 1] << np.uint64(20)) | (c[:, 2] << np.uint64(40)) | (i.astype(np.uint64) << np.uint64(60)) | \
        (neg.astype(np.uint64) << np.uint64(62))
    return code.reshape(sh)


def dec_quat(code):
    code = np.asarray(code, np.uint64)
    sh = code.shape
    code = code.reshape(-1)
    m = np.uint64(_Q3_MAX)
    rest = np.stack([(code >> np.uint64(s)) & m for s in (0, 20, 40)], 1).astype(np.float64) / _Q3_MAX * (2 * _Q3_R) - _Q3_R
    i = ((code >> np.uint64(60)) & np.uint64(3)).astype(np.int64)
    neg = ((code >> np.uint64(62)) & np.uint64(1)).astype(bool)
    big = np.sqrt(np.clip(1.0 - (rest ** 2).sum(1), 0.0, 1.0))
    big = np.where(neg, -big, big)
    q = np.empty((len(code), 4))
    mask = np.ones((len(code), 4), bool); mask[np.arange(len(code)), i] = False
    q[mask] = rest.reshape(-1)
    q[np.arange(len(code)), i] = big
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.astype(np.float32).reshape(sh + (4,))


# ------------------------------------------------------------------------------------------------ joints (uint16 affine)
def joint_quant_params(x: np.ndarray, margin=1e-6):
    """Per-column lo/step covering the data exactly (step >= 1e-9). x [N,nj] float."""
    x = np.asarray(x, np.float64)
    lo = x.min(0) - margin
    hi = x.max(0) + margin
    step = np.maximum((hi - lo) / 65535.0, 1e-9)
    return lo, step


def enc_joints(x, lo, step):
    c = np.round((np.asarray(x, np.float64) - lo) / step)
    if (c < 0).any() or (c > 65535).any():
        raise ValueError('joint value outside quantisation range')
    return c.astype(np.uint16)


def dec_joints(c, lo, step):
    return (np.asarray(lo, np.float64) + np.asarray(c, np.float64) * np.asarray(step, np.float64)).astype(np.float32)


# ------------------------------------------------------------------------------------------------ axis-angle (int16 symmetric)
def aa_scale(x):
    m = float(np.abs(np.asarray(x)).max()) if np.size(x) else 0.0
    return max(m, 1e-6) / 32767.0


def enc_aa(x, scale):
    return np.clip(np.round(np.asarray(x, np.float64) / scale), -32767, 32767).astype(np.int16)


def dec_aa(c, scale):
    return (np.asarray(c, np.float32) * np.float32(scale)).astype(np.float32)


# ------------------------------------------------------------------------------------------------ signed-log distance (int8)
# code = round(K * sign(d) * log1p(|d| / D0)), clipped to [-127, 127]; -128 = missing (inf / nan / far sentinel).
# D0 = 0.5 mm, K = 14.6 -> |code| 127 <-> 3.0 m; resolution 0.034 mm at 0, ~0.07 mm at 1 mm, ~0.7 mm at 1 cm.
SD_D0, SD_K, SD_MISSING = 5e-4, 14.6, -128


def enc_sdist(d, far=9.0):
    d = np.asarray(d, np.float64)
    bad = ~np.isfinite(d) | (np.abs(d) >= far)
    dd = np.where(bad, 0.0, d)
    c = np.clip(np.round(SD_K * np.sign(dd) * np.log1p(np.abs(dd) / SD_D0)), -127, 127)
    return np.where(bad, SD_MISSING, c).astype(np.int8)


def dec_sdist(c):
    c = np.asarray(c, np.float32)
    d = np.sign(c) * SD_D0 * np.expm1(np.abs(c) / SD_K)
    return np.where(c == SD_MISSING, np.float32(np.inf), d).astype(np.float32)


# log speed (uint8): code = round(K * log1p(v / V0)), V0 = 1 mm/s, K = 27.6 (254 <-> 10 m/s); 255 = not judged (nan)
SP_V0, SP_K, SP_MISSING = 1e-3, 27.6, 255


def enc_speed(v):
    v = np.asarray(v, np.float64)
    bad = ~np.isfinite(v)
    c = np.clip(np.round(SP_K * np.log1p(np.maximum(np.where(bad, 0, v), 0) / SP_V0)), 0, 254)
    return np.where(bad, SP_MISSING, c).astype(np.uint8)


def dec_speed(c):
    c = np.asarray(c, np.float32)
    v = SP_V0 * np.expm1(c / SP_K)
    return np.where(c == SP_MISSING, np.float32(np.nan), v).astype(np.float32)


# ------------------------------------------------------------------------------------------------ record dtypes
def robot_dtype(nj: int, tier: str, has_object: bool) -> np.dtype:
    assert tier in TIERS, tier
    f = [('root_pos', '<f4', (3,)), ('root_q', '<u8'), ('joints', '<u2', (nj,)), ('flags', 'u1'),
         ('gc', '<u2'), ('gs', '<u2'), ('hgc', '<u2'), ('hgs', '<u2')]
    if has_object:
        f += [('oc', '<u2'), ('os', '<u2'), ('hoc', '<u2'), ('hos', '<u2')]
    if tier in ('standard', 'full'):
        f += [('gclr', 'i1', (NREG,)), ('hgclr', 'i1', (NREG,)), ('gspd', 'u1', (NREG,))]
        if has_object:
            f += [('oclr', 'i1', (NREG,)), ('ospd', 'u1', (NREG,))]
    if tier == 'full':
        f += [('gclr_f', '<f2', (NREG,)), ('hgclr_f', '<f2', (NREG,)), ('gspd_f', '<f2', (NREG,)), ('pen', '<f2', (3,))]
        if has_object:
            f += [('oclr_f', '<f2', (NREG,)), ('ospd_f', '<f2', (NREG,))]
    return np.dtype(f)


def human_dtype(has_extra: bool, n_obj: int) -> np.dtype:
    f = [('transl', '<f4', (3,)), ('body_aa', '<i2', (22, 3))]
    if has_extra:
        f += [('extra_aa', '<i2', (33, 3))]
    if n_obj:
        f += [('obj_pos', '<f4', (n_obj, 3)), ('obj_quat', '<f4', (n_obj, 4))]
    return np.dtype(f)


def rotmat_to_quat(R):
    """[...,3,3] -> wxyz [...,4] (Shepperd, w >= 0 not enforced; continuity enforced along axis 0 when 2-D+)."""
    R = np.asarray(R, np.float64)
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    tr = m00 + m11 + m22
    q = np.zeros(R.shape[:-2] + (4,))
    c0 = tr > 0
    s = np.sqrt(np.maximum(tr + 1.0, 1e-12)) * 2
    q0 = np.stack([0.25 * s, (R[..., 2, 1] - R[..., 1, 2]) / s, (R[..., 0, 2] - R[..., 2, 0]) / s, (R[..., 1, 0] - R[..., 0, 1]) / s], -1)
    c1 = (~c0) & (m00 > m11) & (m00 > m22)
    s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 1e-12)) * 2
    q1 = np.stack([(R[..., 2, 1] - R[..., 1, 2]) / s1, 0.25 * s1, (R[..., 0, 1] + R[..., 1, 0]) / s1, (R[..., 0, 2] + R[..., 2, 0]) / s1], -1)
    c2 = (~c0) & (~c1) & (m11 > m22)
    s2 = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 1e-12)) * 2
    q2 = np.stack([(R[..., 0, 2] - R[..., 2, 0]) / s2, (R[..., 0, 1] + R[..., 1, 0]) / s2, 0.25 * s2, (R[..., 1, 2] + R[..., 2, 1]) / s2], -1)
    s3 = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 1e-12)) * 2
    q3 = np.stack([(R[..., 1, 0] - R[..., 0, 1]) / s3, (R[..., 0, 2] + R[..., 2, 0]) / s3, (R[..., 1, 2] + R[..., 2, 1]) / s3, 0.25 * s3], -1)
    q = np.where(c0[..., None], q0, np.where(c1[..., None], q1, np.where(c2[..., None], q2, q3)))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    if q.ndim >= 2 and q.shape[0] > 1:          # sign continuity along time: s_t = s_{t-1} * sign(<q_t, q_{t-1}>)
        sg = np.where((q[1:] * q[:-1]).sum(-1) < 0, -1.0, 1.0)
        s = np.concatenate([np.ones((1,) + sg.shape[1:]), np.cumprod(sg, axis=0)], 0)
        q = q * s[..., None]
    return q


def quat_to_rotmat(q):
    q = np.asarray(q, np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))
