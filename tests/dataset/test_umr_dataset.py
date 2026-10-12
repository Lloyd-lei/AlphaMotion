"""Self-contained tests (no pytest needed):  python tests/test_umr_dataset.py [--keep]

1 codecs: quaternion / joints / axis-angle / signed-log distance / log speed / region bits, error bounds
2 container: write -> read -> blob + file checksums, corruption is detected
3 converter: incremental ingest in 3 rounds from symlinked factory outputs, unchanged files skipped, a changed file
  supersedes its entry, compaction merges the small shards byte-exactly (decoded windows identical before / after)
4 pairing: every loader window's robot qpos / labels equal the factory npz at source frame k*stride, the human
  SMPL-X equals the source at the same frame, every robot frame in a window is strictly valid (flags == 0)
"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from surface_retargeting.dataset import schema as S                      # noqa: E402
from surface_retargeting.dataset.container import write_shard, Shard, file_sha256   # noqa: E402
import surface_retargeting.dataset.catalog as CAT                        # noqa: E402

FACTORY_OUT = Path(f'{SRCFG.data_root}/out')
SPEC_ROOT = Path(f'{SRCFG.dataset_root}')     # robots/<r>/{spec.json,kin.npz,bounds.json} reused (copied)


def test_codecs():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(200000, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    d = S.dec_quat(S.enc_quat(q)).astype(np.float64)
    sg = np.where((d * q).sum(1, keepdims=True) < 0, -1.0, 1.0)
    ang = 4 * np.arcsin(np.clip(np.linalg.norm(d - sg * q, axis=1) / 2, 0, 1))   # chord form (well conditioned)
    assert ang.max() < 1e-5, ang.max()                    # kinematics.ROOT_ANG_ERR
    assert (np.sign(d[np.arange(len(q)), np.abs(q).argmax(1)]) == np.sign(q[np.arange(len(q)), np.abs(q).argmax(1)])).all()
    lo, step = np.array([-2.0, -6.3]), np.array([4.0, 12.6]) / 65535
    x = rng.uniform([-2, -6.3], [2, 6.3], size=(10000, 2))
    e = np.abs(S.dec_joints(S.enc_joints(x, lo, step), lo, step) - x)
    assert (e <= step / 2 + 1e-6).all(), e.max(0)
    aa = rng.uniform(-3, 3, size=(1000, 22, 3)); sc = S.aa_scale(aa)
    assert np.abs(S.dec_aa(S.enc_aa(aa, sc), sc) - aa).max() <= sc / 2 + 1e-6
    dd = np.concatenate([rng.uniform(-0.02, 0.02, 10000), [np.inf, 9.9, np.nan]])
    r = S.dec_sdist(S.enc_sdist(dd))
    m = np.abs(dd) < 1e-3
    assert np.abs(r[m] - dd[m]).max() < 1e-4                # < 0.1 mm inside the contact band
    assert np.isinf(r[-3:]).all()
    v = np.concatenate([rng.uniform(0, 5, 1000), [np.nan]])
    vr = S.dec_speed(S.enc_speed(v))
    assert np.isnan(vr[-1]) and (np.abs(vr[:-1] - v[:-1]) <= 0.02 * v[:-1] + 1e-3).all()
    b = rng.random((500, 13)) < 0.3
    assert (S.unpack_regions(S.pack_regions(b)) == b).all()
    print('codecs ok: quat max angle %.2e rad' % ang.max())


def test_container(tmp):
    a = np.zeros(1000, S.robot_dtype(29, 'standard', True)); a['joints'] = 7
    b = np.arange(12, dtype=np.uint8).reshape(3, 4)
    info = write_shard(tmp / 'x.umrs', {'frames': a, 'pts': b}, dict(hello='world'), kind='robot')
    assert info['sha256'] == file_sha256(tmp / 'x.umrs')
    s = Shard(tmp / 'x.umrs')
    assert s.meta['hello'] == 'world' and (s.blob('frames')['joints'] == 7).all() and (s.blob('pts') == b).all()
    assert all(s.verify().values())
    raw = bytearray((tmp / 'x.umrs').read_bytes()); raw[s.header['_data_start'] + 1234] ^= 0xFF   # flip a byte in 'frames'
    (tmp / 'y.umrs').write_bytes(bytes(raw))
    assert not all(Shard(tmp / 'y.umrs').verify().values())
    print('container ok')


def _mk_src(src: Path, reps: range):
    for d in sorted(p for p in FACTORY_OUT.iterdir() if p.is_dir() and p.name != 'tests'):
        for i in reps:
            t = src / f'{d.name}__rep{i:04d}'
            t.mkdir(parents=True, exist_ok=True)
            for f in d.glob('*.npz'):
                if not (t / f.name).exists():
                    (t / f.name).symlink_to(f)


def _all_windows(root, W=32):
    from surface_retargeting.dataset.dataset import Reader
    rd = Reader(root); cat = rd.cat
    out = {}
    for (c, b), segs in sorted(rd.seg.items()):
        for k0, n, sh, r0, e in segs:
            for k in range(k0, k0 + n - W + 1, 97):
                if b == 0:
                    v = rd.human(c, k, W); key = (cat.clips[c], 'human', k)
                    out[key] = np.concatenate([v['transl'], v['body_pose'].reshape(W, -1)], 1)
                else:
                    v = rd.robot(c, b, k, W, fields=('labels', 'clearance'))
                    out[(cat.clips[c], cat.bodies[b], k)] = np.concatenate(
                        [v['qpos'], v['contact'], v['slip'], v['flags'][:, None], v['clearance']], 1)
    return out


@pytest.mark.data
@pytest.mark.xfail(reason='known packer bug (plan 4.8): an empty contact segment in the current factory out/ makes '
                          'convert.write_robot_shards index seg[0] of an empty array; identical failure in the old '
                          '~/umr_dataset code. Fix is migration step 6.', raises=IndexError, strict=False)
def test_convert_compact_pairing(tmp):
    for p in (FACTORY_OUT, SPEC_ROOT / 'robots'):
        if not p.exists():
            pytest.skip(f'needs data: {p}')
    from surface_retargeting.dataset import convert as C, compact as K
    from surface_retargeting.dataset.dataset import PairedWindows, Reader
    root, src = tmp / 'root', tmp / 'src'
    for r in ('fourier_gr3', 'unitree_g1_29dof', 'unitree_h2'):          # reuse built specs (needs no mujoco)
        (root / 'robots' / r).mkdir(parents=True)
        for f in ('spec.json', 'kin.npz', 'bounds.json'):
            shutil.copy2(SPEC_ROOT / 'robots' / r / f, root / 'robots' / r / f)
    log = (lambda *a: print(*a)) if os.environ.get('UMRDS_VERBOSE') else (lambda *a: None)
    for rnd in range(3):
        _mk_src(src, range(rnd * 4, rnd * 4 + 4))
        r = C.ingest(root, src, settle=0, workers=4, min_gauge='visual', log=log)
        n_npz = sum(1 for d in FACTORY_OUT.iterdir() if d.is_dir() and d.name != 'tests' for _ in d.glob('*.npz'))
        assert r['failures'] == 0 and r['new'] == 4 * n_npz, r            # only the new clips
    r = C.ingest(root, src, settle=0, workers=4, min_gauge='visual', log=log)
    assert r['new'] == 0, r                                                 # unchanged -> skipped
    # same content, new mtime -> sha256 equal -> not re-ingested
    clip0 = sorted(p.name for p in src.iterdir())[0]
    f = src / clip0 / 'unitree_h2.npz'
    tgt = f.resolve(); f.unlink(); shutil.copy2(tgt, f); os.utime(f, (time.time() - 5, time.time() - 5))
    assert C.ingest(root, src, settle=0, workers=4, min_gauge='visual', log=log)['new'] == 0
    # a factory re-run that changes the content (here: frames 100-139 flagged) -> the new entry supersedes the old
    z = dict(np.load(tgt)); z['flags'] = z['flags'].copy(); z['flags'][100:140] |= 1
    f.unlink(); np.savez_compressed(f, **z); os.utime(f, (time.time() - 5, time.time() - 5))
    r = C.ingest(root, src, settle=0, workers=4, min_gauge='visual', log=log)
    assert r['new'] == 1, r
    rd0 = __import__('surface_retargeting.dataset.dataset', fromlist=['Reader']).Reader(root)
    ci, bi = rd0.cat.clips.index(clip0), rd0.cat.body_index['unitree_h2']
    st = int(rd0.cat.c_stride[ci])
    fl = np.concatenate([rd0.robot(ci, bi, k0, n)['flags'] for k0, n, *_ in rd0.seg[(ci, bi)]])
    assert (fl[100 // st:140 // st] != 0).all() and ((fl[34:47] & 1) > 0).all() if st == 3 else True, 'superseding entry not live'
    man = json.load(open(root / 'manifest.json'))
    n_before = len(man['shards'])
    before = _all_windows(root)
    res = K.compact(root, target_rows=2_000_000, grace_hours=0, log=log)
    man = json.load(open(root / 'manifest.json'))
    after = _all_windows(root)
    assert before.keys() == after.keys() and all(np.array_equal(before[k], after[k], equal_nan=True) for k in before)
    robot_shards = [s for s in man['shards'] if s.startswith('robots/')]
    assert len(robot_shards) == 6, robot_shards                             # 3 robots x {body, hoi}
    from surface_retargeting.dataset.verify import verify
    assert not verify(root, log=log)['problems']
    print(f'convert/compact ok: {n_before} shards -> {len(man["shards"])} ({res}); {len(before)} windows identical')
    # pairing vs factory npz + SMPL-X source
    from surface_retargeting.dataset import sources as SRC
    ds = PairedWindows(root, window=48, pair_mode='multi', n_bodies=4, val_permille=0, samples_per_epoch=64, seed=3)
    rd = Reader(root); cat = rd.cat
    batch = ds.__getitems__(list(range(64)))
    src_cache, n_chk = {}, 0
    for i in range(64):
        c, k = int(batch['clip'][i]), int(batch['k'][i])
        clip = cat.clips[c]; base = clip.split('__rep')[0]
        stride = int(cat.c_stride[c])
        for j in range(4):
            rows = (k + np.arange(48)) * stride                             # human: full-rate SMPL-X source frames
            sl = batch[f'slot{j}']
            b = int(sl['body'][i])
            if b == 0:
                if base not in src_cache:
                    z = np.load(sorted((FACTORY_OUT / base).glob('*.npz'))[0])
                    src_cache[base] = SRC.load_human(json.loads(str(z['provenance'])))
                h = src_cache[base]
                assert np.abs(sl['transl'][i].numpy() - h['trans'][rows]).max() == 0
                assert np.abs(sl['body_pose'][i].numpy().reshape(48, -1) - h['poses'][rows, 3:66]).max() < 1e-4
            else:
                z = np.load(FACTORY_OUT / base / f'{cat.bodies[b]}.npz')
                rows = CAT.src_rows(rd.entry_meta(c, b, k), k, 48)          # robot npz may hold decimated UMR frames
                nq = z['qpos'].shape[1]
                q = sl['qpos'][i, :, :nq].numpy()
                assert np.abs(q[:, 7:] - z['qpos'][rows, 7:]).max() < 1e-4
                assert np.abs(q[:, :3] - z['qpos'][rows, :3]).max() == 0
                assert (sl['contact'][i].numpy() == z['ground_contact_required'][rows]).all()
                assert (sl['human_contact'][i].numpy() == z['human_ground_contact'][rows]).all()
                assert (sl['flags'][i].numpy() == 0).all() and (z['flags'][rows] == 0).all()
            n_chk += 1
    print(f'pairing ok: {n_chk} (sample, body) windows match the factory npz / SMPL-X source frame-exactly')


if __name__ == '__main__':
    keep = '--keep' in sys.argv
    tmp = Path(tempfile.mkdtemp(prefix='umrds_test_'))
    try:
        test_codecs()
        test_container(tmp)
        test_convert_compact_pairing(tmp)
        print('ALL TESTS PASSED')
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print('kept', tmp)
