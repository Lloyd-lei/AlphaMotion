"""WP-OPS tests: GPU lease, two-partition guard, process receipts (CPU only, seconds)."""
from __future__ import annotations
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from am2exp_v1.common import paths  # noqa: E402
from am2exp_v1.ops import guard as G  # noqa: E402
from am2exp_v1.ops import lease as L  # noqa: E402
from am2exp_v1.ops import receipts as R  # noqa: E402

ENV = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''}


# ------------------------------------------------------------------ paths
def test_paths_contract():
    assert paths.ROOT.name == 'alphamotion_scaling_20261004' and paths.ROOT.exists()
    assert paths.CODE == paths.ROOT / 'am2exp_v1'
    assert str(paths.OUT) == '/var/tmp/alphamotion/am2exp'
    assert Path(paths.PY).exists() and Path(paths.PYROT).exists()
    added = paths.add_sealed_paths()
    for want in ('orchestration', 'base', 'legacy', 'prismatic'):
        assert any(p.endswith(want) for p in added)
    assert str(paths.ROOT) in added
    assert os.stat(paths.GPU_LOCK).st_ino == paths.GPU_LOCK_INODE


# ------------------------------------------------------------------ lease
@pytest.fixture()
def tmp_lock(tmp_path):
    p = tmp_path / 'gpu_queue.lock'
    p.touch()
    return p


def test_second_taker_gets_blocking_io_error(tmp_lock):
    a = L.GpuLease(tmp_lock, expected_inode=os.stat(tmp_lock).st_ino).acquire()
    try:
        assert L.precheck(tmp_lock, None)['held']
        with pytest.raises(BlockingIOError):           # same process, other open file description
            L.GpuLease(tmp_lock, expected_inode=None).acquire()
        code = textwrap.dedent(f"""
            import sys; sys.path.insert(0, {str(ROOT)!r})
            from am2exp_v1.ops.lease import GpuLease
            try:
                GpuLease({str(tmp_lock)!r}, expected_inode=None).acquire()
            except BlockingIOError:
                sys.exit(7)
            sys.exit(0)""")
        rc = subprocess.run([sys.executable, '-c', code], env=ENV, timeout=60).returncode
        assert rc == 7                                     # another process
    finally:
        a.release()
    assert not L.precheck(tmp_lock, None)['held']
    L.GpuLease(tmp_lock, expected_inode=None).acquire().release()  # free again after release


def test_lease_never_recreates_and_checks_inode(tmp_path, tmp_lock):
    missing = tmp_path / 'nope.lock'
    with pytest.raises(L.LeaseError):
        L.GpuLease(missing, expected_inode=None).acquire()
    assert not missing.exists()
    with pytest.raises(L.LeaseError):
        L.GpuLease(tmp_lock, expected_inode=os.stat(tmp_lock).st_ino + 1).acquire()


def test_lease_inherited_by_child(tmp_lock):
    lease = L.GpuLease(tmp_lock, expected_inode=None).acquire()
    try:
        fd = lease.handoff_fd()
        code = textwrap.dedent(f"""
            import json, os, sys; sys.path.insert(0, {str(ROOT)!r})
            from am2exp_v1.ops.lease import GpuLease, precheck
            l = GpuLease.inherit({fd}, path={str(tmp_lock)!r}, expected_inode=None)
            print(json.dumps({{'inherited': l.inherited, 'held': precheck({str(tmp_lock)!r}, None)['held']}}))
            l.release()  # no-op for an inherited lease""")
        out = subprocess.run([sys.executable, '-c', code], env=ENV, timeout=60, pass_fds=(fd,),
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert json.loads(out.stdout) == {'inherited': True, 'held': True}
        assert L.precheck(tmp_lock, None)['held']           # child exit did not drop the parent's lease
    finally:
        lease.release()


def test_real_lock_precheck_is_read_only():
    st = os.stat(paths.GPU_LOCK)
    rep = L.precheck()
    assert rep['inode'] == paths.GPU_LOCK_INODE and isinstance(rep['held'], bool)
    st2 = os.stat(paths.GPU_LOCK)
    assert (st.st_ino, st.st_mtime_ns, st.st_size) == (st2.st_ino, st2.st_mtime_ns, st2.st_size)


def test_real_lock_second_taker(tmp_path):
    """On the real ROOT/gpu_queue.lock: take it for < 1 s, a second taker is refused, release."""
    if L.precheck()['held']:
        pytest.skip('real GPU lease is held by someone else right now (never interfere)')
    with L.GpuLease() as lease:
        assert lease.receipt['inode'] == paths.GPU_LOCK_INODE
        with pytest.raises(BlockingIOError):
            L.GpuLease().acquire()
    assert not L.precheck()['held']


# ------------------------------------------------------------------ guard
def _cap(**kw):
    base = {'expected_unique_bytes': paths.PLAN_IDENTITY_BYTES, 'available_bytes': 200 * G.GiB,
            'remaining_bytes': 100 * G.GiB, 'required_available_bytes': 116 * G.GiB}
    base.update(kw)
    return lambda: base


def test_guard_passes_today_and_prints_both_partitions():
    out = subprocess.run([paths.PY, str(paths.CODE / 'ops' / 'guard.py'), '--json'], env=ENV, timeout=120,
                         capture_output=True, text=True)
    first = out.stdout.splitlines()[0]
    print(first)
    assert out.returncode == 0, out.stdout + out.stderr
    assert first.startswith('[guard PASS]') and '/home free' in first and '| / free' in first
    rep = json.loads(out.stdout.split('\n', 1)[1])
    assert rep['passed'] and rep['home']['plan_identity_ok']
    assert rep['home']['plan_expected_unique_bytes'] == 200_505_865_539
    assert rep['root']['free_bytes'] >= 20 * G.GiB
    assert rep['home']['free_bytes'] >= rep['home']['downloader_remaining_bytes'] + 16 * G.GiB


def test_guard_rules_with_injected_capacity(tmp_path):
    ok = G.check(capacity_fn=_cap(), root_path=tmp_path, home_allocation_bytes=0)
    assert ok['passed'] and ok['home']['required_bytes'] == 116 * G.GiB
    with pytest.raises(G.CapacityError, match='identity'):
        G.check(capacity_fn=_cap(expected_unique_bytes=1), root_path=tmp_path)
    with pytest.raises(G.CapacityError, match='/home free'):
        G.check(capacity_fn=_cap(available_bytes=115 * G.GiB), root_path=tmp_path, home_allocation_bytes=0)
    # downloader's own requirement can only tighten the rule
    with pytest.raises(G.CapacityError, match='/home free'):
        G.check(capacity_fn=_cap(required_available_bytes=250 * G.GiB), root_path=tmp_path, home_allocation_bytes=0)
    # root partition: a pending OUT allocation bigger than free - 20 GiB fails
    import shutil
    free = shutil.disk_usage(tmp_path).free
    with pytest.raises(G.CapacityError, match='/ free'):
        G.check(capacity_fn=_cap(), root_path=tmp_path, out_allocation_bytes=free + G.GiB + G._tree_bytes(paths.OUT))
    rep = G.check(capacity_fn=_cap(expected_unique_bytes=1), root_path=tmp_path, raise_on_fail=False)
    assert not rep['passed'] and 'FAIL' in G.describe(rep)

    def boom():
        raise ValueError('plan changed')
    with pytest.raises(G.CapacityError, match='capacity probe failed'):
        G.check(capacity_fn=boom, root_path=tmp_path)


# ------------------------------------------------------------------ receipts
def test_identity_and_free_snapshot():
    ident = R.identity()
    assert ident['pid'] == os.getpid() and ident['argv'] and len(ident['boot_id']) > 10
    assert R.same_process(ident)
    assert not R.same_process({**ident, 'start_ticks': str(int(ident['start_ticks']) + 1)})
    snap = R.free_snapshot()
    assert 'Mem:' in snap['free_g_text'] and snap['mem']['total'] > 0


def test_receipts_exit_status_on_exception(tmp_path):
    with pytest.raises(ZeroDivisionError):
        with R.ProcessReceipts(tmp_path, 'unit', extra={'arm': 'x'}, rss_interval=0.05) as rec:
            rec.update(phase='work', step=3)
            1 / 0
    ex = R.read_json(tmp_path / 'exit_status.json')
    assert ex['status'] == 'failed' and ex['phase'] == 'work' and ex['step'] == 3
    assert ex['error']['type'] == 'ZeroDivisionError' and ex['rss']['rss_peak_bytes'] > 0
    assert 'Mem:' in ex['free_end']['free_g_text']
    for f in ('process_identity.json', 'state.json', 'free_start.json'):
        assert (tmp_path / f).exists()
    with R.ProcessReceipts(tmp_path / 'ok', 'unit', rss_interval=0) as rec:
        rec.update(phase='done')
    assert R.read_json(tmp_path / 'ok' / 'exit_status.json')['status'] == 'completed'


def test_receipts_exit_status_on_sigterm(tmp_path):
    code = textwrap.dedent(f"""
        import sys, time; sys.path.insert(0, {str(ROOT)!r})
        from am2exp_v1.ops.receipts import ProcessReceipts
        with ProcessReceipts({str(tmp_path)!r}, 'unit', rss_interval=0.1) as rec:
            rec.update(phase='sleeping'); print('ready', flush=True); time.sleep(60)""")
    p = subprocess.Popen([sys.executable, '-c', code], env=ENV, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == 'ready'
    p.terminate()
    assert p.wait(timeout=30) == 143
    ex = R.read_json(tmp_path / 'exit_status.json')
    assert ex['status'] == 'terminated' and ex['phase'] == 'sleeping'


def test_rss_sampler_limit():
    s = R.RssSampler(interval=1, limit_bytes=1)
    s.sample()
    assert s.exceeded and s.peak > 0
