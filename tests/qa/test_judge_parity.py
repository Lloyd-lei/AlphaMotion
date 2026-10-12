"""Independent CPU judge (MuJoCo + numpy): re-judge the first frames of a stored parity run and compare with the
per-frame series the old ~/umr_gpu/eval/judge.py wrote.  Read-only on SR_WORK.  No GPU."""
import numpy as np
import pytest

from surface_retargeting.config import CFG
from conftest import need

CASES = [('hiphi_getup__getup0004_neutral_30fps', 'unitree_g1_29dof'),
         ('lafan1_smplx__dance1_subject2', 'fourier_gr3')]
N = 40


@pytest.mark.data
@pytest.mark.parametrize('clip,robot', CASES)
def test_judge_matches_stored_series(clip, robot):
    from surface_retargeting.qa.judge import Judge, judge_clip
    series = f'{CFG.work}/results/parity/{clip}__{robot}.series.npz'
    xml = f'{CFG.work}/robots/{robot}/{robot}.xml'
    surf = f'{CFG.work}/robots/{robot}/surface.npz'
    for p in (series, xml, surf):
        need(p)
    S = np.load(series, allow_pickle=True)
    if S['obj_pose'].size:
        pytest.skip('object clip: needs the captured problem (GPU-side objpen); covered in the cloud run')
    J = Judge(xml, surf)
    floor, fps = float(S['floor']), float(S['fps'])
    for name in ('cpu_umr', 'gpu_strict'):
        Q = S[f'q_{name}'][:N].copy()
        Q[:, 2] -= floor
        Sj, summ = judge_clip(J, Q, fps, S['human_contact'][:N], S['human_clear'][:N])
        for k in ('ground_min', 'colground_min', 'self_min', 'region_min'):
            ref = S[f'{name}__{k}'][:N]
            assert np.allclose(Sj[k], ref, atol=1e-9, rtol=0), (name, k, np.abs(Sj[k] - ref).max())
        assert summ['frames'] == N
