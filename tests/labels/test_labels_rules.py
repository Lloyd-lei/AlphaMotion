"""Contact-label rules: the CPU factory version (labels/labels.py) and the vectorised GPU-engine copy
(gpu/labels.py, pure numpy) must agree on the shared primitives.  Synthetic inputs, CPU only, no data."""
import numpy as np
import pytest

from surface_retargeting.labels import labels as LC
from surface_retargeting.gpu import labels as LG


def _signals(seed, T=400, R=13):
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0, 0.004, (T, R)), axis=0)
    return np.abs(base) + rng.uniform(0, 0.03, (1, R))


def test_shared_constants():
    assert LC.CFG == LG.CFG


@pytest.mark.parametrize('seed', range(5))
def test_hysteresis_and_drop_short_agree(seed):
    d = _signals(seed)
    hc = LC.hysteresis(d, LC.CFG['enter'], LC.CFG['leave'])
    hg = LG.hysteresis(d, LG.CFG['enter'], LG.CFG['leave'])
    assert np.array_equal(np.asarray(hc, bool), np.asarray(hg, bool))
    assert np.array_equal(np.asarray(LC.drop_short(hc, 3), bool), np.asarray(LG.drop_short(hg, 3), bool))


def test_hysteresis_semantics():
    d = np.array([0.05, 0.009, 0.015, 0.019, 0.021, 0.005])[:, None]
    h = np.asarray(LC.hysteresis(d, 0.01, 0.02), bool)[:, 0]
    assert h.tolist() == [False, True, True, True, False, True]
