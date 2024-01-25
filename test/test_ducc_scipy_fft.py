from nifty8.ducc_dispatch import (_scipy_fftn, _scipy_ifftn, _scipy_hartley,
                                  _scipy_vdot, fftn, ifftn, hartley, vdot)
import pytest
from numpy.testing import assert_allclose
import numpy as np
pytest.importorskip("ducc0")

_sseq = [np.random.SeedSequence(42)]
_rng = np.random.default_rng(_sseq[-1])

def test_ffts():
    array = _rng.normal(0, 1, (32, 32)).astype(np.float64, copy=False)
    assert_allclose(fftn(array), _scipy_fftn(array))

def test_iffts():
    array = _rng.normal(0, 1, (32, 32)).astype(np.float64, copy=False)
    assert_allclose(ifftn(array), _scipy_ifftn(array))

def test_hartleys():
    array = _rng.normal(0, 1, (32, 32)).astype(np.float64, copy=False)
    assert_allclose(hartley(array), _scipy_hartley(array))

def test_vdots():
    a = _rng.normal(0, 1, (32, 32)).astype(np.float64, copy=False)
    b = _rng.normal(0, 1, (32, 32)).astype(np.float64, copy=False)
    assert_allclose(vdot(a, b), _scipy_vdot(a, b))
