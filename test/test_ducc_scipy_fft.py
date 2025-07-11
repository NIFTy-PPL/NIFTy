import numpy as np
import pytest
from numpy.testing import assert_allclose

from nifty8.ducc_dispatch import (
    _scipy_fftn, _scipy_hartley, _scipy_ifftn, _scipy_vdot, fftn, hartley,
    ifftn, vdot
)

pytest.importorskip("ducc0")

pmp = pytest.mark.parametrize


@pmp("seed", [42])
@pmp("shape", [(10), (32, 32), (16, 16, 16)])
@pmp("dtype", [np.float64, np.complex128])
def test_ffts(seed, shape, dtype):
    rng = np.random.default_rng(seed)
    array = rng.normal(0, 1, shape).astype(dtype, copy=False)
    assert_allclose(fftn(array), _scipy_fftn(array))


@pmp("seed", [42])
@pmp("shape", [(10), (32, 32), (16, 16, 16)])
@pmp("dtype", [np.float64, np.complex128])
def test_iffts(seed, shape, dtype):
    rng = np.random.default_rng(seed)
    array = rng.normal(0, 1, shape).astype(dtype, copy=False)
    assert_allclose(ifftn(array), _scipy_ifftn(array))


@pmp("seed", [42])
@pmp("shape", [(10), (32, 32), (16, 16, 16)])
@pmp("dtype", [np.float64])
def test_hartleys(seed, shape, dtype):
    rng = np.random.default_rng(seed)
    array = rng.normal(0, 1, shape).astype(dtype, copy=False)
    assert_allclose(hartley(array), _scipy_hartley(array))


@pmp("seed", [42])
@pmp("shape", [(10), (32, 32), (16, 16, 16)])
@pmp("dtype", [np.float64, np.complex128])
def test_vdots(seed, shape, dtype):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, shape).astype(dtype, copy=False)
    b = rng.normal(0, 1, shape).astype(dtype, copy=False)
    assert_allclose(vdot(a, b), _scipy_vdot(a, b))
