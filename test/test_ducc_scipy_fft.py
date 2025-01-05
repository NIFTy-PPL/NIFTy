import nifty8 as ift
import numpy as np
import pytest
from nifty8.ducc_dispatch import (_scipy_fftn, _scipy_hartley, _scipy_ifftn,
                                  _scipy_vdot, fftn, hartley, ifftn, vdot)
from numpy.testing import assert_allclose

from .common import list2fixture, setup_function, teardown_function

pytest.importorskip("ducc0")

pmp = pytest.mark.parametrize
device_id = list2fixture([-1, 0] if ift.device_available() else [-1])


@pmp("seed", [42])
@pmp("shape", [(10), (32, 32), (16, 16, 16)])
@pmp("dtype", [np.float64, np.complex128])
def test_ffts(seed, shape, dtype, device_id):
    rng = np.random.default_rng(seed)
    array = rng.normal(0, 1, shape).astype(dtype, copy=False)
    array = ift.AnyArray(array).at(device_id)
    _scipy_fftn(array)
    assert_allclose(fftn(array).asnumpy(), _scipy_fftn(array).asnumpy())


@pmp("seed", [42])
@pmp("shape", [(10), (32, 32), (16, 16, 16)])
@pmp("dtype", [np.float64, np.complex128])
def test_iffts(seed, shape, dtype, device_id):
    rng = np.random.default_rng(seed)
    array = rng.normal(0, 1, shape).astype(dtype, copy=False)
    array = ift.AnyArray(array).at(device_id)
    assert_allclose(ifftn(array).asnumpy(), _scipy_ifftn(array).asnumpy())


@pmp("seed", [42])
@pmp("shape", [(10), (32, 32), (16, 16, 16)])
@pmp("dtype", [np.float64])
def test_hartleys(seed, shape, dtype, device_id):
    rng = np.random.default_rng(seed)
    array = rng.normal(0, 1, shape).astype(dtype, copy=False)
    array = ift.AnyArray(array).at(device_id)
    assert_allclose(hartley(array).asnumpy(), _scipy_hartley(array).asnumpy())


@pmp("seed", [42])
@pmp("shape", [(10), (32, 32), (16, 16, 16)])
@pmp("dtype", [np.float64, np.complex128])
def test_vdots(seed, shape, dtype):
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, shape).astype(dtype, copy=False)
    b = rng.normal(0, 1, shape).astype(dtype, copy=False)
    assert_allclose(vdot(a, b), _scipy_vdot(a, b))
