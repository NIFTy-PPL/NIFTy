import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_raises

from nose_parameterized import parameterized
import unittest
import itertools

from nifty import RGSpace, LMSpace, HPSpace, GLSpace
from nifty import transformator
from nifty.transforms.transform import Transform
from nifty.rg.rg_space import gc as RG_GC
import d2o


###############################################################################

def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )


###############################################################################

rg_fft_modules = []
for name in ['gfft', 'gfft_dummy', 'pyfftw']:
    if RG_GC.validQ('fft_module', name):
        rg_fft_modules += [name]


###############################################################################

class TestRGSpaceTransforms(unittest.TestCase):
    @parameterized.expand(rg_fft_modules, testcase_func_name=custom_name_func)
    def test_check_codomain_none(self, module):
        x = RGSpace((8, 8))
        with assert_raises(ValueError):
            transformator.create(x, None, module=module)

    @parameterized.expand(rg_fft_modules, testcase_func_name=custom_name_func)
    def test_check_codomain_mismatch(self, module):
        x = RGSpace((8, 8))
        y = LMSpace(8)
        with assert_raises(TypeError):
            transformator.create(x, y, module=module)

    @parameterized.expand(rg_fft_modules, testcase_func_name=custom_name_func)
    def test_shapemismatch(self, module):
        x = RGSpace((8, 8))
        b = d2o.distributed_data_object(np.ones((8, 8)))
        with assert_raises(ValueError):
            transformator.create(
                x, x.get_codomain(), module=module
            ).transform(b, axes=(0, 1, 2))

    @parameterized.expand(
        itertools.product(rg_fft_modules, [(128, 128), (179, 179), (512, 512)]),
        testcase_func_name=custom_name_func
    )
    def test_local_ndarray(self, module, shape):
        x = RGSpace(shape)
        a = np.ones(shape)
        assert np.allclose(
            transformator.create(
                x, x.get_codomain(), module=module
            ).transform(a),
            np.fft.fftn(a)
        )

    @parameterized.expand(
        itertools.product(rg_fft_modules, [(128, 128), (179, 179), (512, 512)]),
        testcase_func_name=custom_name_func
    )
    def test_local_notzero(self, module, shape):
        x = RGSpace(shape[0])  # all tests along axis 1
        a = np.ones(shape)
        b = d2o.distributed_data_object(a)
        assert np.allclose(
            transformator.create(
                x, x.get_codomain(), module=module
            ).transform(b, axes=(1,)),
            np.fft.fftn(a, axes=(1,))
        )

    @parameterized.expand(
        itertools.product(rg_fft_modules, [(128, 128), (179, 179), (512, 512)]),
        testcase_func_name=custom_name_func
    )
    def test_not(self, module, shape):
        x = RGSpace(shape)
        a = np.ones(shape)
        b = d2o.distributed_data_object(a, distribution_strategy='not')
        assert np.allclose(
            transformator.create(
                x, x.get_codomain(), module=module
            ).transform(b),
            np.fft.fftn(a)
        )

    # ndarray is not contiguous?
    @parameterized.expand(
        itertools.product(rg_fft_modules, [(128, 128), (179, 179), (512, 512)]),
        testcase_func_name=custom_name_func
    )
    def test_mpi_axesnone(self, module, shape):
        x = RGSpace(shape)
        a = np.ones(shape)
        b = d2o.distributed_data_object(a)
        assert np.allclose(
            transformator.create(
                x, x.get_codomain(), module=module
            ).transform(b),
            np.fft.fftn(a)
        )

    #TODO: check what to do when cannot be distributed
if __name__ == '__main__':
    unittest.main()
