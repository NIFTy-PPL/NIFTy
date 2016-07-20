import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_raises

from nose_parameterized import parameterized
import unittest
import itertools

from nifty import RGSpace, LMSpace, HPSpace, GLSpace
from nifty import transformator
from nifty.transformations.rgrgtransformation import RGRGTransformation
from nifty.rg.rg_space import gc as RG_GC
import d2o


###############################################################################

def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )

def weighted_np_transform(val, domain, codomain, axes=None):
    if codomain.harmonic:
        # correct for forward fft
        val = domain.calc_weight(val, power=1)

    # Perform the transformation
    Tval = np.fft.fftn(val, axes=axes)

    if not codomain.harmonic:
        # correct for inverse fft
        Tval = codomain.calc_weight(Tval, power=-1)

    return Tval

###############################################################################

rg_rg_fft_modules = []
for name in ['gfft', 'gfft_dummy', 'pyfftw']:
    if RG_GC.validQ('fft_module', name):
        rg_rg_fft_modules += [name]

rg_rg_test_shapes = [(128, 128), (179, 179), (512, 512)]

rg_rg_test_spaces = [(GLSpace(8),), (HPSpace(8),), (LMSpace(8),)]
gl_hp_lm_test_spaces = [(GLSpace(8),), (HPSpace(8),), (RGSpace(8),)]
lm_gl_hp_test_spaces = [(LMSpace(8),), (RGSpace(8),)]

###############################################################################

class TestRGRGTransformation(unittest.TestCase):
    # all domain/codomain checks
    def test_check_codomain_none(self):
        x = RGSpace((8, 8))
        with assert_raises(ValueError):
            transformator.create(x, None)

    @parameterized.expand(
        rg_rg_test_spaces,
        testcase_func_name=custom_name_func
    )
    def test_check_codomain_mismatch(self, space):
        x = RGSpace((8, 8))
        with assert_raises(TypeError):
            transformator.create(x, space)

    @parameterized.expand(
        itertools.product([0, 1, 2], [None, (1, 1), (10, 10)], [False, True]),
        testcase_func_name=custom_name_func
    )
    def test_check_codomain_rgspecific(self, complexity, distances, harmonic):
        x = RGSpace((8, 8), complexity=complexity,
                    distances=distances, harmonic=harmonic)
        assert (RGRGTransformation.check_codomain(x, x.get_codomain()))
        assert (RGRGTransformation.check_codomain(x, x.get_codomain()))

    @parameterized.expand(rg_rg_fft_modules, testcase_func_name=custom_name_func)
    def test_shapemismatch(self, module):
        x = RGSpace((8, 8))
        b = d2o.distributed_data_object(np.ones((8, 8)))
        with assert_raises(ValueError):
            transformator.create(
                x, x.get_codomain(), module=module
            ).transform(b, axes=(0, 1, 2))

    @parameterized.expand(
        itertools.product(rg_rg_fft_modules, rg_rg_test_shapes),
        testcase_func_name=custom_name_func
    )
    def test_local_ndarray(self, module, shape):
        x = RGSpace(shape)
        a = np.ones(shape)
        assert np.allclose(
            transformator.create(
                x, x.get_codomain(), module=module
            ).transform(a),
            weighted_np_transform(a, x, x.get_codomain())
        )

    @parameterized.expand(
        itertools.product(rg_rg_fft_modules, rg_rg_test_shapes),
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
            weighted_np_transform(a, x, x.get_codomain(), axes=(1,))
        )

    @parameterized.expand(
        itertools.product(rg_rg_fft_modules, rg_rg_test_shapes),
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
            weighted_np_transform(a, x, x.get_codomain())
        )

    @parameterized.expand(
        itertools.product(rg_rg_test_shapes),
        testcase_func_name=custom_name_func
    )
    def test_mpi_axesnone(self, shape):
        x = RGSpace(shape)
        a = np.ones(shape)
        b = d2o.distributed_data_object(a)
        assert np.allclose(
            transformator.create(
                x, x.get_codomain(), module='pyfftw'
            ).transform(b),
            weighted_np_transform(a, x, x.get_codomain())
        )

    @parameterized.expand(
        itertools.product(rg_rg_test_shapes),
        testcase_func_name=custom_name_func
    )
    def test_mpi_axesnone_equal(self, shape):
        x = RGSpace(shape)
        a = np.ones(shape)
        b = d2o.distributed_data_object(a, distribution_strategy='equal')
        assert np.allclose(
            transformator.create(
                x, x.get_codomain(), module='pyfftw'
            ).transform(b),
            weighted_np_transform(a, x, x.get_codomain())
        )

class TestGLLMTransformation(unittest.TestCase):
    # all domain/codomain checks
    def test_check_codomain_none(self):
        x = GLSpace(8)
        with assert_raises(ValueError):
            transformator.create(x, None)

    @parameterized.expand(
        gl_hp_lm_test_spaces,
        testcase_func_name=custom_name_func
    )
    def test_check_codomain_mismatch(self, space):
        x = GLSpace(8)
        with assert_raises(TypeError):
            transformator.create(x, space)

class TestHPLMTransformation(unittest.TestCase):
    # all domain/codomain checks
    def test_check_codomain_none(self):
        x = HPSpace(8)
        with assert_raises(ValueError):
            transformator.create(x, None)

    @parameterized.expand(
        gl_hp_lm_test_spaces,
        testcase_func_name=custom_name_func
    )
    def test_check_codomain_mismatch(self, space):
        x = GLSpace(8)
        with assert_raises(TypeError):
            transformator.create(x, space)

class TestLMTransformation(unittest.TestCase):
    # all domain/codomain checks
    def test_check_codomain_none(self):
        x = LMSpace(8)
        with assert_raises(ValueError):
            transformator.create(x, None)

    @parameterized.expand(
        lm_gl_hp_test_spaces,
        testcase_func_name=custom_name_func
    )
    def test_check_codomain_mismatch(self, space):
        x = LMSpace(8)
        with assert_raises(ValueError):
            transformator.create(x, space)


if __name__ == '__main__':
    unittest.main()
