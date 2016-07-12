import numpy as np
from numpy.testing import assert_equal, assert_almost_equal, assert_raises

from nose_parameterized import parameterized
import unittest
import itertools

from nifty import RGSpace, LMSpace, HPSpace, GLSpace
from nifty import transformator
from nifty.transformations.transformation import Transformation
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


rg_test_shapes = [(128, 128), (179, 179), (512, 512)]

rg_test_data = np.array(
    [[0.38405405 + 0.32460996j, 0.02718878 + 0.08326207j,
      0.78792080 + 0.81192595j, 0.17535687 + 0.68054781j,
      0.93044845 + 0.71942995j, 0.21179999 + 0.00637665j],
     [0.10905553 + 0.3027462j, 0.37361237 + 0.68434316j,
      0.94070232 + 0.34129582j, 0.04658034 + 0.4575192j,
      0.45057929 + 0.64297612j, 0.01007361 + 0.24953504j],
     [0.39579662 + 0.70881906j, 0.01614435 + 0.82603832j,
      0.84036344 + 0.50321592j, 0.87699553 + 0.40337862j,
      0.11816016 + 0.43332373j, 0.76627757 + 0.66327959j],
     [0.77272335 + 0.18277367j, 0.93341953 + 0.58105518j,
      0.27227913 + 0.17458168j, 0.70204032 + 0.81397425j,
      0.12422993 + 0.19215286j, 0.30897158 + 0.47364969j],
     [0.24702012 + 0.54534373j, 0.55206013 + 0.98406613j,
      0.57408167 + 0.55685406j, 0.87991341 + 0.52534323j,
      0.93912604 + 0.97186519j, 0.77778942 + 0.45812051j],
     [0.79367868 + 0.48149411j, 0.42484378 + 0.74870011j,
      0.79611264 + 0.50926774j, 0.35372794 + 0.10468412j,
      0.46140736 + 0.09449825j, 0.82044644 + 0.95992843j]])

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

    @parameterized.expand(
        itertools.product([0, 1, 2], [None, (1, 1), (10, 10)], [False, True]),
        testcase_func_name=custom_name_func
    )
    def test_check_codomain_rgspecific(self, complexity, distances, harmonic):
        x = RGSpace((8, 8), complexity=complexity,
                    distances=distances, harmonic=harmonic)
        assert(Transformation.check_codomain(x, x.get_codomain()))
        assert(Transformation.check_codomain(x, x.get_codomain()))

    @parameterized.expand(rg_fft_modules, testcase_func_name=custom_name_func)
    def test_shapemismatch(self, module):
        x = RGSpace((8, 8))
        b = d2o.distributed_data_object(np.ones((8, 8)))
        with assert_raises(ValueError):
            transformator.create(
                x, x.get_codomain(), module=module
            ).transform(b, axes=(0, 1, 2))

    @parameterized.expand(
        itertools.product(rg_fft_modules, rg_test_shapes),
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
        itertools.product(rg_fft_modules, rg_test_shapes),
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
        itertools.product(rg_fft_modules, rg_test_shapes),
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

    @parameterized.expand(
        itertools.product(rg_test_shapes),
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
            np.fft.fftn(a)
        )

    @parameterized.expand(
        itertools.product(rg_test_shapes),
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
            np.fft.fftn(a)
        )

if __name__ == '__main__':
    unittest.main()
