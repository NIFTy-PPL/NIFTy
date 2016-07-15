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
        assert (Transformation.check_codomain(x, x.get_codomain()))
        assert (Transformation.check_codomain(x, x.get_codomain()))

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
            weighted_np_transform(a, x, x.get_codomain())
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
            weighted_np_transform(a, x, x.get_codomain(), axes=(1,))
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
            weighted_np_transform(a, x, x.get_codomain())
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
            weighted_np_transform(a, x, x.get_codomain())
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
            weighted_np_transform(a, x, x.get_codomain())
        )

    @parameterized.expand(
        itertools.product(rg_fft_modules),
        testcase_func_name=custom_name_func)
    def test_calc_transform_explicit(self, module):
        data = rg_test_data.copy()
        d2o_data = d2o.distributed_data_object(data)
        shape = data.shape

        x = RGSpace(shape, complexity=2, zerocenter=False)
        assert np.allclose(
            transformator.create(
                x, x.get_codomain(), module=module
            ).transform(d2o_data),
            np.array([[0.50541615 + 0.50558267j, -0.01458536 - 0.01646137j,
                       0.01649006 + 0.01990988j, 0.04668049 - 0.03351745j,
                       -0.04382765 - 0.06455639j, -0.05978564 + 0.01334044j],
                      [-0.05347464 + 0.04233343j, -0.05167177 + 0.00643947j,
                       -0.01995970 - 0.01168872j, 0.10653817 + 0.03885947j,
                       -0.03298075 - 0.00374715j, 0.00622585 - 0.01037453j],
                      [-0.01128964 - 0.02424692j, -0.03347793 - 0.0358814j,
                       -0.03924164 - 0.01978305j, 0.03821242 - 0.00435542j,
                       0.07533170 + 0.14590143j, -0.01493027 - 0.02664675j],
                      [0.02238926 + 0.06140625j, -0.06211313 + 0.03317753j,
                       0.01519073 + 0.02842563j, 0.00517758 + 0.08601604j,
                       -0.02246912 - 0.01942764j, -0.06627311 - 0.08763801j],
                      [-0.02492378 - 0.06097411j, 0.06365649 - 0.09346585j,
                       0.05031486 + 0.00858656j, -0.00881969 + 0.01842357j,
                       -0.01972641 - 0.00994365j, 0.05289453 - 0.06822038j],
                      [-0.01865586 - 0.08640926j, 0.03414096 - 0.02605602j,
                       -0.09492552 + 0.01306734j, 0.09355730 + 0.07553701j,
                       -0.02395259 - 0.02185743j, -0.03107832 - 0.04714527j]])
        )

if __name__ == '__main__':
    unittest.main()
