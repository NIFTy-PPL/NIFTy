# -*- coding: utf-8 -*-

from numpy.testing import assert_equal,\
    assert_almost_equal,\
    assert_raises

from nose_parameterized import parameterized
import unittest
import itertools
import numpy as np

from nifty.nifty_field import field

from d2o import distributed_data_object

from nifty.nifty_paradict import space_paradict
from nifty.nifty_core import POINT_DISTRIBUTION_STRATEGIES,\
    space,\
    point_space

from nifty.rg.nifty_rg import RG_DISTRIBUTION_STRATEGIES,\
                              gc as RG_GC,\
                              rg_space
from nifty.lm.nifty_lm import LM_DISTRIBUTION_STRATEGIES,\
                              GL_DISTRIBUTION_STRATEGIES,\
                              HP_DISTRIBUTION_STRATEGIES
from nifty.nifty_power_indices import power_indices
from nifty.nifty_utilities import _hermitianize_inverter as \
                                                        hermitianize_inverter

from nifty.operators.nifty_operators import power_operator

available = []
try:
    from nifty import lm_space
except ImportError:
    pass
else:
    available += ['lm_space']
try:
    from nifty import gl_space
except ImportError:
    pass
else:
    available += ['gl_space']
try:
    from nifty import hp_space
except ImportError:
    pass
else:
    available += ['hp_space']



###############################################################################

def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )

###############################################################################
###############################################################################

all_point_datatypes = [np.dtype('bool'),
                       np.dtype('int16'),
                       np.dtype('int32'),
                       np.dtype('int64'),
                       np.dtype('float32'),
                       np.dtype('float64'),
                       np.dtype('complex64'),
                       np.dtype('complex128')]

all_lm_datatypes = [np.dtype('complex64'),
                    np.dtype('complex128')]

all_gl_datatypes = [np.dtype('float64'),
                    np.dtype('float128')]

all_hp_datatypes = [np.dtype('float64')]

###############################################################################

DATAMODELS = {}
DATAMODELS['point_space'] = POINT_DISTRIBUTION_STRATEGIES
DATAMODELS['rg_space'] = RG_DISTRIBUTION_STRATEGIES
DATAMODELS['lm_space'] = LM_DISTRIBUTION_STRATEGIES
DATAMODELS['gl_space'] = GL_DISTRIBUTION_STRATEGIES
DATAMODELS['hp_space'] = HP_DISTRIBUTION_STRATEGIES

###############################################################################

fft_modules = []
for name in ['gfft', 'gfft_dummy', 'pyfftw']:
    if RG_GC.validQ('fft_module', name):
        fft_modules += [name]

###############################################################################

all_spaces = ['space', 'point_space', 'rg_space']
if 'lm_space' in available:
    all_spaces += ['lm_space']
if 'gl_space' in available:
    all_spaces += ['gl_space']
if 'hp_space' in available:
    all_spaces += ['hp_space']


point_like_spaces = ['point_space', 'rg_space']
if 'lm_space' in available:
    point_like_spaces += ['lm_space']
if 'gl_space' in available:
    point_like_spaces += ['gl_space']
if 'hp_space' in available:
    point_like_spaces += ['hp_space']

###############################################################################

np_spaces = point_like_spaces
d2o_spaces = []
if POINT_DISTRIBUTION_STRATEGIES != []:
    d2o_spaces += ['point_space']
if RG_DISTRIBUTION_STRATEGIES != []:
    d2o_spaces += ['rg_space']
if LM_DISTRIBUTION_STRATEGIES != []:
    d2o_spaces += ['lm_space']
if GL_DISTRIBUTION_STRATEGIES != []:
    d2o_spaces += ['gl_space']
if HP_DISTRIBUTION_STRATEGIES != []:
    d2o_spaces += ['hp_space']


unary_operations = ['pos', 'neg', 'abs', 'real', 'imag', 'nanmin', 'amin',
                    'nanmax', 'amax', 'median', 'mean', 'std', 'var', 'argmin',
                    'argmin_nonflat', 'argmax', 'argmax_nonflat', 'conjugate',
                    'sum', 'prod', 'unique', 'copy', 'copy_empty', 'isnan',
                    'isinf', 'isfinite', 'nan_to_num', 'all', 'any', 'None']

binary_operations = ['add', 'radd', 'iadd', 'sub', 'rsub', 'isub', 'mul',
                     'rmul', 'imul', 'div', 'rdiv', 'idiv', 'pow', 'rpow',
                     'ipow', 'ne', 'lt', 'le', 'eq', 'ge', 'gt', 'None']

###############################################################################

fft_test_data = np.array(
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


def generate_space(name):
    space_dict = {'space': space(),
                  'point_space': point_space(10),
                  'rg_space': rg_space((8, 8)),
                  }
    if 'lm_space' in available:
        space_dict['lm_space'] = lm_space(mmax=11, lmax=11)
    if 'hp_space' in available:
        space_dict['hp_space'] = hp_space(8)
    if 'gl_space' in available:
        space_dict['gl_space'] = gl_space(nlat=10, nlon=19)

    return space_dict[name]


def generate_space_with_size(name, num):
    space_dict = {'space': space(),
                  'point_space': point_space(num),
                  'rg_space': rg_space((num, num)),
                  }
    if 'lm_space' in available:
        space_dict['lm_space'] = lm_space(mmax=num, lmax=num)
    if 'hp_space' in available:
        space_dict['hp_space'] = hp_space(num)
    if 'gl_space' in available:
        space_dict['gl_space'] = gl_space(nlat=num, nlon=num)

    return space_dict[name]

def generate_data(space):
    a = np.arange(space.get_dim()).reshape(space.get_shape())
    data = space.cast(a)
    return data


def check_equality(space, data1, data2):
    return space.unary_operation(space.binary_operation(data1, data2, 'eq'),
                                 'all')


def check_almost_equality(space, data1, data2, integers=7):
    return space.unary_operation(
        space.binary_operation(
            space.unary_operation(
                space.binary_operation(data1, data2, 'sub'),
                'abs'),
            10.**(-1. * integers), 'le'),
        'all')


def flip(space, data):
    return space.unary_operation(hermitianize_inverter(data), 'conjugate')


###############################################################################
###############################################################################

class Test_Common_Space_Features(unittest.TestCase):

    @parameterized.expand(all_spaces,
                          testcase_func_name=custom_name_func)
    def test_successfull_init_and_attributes(self, name):
        s = generate_space(name)
        assert(isinstance(s.paradict, space_paradict))

    @parameterized.expand(all_spaces,
                          testcase_func_name=custom_name_func)
    def test_successfull_init_and_methods(self, name):
        s = generate_space(name)
        assert(callable(s.__hash__))
        assert(callable(s.__eq__))
        assert(callable(s.__ne__))
        assert(callable(s.__len__))
        assert(callable(s.copy))
        assert(callable(s.getitem))
        assert(callable(s.setitem))
        assert(callable(s.apply_scalar_function))
        assert(callable(s.unary_operation))
        assert(callable(s.binary_operation))
        assert(callable(s.get_shape))
        assert(callable(s.get_dim))
        assert(callable(s.get_dof))
        assert(callable(s.cast))
        assert(callable(s.enforce_power))
        assert(callable(s.check_codomain))
        assert(callable(s.get_codomain))
        assert(callable(s.get_random_values))
        assert(callable(s.calc_weight))
        assert(callable(s.get_weight))
        assert(callable(s.calc_norm))
        assert(callable(s.calc_dot))
        assert(callable(s.calc_transform))
        assert(callable(s.calc_smooth))
        assert(callable(s.calc_power))
        assert(callable(s.calc_real_Q))
        assert(callable(s.calc_bincount))
        assert(callable(s.get_plot))
        assert(callable(s.__repr__))
        assert(callable(s.__str__))

        assert(s.check_codomain(None) == False)
        assert(isinstance(repr(s), str))

    @parameterized.expand(all_spaces,
                          testcase_func_name=custom_name_func)
    def test_successfull_hashing(self, name):
        s1 = generate_space(name)
        s2 = generate_space(name)
        assert(s1.__hash__() == s2.__hash__())


###############################################################################
###############################################################################

class Test_Common_Point_Like_Space_Interface(unittest.TestCase):

    @parameterized.expand(point_like_spaces,
                          testcase_func_name=custom_name_func)
    def test_successfull_init_and_attributes(self, name):
        s = generate_space(name)

        assert(isinstance(s.paradict, space_paradict))
        assert(isinstance(s.paradict, space_paradict))
        assert(isinstance(s.dtype, np.dtype))
        assert(isinstance(s.discrete, bool))
#        assert(isinstance(s.harmonic, bool))
        assert(isinstance(s.distances, tuple))
        if hasattr(s, 'harmonic'):
            if s.harmonic:
                assert(isinstance(s.power_indices, power_indices))

    @parameterized.expand(point_like_spaces,
                          testcase_func_name=custom_name_func)
    def test_getters(self, name):
        s = generate_space(name)
        assert(isinstance(s.get_shape(), tuple))
        assert(isinstance(s.get_dim(), np.int))

        assert(isinstance(s.get_dof(), np.int))
        assert(isinstance(s.get_dof(split=True), tuple))
        assert_equal(s.get_dof(), np.prod(s.get_dof(split=True)))

        assert(isinstance(s.get_vol(), np.float))
        assert(isinstance(s.get_dof(split=True), tuple))

        assert(isinstance(s.get_meta_volume(), np.float))
        print(s.get_meta_volume(split=True), type(s.cast(1)))
        assert(isinstance(s.get_meta_volume(split=True), type(s.cast(1))))
        assert_almost_equal(
            s.get_meta_volume(), s.get_meta_volume(split=True).sum(), 2)

    @parameterized.expand(point_like_spaces,
                          testcase_func_name=custom_name_func)
    def test_copy(self, name):
        s = generate_space(name)
        t = s.copy()
        assert(s == t)
        assert(id(s) != id(t))


###############################################################################
###############################################################################

class Test_Point_Space(unittest.TestCase):

    @parameterized.expand(
        itertools.product([0, 1, 10],
                          all_point_datatypes),
        testcase_func_name=custom_name_func)
    def test_successfull_init(self, num, dtype):
        p = point_space(num, dtype)
        assert_equal(p.paradict['num'], num)
        assert_equal(p.dtype, dtype)

        assert_equal(p.discrete, True)
        assert_equal(p.distances, (np.float(1.),))

###############################################################################

    def test_para(self):
        num = 10
        p = point_space(num)
        assert_equal(p.para[0], num)

        new_num = 15
        p.para = np.array([new_num])
        assert_equal(p.para[0], new_num)

###############################################################################

    def test_init_fail(self):
        assert_raises(ValueError, lambda: point_space(-5))
        assert_raises(ValueError, lambda: point_space((10, 10)))
        assert_raises(ValueError, lambda: point_space(10, np.uint))

###############################################################################

    @parameterized.expand(
        itertools.product([0, 1, 10]),
        testcase_func_name=custom_name_func)
    def test_apply_scalar_function(self, num):
        s = point_space(num)
        d = generate_data(s)
        t = s.apply_scalar_function(d, lambda x: x**2)
        assert(check_equality(s, d**2, t))
        assert(id(d) != id(t))

        t = s.apply_scalar_function(d, lambda x: x**2, inplace=True)
        assert(check_equality(s, d, t))
        assert(id(d) == id(t))

###############################################################################

    @parameterized.expand(
        itertools.product([1, 10],
                          unary_operations),
        testcase_func_name=custom_name_func)
    def test_unary_operations(self, num, op):
        s = point_space(num)
        d = s.cast(np.arange(num))
        s.unary_operation(d, op)
        # TODO: Implement value verification

    @parameterized.expand(
        itertools.product([1, 10],
                          binary_operations),
        testcase_func_name=custom_name_func)
    def test_binary_operations(self, num, op):
        s = point_space(num)
        d = s.cast(np.arange(num))
        d2 = d[::-1]
        s.binary_operation(d, d2, op)
        # TODO: Implement value verification

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes),
        testcase_func_name=custom_name_func)
    def test_get_shape_dim(self, dtype):
        num = 10
        s = point_space(num, dtype)

        assert_equal(s.get_shape(), (num,))
        assert_equal(s.get_dim(), num)

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes),
        testcase_func_name=custom_name_func)
    def test_get_shape_dof(self, dtype):
        num = 10
        s = point_space(num, dtype)

        if issubclass(dtype.type, np.complexfloating):
            assert_equal(s.get_dof(), 2 * num)
            assert_equal(s.get_dof(split=True), (2 * num,))
        else:
            assert_equal(s.get_dof(), num)
            assert_equal(s.get_dof(split=True), (num,))

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes),
        testcase_func_name=custom_name_func)
    def test_get_shape_vol(self, dtype):
        num = 10
        s = point_space(num, dtype)

        assert_equal(s.get_vol(), 1.)
        assert_equal(s.get_vol(split=True), (1.,))

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes),
        testcase_func_name=custom_name_func)
    def test_get_shape_metavolume(self, dtype):
        num = 10
        s = point_space(num, dtype)

        assert_equal(s.get_meta_volume(), 10.)
        assert(check_equality(s, s.get_meta_volume(split=True), s.cast(1)))

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes),
        testcase_func_name=custom_name_func)
    def test_cast_from_scalar(self, dtype):
        num = 10
        scalar = 4
        s = point_space(num, dtype)
        d = distributed_data_object(scalar,
                                    global_shape=(num,),
                                    dtype=dtype)

        casted_scalar = s.cast(scalar)
        assert(check_equality(s, casted_scalar, d))
        assert(d.equal(casted_scalar))

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes),
        testcase_func_name=custom_name_func)
    def test_cast_from_field(self, dtype):
        num = 10
        a = np.arange(num,).astype(dtype)
        s = point_space(num, dtype)
        f = field(s, val=a)

        d = distributed_data_object(a, dtype=dtype)

        casted_f = s.cast(f)
        assert(check_equality(s, casted_f, d))
        assert(d.equal(casted_f))

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes),
        testcase_func_name=custom_name_func)
    def test_cast_from_ndarray(self, dtype):
        num = 10
        a = np.arange(num,)
        s = point_space(num, dtype)

        d = distributed_data_object(a, dtype=dtype)

        casted_a = s.cast(a)
        assert(check_equality(s, casted_a, d))
        assert(d.equal(casted_a))

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes),
        testcase_func_name=custom_name_func)
    def test_cast_from_d2o(self, dtype):
        num = 10
        pre_a = np.arange(num,)
        a = distributed_data_object(pre_a)
        s = point_space(num, dtype)

        d = distributed_data_object(a, dtype=dtype)

        casted_a = s.cast(a)
        assert(check_equality(s, casted_a, d))
        assert(d.equal(casted_a))


###############################################################################

    def test_raise_on_not_implementable_methods(self):
        s = point_space(10)
        assert_raises(AttributeError, lambda: s.enforce_power(1))
        assert_raises(AttributeError, lambda: s.calc_smooth(1))
        assert_raises(AttributeError, lambda: s.calc_power(1))
        assert_raises(AttributeError, lambda: s.calc_transform(1))

###############################################################################

    @parameterized.expand(
        [[10, np.dtype('float64')],
         [10, np.dtype('float32')],
         [12, np.dtype('float64')]],
        testcase_func_name=custom_name_func)
    def test_get_check_codomain(self, num, dtype):
        s = point_space(10, dtype=np.dtype('float64'))

        t = s.get_codomain()
        assert(s.check_codomain(t))

        t_bad = point_space(num, dtype=dtype)
        assert(s.check_codomain(t_bad) == False)

        assert(s.check_codomain(None) == False)

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes),
        testcase_func_name=custom_name_func)
    def test_get_random_values(self, dtype):
        if dtype == np.dtype('bool'):
            return None

        num = 100000
        s = point_space(num, dtype)

        pm = s.get_random_values(random='pm1')
        assert(abs(s.unary_operation(pm, op='mean')) < 0.1)

        std = 4
        mean = 5
        gau = s.get_random_values(random='gau', mean=mean, std=std)
        assert(abs(gau.std() - std) / std < 0.2)
        assert(abs(gau.mean() - mean) / mean < 0.2)

        vmin = -4
        vmax = 10
        uni = s.get_random_values(random='uni', vmin=vmin, vmax=vmax)
        assert(abs(uni.real.mean() - 3.) / 3. < 0.1)
        assert(abs(uni.real.std() - 4.) / 4. < 0.1)

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes),
        testcase_func_name=custom_name_func)
    def test_get_calc_weight(self, dtype):
        num = 100
        s = point_space(num, dtype)
        weight = 1
        assert_equal(s.get_weight(), weight)
        assert_equal(s.get_weight(power=4), weight)
        assert_equal(s.get_weight(power=4, split=True), (weight,))

        data = s.cast(2)
        assert(check_equality(s, data, s.calc_weight(data)))

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes),
        testcase_func_name=custom_name_func)
    def test_calc_dot(self, dtype):
        num = 100
        s = point_space(num, dtype)
        if dtype == np.dtype('bool'):
            assert_equal(s.calc_dot(1, 1), 1)
        else:
            assert_equal(s.calc_dot(1, 1), num)
            assert_equal(s.calc_dot(np.arange(num), 1), num * (num - 1.) / 2.)

###############################################################################

    @parameterized.expand(
        itertools.product(),
        testcase_func_name=custom_name_func)
    def test_calc_norm(self):
        num = 10
        s = point_space(num)
        d = s.cast(np.arange(num))
        assert_almost_equal(s.calc_norm(d), 16.881943016134134)
        assert_almost_equal(s.calc_norm(d, q=3), 12.651489979526238)

###############################################################################

    @parameterized.expand(
        itertools.product(),
        testcase_func_name=custom_name_func)
    def test_calc_real_Q(self):
        num = 100
        s = point_space(num, dtype=np.complex)
        real_data = s.cast(1)
        assert(s.calc_real_Q(real_data))
        complex_data = s.cast(1 + 1j)
        assert(s.calc_real_Q(complex_data) == False)

###############################################################################

    @parameterized.expand(
        itertools.product(),
        testcase_func_name=custom_name_func)
    def test_calc_bincount(self):
        num = 10
        s = point_space(num, dtype=np.int)
        data = s.cast(np.array([1, 1, 2, 0, 5, 8, 4, 5, 4, 5]))
        weights = np.arange(10) / 10.
        assert_equal(s.calc_bincount(data),
                     np.array([1, 2, 1, 0, 2, 3, 0, 0, 1]))
        assert_equal(s.calc_bincount(data, weights=weights),
                     np.array([0.3, 0.1, 0.2, 0, 1.4, 2, 0, 0, 0.5]))


###############################################################################
###############################################################################

class Test_RG_Space(unittest.TestCase):

    @parameterized.expand(
        itertools.product([(1,), (10, 10)],
                          [0, 1, 2],
                          [True, False],
                          [None, 0.5],
                          [True, False],
                          fft_modules),
        testcase_func_name=custom_name_func)
    def test_successfull_init(self, shape, complexity, zerocenter, distances,
                              harmonic, fft_module):
        x = rg_space(shape,
                     complexity=complexity,
                     zerocenter=zerocenter,
                     distances=distances,
                     harmonic=harmonic,
                     fft_module=fft_module)
        assert(isinstance(x.harmonic, bool))
        assert_equal(x.get_shape(), shape)
        assert_equal(x.dtype,
                     np.dtype('float64') if complexity == 0 else
                     np.dtype('complex128'))
        assert_equal(x.distances,
                     1. / np.array(shape) if distances is None else
                     np.ones(len(shape)) * distances)

###############################################################################

    def test_para(self):
        shape = (10, 10)
        zerocenter = True
        complexity = 2
        x = rg_space(shape, zerocenter=zerocenter, complexity=complexity)
        assert_equal(x.para, np.array([10, 10, 2, 1, 1]))

        new_para = np.array([6, 6, 1, 0, 1])
        x.para = new_para
        assert_equal(x.para, new_para)

###############################################################################

    def test_init_fail(self):
        assert_raises(ValueError, lambda: rg_space((-3, 10)))
        assert_raises(ValueError, lambda: rg_space((10, 10), complexity=3))
        assert_raises(ValueError, lambda: rg_space((10, 10),
                                                   distances=[1, 1, 1]))
        assert_raises(ValueError, lambda: rg_space((10, 10),
                                                   zerocenter=[1, 1, 1]))

###############################################################################

    @parameterized.expand([], testcase_func_name=custom_name_func)
    def test_cast_to_hermitian(self):
        shape = (10, 10)
        x = rg_space(shape, complexity=1)
        data = np.random.random(shape) + np.random.random(shape) * 1j
        casted_data = x.cast(data)
        flipped_data = flip(x, casted_data)
        assert(check_equality(x, flipped_data, casted_data))

###############################################################################

    @parameterized.expand([], testcase_func_name=custom_name_func)
    def test_enforce_power(self):
        shape = (6, 6)
        x = rg_space(shape)

        assert_equal(x.enforce_power(2),
                     np.array([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.]))
        assert_almost_equal(
            x.enforce_power(lambda x: 42 / (1 + x)**5),
            np.array([4.20000000e+01, 1.31250000e+00, 5.12118970e-01,
                      1.72839506e-01, 1.18348051e-01, 5.10678257e-02,
                      4.10156250e-02, 3.36197167e-02, 2.02694134e-02,
                      1.06047106e-02]))

###############################################################################

    @parameterized.expand(
        itertools.product([0, 1, 2],
                          [None, 1, 10],
                          [False, True]),
        testcase_func_name=custom_name_func)
    def test_get_check_codomain(self, complexity, distances, harmonic):
        shape = (6, 6)
        x = rg_space(shape, complexity=complexity, distances=distances,
                     harmonic=harmonic)
        y = x.get_codomain()
        assert(x.check_codomain(y))
        assert(y.check_codomain(x))

###############################################################################

#    @parameterized.expand(
#        itertools.product([True], #[True, False],
#                          ['fftw']),
#                          #DATAMODELS['rg_space']),
#        testcase_func_name=custom_name_func)
#    def test_get_random_values(self, harmonic, datamodel):
#        x = rg_space((4, 4), complexity=1, harmonic=harmonic,
#                     datamodel=datamodel)
#
#        # pm1
#        data = x.get_random_values(random='pm1')
#        flipped_data = flip(x, data)
#        assert(check_almost_equality(x, data, flipped_data))
#
#        # gau
#        data = x.get_random_values(random='gau', mean=4 + 3j, std=2)
#        flipped_data = flip(x, data)
#        assert(check_almost_equality(x, data, flipped_data))
#
#        # uni
#        data = x.get_random_values(random='uni', vmin=-2, vmax=4)
#        flipped_data = flip(x, data)
#        assert(check_almost_equality(x, data, flipped_data))
#
#        # syn
#        data = x.get_random_values(random='syn',
#                                   spec=lambda x: 42 / (1 + x)**3)
#        flipped_data = flip(x, data)
#        assert(check_almost_equality(x, data, flipped_data))

###############################################################################

    @parameterized.expand([], testcase_func_name=custom_name_func)
    def test_calc_dot(self):
        shape = (8, 8)
        a = np.arange(np.prod(shape)).reshape(shape)
        x = rg_space(shape)
        assert_equal(x.calc_dot(a, a), 85344)
        assert_equal(x.calc_dot(a, 1), 2016)
        assert_equal(x.calc_dot(1, a), 2016)

###############################################################################

    @parameterized.expand(
        itertools.product([0, 1]),
        testcase_func_name=custom_name_func)
    def test_calc_transform_general(self, complexity):
        data = fft_test_data.copy()
        shape = data.shape

        x = rg_space(shape, complexity=complexity)
        data = fft_test_data.copy()
        data = x.cast(data)
        check_equality(x, data, x.calc_transform(x.calc_transform(data)))

###############################################################################

    @parameterized.expand(
        itertools.product(fft_modules),
        testcase_func_name=custom_name_func)
    def test_calc_transform_explicit(self, fft_module):
        data = fft_test_data.copy()
        shape = data.shape

        x = rg_space(shape, complexity=2, zerocenter=False,
                     fft_module=fft_module)
        casted_data = x.cast(data)
        assert(check_almost_equality(x, x.calc_transform(casted_data),
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
                                                -0.02395259 - 0.02185743j, -0.03107832 - 0.04714527j]])))

        x = rg_space(shape, complexity=2, zerocenter=True,
                     fft_module=fft_module)
        casted_data = x.cast(data)
        assert(check_almost_equality(x, x.calc_transform(casted_data),
                                     np.array([[0.00517758 + 0.08601604j, 0.02246912 + 0.01942764j,
                                                -0.06627311 - 0.08763801j, -0.02238926 - 0.06140625j,
                                                -0.06211313 + 0.03317753j, -0.01519073 - 0.02842563j],
                                               [0.00881969 - 0.01842357j, -0.01972641 - 0.00994365j,
                                                -0.05289453 + 0.06822038j, -0.02492378 - 0.06097411j,
                                                -0.06365649 + 0.09346585j, 0.05031486 + 0.00858656j],
                                               [0.09355730 + 0.07553701j, 0.02395259 + 0.02185743j,
                                                -0.03107832 - 0.04714527j, 0.01865586 + 0.08640926j,
                                                0.03414096 - 0.02605602j, 0.09492552 - 0.01306734j],
                                               [-0.04668049 + 0.03351745j, -0.04382765 - 0.06455639j,
                                                0.05978564 - 0.01334044j, 0.50541615 + 0.50558267j,
                                                0.01458536 + 0.01646137j, 0.01649006 + 0.01990988j],
                                               [0.10653817 + 0.03885947j, 0.03298075 + 0.00374715j,
                                                0.00622585 - 0.01037453j, 0.05347464 - 0.04233343j,
                                                -0.05167177 + 0.00643947j, 0.01995970 + 0.01168872j],
                                               [-0.03821242 + 0.00435542j, 0.07533170 + 0.14590143j,
                                                0.01493027 + 0.02664675j, -0.01128964 - 0.02424692j,
                                                0.03347793 + 0.0358814j, -0.03924164 - 0.01978305j]])))

        x = rg_space(shape, complexity=2, zerocenter=[True, False],
                     fft_module=fft_module)
        casted_data = x.cast(data)
        assert(check_almost_equality(x, x.calc_transform(casted_data),
                                     np.array([[-0.02238926 - 0.06140625j, 0.06211313 - 0.03317753j,
                                                -0.01519073 - 0.02842563j, -0.00517758 - 0.08601604j,
                                                0.02246912 + 0.01942764j, 0.06627311 + 0.08763801j],
                                               [-0.02492378 - 0.06097411j, 0.06365649 - 0.09346585j,
                                                0.05031486 + 0.00858656j, -0.00881969 + 0.01842357j,
                                                -0.01972641 - 0.00994365j, 0.05289453 - 0.06822038j],
                                               [0.01865586 + 0.08640926j, -0.03414096 + 0.02605602j,
                                                0.09492552 - 0.01306734j, -0.09355730 - 0.07553701j,
                                                0.02395259 + 0.02185743j, 0.03107832 + 0.04714527j],
                                               [0.50541615 + 0.50558267j, -0.01458536 - 0.01646137j,
                                                0.01649006 + 0.01990988j, 0.04668049 - 0.03351745j,
                                                -0.04382765 - 0.06455639j, -0.05978564 + 0.01334044j],
                                               [0.05347464 - 0.04233343j, 0.05167177 - 0.00643947j,
                                                0.01995970 + 0.01168872j, -0.10653817 - 0.03885947j,
                                                0.03298075 + 0.00374715j, -0.00622585 + 0.01037453j],
                                               [-0.01128964 - 0.02424692j, -0.03347793 - 0.0358814j,
                                                -0.03924164 - 0.01978305j, 0.03821242 - 0.00435542j,
                                                0.07533170 + 0.14590143j, -0.01493027 - 0.02664675j]])))

        x = rg_space(shape, complexity=2, zerocenter=[True, False],
                     fft_module=fft_module)
        y = rg_space(shape, complexity=2, zerocenter=[False, True],
                     distances=[1, 1], harmonic=True,
                     fft_module=fft_module)
        casted_data = x.cast(data)
        assert(check_almost_equality(x, x.calc_transform(casted_data,
                                                         codomain=y),
                                     np.array([[0.04668049 - 0.03351745j, -0.04382765 - 0.06455639j,
                                                -0.05978564 + 0.01334044j, 0.50541615 + 0.50558267j,
                                                -0.01458536 - 0.01646137j, 0.01649006 + 0.01990988j],
                                               [-0.10653817 - 0.03885947j, 0.03298075 + 0.00374715j,
                                                -0.00622585 + 0.01037453j, 0.05347464 - 0.04233343j,
                                                0.05167177 - 0.00643947j, 0.01995970 + 0.01168872j],
                                               [0.03821242 - 0.00435542j, 0.07533170 + 0.14590143j,
                                                -0.01493027 - 0.02664675j, -0.01128964 - 0.02424692j,
                                                -0.03347793 - 0.0358814j, -0.03924164 - 0.01978305j],
                                               [-0.00517758 - 0.08601604j, 0.02246912 + 0.01942764j,
                                                0.06627311 + 0.08763801j, -0.02238926 - 0.06140625j,
                                                0.06211313 - 0.03317753j, -0.01519073 - 0.02842563j],
                                               [-0.00881969 + 0.01842357j, -0.01972641 - 0.00994365j,
                                                0.05289453 - 0.06822038j, -0.02492378 - 0.06097411j,
                                                0.06365649 - 0.09346585j, 0.05031486 + 0.00858656j],
                                               [-0.09355730 - 0.07553701j, 0.02395259 + 0.02185743j,
                                                0.03107832 + 0.04714527j, 0.01865586 + 0.08640926j,
                                                -0.03414096 + 0.02605602j, 0.09492552 - 0.01306734j]])))

###############################################################################

    @parameterized.expand(
        itertools.product(fft_modules,
                          [(6, 6), (8, 8), (6, 8)],
                          [(True, True), (False, False),
                           (True, False), (False, True)],
                          [(True, True), (False, False),
                           (True, False), (False, True)]),
        testcase_func_name=custom_name_func)
    def test_calc_transform_variations(self, fft_module, shape, zerocenter_in,
                                       zerocenter_out):
        data = np.arange(np.prod(shape)).reshape(shape)
        x = rg_space(shape, complexity=2, zerocenter=zerocenter_in,
                     fft_module=fft_module)
        y = x.get_codomain()
        y.paradict['zerocenter'] = zerocenter_out

        casted_data = x.cast(data)
        x_result = x.calc_transform(casted_data, codomain=y)

        np_data = data.copy()
        np_data = np.fft.fftshift(np_data, axes=np.where(zerocenter_in)[0])
        np_data = np.fft.fftn(np_data)
        np_data = np.fft.fftshift(np_data, axes=np.where(zerocenter_out)[0])
        np_result = np_data/np.prod(shape)
        assert(check_almost_equality(x, x_result, np_result))

###############################################################################

    @parameterized.expand([],testcase_func_name=custom_name_func)
    def test_calc_smooth(self):
        sigma = 0.01
        shape = (8, 8)
        a = np.arange(np.prod(shape)).reshape(shape)
        x = rg_space(shape)
        casted_a = x.cast(a)
        assert(check_almost_equality(x, x.calc_smooth(casted_a, sigma=sigma),
                                     np.array([[0.3869063,   1.33370382,   2.34906384,   3.3400879,
                                                4.34774552,   5.33876958,   6.3541296,   7.30092712],
                                               [7.96128648,   8.90808401,   9.92344403,  10.91446809,
                                                11.9221257,  12.91314976,  13.92850978,  14.87530731],
                                               [16.08416664,  17.03096417,  18.04632419,  19.03734824,
                                                20.04500586,  21.03602992,  22.05138994,  22.99818747],
                                               [24.01235911,  24.95915664,  25.97451666,  26.96554072,
                                                27.97319833,  28.96422239,  29.97958241,  30.92637994],
                                               [32.07362006,  33.02041759,  34.03577761,  35.02680167,
                                                36.03445928,  37.02548334,  38.04084336,  38.98764089],
                                               [40.00181253,  40.94861006,  41.96397008,  42.95499414,
                                                43.96265176,  44.95367581,  45.96903583,  46.91583336],
                                               [48.12469269,  49.07149022,  50.08685024,  51.0778743,
                                                52.08553191,  53.07655597,  54.09191599,  55.03871352],
                                               [55.69907288,  56.6458704,  57.66123042,  58.65225448,
                                                59.6599121,  60.65093616,  61.66629618,  62.6130937]])))

###############################################################################

    @parameterized.expand([], testcase_func_name=custom_name_func)
    def test_calc_power(self):
        shape = (8, 8)
        a = np.arange(np.prod(shape)).reshape(shape)
        x = rg_space(shape)
        assert_almost_equal(x.calc_power(a),
                            np.array([992.25, 55.48097039, 0., 16.25,
                                      0., 0., 9.51902961, 0.,
                                      0., 8.125, 0., 0.,
                                      0., 0., 0.]))


###############################################################################
###############################################################################

class Test_Lm_Space(unittest.TestCase):

    @parameterized.expand(
        itertools.product([1, 17],
                          [None, 12, 17],
                          all_lm_datatypes),
        testcase_func_name=custom_name_func)
    def test_successfull_init(self, lmax, mmax, dtype):
        # TODO Look at this
        if datamodel in ['not']:
            l = lm_space(lmax, mmax=mmax, dtype=dtype)
            assert(isinstance(l.harmonic, bool))
            assert_equal(l.paradict['lmax'], lmax)
            if mmax is None or mmax > lmax:
                assert_equal(l.paradict['mmax'], lmax)
            else:
                assert_equal(l.paradict['mmax'], mmax)
            assert_equal(l.dtype, dtype)
            assert_equal(l.discrete, True)
            assert_equal(l.harmonic, True)
            assert_equal(l.distances, (np.float(1),))
        else:
            with assert_raises(NotImplementedError): lm_space(lmax, mmax=mmax, dtype=dtype)


###############################################################################

    def test_para(self):
        lmax = 17
        mmax = 12
        l = lm_space(lmax, mmax=mmax)
        assert_equal(l.para, np.array([lmax, mmax]))

        new_para = np.array([9, 12])
        l.para = new_para
        assert_equal(l.para, np.array([9, 9]))

    def test_get_shape_dof_meta_volume(self):
        lmax = 17
        mmax = 12
        l = lm_space(lmax, mmax=mmax)

        assert_equal(l.get_shape(), (156,))
        assert_equal(l.get_dof(), 294)
        assert_equal(l.get_dof(split=True), (294,))
        assert_equal(l.get_meta_volume(), 294.)
        assert_equal(l.get_meta_volume(split=True),
                     l.cast(np.concatenate([np.ones(18), np.ones(138)*2])))

    def test_cast(self):
        lmax = 17
        mmax = 12
        l = lm_space(lmax, mmax=mmax)

        casted = l.cast(1+1j)
        real_part = casted[:18]
        assert(real_part,  l.unary_operation(real_part, 'real'))

###############################################################################

    @parameterized.expand([], testcase_func_name=custom_name_func)
    def test_enforce_power(self):
        lmax = 17
        mmax = 12
        # TODO Look at this
        if datamodel in ['not']:
            l = lm_space(lmax, mmax=mmax, datamodel=datamodel)

            assert_equal(l.enforce_power(2),
                         np.ones(18)*2)
            assert_almost_equal(
                l.enforce_power(lambda x: 42 / (1 + x)**5),
                np.array([  4.20000000e+01,   1.31250000e+00,   1.72839506e-01,
             4.10156250e-02,   1.34400000e-02,   5.40123457e-03,
             2.49895877e-03,   1.28173828e-03,   7.11273688e-04,
             4.20000000e-04,   2.60786956e-04,   1.68788580e-04,
             1.13118211e-04,   7.80924615e-05,   5.53086420e-05,
             4.00543213e-05,   2.95804437e-05,   2.22273027e-05]))
        else:
            with assert_raises(NotImplementedError): lm_space(lmax, mmax=mmax, datamodel=datamodel)

##############################################################################

    @parameterized.expand([], testcase_func_name=custom_name_func)
    def test_get_check_codomain(self):
        lmax = 23
        mmax = 23
        # TODO Look at this
        if datamodel in ['not']:
            l = lm_space(lmax, mmax=mmax)

            y = l.get_codomain()
            assert(l.check_codomain(y))
            assert(y.check_codomain(l))

            if 'hp_space' in available:
                y = l.get_codomain('hp')
                assert(l.check_codomain(y))
                assert(y.check_codomain(l))
            if 'gl_space' in available:
                y = l.get_codomain('gl')
                assert(l.check_codomain(y))
                assert(y.check_codomain(l))
        else:
            with assert_raises(NotImplementedError): lm_space(lmax, mmax=mmax)


###############################################################################
#
#    @parameterized.expand(
#        itertools.product([True], #[True, False],
#                          ['pyfftw']),
#                          #DATAMODELS['rg_space']),
#        testcase_func_name=custom_name_func)
#    def test_get_random_values(self, harmonic, datamodel):
#        x = rg_space((4, 4), complexity=1, harmonic=harmonic,
#                     datamodel=datamodel)
#
#        # pm1
#        data = x.get_random_values(random='pm1')
#        flipped_data = flip(x, data)
#        assert(check_almost_equality(x, data, flipped_data))
#
#        # gau
#        data = x.get_random_values(random='gau', mean=4 + 3j, std=2)
#        flipped_data = flip(x, data)
#        assert(check_almost_equality(x, data, flipped_data))
#
#        # uni
#        data = x.get_random_values(random='uni', vmin=-2, vmax=4)
#        flipped_data = flip(x, data)
#        assert(check_almost_equality(x, data, flipped_data))
#
#        # syn
#        data = x.get_random_values(random='syn',
#                                   spec=lambda x: 42 / (1 + x)**3)
#        flipped_data = flip(x, data)
#        assert(check_almost_equality(x, data, flipped_data))
#
################################################################################
#
#    @parameterized.expand(
#        DATAMODELS['rg_space'],
#        testcase_func_name=custom_name_func)
#    def test_calc_dot(self, datamodel):
#        shape = (8, 8)
#        a = np.arange(np.prod(shape)).reshape(shape)
#        x = rg_space(shape)
#        assert_equal(x.calc_dot(a, a), 85344)
#        assert_equal(x.calc_dot(a, 1), 2016)
#        assert_equal(x.calc_dot(1, a), 2016)
#
################################################################################
#
#    @parameterized.expand(
#        itertools.product([0, 1],
#                          DATAMODELS['rg_space']),
#        testcase_func_name=custom_name_func)
#    def test_calc_transform_general(self, complexity, datamodel):
#        data = fft_test_data.copy()
#        shape = data.shape
#
#        x = rg_space(shape, complexity=complexity, datamodel=datamodel)
#        data = fft_test_data.copy()
#        data = x.cast(data)
#        check_equality(x, data, x.calc_transform(x.calc_transform(data)))
#
################################################################################
#
#    @parameterized.expand(
#        itertools.product(fft_modules,
#                          DATAMODELS['rg_space']),
#        testcase_func_name=custom_name_func)
#    def test_calc_transform_explicit(self, fft_module, datamodel):
#        data = fft_test_data.copy()
#        shape = data.shape
#
#        x = rg_space(shape, complexity=2, zerocenter=False,
#                     fft_module=fft_module, datamodel=datamodel)
#        casted_data = x.cast(data)
#        assert(check_almost_equality(x, x.calc_transform(casted_data),
#                                     np.array([[0.50541615 + 0.50558267j, -0.01458536 - 0.01646137j,
#                                                0.01649006 + 0.01990988j, 0.04668049 - 0.03351745j,
#                                                -0.04382765 - 0.06455639j, -0.05978564 + 0.01334044j],
#                                               [-0.05347464 + 0.04233343j, -0.05167177 + 0.00643947j,
#                                                -0.01995970 - 0.01168872j, 0.10653817 + 0.03885947j,
#                                                -0.03298075 - 0.00374715j, 0.00622585 - 0.01037453j],
#                                               [-0.01128964 - 0.02424692j, -0.03347793 - 0.0358814j,
#                                                -0.03924164 - 0.01978305j, 0.03821242 - 0.00435542j,
#                                                0.07533170 + 0.14590143j, -0.01493027 - 0.02664675j],
#                                               [0.02238926 + 0.06140625j, -0.06211313 + 0.03317753j,
#                                                0.01519073 + 0.02842563j, 0.00517758 + 0.08601604j,
#                                                -0.02246912 - 0.01942764j, -0.06627311 - 0.08763801j],
#                                               [-0.02492378 - 0.06097411j, 0.06365649 - 0.09346585j,
#                                                0.05031486 + 0.00858656j, -0.00881969 + 0.01842357j,
#                                                -0.01972641 - 0.00994365j, 0.05289453 - 0.06822038j],
#                                               [-0.01865586 - 0.08640926j, 0.03414096 - 0.02605602j,
#                                                -0.09492552 + 0.01306734j, 0.09355730 + 0.07553701j,
#                                                -0.02395259 - 0.02185743j, -0.03107832 - 0.04714527j]])))
#
#        x = rg_space(shape, complexity=2, zerocenter=True,
#                     fft_module=fft_module, datamodel=datamodel)
#        casted_data = x.cast(data)
#        assert(check_almost_equality(x, x.calc_transform(casted_data),
#                                     np.array([[0.00517758 + 0.08601604j, 0.02246912 + 0.01942764j,
#                                                -0.06627311 - 0.08763801j, -0.02238926 - 0.06140625j,
#                                                -0.06211313 + 0.03317753j, -0.01519073 - 0.02842563j],
#                                               [0.00881969 - 0.01842357j, -0.01972641 - 0.00994365j,
#                                                -0.05289453 + 0.06822038j, -0.02492378 - 0.06097411j,
#                                                -0.06365649 + 0.09346585j, 0.05031486 + 0.00858656j],
#                                               [0.09355730 + 0.07553701j, 0.02395259 + 0.02185743j,
#                                                -0.03107832 - 0.04714527j, 0.01865586 + 0.08640926j,
#                                                0.03414096 - 0.02605602j, 0.09492552 - 0.01306734j],
#                                               [-0.04668049 + 0.03351745j, -0.04382765 - 0.06455639j,
#                                                0.05978564 - 0.01334044j, 0.50541615 + 0.50558267j,
#                                                0.01458536 + 0.01646137j, 0.01649006 + 0.01990988j],
#                                               [0.10653817 + 0.03885947j, 0.03298075 + 0.00374715j,
#                                                0.00622585 - 0.01037453j, 0.05347464 - 0.04233343j,
#                                                -0.05167177 + 0.00643947j, 0.01995970 + 0.01168872j],
#                                               [-0.03821242 + 0.00435542j, 0.07533170 + 0.14590143j,
#                                                0.01493027 + 0.02664675j, -0.01128964 - 0.02424692j,
#                                                0.03347793 + 0.0358814j, -0.03924164 - 0.01978305j]])))
#
#        x = rg_space(shape, complexity=2, zerocenter=[True, False],
#                     fft_module=fft_module, datamodel=datamodel)
#        casted_data = x.cast(data)
#        assert(check_almost_equality(x, x.calc_transform(casted_data),
#                                     np.array([[-0.02238926 - 0.06140625j, 0.06211313 - 0.03317753j,
#                                                -0.01519073 - 0.02842563j, -0.00517758 - 0.08601604j,
#                                                0.02246912 + 0.01942764j, 0.06627311 + 0.08763801j],
#                                               [-0.02492378 - 0.06097411j, 0.06365649 - 0.09346585j,
#                                                0.05031486 + 0.00858656j, -0.00881969 + 0.01842357j,
#                                                -0.01972641 - 0.00994365j, 0.05289453 - 0.06822038j],
#                                               [0.01865586 + 0.08640926j, -0.03414096 + 0.02605602j,
#                                                0.09492552 - 0.01306734j, -0.09355730 - 0.07553701j,
#                                                0.02395259 + 0.02185743j, 0.03107832 + 0.04714527j],
#                                               [0.50541615 + 0.50558267j, -0.01458536 - 0.01646137j,
#                                                0.01649006 + 0.01990988j, 0.04668049 - 0.03351745j,
#                                                -0.04382765 - 0.06455639j, -0.05978564 + 0.01334044j],
#                                               [0.05347464 - 0.04233343j, 0.05167177 - 0.00643947j,
#                                                0.01995970 + 0.01168872j, -0.10653817 - 0.03885947j,
#                                                0.03298075 + 0.00374715j, -0.00622585 + 0.01037453j],
#                                               [-0.01128964 - 0.02424692j, -0.03347793 - 0.0358814j,
#                                                -0.03924164 - 0.01978305j, 0.03821242 - 0.00435542j,
#                                                0.07533170 + 0.14590143j, -0.01493027 - 0.02664675j]])))
#
#        x = rg_space(shape, complexity=2, zerocenter=[True, False],
#                     fft_module=fft_module, datamodel=datamodel)
#        y = rg_space(shape, complexity=2, zerocenter=[False, True],
#                     distances=[1, 1], harmonic=True,
#                     fft_module=fft_module, datamodel=datamodel)
#        casted_data = x.cast(data)
#        assert(check_almost_equality(x, x.calc_transform(casted_data,
#                                                         codomain=y),
#                                     np.array([[0.04668049 - 0.03351745j, -0.04382765 - 0.06455639j,
#                                                -0.05978564 + 0.01334044j, 0.50541615 + 0.50558267j,
#                                                -0.01458536 - 0.01646137j, 0.01649006 + 0.01990988j],
#                                               [-0.10653817 - 0.03885947j, 0.03298075 + 0.00374715j,
#                                                -0.00622585 + 0.01037453j, 0.05347464 - 0.04233343j,
#                                                0.05167177 - 0.00643947j, 0.01995970 + 0.01168872j],
#                                               [0.03821242 - 0.00435542j, 0.07533170 + 0.14590143j,
#                                                -0.01493027 - 0.02664675j, -0.01128964 - 0.02424692j,
#                                                -0.03347793 - 0.0358814j, -0.03924164 - 0.01978305j],
#                                               [-0.00517758 - 0.08601604j, 0.02246912 + 0.01942764j,
#                                                0.06627311 + 0.08763801j, -0.02238926 - 0.06140625j,
#                                                0.06211313 - 0.03317753j, -0.01519073 - 0.02842563j],
#                                               [-0.00881969 + 0.01842357j, -0.01972641 - 0.00994365j,
#                                                0.05289453 - 0.06822038j, -0.02492378 - 0.06097411j,
#                                                0.06365649 - 0.09346585j, 0.05031486 + 0.00858656j],
#                                               [-0.09355730 - 0.07553701j, 0.02395259 + 0.02185743j,
#                                                0.03107832 + 0.04714527j, 0.01865586 + 0.08640926j,
#                                                -0.03414096 + 0.02605602j, 0.09492552 - 0.01306734j]])))
#
################################################################################
#
#    @parameterized.expand(DATAMODELS['rg_space'],
#                          testcase_func_name=custom_name_func)
#    def test_calc_smooth(self, datamodel):
#        sigma = 0.01
#        shape = (8, 8)
#        a = np.arange(np.prod(shape)).reshape(shape)
#        x = rg_space(shape)
#        casted_a = x.cast(a)
#        assert(check_almost_equality(x, x.calc_smooth(casted_a, sigma=sigma),
#                                     np.array([[0.3869063,   1.33370382,   2.34906384,   3.3400879,
#                                                4.34774552,   5.33876958,   6.3541296,   7.30092712],
#                                               [7.96128648,   8.90808401,   9.92344403,  10.91446809,
#                                                11.9221257,  12.91314976,  13.92850978,  14.87530731],
#                                               [16.08416664,  17.03096417,  18.04632419,  19.03734824,
#                                                20.04500586,  21.03602992,  22.05138994,  22.99818747],
#                                               [24.01235911,  24.95915664,  25.97451666,  26.96554072,
#                                                27.97319833,  28.96422239,  29.97958241,  30.92637994],
#                                               [32.07362006,  33.02041759,  34.03577761,  35.02680167,
#                                                36.03445928,  37.02548334,  38.04084336,  38.98764089],
#                                               [40.00181253,  40.94861006,  41.96397008,  42.95499414,
#                                                43.96265176,  44.95367581,  45.96903583,  46.91583336],
#                                               [48.12469269,  49.07149022,  50.08685024,  51.0778743,
#                                                52.08553191,  53.07655597,  54.09191599,  55.03871352],
#                                               [55.69907288,  56.6458704,  57.66123042,  58.65225448,
#                                                59.6599121,  60.65093616,  61.66629618,  62.6130937]])))
#
################################################################################
#
#    @parameterized.expand(DATAMODELS['rg_space'],
#                          testcase_func_name=custom_name_func)
#    def test_calc_power(self, datamodel):
#        shape = (8, 8)
#        a = np.arange(np.prod(shape)).reshape(shape)
#        x = rg_space(shape)
#        assert_almost_equal(x.calc_power(a),
#                            np.array([992.25, 55.48097039, 0., 16.25,
#                                      0., 0., 9.51902961, 0.,
#                                      0., 8.125, 0., 0.,
#                                      0., 0., 0.]))
#


print all_spaces
print generate_space('rg_space')

class Test_axis(unittest.TestCase):
    @parameterized.expand(
        itertools.product(point_like_spaces, [4],
                          ['sum', 'prod', 'mean', 'var', 'std', 'median', 'all',
                           'any', 'amin', 'nanmin', 'argmin', 'amax', 'nanmax',
                           'argmax'],
                          [None, (0,)]),
        testcase_func_name=custom_name_func)
    def test_unary_operations(self, name, num, op, axis):
        s = generate_space_with_size(name, num)
        d = generate_data(s)
        a = d.get_full_data()
        if op in ['argmin', 'argmax'] and axis is not None:
            assert_raises(NotImplementedError, lambda: s.unary_operation
                          (d, op, axis=axis))
        else:
            assert_almost_equal(s.unary_operation(d, op, axis=axis),
                                getattr(np, op)(a, axis=axis), decimal=4)
            if name in ['rg_space']:
                if op in ['argmin', 'argmax']:
                    assert_raises(NotImplementedError, lambda: s.unary_operation
                                  (d, op, axis=(0, 1)))
                    assert_raises(NotImplementedError, lambda: s.unary_operation
                                  (d, op, axis=(1, )))
                else:
                    assert_almost_equal(s.unary_operation(d, op, axis=(0, 1)),
                                        getattr(np, op)(a, axis=(0, 1)),
                                        decimal=4)
                    assert_almost_equal(s.unary_operation(d, op, axis=(1,)),
                                        getattr(np, op)(a, axis=(1,)),
                                        decimal=4)
