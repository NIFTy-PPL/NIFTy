# -*- coding: utf-8 -*-

from numpy.testing import assert_equal,\
    assert_almost_equal,\
    assert_raises

from nose_parameterized import parameterized
import unittest
import itertools
import numpy as np

from nifty import space,\
                  point_space,\
                  rg_space,\
                  lm_space,\
                  hp_space,\
                  gl_space,\
                  field,\
                  distributed_data_object

from nifty.nifty_paradict import space_paradict
from nifty.nifty_core import POINT_DISTRIBUTION_STRATEGIES

from nifty.rg.nifty_rg import RG_DISTRIBUTION_STRATEGIES
from nifty.lm.nifty_lm import LM_DISTRIBUTION_STRATEGIES,\
                              GL_DISTRIBUTION_STRATEGIES,\
                              HP_DISTRIBUTION_STRATEGIES
from nifty.nifty_power_indices import power_indices


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
DATAMODELS['point_space'] = ['np'] + POINT_DISTRIBUTION_STRATEGIES
DATAMODELS['rg_space'] = ['np'] + RG_DISTRIBUTION_STRATEGIES
DATAMODELS['lm_space'] = ['np'] + LM_DISTRIBUTION_STRATEGIES
DATAMODELS['gl_space'] = ['np'] + GL_DISTRIBUTION_STRATEGIES
DATAMODELS['hp_space'] = ['np'] + HP_DISTRIBUTION_STRATEGIES

###############################################################################

all_spaces = ['space', 'point_space', 'rg_space', 'lm_space', 'hp_space',
              'gl_space']

point_like_spaces = ['point_space', 'rg_space', 'lm_space', 'hp_space',
                     'gl_space']

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
                    'argmin_flat', 'argmax', 'argmax_flat', 'conjugate', 'sum',
                    'prod', 'unique', 'copy', 'copy_empty', 'isnan', 'isinf',
                    'isfinite', 'nan_to_num', 'all', 'any', 'None']

binary_operations = ['add', 'radd', 'iadd', 'sub', 'rsub', 'isub', 'mul',
                     'rmul', 'imul', 'div', 'rdiv', 'idiv', 'pow', 'rpow',
                     'ipow', 'ne', 'lt', 'le', 'eq', 'ge', 'gt', 'None']

###############################################################################

def generate_space(name):
    space_dict = {'space': space(),
                  'point_space': point_space(10),
                  'rg_space': rg_space((8, 8)),
                  'lm_space': lm_space(mmax=11, lmax=11),
                  'hp_space': hp_space(8),
                  'gl_space': gl_space(nlat=10, nlon=19),
                  }
    return space_dict[name]


def generate_data(space):
    a = np.arange(space.get_dim()).reshape(space.get_shape())
    data = space.cast(a)
    return data


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
        assert(callable(s._identifier))
        assert(callable(s.__eq__))
        assert(callable(s.__ne__))
        assert(callable(s.__len__))
        assert(callable(s.copy))
        assert(callable(s.getitem))
        assert(callable(s.setitem))
        assert(callable(s.apply_scalar_function))
        assert(callable(s.unary_operation))
        assert(callable(s.binary_operation))
        assert(callable(s.get_norm))
        assert(callable(s.get_shape))
        assert(callable(s.get_dim))
        assert(callable(s.get_dof))
        assert(callable(s.get_meta_volume))
        assert(callable(s.cast))
        assert(callable(s.enforce_power))
        assert(callable(s.check_codomain))
        assert(callable(s.get_codomain))
        assert(callable(s.get_random_values))
        assert(callable(s.calc_weight))
        assert(callable(s.get_weight))
        assert(callable(s.calc_dot))
        assert(callable(s.calc_transform))
        assert(callable(s.calc_smooth))
        assert(callable(s.calc_power))
        assert(callable(s.calc_real_Q))
        assert(callable(s.calc_bincount))
        assert(callable(s.get_plot))
        assert(callable(s.__repr__))
        assert(callable(s.__str__))


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
        assert(isinstance(s.datamodel, str))
        assert(isinstance(s.discrete, bool))
        assert(isinstance(s.harmonic, bool))
        assert(isinstance(s.distances, tuple))
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
        assert(isinstance(s.get_meta_volume(split=True), type(s.cast(1))))
        assert_almost_equal(
            s.get_meta_volume(), s.get_meta_volume(split=True).sum(), 2)

#
#class Test_Common_Point_Like_Space_Functions(unittest.TestCase):
#
#    @parameterized.expand(point_like_spaces,
#                          testcase_func_name=custom_name_func)
#    def test_copy(self, name):
#        s = generate_space(name)
#        t = s.copy()
#        assert(s == t)
#        assert(id(s) != id(t))
#
#    @parameterized.expand(point_like_spaces,
#                          testcase_func_name=custom_name_func)
#    def test_unary_operations(self, name):
#        s = generate_space(name)
#        t = s.copy()
#
#
#    @parameterized.expand(point_like_spaces,
#                          testcase_func_name=custom_name_func)
#    def test_apply_scalar_function(self, name):
#        s = generate_space(name)
#        d = generate_data(s)
#        d2 = s.apply_scalar_function(d, lambda x: x**2)
#        assert(isinstance(d2, type(d)))


###############################################################################
###############################################################################

class Test_Point_Space(unittest.TestCase):

    @parameterized.expand(
        itertools.product([0, 1, 10],
                          all_point_datatypes,
                          DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_successfull_init(self, num, dtype, datamodel):
        p = point_space(num, dtype, datamodel)
        assert_equal(p.paradict['num'], num)
        assert_equal(p.dtype, dtype)
        assert_equal(p.datamodel, datamodel)

        assert_equal(p.discrete, True)
        assert_equal(p.harmonic, False)
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
        itertools.product([0, 1, 10],
                          DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_apply_scalar_function(self, num, datamodel):
        s = point_space(num, datamodel=datamodel)
        d = generate_data(s)
        t = s.apply_scalar_function(d, lambda x: x**2)
        assert(s.unary_operation(s.binary_operation(d**2, t, 'eq'), 'all'))
        assert(id(d) != id(t))

        t = s.apply_scalar_function(d, lambda x: x**2, inplace=True)
        assert(s.unary_operation(s.binary_operation(d, t, 'eq'), 'all'))
        assert(id(d) == id(t))

###############################################################################

    @parameterized.expand(
        itertools.product([1, 10],
                          unary_operations,
                          DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_unary_operations(self, num, op, datamodel):
        s = point_space(num, datamodel=datamodel)
        d = s.cast(np.arange(num))
        s.unary_operation(d, op)
        # TODO: Implement value verification

    @parameterized.expand(
        itertools.product([1, 10],
                          binary_operations,
                          DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_binary_operations(self, num, op, datamodel):
        s = point_space(num, datamodel=datamodel)
        d = s.cast(np.arange(num))
        d2 = d[::-1]
        s.binary_operation(d, d2, op)
        # TODO: Implement value verification

###############################################################################

    @parameterized.expand(
        itertools.product(DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_get_norm(self, datamodel):
        num = 10
        s = point_space(num, datamodel=datamodel)
        d = s.cast(np.arange(num))
        assert_almost_equal(s.get_norm(d), 16.881943016134134)
        assert_almost_equal(s.get_norm(d, q=3), 12.651489979526238)

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes,
                          DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_get_shape_dim(self, dtype, datamodel):
        num = 10
        s = point_space(num, dtype, datamodel=datamodel)

        assert_equal(s.get_shape(), (num,))
        assert_equal(s.get_dim(), num)

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes,
                          DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_get_shape_dof(self, dtype, datamodel):
        num = 10
        s = point_space(num, dtype, datamodel=datamodel)

        if issubclass(dtype.type, np.complexfloating):
            assert_equal(s.get_dof(), 2*num)
            assert_equal(s.get_dof(split=True), (2*num,))
        else:
            assert_equal(s.get_dof(), num)
            assert_equal(s.get_dof(split=True), (num,))

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes,
                          DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_get_shape_vol(self, dtype, datamodel):
        num = 10
        s = point_space(num, dtype, datamodel=datamodel)

        assert_equal(s.get_vol(), 1.)
        assert_equal(s.get_vol(split=True), (1.,))

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes,
                          DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_get_shape_metavolume(self, dtype, datamodel):
        num = 10
        s = point_space(num, dtype, datamodel=datamodel)

        assert_equal(s.get_meta_volume(), 10.)
        assert(s.unary_operation(s.binary_operation(
                                                s.get_meta_volume(split=True),
                                                s.cast(1), 'eq'),
                                 'all'))

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes,
                          DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_cast_from_scalar(self, dtype, datamodel):
        num = 10
        scalar = 4
        s = point_space(num, dtype, datamodel=datamodel)
        if datamodel == 'np':
            d = (np.ones((num,)) * scalar).astype(dtype=dtype)
        else:
            d = distributed_data_object(scalar,
                                        global_shape=(num,),
                                        dtype=dtype,
                                        distribution_strategy=datamodel)

        casted_scalar = s.cast(scalar)
        assert(s.unary_operation(s.binary_operation(casted_scalar, d, 'eq'),
                                 'all'))
        if datamodel != 'np':
            assert(d.equal(casted_scalar))

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes,
                          DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_cast_from_field(self, dtype, datamodel):
        num = 10
        a = np.arange(num,).astype(dtype)
        s = point_space(num, dtype, datamodel=datamodel)
        f = field(s, val=a)

        if datamodel == 'np':
            d = a
        else:
            d = distributed_data_object(a, dtype=dtype,
                                        distribution_strategy=datamodel)

        casted_f = s.cast(f)
        assert(s.unary_operation(s.binary_operation(casted_f, d, 'eq'),
                                 'all'))
        if datamodel != 'np':
            assert(d.equal(casted_f))

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes,
                          DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_cast_from_ndarray(self, dtype, datamodel):
        num = 10
        a = np.arange(num,)
        s = point_space(num, dtype, datamodel=datamodel)

        if datamodel == 'np':
            d = a.astype(dtype)
        else:
            d = distributed_data_object(a, dtype=dtype,
                                        distribution_strategy=datamodel)

        casted_a = s.cast(a)
        assert(s.unary_operation(s.binary_operation(casted_a, d, 'eq'),
                                 'all'))
        if datamodel != 'np':
            assert(d.equal(casted_a))

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes,
                          DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_cast_from_d2o(self, dtype, datamodel):
        num = 10
        pre_a = np.arange(num,)
        a = distributed_data_object(pre_a)
        s = point_space(num, dtype, datamodel=datamodel)

        if datamodel == 'np':
            d = pre_a.astype(dtype)
        else:
            d = distributed_data_object(a, dtype=dtype,
                                        distribution_strategy=datamodel)

        casted_a = s.cast(a)
        assert(s.unary_operation(s.binary_operation(casted_a, d, 'eq'),
                                 'all'))
        if datamodel != 'np':
            assert(d.equal(casted_a))


###############################################################################

    def test_raise_on_not_implementable_methods(self):
        s = point_space(10)
        assert_raises(lambda: s.enforce_power)
        assert_raises(lambda: s.calc_smooth)
        assert_raises(lambda: s.calc_power)

###############################################################################

    @parameterized.expand(
        [[10, np.dtype('float64'), 'equal'],
         [10, np.dtype('float32'), 'np'],
         [12, np.dtype('float64'), 'np']],
        testcase_func_name=custom_name_func)
    def test_get_check_codomain(self, num, dtype, datamodel):
        s = point_space(10, dtype=np.dtype('float64'), datamodel='np')

        t = s.get_codomain()
        assert(s.check_codomain(t))

        t_bad = point_space(num, dtype=dtype, datamodel=datamodel)
        assert(s.check_codomain(t_bad) == False)

        assert(s.check_codomain(None) == False)

###############################################################################

    @parameterized.expand(
        itertools.product(all_point_datatypes,
                          DATAMODELS['point_space']),
        testcase_func_name=custom_name_func)
    def test_cast_from_d2o(self, dtype, datamodel):
        num = 100000
        s = point_space(num, dtype, datamodel=datamodel)

        pm = s.get_random_values(random='pm1')
        assert_almost_equal(0, s.unary_operation(pm, op='mean'), 2)

        gau = s.get_random_values(random='gau',
                                  mean=10,
                                  dev=)














