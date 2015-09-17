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
                  gl_space

from nifty.nifty_paradict import space_paradict
from nifty.nifty_core import POINT_DISTRIBUTION_STRATEGIES


###############################################################################

def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )

###############################################################################
###############################################################################

all_datatypes = [np.dtype('bool'),
                 np.dtype('int8'),
                 np.dtype('int16'),
                 np.dtype('int32'),
                 np.dtype('int64'),
                 np.dtype('float16'),
                 np.dtype('float32'),
                 np.dtype('float64'),
                 np.dtype('complex64'),
                 np.dtype('complex128')]

###############################################################################

point_datamodels = ['np'] + POINT_DISTRIBUTION_STRATEGIES

###############################################################################

all_spaces = ['space', 'point_space', 'rg_space', 'lm_space', 'hp_space',
              'gl_space']

point_like_spaces = ['point_space', 'rg_space', 'lm_space', 'hp_space',
                     'gl_space']


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

class Test_Common_Point_Like_Space_Features(unittest.TestCase):

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
        # TODO: Make power_indices class for lm_space
        # if s.harmonic:
        #     assert(isinstance(s.power_indices, power_indices))

    @parameterized.expand(point_like_spaces,
                          testcase_func_name=custom_name_func)
    def test_global_parameters(self, name):
        s = generate_space(name)
        assert(isinstance(s.get_shape(), tuple))

        assert(isinstance(s.get_dim(), np.int))
        assert(isinstance(s.get_dim(split=True), tuple))
        assert_equal(s.get_dim(), np.prod(s.get_dim(split=True)))

        assert(isinstance(s.get_dof(), np.int))
        assert(isinstance(s.get_dof(split=True), tuple))
        assert_equal(s.get_dof(), np.prod(s.get_dof(split=True)))

        assert(isinstance(s.get_meta_volume(), np.float))
        assert(isinstance(s.get_meta_volume(split=True), type(s.cast(1))))
        assert_almost_equal(
            s.get_meta_volume(), s.get_meta_volume(split=True).sum(), 2)


###############################################################################
###############################################################################

class Test_Point_Initialization(unittest.TestCase):

    @parameterized.expand(
        itertools.product([0, 1, 10],
                          all_datatypes,
                          point_datamodels),
        testcase_func_name=custom_name_func)
    def test_successfull_init(self, num, dtype, datamodel):
        p = point_space(num, dtype, datamodel)
        assert_equal(p.paradict['num'], num)
        assert_equal(p.dtype, dtype)
        assert_equal(p.datamodel, datamodel)

        assert_equal(p.discrete, True)
        assert_equal(p.harmonic, False)
        assert_equal(p.distances, (np.float(1.),))

    def test_para(self):
        num = 10
        p = point_space(num)
        assert_equal(p.para[0], num)

        new_num = 15
        p.para = np.array([new_num])
        assert_equal(p.para[0], new_num)

    def test_init_fail(self):
        assert_raises(ValueError, lambda: point_space(-5))
        assert_raises(ValueError, lambda: point_space((10, 10)))
        assert_raises(ValueError, lambda: point_space(10, np.uint))




