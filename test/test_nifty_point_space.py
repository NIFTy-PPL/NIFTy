# -*- coding: utf-8 -*-

from numpy.testing import assert_equal,\
    assert_almost_equal,\
    assert_raises

from nose_parameterized import parameterized
import unittest
import itertools
import numpy as np

def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )

from nifty import point_space
from nifty.nifty_core import POINT_DISTRIBUTION_STRATEGIES

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

all_datamodels = ['np'] + POINT_DISTRIBUTION_STRATEGIES


###############################################################################
###############################################################################

class Test_Initialization(unittest.TestCase):

    @parameterized.expand(
        itertools.product([0, 1, 10],
                          all_datatypes,
                          all_datamodels),
        testcase_func_name=custom_name_func)
    def test_successfull_init(self, num, dtype, datamodel):
        p = point_space(num, dtype, datamodel)
        assert_equal(p.paradict['num'], num)
        assert_equal(p.dtype, dtype)
        assert_equal(p.datamodel, datamodel)

        assert_equal(p.discrete, True)
        assert_equal(p.harmonic, False)
        assert_equal(p.vol, np.real(np.array([1], dtype=dtype)))

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


