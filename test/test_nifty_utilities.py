# -*- coding: utf-8 -*-

import numpy as np
from numpy.testing import assert_equal,\
    assert_almost_equal,\
    assert_raises

from nose_parameterized import parameterized
import unittest

from nifty.nifty_utilities import hermitianize,\
    _hermitianize_inverter

from nifty.d2o import distributed_data_object,\
                      STRATEGIES

###############################################################################

def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )

###############################################################################

test_data = np.array([[-10,   9,  10,   2,  -7,  -8],
                      [ -5,   5,   5,  -1,   9,   3],
                      [ -2,  -2,   8,   9,   9, -10],
                      [ -8,  -5,  -2, -10,  -7,   7],
                      [ 10,   6,  -2,   6,  -3,  -1],
                      [  8,   1,  10,  -7,   6,  -6]])

flipped_data = np.array([[-10,  -8,  -7,   2,  10,   9],
                      [  8,  -6,   6,  -7,  10,   1],
                      [ 10,  -1,  -3,   6,  -2,   6],
                      [ -8,   7,  -7, -10,  -2,  -5],
                      [ -2, -10,   9,   9,   8,  -2],
                      [ -5,   3,   9,  -1,   5,   5]])

###############################################################################
###############################################################################

class Test_hermitianize_inverter(unittest.TestCase):
    def test_with_ndarray(self):
       assert_equal(_hermitianize_inverter(test_data), flipped_data)

    @parameterized.expand(STRATEGIES['global'],
                          testcase_func_name=custom_name_func)
    def test_with_d2o(self, distribution_strategy):
        d = distributed_data_object(
                                test_data,
                                distribution_strategy=distribution_strategy)
        assert_equal(_hermitianize_inverter(d).get_full_data(), flipped_data)