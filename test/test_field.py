# -*- coding: utf-8 -*-

import unittest

import numpy as np
from numpy.testing import assert_,\
                          assert_equal

from itertools import product

from nifty import Field,\
                  RGSpace,\
                  FieldArray

from d2o import distributed_data_object,\
                STRATEGIES

from test.common import expand

np.random.seed(123)

SPACES = [RGSpace((4,), dtype=np.float), RGSpace((5), dtype=np.complex)]
SPACE_COMBINATIONS = [(), SPACES[0], SPACES[1], SPACES]


class Test_Interface(unittest.TestCase):
    @expand(product(SPACE_COMBINATIONS,
                    [['dtype', np.dtype],
                     ['distribution_strategy', str],
                     ['domain', tuple],
                     ['domain_axes', tuple],
                     ['val', distributed_data_object],
                     ['shape', tuple],
                     ['dim', np.int],
                     ['dof', np.int],
                     ['total_volume', np.float]]))
    def test_return_types(self, domain, attribute_desired_type):
        attribute = attribute_desired_type[0]
        desired_type = attribute_desired_type[1]
        f = Field(domain=domain)
        assert_(isinstance(getattr(f, attribute), desired_type))

#class Test_Initialization(unittest.TestCase):
#
#    @parameterized.expand(
#        itertools.product(SPACE_COMBINATIONS,
#                          []
#                          )
#    def test_
