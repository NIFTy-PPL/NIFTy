# -*- coding: utf-8 -*-

import unittest

import numpy as np
from numpy.testing import assert_,\
                          assert_equal

import itertools

from nifty import Field,\
                  RGSpace,\
                  FieldArray

from d2o import distributed_data_object,\
                STRATEGIES

from test.common import expand

np.random.seed(123)

SPACES = [RGSpace((4,), dtype=np.float), RGSpace((5), dtype=np.complex)]
SPACE_COMBINATIONS = [SPACES[0], SPACES[1], SPACES]


class Test_Interface(unittest.TestCase):
    @expand([['dtype', np.dtype],
             ['distribution_strategy', str],
             ['domain', tuple],
             ['field_type', tuple],
             ['domain_axes', tuple],
             ['field_type_axes', tuple],
             ['val', distributed_data_object],
             ['shape', tuple],
             ['dim', np.int],
             ['dof', np.int],
             ['total_volume', np.float]])
    def test_return_types(self, attribute, desired_type):
        x = RGSpace(shape=(4,))
        ft = FieldArray(shape=(2,), dtype=np.complex)
        f = Field(domain=x, field_type=ft)
        assert_(isinstance(getattr(f, attribute), desired_type))

#class Test_Initialization(unittest.TestCase):
#
#    @parameterized.expand(
#        itertools.product(SPACE_COMBINATIONS,
#                          []
#                          )
#    def test_
