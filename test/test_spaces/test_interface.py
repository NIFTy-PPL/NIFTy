import unittest
import numpy as np

from itertools import product
from numpy.testing import assert_
from nifty import RGSpace, LMSpace, GLSpace, HPSpace
from test.common import expand


SPACES = [RGSpace((4,)), LMSpace(5), GLSpace(4), HPSpace(4)]


class SpaceInterfaceTestCase(unittest.TestCase):
    @expand(product(SPACES, [['dtype', np.dtype],
            ['harmonic', bool],
            ['shape', tuple],
            ['dim', int],
            ['total_volume', np.float]]))
    def test_return_types(self, space, attr_expected_type):
        assert_(isinstance(getattr(
                space, attr_expected_type[0]), attr_expected_type[1]))
