# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest
import numpy as np

from itertools import product
from types import LambdaType
from numpy.testing import assert_, assert_raises, assert_equal
from nifty import LMSpace, GLSpace, HPSpace
from nifty.config import dependency_injector as di
from test.common import expand, generate_spaces, generate_harmonic_spaces


class SpaceInterfaceTests(unittest.TestCase):
    @expand(product(generate_spaces(), [
                    ['harmonic', bool],
                    ['shape', tuple],
                    ['dim', int],
                    ['total_volume', np.float]]))
    def test_property_ret_type(self, space, attr_expected_type):
        assert_(
            isinstance(getattr(
                space,
                attr_expected_type[0]
            ), attr_expected_type[1])
        )

    @expand(product(generate_harmonic_spaces(), [
        ['get_fft_smoothing_kernel_function', None, LambdaType],
        ['get_fft_smoothing_kernel_function', 2.0, LambdaType],
        ]))
    def test_method_ret_type(self, space, method_expected_type):
        getattr(
            space, method_expected_type[0])(*method_expected_type[1:-1])

        assert_equal(
            type(getattr(
                space,
                method_expected_type[0])(*method_expected_type[1:-1])
            ),
            method_expected_type[-1]
        )

    @expand([[space] for space in generate_spaces()])
    def test_copy(self, space):
        # make sure it's a deep copy
        assert_(space is not space.copy())
        # make sure contents are the same
        assert_equal(space, space.copy())
