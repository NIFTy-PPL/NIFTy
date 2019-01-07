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
#
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import unittest
from itertools import product
from test.common import expand
from types import LambdaType

import nifty5 as ift
from numpy.testing import assert_

spaces = [ift.RGSpace(4),
          ift.PowerSpace(ift.RGSpace((4, 4), harmonic=True)),
          ift.LMSpace(5), ift.HPSpace(4), ift.GLSpace(4)]


class SpaceInterfaceTests(unittest.TestCase):
    @expand(product(spaces, [
                    ['harmonic', bool],
                    ['shape', tuple],
                    ['size', int]]))
    def test_property_ret_type(self, space, attr_expected_type):
        assert_(isinstance(getattr(space, attr_expected_type[0]),
                           attr_expected_type[1]))

    @expand(product([ift.RGSpace(4, harmonic=True), ift.LMSpace(5)], [
        ['get_k_length_array', ift.Field],
        ['get_fft_smoothing_kernel_function', 2.0, LambdaType],
        ]))
    def test_method_ret_type(self, space, method_expected_type):
        assert_(type(getattr(space, method_expected_type[0])(
                          *method_expected_type[1:-1])) is
                method_expected_type[-1])
