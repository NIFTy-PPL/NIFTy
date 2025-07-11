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

from types import LambdaType

import nifty8 as ift
import pytest

from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize


@pmp('attr_expected_type',
     [['harmonic', bool], ['shape', tuple], ['size', int]])
@pmp('space', [
    ift.RGSpace(4),
    ift.PowerSpace(ift.RGSpace((4, 4), harmonic=True)),
    ift.LMSpace(5),
    ift.HPSpace(4),
    ift.GLSpace(4)
])
def test_property_ret_type(space, attr_expected_type):
    ift.myassert(
        isinstance(
            getattr(space, attr_expected_type[0]), attr_expected_type[1]))


@pmp('method_expected_type',
     [['get_k_length_array', ift.Field],
      ['get_fft_smoothing_kernel_function', 2.0, LambdaType]])
@pmp('space', [ift.RGSpace(4, harmonic=True), ift.LMSpace(5)])
def test_method_ret_type(space, method_expected_type):
    ift.myassert(
        type(
            getattr(space, method_expected_type[0])
            (*method_expected_type[1:-1])) is method_expected_type[-1])
