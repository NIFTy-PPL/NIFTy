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
# Copyright(C) 2025 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest

from .common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize

device_id = list2fixture([-1, 0] if ift.device_available() else [-1])

@pmp("op, args", [
    (np.add, (np.array([1, 2, 3]), np.array([4, 5, 6]))),
    (np.subtract, (np.array([10, 20, 30]), np.array([1, 2, 3]))),
    (np.multiply, (np.array([2, 3, 4]), np.array([5, 6, 7]))),
    (np.maximum, (np.array([1, 5, 3]), np.array([4, 2, 6]))),
])
def test_numpy_operations_with_out(op, args, device_id):
    arr1, arr2 = map(ift.AnyArray, args)
    out = np.zeros_like(arr1)

    out2 = op(arr1, arr2, out=out)
    assert len(out2) == 1
    assert out2[0] is out
    assert out2[0].val is out.val

    expected = op(arr1.val, arr2.val)
    np.testing.assert_array_equal(out.val, expected)
