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
# Copyright(C) 2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal, assert_raises

import nifty6 as ift
from .common import setup_function, teardown_function


def test_simple():
    a = ift.Field.full(ift.RGSpace(10),3.)
    b = ift.Field.full(ift.RGSpace(11),3.)
    t = ift.DiffTensor.makeDiagonal(a, 3)
    t = t.contract((a,))
    t = t + t
    print(t)
    print(t.linop(a).val)
    t = t.contract((a,))
    print(t._arg, t.rank)
    print(t.vec)
