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

import nifty5 as ift
import numpy as np
from numpy.testing import assert_allclose


class LaplaceOperatorTests(unittest.TestCase):
    @expand(product([None, False, True], [False, True], [10, 100, 1000]))
    def test_Laplace(self, log1, log2, sz):
        s = ift.RGSpace(sz, harmonic=True)
        bb = ift.PowerSpace.useful_binbounds(s, logarithmic=log1)
        p = ift.PowerSpace(s, binbounds=bb)
        L = ift.LaplaceOperator(p, logarithmic=log2)
        fp = ift.Field.from_random("normal", domain=p, dtype=np.float64)
        assert_allclose(L(fp).vdot(L(fp)), L.adjoint_times(L(fp)).vdot(fp))

    @expand(product([10, 100, 1000]))
    def test_Laplace2(self, sz):
        s = ift.RGSpace(sz, harmonic=True, distances=0.764)
        bb = ift.PowerSpace.useful_binbounds(s, logarithmic=False)
        p = ift.PowerSpace(s, binbounds=bb)

        foo = ift.PS_field(p, lambda k: 2*k**2)
        L = ift.LaplaceOperator(p, logarithmic=False)

        result = np.full(p.shape, 2*2.)
        result[0] = result[1] = result[-1] = 0.

        assert_allclose(L(foo).to_global_data(), result)
