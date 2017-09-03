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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import unittest
import numpy as np
import nifty2go as ift
from numpy.testing import assert_allclose
from itertools import product
from test.common import expand, marco_binbounds


class LaplaceOperatorTests(unittest.TestCase):
    @expand(product([None, False, True], [False, True], [10, 100, 1000]))
    def test_Laplace(self, log1, log2, sz):
        s = ift.RGSpace(sz, harmonic=True)
        p = ift.PowerSpace(s, binbounds=marco_binbounds(s, logarithmic=log1))
        L = ift.LaplaceOperator(p, logarithmic=log2)
        arr = np.random.random(p.shape[0])
        fp = ift.Field(p, val=arr)
        assert_allclose(L(fp).vdot(L(fp)), L.adjoint_times(L(fp)).vdot(fp))
