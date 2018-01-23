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
from numpy.testing import assert_allclose
import nifty4 as ift
from itertools import product
from test.common import expand


class ResponseOperator_Tests(unittest.TestCase):
    spaces = [ift.RGSpace(128), ift.GLSpace(nlat=37)]

    @expand(product(spaces, [0.,  5., 1.], [0., 1., .33]))
    def test_property(self, space, sigma, sensitivity):
        op = ift.ResponseOperator(space, sigma=[sigma],
                                  sensitivity=[sensitivity])
        if op.domain[0] != space:
            raise TypeError

    @expand(product(spaces, [0.,  5., 1.], [0., 1., .33]))
    def test_times_adjoint_times(self, space, sigma, sensitivity):
        if not isinstance(space, ift.RGSpace):  # no smoothing supported
            sigma = 0.
        op = ift.ResponseOperator(space, sigma=[sigma],
                                  sensitivity=[sensitivity])
        rand1 = ift.Field.from_random('normal', domain=space)
        rand2 = ift.Field.from_random('normal', domain=op.target[0])
        tt1 = rand2.vdot(op.times(rand1))
        tt2 = rand1.vdot(op.adjoint_times(rand2))
        assert_allclose(tt1, tt2)
