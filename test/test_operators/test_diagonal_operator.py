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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import division

import unittest
from itertools import product
from test.common import expand

import nifty5 as ift
from numpy.testing import assert_allclose, assert_equal


class DiagonalOperator_Tests(unittest.TestCase):
    spaces = [ift.RGSpace(4),
              ift.PowerSpace(ift.RGSpace((4, 4), harmonic=True)),
              ift.LMSpace(5), ift.HPSpace(4), ift.GLSpace(4)]

    @expand(product(spaces))
    def test_property(self, space):
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        if D.domain[0] != space:
            raise TypeError

    @expand(product(spaces))
    def test_times_adjoint(self, space):
        rand1 = ift.Field.from_random('normal', domain=space)
        rand2 = ift.Field.from_random('normal', domain=space)
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        tt1 = rand1.vdot(D.times(rand2))
        tt2 = rand2.vdot(D.times(rand1))
        assert_allclose(tt1, tt2)

    @expand(product(spaces))
    def test_times_inverse(self, space):
        rand1 = ift.Field.from_random('normal', domain=space)
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        tt1 = D.times(D.inverse_times(rand1))
        assert_allclose(rand1.local_data, tt1.local_data)

    @expand(product(spaces))
    def test_times(self, space):
        rand1 = ift.Field.from_random('normal', domain=space)
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        tt = D.times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces))
    def test_adjoint_times(self, space):
        rand1 = ift.Field.from_random('normal', domain=space)
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        tt = D.adjoint_times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces))
    def test_inverse_times(self, space):
        rand1 = ift.Field.from_random('normal', domain=space)
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        tt = D.inverse_times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces))
    def test_adjoint_inverse_times(self, space):
        rand1 = ift.Field.from_random('normal', domain=space)
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        tt = D.adjoint_inverse_times(rand1)
        assert_equal(tt.domain[0], space)

    @expand(product(spaces))
    def test_diagonal(self, space):
        diag = ift.Field.from_random('normal', domain=space)
        D = ift.DiagonalOperator(diag)
        diag_op = D(ift.Field.full(space, 1.))
        assert_allclose(diag.local_data, diag_op.local_data)
