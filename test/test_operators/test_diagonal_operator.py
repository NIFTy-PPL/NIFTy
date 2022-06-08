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

import nifty8 as ift
from numpy.testing import assert_allclose, assert_equal

from ..common import list2fixture, setup_function, teardown_function

space = list2fixture([
    ift.RGSpace(4),
    ift.PowerSpace(ift.RGSpace((4, 4), harmonic=True)),
    ift.LMSpace(5),
    ift.HPSpace(4),
    ift.GLSpace(4)
])


def test_property(space):
    diag = ift.Field.from_random(domain=space, random_type='normal')
    D = ift.DiagonalOperator(diag)
    if D.domain[0] != space:
        raise TypeError


def test_times_adjoint(space):
    rand1 = ift.Field.from_random(domain=space, random_type='normal')
    rand2 = ift.Field.from_random(domain=space, random_type='normal')
    diag = ift.Field.from_random(domain=space, random_type='normal')
    D = ift.DiagonalOperator(diag)
    tt1 = rand1.s_vdot(D.times(rand2))
    tt2 = rand2.s_vdot(D.times(rand1))
    assert_allclose(tt1, tt2)


def test_times_inverse(space):
    rand1 = ift.Field.from_random(domain=space, random_type='normal')
    diag = ift.Field.from_random(domain=space, random_type='normal')
    D = ift.DiagonalOperator(diag)
    tt1 = D.times(D.inverse_times(rand1))
    assert_allclose(rand1.val, tt1.val)


def test_times(space):
    rand1 = ift.Field.from_random(domain=space, random_type='normal')
    diag = ift.Field.from_random(domain=space, random_type='normal')
    D = ift.DiagonalOperator(diag)
    tt = D.times(rand1)
    assert_equal(tt.domain[0], space)


def test_adjoint_times(space):
    rand1 = ift.Field.from_random(domain=space, random_type='normal')
    diag = ift.Field.from_random(domain=space, random_type='normal')
    D = ift.DiagonalOperator(diag)
    tt = D.adjoint_times(rand1)
    assert_equal(tt.domain[0], space)


def test_inverse_times(space):
    rand1 = ift.Field.from_random(domain=space, random_type='normal')
    diag = ift.Field.from_random(domain=space, random_type='normal')
    D = ift.DiagonalOperator(diag)
    tt = D.inverse_times(rand1)
    assert_equal(tt.domain[0], space)


def test_adjoint_inverse_times(space):
    rand1 = ift.Field.from_random(domain=space, random_type='normal')
    diag = ift.Field.from_random(domain=space, random_type='normal')
    D = ift.DiagonalOperator(diag)
    tt = D.adjoint_inverse_times(rand1)
    assert_equal(tt.domain[0], space)


def test_diagonal(space):
    diag = ift.Field.from_random(domain=space, random_type='normal')
    D = ift.DiagonalOperator(diag)
    diag_op = D(ift.Field.full(space, 1.))
    assert_allclose(diag.val, diag_op.val)
