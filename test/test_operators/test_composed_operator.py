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

space1 = list2fixture([
    ift.RGSpace(4),
    ift.PowerSpace(ift.RGSpace((4, 4), harmonic=True)),
    ift.LMSpace(5),
    ift.HPSpace(4),
    ift.GLSpace(4)
])
space2 = space1


def test_times_adjoint_times(space1, space2):
    cspace = (space1, space2)
    diag1 = ift.Field.from_random(domain=space1, random_type='normal')
    diag2 = ift.Field.from_random(domain=space2, random_type='normal')
    op1 = ift.DiagonalOperator(diag1, cspace, spaces=(0,))
    op2 = ift.DiagonalOperator(diag2, cspace, spaces=(1,))

    op = op2(op1)

    rand1 = ift.Field.from_random(domain=(space1, space2), random_type='normal')
    rand2 = ift.Field.from_random(domain=(space1, space2), random_type='normal')

    tt1 = rand2.s_vdot(op.times(rand1))
    tt2 = rand1.s_vdot(op.adjoint_times(rand2))
    assert_allclose(tt1, tt2)


def test_times_inverse_times(space1, space2):
    cspace = (space1, space2)
    diag1 = ift.Field.from_random(domain=space1, random_type='normal')
    diag2 = ift.Field.from_random(domain=space2, random_type='normal')
    op1 = ift.DiagonalOperator(diag1, cspace, spaces=(0,))
    op2 = ift.DiagonalOperator(diag2, cspace, spaces=(1,))

    op = op2(op1)

    rand1 = ift.Field.from_random(domain=(space1, space2), random_type='normal')
    tt1 = op.inverse_times(op.times(rand1))

    assert_allclose(tt1.val, rand1.val)


def test_sum(space1):
    op1 = ift.makeOp(ift.Field.full(space1, 2.))
    op2 = ift.ScalingOperator(space1, 3.)
    full_op = op1 + op2 - (op2 - op1) + op1 + op1 + op2
    x = ift.Field.full(space1, 1.)
    res = full_op(x)
    assert_equal(isinstance(full_op, ift.DiagonalOperator), True)
    assert_allclose(res.val, 11.)


def test_chain(space1):
    op1 = ift.makeOp(ift.Field.full(space1, 2.))
    op2 = ift.ScalingOperator(space1, 3.,)
    full_op = op1(op2)(op2)(op1)(op1)(op1)(op2)
    x = ift.Field.full(space1, 1.)
    res = full_op(x)
    assert_equal(isinstance(full_op, ift.DiagonalOperator), True)
    assert_allclose(res.val, 432.)


def test_mix(space1):
    op1 = ift.makeOp(ift.Field.full(space1, 2.))
    op2 = ift.ScalingOperator(space1, 3.)
    full_op = op1(op2 + op2)(op1)(op1) - op1(op2)
    x = ift.Field.full(space1, 1.)
    res = full_op(x)
    assert_equal(isinstance(full_op, ift.DiagonalOperator), True)
    assert_allclose(res.val, 42.)
