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

import pytest
from numpy.testing import assert_allclose, assert_equal

import nifty5 as ift


def test_simplification():
    from nifty5.operators.operator import _ConstantOperator
    f1 = ift.Field.full(ift.RGSpace(10),2.)
    op = ift.FFTOperator(f1.domain)
    _, op2 = op.simplify_for_constant_input(f1)
    assert_equal(isinstance(op2, _ConstantOperator), True)
    assert_allclose(op(f1).local_data, op2(f1).local_data)

    dom = {"a": ift.RGSpace(10)}
    f1 = ift.full(dom,2.)
    op = ift.FFTOperator(f1.domain["a"]).ducktape("a")
    _, op2 = op.simplify_for_constant_input(f1)
    assert_equal(isinstance(op2, _ConstantOperator), True)
    assert_allclose(op(f1).local_data, op2(f1).local_data)

    dom = {"a": ift.RGSpace(10), "b": ift.RGSpace(5)}
    f1 = ift.full(dom,2.)
    pdom = {"a": ift.RGSpace(10)}
    f2 = ift.full(pdom,2.)
    o1 = ift.FFTOperator(f1.domain["a"])
    o2 = ift.FFTOperator(f1.domain["b"])
    op = (o1.ducktape("a").ducktape_left("a") +
          o2.ducktape("b").ducktape_left("b"))
    _, op2 = op.simplify_for_constant_input(f2)
    assert_equal(isinstance(op2._op1, _ConstantOperator), True)
    assert_allclose(op(f1)["a"].local_data, op2(f1)["a"].local_data)
    assert_allclose(op(f1)["b"].local_data, op2(f1)["b"].local_data)
