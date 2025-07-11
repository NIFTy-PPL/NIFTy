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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
from nifty8.operators.simplify_for_const import ConstantOperator
from numpy.testing import assert_allclose, assert_raises


def test_simplification():
    dom = {"a": ift.RGSpace(10)}
    f1 = ift.full(dom, 2.)
    op = ift.FFTOperator(f1.domain["a"]).ducktape("a")
    _, op2 = op.simplify_for_constant_input(f1)
    ift.myassert(isinstance(op2, ConstantOperator))
    assert_allclose(op(f1).val, op2.force(f1).val)

    dom = {"a": ift.RGSpace(10), "b": ift.RGSpace(5)}
    f1 = ift.full(dom, 2.)
    pdom = {"a": ift.RGSpace(10)}
    f2 = ift.full(pdom, 2.)
    o1 = ift.FFTOperator(f1.domain["a"])
    o2 = ift.FFTOperator(f1.domain["b"])
    op = (o1.ducktape("a").ducktape_left("a") +
          o2.ducktape("b").ducktape_left("b"))
    _, op2 = op.simplify_for_constant_input(f2)
    assert_allclose(op(f1)["a"].val, op2.force(f1)["a"].val)
    assert_allclose(op(f1)["b"].val, op2.force(f1)["b"].val)
    # FIXME Add test for ChainOperator._simplify_for_constant_input_nontrivial()
