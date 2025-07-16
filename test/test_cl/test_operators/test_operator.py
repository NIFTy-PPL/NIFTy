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
# Copyright(C) 2025 LambdaFields GmbH
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty.cl as ift
import numpy as np
import pytest
from numpy.testing import assert_allclose

from ..common import setup_function, teardown_function, list2fixture

pmp = pytest.mark.parametrize


def test_ducktape():
    dom = ift.RGSpace(10)
    op = ift.FFTOperator(dom)
    fld = ift.full(dom, 2.)
    lin = ift.Linearization.make_var(fld)
    a = "foo"
    op1 = op.ducktape(a)
    mdom = ift.MultiDomain.make({a: dom})
    assert op1.domain == mdom
    assert op1.target == op.target
    op1 = op.ducktape_left(a)
    assert op1.target == ift.MultiDomain.make({a: op.target})
    assert op1.domain == op.domain
    with pytest.raises(RuntimeError):
        fld.ducktape(a)
    with pytest.raises(RuntimeError):
        lin.ducktape(a)
    fld0 = fld.ducktape_left(a)
    assert fld0.domain == mdom
    lin0 = lin.ducktape_left(a)
    assert lin0.domain == lin.domain
    assert lin0.target == mdom


def test_broadcast():
    dom = ift.RGSpace((12,2), (0.2,11.))

    op = ift.ScalingOperator(dom, 1.).ducktape("inp").exp()

    # Operator
    op1 = op.broadcast(0, ift.UnstructuredDomain(3))
    loc = ift.from_random(op1.domain)
    ift.extra.check_operator(op1, loc, ntries=3)
    res = op1(loc).asnumpy()
    ref = np.broadcast_to(op(loc).asnumpy()[None], op1.target.shape)
    assert_allclose(res, ref)

    # Linearization
    lin = op(ift.Linearization.make_var(loc))
    lin1 = lin.broadcast(0, ift.UnstructuredDomain(3))
    assert_allclose(lin1.val.asnumpy(), ref)

    # Field
    fld1 = loc["inp"].exp().broadcast(0, ift.UnstructuredDomain(3))
    assert_allclose(fld1.asnumpy(), ref)


func = list2fixture(
    [
        lambda x, y: x + y,
        lambda x, y: y + x,
        lambda x, y: x / y,
        lambda x, y: y / x,
        lambda x, y: x * y,
        lambda x, y: y * x,
        lambda x, y: x - y,
        lambda x, y: y - x,
        lambda x, y: x**y,
        lambda x, y: y**x,
    ]
)

@pmp("cst_ind", [0, 1])
def test_binary_ops(func, cst_ind):
    dom1 = ift.RGSpace(10)

    X = ift.Variable(dom1, "xx")
    Y = ift.Variable(dom1, "yy")

    cst = [ift.from_random(dom1).exp(), 3.2][cst_ind]

    # (Operator, Operator)
    op0 = func(X, Y)
    fld0 = ift.from_random(op0.domain)
    res0 = op0(fld0)
    np.testing.assert_allclose(
        res0.asnumpy(),
        func(X.force(fld0).asnumpy(), Y.force(fld0).asnumpy())
    )
    ift.extra.check_operator(op0, ift.from_random(op0.domain).exp(), ntries=3)

    # (Operator, Linearization) -> should raise
    lin1 = X(ift.Linearization.make_var(fld0.extract(X.domain)))
    with pytest.raises(TypeError, match="unsupported operand type"):
        func(op0, lin1)

    # (Operator, Field/number)
    op1 = func(X, cst)
    ift.extra.check_operator(op1, ift.from_random(op1.domain).exp(), ntries=3)

    fld = ift.from_random(op1.domain)
    res1 = op1(fld)
    ref = cst.asnumpy() if isinstance(cst, ift.Field) else cst
    np.testing.assert_allclose(
        res1.asnumpy(),
        func(X(fld).asnumpy(), ref)
    )

    # (Linearization, Linearization)
    lin1 = X(ift.Linearization.make_var(fld0.extract(X.domain)))
    lin2 = Y(ift.Linearization.make_var(fld0.extract(Y.domain)))
    ift.extra.assert_allclose(res0, func(lin1, lin2).val)
    # TODO: Test Jacobian

    # (Linearization, Field/number)
    lin = X(ift.Linearization.make_var(fld))
    res5 = func(lin, cst)
    ift.extra.assert_allclose(res1, res5.val)
    # TODO: Test Jacobian

    # (Field, Field/number)
    res3 = func(X(fld), cst)
    ref = cst.asnumpy() if isinstance(cst, ift.Field) else cst
    np.testing.assert_allclose(
        res3.asnumpy(),
        func(X(fld).asnumpy(), ref)
    )


def test_unary_ops():
    dom1 = ift.RGSpace(10)
    X = ift.Variable(dom1, "xx")
    fld = ift.from_random(X.domain)

    res1 = (-X)(fld)
    res2 = -(X(fld))
    res3 = X.scale(-1)(fld)
    ift.extra.assert_allclose(res1, res2)
    ift.extra.assert_allclose(res1, res3)

    lin = ift.Linearization.make_var(fld)
    res4 = (-X)(lin)
    res5 = -(X(lin))
    ift.extra.assert_allclose(res4.val, res5.val)
    ift.extra.assert_allclose(res4.val, res1)
    ift.extra.check_operator(-X, ift.from_random(X.domain), ntries=3)


class NoOp:
    pass

@pmp("obj", [
    None,
    "foo", b"bar",
    [1, 2], (3, 4), {5, 6}, {"a": 7},
    # np.array([1, 2, 3]),  it is unclear how to catch this
    lambda x: x, object(), range(5), iter([1]),
    NoOp()
])
def test_interface_checks(func, obj):
    dom = ift.RGSpace(2)
    op = ift.Variable(dom, "a").exp()
    with pytest.raises(TypeError, match=r"(unsupported operand type|can only concatenate|can't concat|can't multiply)"):
        func(obj, op)

@pmp("obj", [True, np.int64(42), np.float32(3.14), 12.1, -2])
def test_interface_checks2(func, obj):
    dom = ift.RGSpace(2)
    op = ift.Variable(dom, "a").exp()
    func(obj, op)
