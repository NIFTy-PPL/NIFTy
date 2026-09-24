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
# Copyright(C) 2026 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty.cl as ift
import numpy as np
import pytest
from numpy.testing import assert_allclose

from .common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize

device_id = list2fixture([-1, 0] if ift.device_available() else [-1])


def _lin2grad(lin):
    return lin.jac(ift.full(lin.domain, 1.)).asnumpy()


def jt(lin, check):
    assert_allclose(_lin2grad(lin), check)


def test_special_gradients():
    dom = ift.UnstructuredDomain((1,))
    f = ift.full(dom, 2.4)
    var = ift.Linearization.make_var(f)
    s = f.asnumpy()

    jt(var.clip(0, 10), np.ones_like(s))
    jt(var.clip(-1, 0), np.zeros_like(s))

    assert_allclose(
        _lin2grad(ift.Linearization.make_var(0*f).ptw("sinc")), np.zeros(s.shape))
    ift.myassert(np.isnan(_lin2grad(ift.Linearization.make_var(0*f).ptw("abs"))))
    assert_allclose(
        _lin2grad(ift.Linearization.make_var(0*f + 10).ptw("abs")),
        np.ones(s.shape))
    assert_allclose(
        _lin2grad(ift.Linearization.make_var(0*f - 10).ptw("abs")),
        -np.ones(s.shape))


@pmp('f', [
    'log', 'exp', 'sqrt', 'sin', 'cos', 'tan', 'sinc', 'sinh', 'cosh', 'tanh',
    'absolute', 'reciprocal', 'sigmoid', 'log10', 'log1p', 'expm1', 'softplus',
    ('power', 2.), ('exponentiate', 1.1)
])
@pmp('cplxpos', [True, False])
@pmp('cplxdir', [True, False])
@pmp('holomorphic', [True, False])
def test_actual_gradients(f, cplxpos, cplxdir, holomorphic, device_id):
    if (cplxpos or cplxdir) and f in ['absolute']:
        return
    if holomorphic and f in ['absolute']:
        # These function are not holomorphic
        return
    dom = ift.UnstructuredDomain((1,))
    fld = ift.full(dom, 2.4, device_id)
    if cplxpos:
        fld = fld + 0.21j
    eps = 1e-7
    if cplxdir:
        eps *= 1j
    if holomorphic:
        eps *= (1+0.78j)
    var0 = ift.Linearization.make_var(fld)
    var1 = ift.Linearization.make_var(fld + eps)
    if not isinstance(f, tuple):
        f = (f,)
    f0 = var0.ptw(*f).val.asnumpy()
    f1 = var1.ptw(*f).val.asnumpy()
    df1 = _lin2grad(var0.ptw(*f))
    df0 = (f1 - f0)/eps
    assert_allclose(df0, df1, rtol=100*np.abs(eps))


@pmp('f', [
    'log', 'exp', 'sqrt', 'sin', 'cos', 'tan', 'sinc', 'sinh', 'cosh', 'tanh',
    'absolute', 'reciprocal', 'sigmoid', 'log10', 'log1p', 'expm1', 'softplus', 'abs',
    ('power', 2.), ('exponentiate', 1.1)
])
@pmp('dtype', [float, complex])
def test_actual_gradients2(f, dtype):
    dom = ift.UnstructuredDomain((10,))
    fld = ift.from_random(dom, dtype=dtype)
    only_r_differentiable = False

    if f in ["sqrt", "log", "log10", "log1p"] and dtype is float:
        fld = fld.exp()
    if f in ["absolute", "abs"]:
        if dtype is complex:
            with pytest.raises(TypeError):
                ift.ScalingOperator(dom, 1.).ptw(f)(ift.Linearization.make_var(fld))
            return
        only_r_differentiable = True

    if not isinstance(f, tuple):
        f = (f,)
    ift.extra.check_operator(ift.ScalingOperator(dom, 1.).ptw(*f), fld, ntries=5,
                             only_r_differentiable=only_r_differentiable)


def test_outer_with_field():
    dom0, dom1 = ift.RGSpace((3, 2)), ift.UnstructuredDomain(5)
    a, b, da = (ift.from_random(dd) for dd in (dom0, dom1, dom0))

    lin = ift.Linearization.make_var(a).outer(b)

    assert_allclose(lin.val.asnumpy(), np.multiply.outer(a.asnumpy(), b.asnumpy()))
    assert_allclose(lin.jac(da).asnumpy(),
                    np.multiply.outer(da.asnumpy(), b.asnumpy()))


def test_outer_with_linearization():
    dom = ift.makeDomain({"a": ift.RGSpace((3, 2)), "b": ift.UnstructuredDomain(5)})
    pos, dpos = ift.from_random(dom), ift.from_random(dom)

    lin = ift.Linearization.make_var(pos)
    lin = lin["a"].outer(lin["b"])

    a, b = pos["a"].asnumpy(), pos["b"].asnumpy()
    da, db = dpos["a"].asnumpy(), dpos["b"].asnumpy()
    assert_allclose(lin.val.asnumpy(), np.multiply.outer(a, b))
    assert_allclose(lin.jac(dpos).asnumpy(),
                    np.multiply.outer(da, b) + np.multiply.outer(a, db))


class _OuterProductModel0(ift.Operator):
    def __init__(self, domain, second):
        self._domain = ift.makeDomain(domain)
        self._second = second
        self._target = ift.makeDomain(
            tuple(self._domain["a"]) + tuple(self._second.domain)
        )

    def _device_preparation(self, x):
        self._second = self._second.at(x["a"].device_id)

    def apply(self, x):
        self._check_input(x)
        self._device_preparation(x)
        return x["a"].ptw("exp").outer(self._second)


class _OuterProductModel1(ift.Operator):
    def __init__(self, domain, second):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(
            tuple(self._domain["a"]) + tuple(self._domain["b"])
        )

    def apply(self, x):
        self._check_input(x)
        return x["a"].ptw("exp").outer(x["b"].ptw("sin"))


_dom_a = ift.RGSpace((3, 2))
_dom_b = ift.makeDomain((ift.UnstructuredDomain(5), ift.RGSpace(4)))


@pmp('dtype', [np.float64, np.complex128])
@pmp('const_dtype', [np.float64, np.complex128])
def test_outer_jacobian_with_field(dtype, const_dtype):
    op = _OuterProductModel0({"a": _dom_a},
                             ift.from_random(_dom_b, dtype=const_dtype))
    pos = ift.from_random(op.domain, dtype=dtype)
    ift.extra.check_operator(op, pos, ntries=5, only_r_differentiable=False)


@pmp('dtype', [np.float64, np.complex128, {"a": np.float64, "b": np.complex128}])
def test_outer_jacobian_with_linearization(dtype):
    op = _OuterProductModel1({"a": _dom_a, "b": _dom_b}, "b")
    pos = ift.from_random(op.domain, dtype=dtype)
    ift.extra.check_operator(op, pos, ntries=5, only_r_differentiable=False)
