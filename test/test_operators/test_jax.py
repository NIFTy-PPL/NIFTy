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
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift
import numpy as np
import pytest

try:
    import jax.numpy as jnp
except ImportError:
    pass

from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize


@pmp("dom", [ift.RGSpace((10, 8)), (ift.RGSpace(10), ift.RGSpace(8))])
@pmp("func", [(lambda x: x, True), (lambda x: x**2, False), (lambda x: x*x, False),
              (lambda x: x*x[0, 0], False), (lambda x: x+x[0, 0], True),
              (lambda x: jnp.sin(x), False), (lambda x: x*x.sum(), False),
              (lambda x: x+x.sum(), True)])
def test_jax(dom, func):
    pytest.importorskip("jax")
    loc = ift.from_random(dom)
    f, linear = func
    res0 = np.array(f(loc.val))
    op = ift.JaxOperator(dom, dom, f)
    np.testing.assert_allclose(res0, op(loc).val)
    ift.extra.check_operator(op, ift.from_random(op.domain))

    op = ift.JaxLinearOperator(dom, dom, f, np.float64)
    if linear:
        ift.extra.check_linear_operator(op)
    else:
        with pytest.raises(Exception):
            ift.extra.check_linear_operator(op)


def test_mf_jax():
    pytest.importorskip("jax")
    dom = ift.makeDomain({"a": ift.RGSpace(10), "b": ift.UnstructuredDomain(2)})

    func = lambda x: x["a"]*x["b"][0]
    op = ift.JaxOperator(dom, dom["a"], func)
    loc = ift.from_random(op.domain)
    np.testing.assert_allclose(np.array(func(loc.val)), op(loc).val)
    ift.extra.check_operator(op, loc)

    func = lambda x: {"a": jnp.full(dom["a"].shape, 2.)*x[0]*x[1], "b": jnp.full(dom["b"].shape, 1.)*jnp.exp(x[0])}
    op = ift.JaxOperator(dom["b"], dom, func)
    loc = ift.from_random(op.domain)
    for kk in dom.keys():
        np.testing.assert_allclose(np.array(func(loc.val)[kk]), op(loc)[kk].val)
    ift.extra.check_operator(op, loc)


@pmp("dom", [ift.RGSpace((10, 8)),
    {"a": ift.RGSpace(10), "b": ift.UnstructuredDomain(2)}])
def test_jax_energy(dom):
    pytest.importorskip("jax")
    dom = ift.makeDomain(dom)
    e0 = ift.GaussianEnergy(domain=dom, sampling_dtype=np.float64)
    def func(x):
        return 0.5*jnp.vdot(x, x)
    def funcmf(x):
        res = 0
        for kk, vv in x.items():
            res += jnp.vdot(vv, vv)
        return 0.5*res
    e = ift.JaxLikelihoodEnergyOperator(dom,
            funcmf if isinstance(dom, ift.MultiDomain) else func,
            transformation=ift.ScalingOperator(dom, 1.),
            sampling_dtype=np.float64)
    ift.extra.check_operator(e, ift.from_random(e.domain))
    for wm in [False, True]:
        pos = ift.from_random(e.domain)
        lin = ift.Linearization.make_var(pos, wm)
        ift.extra.assert_allclose(e0(pos), e(pos))
        ift.extra.assert_allclose(e0(lin).val, e(lin).val)
        ift.extra.assert_allclose(e0(lin).gradient, e(lin).gradient)
        if not wm:
            continue
        pos1 = ift.from_random(e.domain)
        ift.extra.assert_allclose(e0(lin).metric(pos1), e(lin).metric(pos1))


def test_jax_errors():
    pytest.importorskip("jax")
    dom = ift.UnstructuredDomain(2)
    mdom = {"a": dom}
    op = ift.JaxOperator(dom, dom, lambda x: {"a": x})
    fld = ift.full(dom, 0.)
    with pytest.raises(TypeError):
        op(fld)
    op = ift.JaxOperator(dom, mdom, lambda x: x)
    with pytest.raises(TypeError):
        op(fld)
    op = ift.JaxOperator(dom, dom, lambda x: x[0])
    with pytest.raises(ValueError):
        op(fld)
    op = ift.JaxOperator(dom, mdom, lambda x: {"a": x[0]})
    with pytest.raises(ValueError):
        op(fld)


def test_jax_complex():
    pytest.importorskip("jax")
    dom = ift.UnstructuredDomain(1)
    a = ift.ducktape(dom, None, "a")
    b = ift.ducktape(dom, None, "b")
    op = a.real+1j*b.real
    op1 = ift.JaxOperator(op.domain, op.target, lambda x: x["a"] + 1j*x["b"])
    _op_equal(op, op1, ift.from_random(op.domain))
    ift.extra.check_operator(op, ift.from_random(op.domain), ntries=10)
    ift.extra.check_operator(op1, ift.from_random(op.domain), ntries=10)

    op = op.imag
    op1 = op1.imag
    _op_equal(op, op1, ift.from_random(op.domain))
    ift.extra.check_operator(op, ift.from_random(op.domain), ntries=10)
    ift.extra.check_operator(op1, ift.from_random(op.domain), ntries=10)

    lin = ift.Linearization.make_var(ift.from_random(op.domain))
    test_vec = ift.full(op.target, 1.)
    grad = op(lin).jac.adjoint(test_vec)
    grad1 = op1(lin).jac.adjoint(test_vec)
    ift.extra.assert_equal(grad, grad1)
    ift.extra.assert_equal(grad, ift.makeField(grad.domain, {"a": 0., "b": 1.}))


def _op_equal(op0, op1, loc):
    assert op0.domain is op1.domain
    assert op0.target is op1.target
    ift.extra.assert_allclose(op0(loc), op1(loc))
    lin = ift.Linearization.make_var(loc)
    res = op0(lin)
    res1 = op1(lin)
    ift.extra.assert_allclose(res.val, res1.val)
    fld = ift.from_random(op0.domain, dtype=loc.dtype)
    ift.extra.assert_allclose(res.jac(fld), res1.jac(fld))
    fld = ift.from_random(op0.target, dtype=res.val.dtype)
    ift.extra.assert_allclose(res.jac.adjoint(fld), res1.jac.adjoint(fld))
