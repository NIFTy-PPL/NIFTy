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
import matplotlib.pyplot as plt
import pytest
try:
    import jax.numpy as jnp
    _skip = False
except ImportError:
    import numpy as np
    _skip = True

from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize


@pmp("dom", [ift.RGSpace((10, 8)), (ift.RGSpace(10), ift.RGSpace(8))])
@pmp("func", [lambda x: x, lambda x: x**2, lambda x: x*x, lambda x: x*x[0, 0],
              lambda x: jnp.sin(x), lambda x: x*x.sum()])
def test_jax(dom, func):
    if _skip:
        pytest.skip()
    loc = ift.from_random(dom)
    res0 = np.array(func(loc.val))
    op = ift.JaxOperator(dom, dom, func)
    np.testing.assert_allclose(res0, op(loc).val)
    ift.extra.check_operator(op, ift.from_random(op.domain))


def test_mf_jax():
    if _skip:
        pytest.skip()
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
    if _skip:
        pytest.skip()
    dom = ift.makeDomain(dom)
    e0 = ift.GaussianEnergy(domain=dom)
    def func(x):
        return 0.5*jnp.vdot(x, x)
    def funcmf(x):
        res = 0
        for kk, vv in x.items():
            res += jnp.vdot(vv, vv)
        return 0.5*res
    e = ift.JaxLikelihoodEnergyOperator(dom,
            funcmf if isinstance(dom, ift.MultiDomain) else func,
            transformation=ift.ScalingOperator(dom, 1.))
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
