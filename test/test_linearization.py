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
import numpy as np
import pytest
from numpy.testing import assert_allclose

from .common import setup_function, teardown_function

pmp = pytest.mark.parametrize


def _lin2grad(lin):
    return lin.jac(ift.full(lin.domain, 1.)).val


def jt(lin, check):
    assert_allclose(_lin2grad(lin), check)


def test_special_gradients():
    dom = ift.UnstructuredDomain((1,))
    f = ift.full(dom, 2.4)
    var = ift.Linearization.make_var(f)
    s = f.val

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
def test_actual_gradients(f, cplxpos, cplxdir, holomorphic):
    if (cplxpos or cplxdir) and f in ['absolute']:
        return
    if holomorphic and f in ['absolute']:
        # These function are not holomorphic
        return
    dom = ift.UnstructuredDomain((1,))
    fld = ift.full(dom, 2.4)
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
    f0 = var0.ptw(*f).val.val
    f1 = var1.ptw(*f).val.val
    df1 = _lin2grad(var0.ptw(*f))
    df0 = (f1 - f0)/eps
    assert_allclose(df0, df1, rtol=100*np.abs(eps))
