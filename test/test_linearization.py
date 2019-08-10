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

import numpy as np
import pytest
from numpy.testing import assert_, assert_allclose

import nifty5 as ift

pmp = pytest.mark.parametrize


def _lin2grad(lin):
    return lin.jac(ift.full(lin.domain, 1.)).to_global_data()


def jt(lin, check):
    assert_allclose(_lin2grad(lin), check)


def test_special_gradients():
    dom = ift.UnstructuredDomain((1,))
    f = ift.full(dom, 2.4)
    var = ift.Linearization.make_var(f)
    s = f.to_global_data()

    jt(var.clip(0, 10), np.ones_like(s))
    jt(var.clip(-1, 0), np.zeros_like(s))

    assert_allclose(
        _lin2grad(ift.Linearization.make_var(0*f).sinc()), np.zeros(s.shape))
    assert_(np.isnan(_lin2grad(ift.Linearization.make_var(0*f).absolute())))
    assert_allclose(
        _lin2grad(ift.Linearization.make_var(0*f + 10).absolute()),
        np.ones(s.shape))
    assert_allclose(
        _lin2grad(ift.Linearization.make_var(0*f - 10).absolute()),
        -np.ones(s.shape))


@pmp('f', [
    'log', 'exp', 'sqrt', 'sin', 'cos', 'tan', 'sinc', 'sinh', 'cosh', 'tanh',
    'absolute', 'one_over', 'sigmoid'
])
def test_actual_gradients(f):
    dom = ift.UnstructuredDomain((1,))
    fld = ift.full(dom, 2.4)
    eps = 1e-8
    var0 = ift.Linearization.make_var(fld)
    var1 = ift.Linearization.make_var(fld + eps)
    f0 = getattr(var0, f)().val.to_global_data()
    f1 = getattr(var1, f)().val.to_global_data()
    df0 = (f1 - f0)/eps
    df1 = _lin2grad(getattr(var0, f)())
    assert_allclose(df0, df1, rtol=100*eps)
