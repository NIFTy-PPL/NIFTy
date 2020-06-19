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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
import pytest

import nifty7 as ift

from .common import setup_function, teardown_function


def _flat_PS(k):
    return np.ones_like(k)


pmp = pytest.mark.parametrize
ntries = 10


@pmp('space', [ift.GLSpace(5),
               ift.RGSpace(5, distances=.789),
               ift.RGSpace([2, 2], distances=.789)])
@pmp('nonlinearity', ["tanh", "exp", ""])
@pmp('noise', [1, 1e-2, 1e2])
@pmp('seed', [4, 78])
def test_gaussian_energy(space, nonlinearity, noise, seed):
    with ift.random.Context(seed):
        dim = len(space.shape)
        hspace = space.get_default_codomain()
        ht = ift.HarmonicTransformOperator(hspace, target=space)
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=False)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        Dist = ift.PowerDistributor(target=hspace, power_space=pspace)
        xi0 = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k):
            return 1/(1 + k**2)**dim

        pspec = ift.PS_field(pspace, pspec)
        A = Dist(pspec.ptw("sqrt"))
        N = ift.ScalingOperator(space, noise)
        n = N.draw_sample_with_dtype(dtype=np.float64)
        R = ift.ScalingOperator(space, 10.)

        def d_model():
            if nonlinearity == "":
                return R @ ht @ ift.makeOp(A)
            else:
                tmp = ht @ ift.makeOp(A)
                nonlin = tmp.ptw(nonlinearity)
                return R @ nonlin

        d = d_model()(xi0) + n

        if noise == 1:
            N = None

        energy = ift.GaussianEnergy(d, N) @ d_model()
        ift.extra.check_operator(
            energy, xi0, ntries=ntries, tol=1e-6)


@pmp('cplx', [True, False])
def testgaussianenergy_compatibility(cplx):
    dt = np.complex128 if cplx else np.float64
    dom = ift.UnstructuredDomain(3)
    e = ift.VariableCovarianceGaussianEnergy(dom, 'resi', 'icov', dt)
    resi = ift.from_random(dom)
    if cplx:
        resi = resi + 1j*ift.from_random(dom)
    loc0 = ift.MultiField.from_dict({'resi': resi})
    loc1 = ift.MultiField.from_dict({'icov': ift.from_random(dom).exp()})
    loc = loc0.unite(loc1)
    val0 = e(loc).val

    _, e0 = e.simplify_for_constant_input(loc0)
    val1 = e0(loc).val
    val2 = e0(loc.unite(loc0)).val
    np.testing.assert_equal(val1, val2)
    np.testing.assert_equal(val0, val1)

    _, e1 = e.simplify_for_constant_input(loc1)
    val1 = e1(loc).val
    val2 = e1(loc.unite(loc1)).val
    np.testing.assert_equal(val0, val1)
    np.testing.assert_equal(val1, val2)

    ift.extra.check_operator(e, loc, ntries=ntries)
    ift.extra.check_operator(e0, loc, ntries=ntries, tol=1e-7)
    ift.extra.check_operator(e1, loc, ntries=ntries)

    # Test jacobian is zero
    lin = ift.Linearization.make_var(loc, want_metric=True)
    grad = e(lin).gradient.val
    grad0 = e0(lin).gradient.val
    grad1 = e1(lin).gradient.val
    samp = e(lin).metric.draw_sample().val
    samp0 = e0(lin).metric.draw_sample().val
    samp1 = e1(lin).metric.draw_sample().val
    np.testing.assert_equal(samp0['resi'], 0.)
    np.testing.assert_equal(samp1['icov'], 0.)
    np.testing.assert_equal(grad0['resi'], 0.)
    np.testing.assert_equal(grad1['icov'], 0.)
    np.testing.assert_(all(samp['resi'] != 0))
    np.testing.assert_(all(samp['icov'] != 0))
    np.testing.assert_(all(samp0['icov'] != 0))
    np.testing.assert_(all(samp1['resi'] != 0))
    np.testing.assert_(all(grad['resi'] != 0))
    np.testing.assert_(all(grad['icov'] != 0))
    np.testing.assert_(all(grad0['icov'] != 0))
    np.testing.assert_(all(grad1['resi'] != 0))
