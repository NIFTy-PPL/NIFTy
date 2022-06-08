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

import nifty8 as ift
import numpy as np
import pytest

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
        N = ift.ScalingOperator(space, noise, float)
        n = N.draw_sample()
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

        energy = ift.GaussianEnergy(d, N, sampling_dtype=float) @ d_model()
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
    ift.extra.check_operator(e, loc, ntries=20)
