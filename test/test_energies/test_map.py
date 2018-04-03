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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import unittest
import nifty4 as ift
import numpy as np
from itertools import product
from test.common import expand
from numpy.testing import assert_allclose


def _flat_PS(k):
    return np.ones_like(k)


class Energy_Tests(unittest.TestCase):
    @expand(product([ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    [4, 78, 23]))
    def testLinearMap(self, space, seed):
        np.random.seed(seed)
        dim = len(space.shape)
        hspace = space.get_default_codomain()
        ht = ift.HarmonicTransformOperator(hspace, target=space)
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=False)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        Dist = ift.PowerDistributor(target=hspace, power_space=pspace)
        xi0 = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 1 / (1 + k**2)**dim
        pspec = ift.PS_field(pspace, pspec)
        A = Dist(ift.sqrt(pspec))
        n = ift.Field.from_random(domain=space, random_type='normal')
        s0 = xi0 * A
        Instrument = ift.ScalingOperator(10., space)
        R = Instrument * ht
        N = ift.ScalingOperator(1., space)
        d = R(s0) + n

        IC = ift.GradientNormController(
            iteration_limit=100,
            tol_abs_gradnorm=1e-5)
        inverter = ift.ConjugateGradient(IC)

        S = ift.create_power_operator(hspace, power_spectrum=_flat_PS)
        energy = ift.library.WienerFilterEnergy(
            position=s0, d=d, R=R, N=N, S=S, inverter=inverter)
        ift.extra.check_value_gradient_curvature_consistency(
            energy, tol=1e-4, ntries=10)

    @expand(product([ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    [ift.library.Tanh, ift.library.Exponential,
                     ift.library.Linear],
                    [4, 78, 23]))
    def testNonlinearMap(self, space, nonlinearity, seed):
        np.random.seed(seed)
        f = nonlinearity()
        dim = len(space.shape)
        hspace = space.get_default_codomain()
        ht = ift.HarmonicTransformOperator(hspace, target=space)
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=False)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        Dist = ift.PowerDistributor(target=hspace, power_space=pspace)
        xi0 = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 1 / (1 + k**2)**dim
        pspec = ift.PS_field(pspace, pspec)
        A = Dist(ift.sqrt(pspec))
        n = ift.Field.from_random(domain=space, random_type='normal')
        s = ht(xi0 * A)
        R = ift.ScalingOperator(10., space)
        N = ift.ScalingOperator(1., space)
        d = R(f(s)) + n

        S = ift.create_power_operator(hspace, power_spectrum=_flat_PS)
        energy = ift.library.NonlinearWienerFilterEnergy(
            position=xi0, d=d, Instrument=R, nonlinearity=f, ht=ht, power=A,
            N=N, S=S)
        if isinstance(nonlinearity, ift.library.Linear):
            ift.extra.check_value_gradient_curvature_consistency(
                energy, tol=1e-4, ntries=10)
        else:
            ift.extra.check_value_gradient_consistency(
                energy, tol=1e-4, ntries=10)
