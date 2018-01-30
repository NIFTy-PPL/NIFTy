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
# Copyright(C) 2013-2017 Max-Planck-Society
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
                    [132, 42, 3]))
    def testLinearPower(self, space, seed):
        np.random.seed(seed)
        dim = len(space.shape)
        hspace = space.get_default_codomain()
        ht = ift.HarmonicTransformOperator(hspace, space)
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=True)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        P = ift.PowerProjectionOperator(domain=hspace, power_space=pspace)
        xi = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 64 / (1 + k**2)**dim
        pspec = ift.PS_field(pspace, pspec)
        tau0 = ift.log(pspec)
        A = P.adjoint_times(ift.sqrt(pspec))
        n = ift.Field.from_random(domain=space, random_type='normal', std=.01)
        N = ift.DiagonalOperator(n**2)
        s = xi * A
        Instrument = ift.ScalingOperator(1., space)
        R = Instrument * ht
        d = R(s) + n

        direction = ift.Field.from_random('normal', pspace)
        direction /= np.sqrt(direction.var())
        eps = 1e-7
        tau1 = tau0 + eps * direction

        IC = ift.GradientNormController(
            iteration_limit=100,
            tol_abs_gradnorm=1e-5)
        inverter = ift.ConjugateGradient(IC)

        S = ift.create_power_operator(
            hspace, power_spectrum=lambda k: 1. / (1 + k**2))

        D = ift.library.WienerFilterEnergy(position=s, d=d, R=R, N=N, S=S,
                                           inverter=inverter).curvature

        energy0 = ift.library.CriticalPowerEnergy(
            position=tau0, m=s, inverter=inverter, D=D, samples=10,
            smoothness_prior=1.)
        energy1 = energy0.at(tau1)

        a = (energy1.value - energy0.value) / eps
        b = energy0.gradient.vdot(direction)
        tol = 1e-4
        assert_allclose(a, b, rtol=tol, atol=tol)

    @expand(product([ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    [ift.library.Exponential, ift.library.Linear],
                    [132, 42, 3]))
    def testNonlinearPower(self, space, nonlinearity, seed):
        np.random.seed(seed)
        f = nonlinearity()
        dim = len(space.shape)
        hspace = space.get_default_codomain()
        ht = ift.HarmonicTransformOperator(hspace, space)
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=True)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        P = ift.PowerProjectionOperator(domain=hspace, power_space=pspace)
        xi = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 1 / (1 + k**2)**dim
        tau0 = ift.PS_field(pspace, pspec)
        A = P.adjoint_times(ift.sqrt(tau0))
        n = ift.Field.from_random(domain=space, random_type='normal')
        s = ht(xi * A)
        R = ift.ScalingOperator(10., space)
        diag = ift.Field.ones(space)
        N = ift.DiagonalOperator(diag)
        d = R(f(s)) + n

        direction = ift.Field.from_random('normal', pspace)
        direction /= np.sqrt(direction.var())
        eps = 1e-7
        tau1 = tau0 + eps * direction

        IC = ift.GradientNormController(
            iteration_limit=100,
            tol_abs_gradnorm=1e-5)
        inverter = ift.ConjugateGradient(IC)

        S = ift.create_power_operator(hspace, power_spectrum=_flat_PS)
        D = ift.library.NonlinearWienerFilterEnergy(
            position=xi,
            d=d,
            Instrument=R,
            nonlinearity=f,
            power=A,
            N=N,
            S=S,
            ht=ht,
            inverter=inverter).curvature

        energy0 = ift.library.NonlinearPowerEnergy(
            position=tau0,
            d=d,
            xi=xi,
            D=D,
            Instrument=R,
            Projection=P,
            nonlinearity=f,
            ht=ht,
            N=N,
            samples=10)
        energy1 = energy0.at(tau1)

        a = (energy1.value - energy0.value) / eps
        b = energy0.gradient.vdot(direction)
        tol = 1e-4
        assert_allclose(a, b, rtol=tol, atol=tol)


class Curvature_Tests(unittest.TestCase):
    # Note: It is only possible to test the linear power curvatures since the
    # non-linear curvatures are not the exact second derivative but only a part
    # of it. One term is neglected which would render the second derivative
    # non-positive definite.
    @expand(product([ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    [132, 42, 3]))
    def testLinearPowerCurvature(self, space, seed):
        np.random.seed(seed)
        dim = len(space.shape)
        hspace = space.get_default_codomain()
        ht = ift.HarmonicTransformOperator(hspace, space)
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=True)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        P = ift.PowerProjectionOperator(domain=hspace, power_space=pspace)
        xi = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 64 / (1 + k**2)**dim
        pspec = ift.PS_field(pspace, pspec)
        tau0 = ift.log(pspec)
        A = P.adjoint_times(ift.sqrt(pspec))
        n = ift.Field.from_random(domain=space, random_type='normal', std=.01)
        N = ift.DiagonalOperator(n**2)
        s = xi * A
        diag = ift.Field.ones(space)
        Instrument = ift.DiagonalOperator(diag)
        R = Instrument * ht
        d = R(s) + n

        direction = ift.Field.from_random('normal', pspace)
        direction /= np.sqrt(direction.var())
        eps = 1e-7
        tau1 = tau0 + eps * direction

        IC = ift.GradientNormController(
            iteration_limit=100,
            tol_abs_gradnorm=1e-5)
        inverter = ift.ConjugateGradient(IC)

        S = ift.create_power_operator(hspace, power_spectrum=_flat_PS)

        D = ift.library.WienerFilterEnergy(position=s, d=d, R=R, N=N, S=S,
                                           inverter=inverter).curvature

        energy0 = ift.library.CriticalPowerEnergy(
            position=tau0, m=s, inverter=inverter, samples=10, D=D, alpha=2.)

        gradient0 = energy0.gradient
        gradient1 = energy0.at(tau1).gradient

        a = (gradient1 - gradient0) / eps
        b = energy0.curvature(direction)
        tol = 1e-5
        assert_allclose(ift.dobj.to_global_data(a.val),
                        ift.dobj.to_global_data(b.val), rtol=tol, atol=tol)
