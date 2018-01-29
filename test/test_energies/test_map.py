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


# TODO Add also other space types
# TODO Set tolerances and eps to reasonable values


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
        P = ift.PowerProjectionOperator(domain=hspace, power_space=pspace)
        xi0 = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 1 / (1 + k**2)**dim
        pspec = ift.PS_field(pspace, pspec)
        A = P.adjoint_times(ift.sqrt(pspec))
        n = ift.Field.from_random(domain=space, random_type='normal')
        s0 = xi0 * A
        Instrument = ift.ScalingOperator(10., space)
        R = Instrument * ht
        N = ift.ScalingOperator(1., space)
        d = R(s0) + n

        direction = ift.Field.from_random('normal', hspace)
        direction /= np.sqrt(direction.var())
        eps = 1e-7
        s1 = s0 + eps * direction

        IC = ift.GradientNormController(
            iteration_limit=100,
            tol_abs_gradnorm=1e-5)
        inverter = ift.ConjugateGradient(IC)

        S = ift.create_power_operator(hspace, power_spectrum=lambda k: 1.)
        energy0 = ift.library.WienerFilterEnergy(
            position=s0, d=d, R=R, N=N, S=S, inverter=inverter)
        energy1 = ift.library.WienerFilterEnergy(
            position=s1, d=d, R=R, N=N, S=S, inverter=inverter)

        a = (energy1.value - energy0.value) / eps
        b = energy0.gradient.vdot(direction)
        tol = 1e-5
        assert_allclose(a, b, rtol=tol, atol=tol)

    @expand(product([ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    [4, 78, 23]))
    def testLognormalMap(self, space, seed):
        np.random.seed(seed)
        dim = len(space.shape)
        hspace = space.get_default_codomain()
        ht = ift.HarmonicTransformOperator(hspace, target=space)
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=False)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        P = ift.PowerProjectionOperator(domain=hspace, power_space=pspace)
        xi0 = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 1 / (1 + k**2)**dim
        pspec = ift.PS_field(pspace, pspec)
        A = P.adjoint_times(ift.sqrt(pspec))
        n = ift.Field.from_random(domain=space, random_type='normal')
        sh0 = xi0 * A
        s = ht(sh0)
        Instrument = ift.ScalingOperator(10., space)
        R = Instrument * ht
        N = ift.ScalingOperator(1., space)
        d = Instrument(ift.exp(s)) + n

        direction = ift.Field.from_random('normal', hspace)
        direction /= np.sqrt(direction.var())
        eps = 1e-6
        sh1 = sh0 + eps * direction

        IC = ift.GradientNormController(
            iteration_limit=100,
            tol_abs_gradnorm=1e-5)
        inverter = ift.ConjugateGradient(IC)

        S = ift.create_power_operator(hspace, power_spectrum=lambda k: 1.)
        energy0 = ift.library.LogNormalWienerFilterEnergy(
            position=sh0, d=d, R=R, N=N, S=S, inverter=inverter)
        energy1 = ift.library.LogNormalWienerFilterEnergy(
            position=sh1, d=d, R=R, N=N, S=S, inverter=inverter)

        a = (energy1.value - energy0.value) / eps
        b = energy0.gradient.vdot(direction)
        tol = 1e-3
        assert_allclose(a, b, rtol=tol, atol=tol)

    @expand(product([ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    [ift.library.Exponential, ift.library.Linear],
                    [4, 78, 23]))
    def testNonlinearMap(self, space, nonlinearity, seed):
        np.random.seed(seed)
        f = nonlinearity()
        dim = len(space.shape)
        hspace = space.get_default_codomain()
        ht = ift.HarmonicTransformOperator(hspace, target=space)
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=False)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        P = ift.PowerProjectionOperator(domain=hspace, power_space=pspace)
        xi0 = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 1 / (1 + k**2)**dim
        pspec = ift.PS_field(pspace, pspec)
        A = P.adjoint_times(ift.sqrt(pspec))
        n = ift.Field.from_random(domain=space, random_type='normal')
        s = ht(xi0 * A)
        R = ift.ScalingOperator(10., space)
        N = ift.ScalingOperator(1., space)
        d = R(f(s)) + n

        direction = ift.Field.from_random('normal', hspace)
        direction /= np.sqrt(direction.var())
        eps = 1e-7
        xi1 = xi0 + eps * direction

        S = ift.create_power_operator(hspace, power_spectrum=lambda k: 1.)
        energy0 = ift.library.NonlinearWienerFilterEnergy(
            position=xi0, d=d, Instrument=R, nonlinearity=f, ht=ht, power=A, N=N, S=S)
        energy1 = ift.library.NonlinearWienerFilterEnergy(
            position=xi1, d=d, Instrument=R, nonlinearity=f, ht=ht, power=A, N=N, S=S)

        a = (energy1.value - energy0.value) / eps
        b = energy0.gradient.vdot(direction)
        tol = 1e-4
        assert_allclose(a, b, rtol=tol, atol=tol)


class Curvature_Tests(unittest.TestCase):
    @expand(product([ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    [4, 78, 23]))
    def testLinearMapCurvature(self, space, seed):
        np.random.seed(seed)
        dim = len(space.shape)
        hspace = space.get_default_codomain()
        ht = ift.HarmonicTransformOperator(hspace, target=space)
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=False)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        P = ift.PowerProjectionOperator(domain=hspace, power_space=pspace)
        xi0 = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 1 / (1 + k**2)**dim
        pspec = ift.PS_field(pspace, pspec)
        A = P.adjoint_times(ift.sqrt(pspec))
        n = ift.Field.from_random(domain=space, random_type='normal')
        s0 = xi0 * A
        Instrument = ift.ScalingOperator(10., space)
        R = Instrument * ht
        N = ift.ScalingOperator(1., space)
        d = R(s0) + n

        direction = ift.Field.from_random('normal', hspace)
        direction /= np.sqrt(direction.var())
        eps = 1e-7
        s1 = s0 + eps * direction

        IC = ift.GradientNormController(
            iteration_limit=100,
            tol_abs_gradnorm=1e-5)
        inverter = ift.ConjugateGradient(IC)

        S = ift.create_power_operator(hspace, power_spectrum=lambda k: 1.)
        energy0 = ift.library.WienerFilterEnergy(
            position=s0, d=d, R=R, N=N, S=S, inverter=inverter)
        gradient0 = energy0.gradient
        gradient1 = energy0.at(s1).gradient

        a = (gradient1 - gradient0) / eps
        b = energy0.curvature(direction)
        tol = 1e-7
        assert_allclose(a.val, b.val, rtol=tol, atol=tol)

    @expand(product([ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    [4, 78, 23]))
    def testLognormalMapCurvature(self, space, seed):
        np.random.seed(seed)
        dim = len(space.shape)
        hspace = space.get_default_codomain()
        ht = ift.HarmonicTransformOperator(hspace, target=space)
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=False)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        P = ift.PowerProjectionOperator(domain=hspace, power_space=pspace)
        xi0 = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 1 / (1 + k**2)**dim
        pspec = ift.PS_field(pspace, pspec)
        A = P.adjoint_times(ift.sqrt(pspec))
        n = ift.Field.from_random(domain=space, random_type='normal')
        sh0 = xi0 * A
        s = ht(sh0)
        Instrument = ift.ScalingOperator(10., space)
        R = Instrument * ht
        N = ift.ScalingOperator(1., space)
        d = Instrument(ift.exp(s)) + n

        direction = ift.Field.from_random('normal', hspace)
        direction /= np.sqrt(direction.var())
        eps = 1e-7
        sh1 = sh0 + eps * direction

        IC = ift.GradientNormController(
            iteration_limit=100,
            tol_abs_gradnorm=1e-5)
        inverter = ift.ConjugateGradient(IC)

        S = ift.create_power_operator(hspace, power_spectrum=lambda k: 1.)

        energy0 = ift.library.LogNormalWienerFilterEnergy(
            position=sh0, d=d, R=R, N=N, S=S, inverter=inverter)
        gradient0 = energy0.gradient
        gradient1 = energy0.at(sh1).gradient

        a = (gradient1 - gradient0) / eps
        b = energy0.curvature(direction)
        tol = 1e-3
        assert_allclose(a.val, b.val, rtol=tol, atol=tol)

    @expand(product([ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    [ift.library.Exponential, ift.library.Linear],
                    [4, 78, 23]))
    def testNonlinearMapCurvature(self, space, nonlinearity, seed):
        np.random.seed(seed)
        f = nonlinearity()
        dim = len(space.shape)
        hspace = space.get_default_codomain()
        ht = ift.HarmonicTransformOperator(hspace, target=space)
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=False)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        P = ift.PowerProjectionOperator(domain=hspace, power_space=pspace)
        xi0 = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 1 / (1 + k**2)**dim
        pspec = ift.PS_field(pspace, pspec)
        A = P.adjoint_times(ift.sqrt(pspec))
        n = ift.Field.from_random(domain=space, random_type='normal')
        s = ht(xi0 * A)
        R = ift.ScalingOperator(10., space)
        N = ift.ScalingOperator(1., space)
        d = R(f(s)) + n

        direction = ift.Field.from_random('normal', hspace)
        direction /= np.sqrt(direction.var())
        eps = 1e-7
        xi1 = xi0 + eps * direction

        S = ift.create_power_operator(hspace, power_spectrum=lambda k: 1.)

        IC = ift.GradientNormController(
            iteration_limit=500,
            tol_abs_gradnorm=1e-7)
        inverter = ift.ConjugateGradient(IC)
        energy0 = ift.library.NonlinearWienerFilterEnergy(
            position=xi0,
            d=d,
            Instrument=R,
            nonlinearity=f,
            ht=ht,
            power=A,
            N=N,
            S=S,
            inverter=inverter)
        gradient0 = energy0.gradient
        gradient1 = energy0.at(xi1).gradient

        a = (gradient1 - gradient0) / eps
        b = energy0.curvature(direction)
        tol = 1e-3
        assert_allclose(a.val, b.val, rtol=tol, atol=tol)
