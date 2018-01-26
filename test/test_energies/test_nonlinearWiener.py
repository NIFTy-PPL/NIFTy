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


_harmonic_spaces = [ift.RGSpace(7, distances=0.2, harmonic=True),
                    ift.RGSpace((12, 46), distances=(0.2, 0.3), harmonic=True),
                    ift.LMSpace(17)]

_position_spaces = [ift.RGSpace(19, distances=0.7),
                    ift.RGSpace((1, 2, 3, 6), distances=(0.2, 0.25, 0.34, .8)),
                    ift.HPSpace(17),
                    ift.GLSpace(8, 13)]


class Energy_Tests(unittest.TestCase):
    @expand(product([ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    [ift.library.Exponential, ift.library.Linear]))
    def testNonlinearMap(self, space, nonlinearity):
        f = nonlinearity()
        dim = len(space.shape)
        fft = ift.FFTOperator(space)
        hspace = fft.target[0]
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=False)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        P = ift.PowerProjectionOperator(domain=hspace, power_space=pspace)
        xi0 = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 1 / (1 + k**2)**dim
        pspec = ift.PS_field(pspace, pspec)
        A = P.adjoint_times(ift.sqrt(pspec))
        n = ift.Field.from_random(domain=space, random_type='normal')
        s = fft.inverse_times(xi0 * A)
        diag = ift.Field.ones(space) * 10
        R = ift.DiagonalOperator(diag)
        diag = ift.Field.ones(space)
        N = ift.DiagonalOperator(diag)
        d = R(f(s)) + n

        direction = ift.Field.from_random('normal', hspace)
        direction /= np.sqrt(direction.var())
        eps = 1e-10
        xi1 = xi0 + eps * direction

        S = ift.create_power_operator(hspace, power_spectrum=lambda k: 1.)
        energy0 = ift.library.NonlinearWienerFilterEnergy(
            position=xi0, d=d, Instrument=R, nonlinearity=f, FFT=fft, power=A, N=N, S=S)
        energy1 = ift.library.NonlinearWienerFilterEnergy(
            position=xi1, d=d, Instrument=R, nonlinearity=f, FFT=fft, power=A, N=N, S=S)

        a = (energy1.value - energy0.value) / eps
        b = energy0.gradient.vdot(direction)
        tol = 1e-2
        assert_allclose(a, b, rtol=tol, atol=tol)

    @expand(product([ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    [ift.library.Exponential, ift.library.Linear]))
    def testNonlinearPower(self, space, nonlinearity):
        f = nonlinearity()
        dim = len(space.shape)
        fft = ift.FFTOperator(space)
        hspace = fft.target[0]
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=True)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        P = ift.PowerProjectionOperator(domain=hspace, power_space=pspace)
        xi = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 1 / (1 + k**2)**dim
        tau0 = ift.PS_field(pspace, pspec)
        A = P.adjoint_times(ift.sqrt(tau0))
        n = ift.Field.from_random(domain=space, random_type='normal')
        s = fft.inverse_times(xi * A)
        diag = ift.Field.ones(space) * 10
        R = ift.DiagonalOperator(diag)
        diag = ift.Field.ones(space)
        N = ift.DiagonalOperator(diag)
        d = R(f(s)) + n

        direction = ift.Field.from_random('normal', pspace)
        direction /= np.sqrt(direction.var())
        eps = 1e-10
        tau1 = tau0 + eps * direction

        IC = ift.GradientNormController(name='IC', verbose=False, iteration_limit=100, tol_abs_gradnorm=1e-5)
        inverter = ift.ConjugateGradient(IC)

        S = ift.create_power_operator(hspace, power_spectrum=lambda k: 1.)
        D = ift.library.NonlinearWienerFilterEnergy(position=xi, d=d, Instrument=R, nonlinearity=f, FFT=fft, power=A, N=N, S=S, inverter=inverter).curvature

        energy0 = ift.library.NonlinearPowerEnergy(
            position=tau0, d=d, m=xi, D=D, Instrument=R, Projection=P, nonlinearity=f, FFT=fft, N=N, inverter=inverter)
        energy1 = ift.library.NonlinearPowerEnergy(
            position=tau1, d=d, m=xi, D=D, Instrument=R, Projection=P, nonlinearity=f, FFT=fft, N=N, inverter=inverter)

        a = (energy1.value - energy0.value) / eps
        b = energy0.gradient.vdot(direction)
        tol = 1e-2
        assert_allclose(a, b, rtol=tol, atol=tol)
