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


class Noise_Energy_Tests(unittest.TestCase):
    @expand(product([ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    [ift.library.Exponential, ift.library.Linear],
                    [23, 131, 32]))
    def testNoise(self, space, nonlinearity, seed):
        np.random.seed(seed)
        f = nonlinearity()
        dim = len(space.shape)
        hspace = space.get_default_codomain()
        ht = ift.HarmonicTransformOperator(hspace, target=space)
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=False)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        P = ift.PowerProjectionOperator(domain=hspace, power_space=pspace)
        xi = ift.Field.from_random(domain=hspace, random_type='normal')

        def pspec(k): return 1 / (1 + k**2)**dim
        tau = ift.PS_field(pspace, pspec)
        A = P.adjoint_times(ift.sqrt(tau))
        var = 1.
        n = ift.Field.from_random(domain=space, random_type='normal', std=np.sqrt(var))
        var = ift.Field(n.domain, val=var)
        N = ift.DiagonalOperator(var)
        eta0 = ift.log(var)
        s = ht(xi * A)
        R = ift.ScalingOperator(10., space)
        d = R(f(s)) + n

        alpha = ift.Field(d.domain, val=2.)
        q = ift.Field(d.domain, val=1e-5)

        direction = ift.Field.from_random('normal', d.domain)
        direction /= np.sqrt(direction.var())
        eps = 1e-8
        eta1 = eta0 + eps * direction

        IC = ift.GradientNormController(
            iteration_limit=100,
            tol_abs_gradnorm=1e-5)
        inverter = ift.ConjugateGradient(IC)

        S = ift.create_power_operator(hspace, power_spectrum=lambda k: 1.)
        D = ift.library.NonlinearWienerFilterEnergy(
            position=xi,
            d=d,
            Instrument=R,
            nonlinearity=f,
            ht=ht,
            power=A,
            N=N,
            S=S,
            inverter=inverter).curvature
        Nsamples = 10
        sample_list = [D.generate_posterior_sample() + xi for i in range(Nsamples)]

        energy0 = ift.library.NoiseEnergy(
            position=eta0, d=d, m=xi, D=D, t=tau, Instrument=R,
            alpha=alpha, q=q, Projection=P, nonlinearity=f,
            ht=ht, sample_list=sample_list)
        energy1 = ift.library.NoiseEnergy(
            position=eta1, d=d, m=xi, D=D, t=tau, Instrument=R,
            alpha=alpha, q=q, Projection=P, nonlinearity=f,
            ht=ht, sample_list=sample_list)

        a = (energy1.value - energy0.value) / eps
        b = energy0.gradient.vdot(direction)
        tol = 1e-5
        assert_allclose(a, b, rtol=tol, atol=tol)
