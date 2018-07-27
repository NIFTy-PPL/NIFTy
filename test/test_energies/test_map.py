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
from itertools import product
from test.common import expand

import nifty5 as ift
import numpy as np


def _flat_PS(k):
    return np.ones_like(k)


class Energy_Tests(unittest.TestCase):
    @expand(product([ift.GLSpace(15),
                     ift.RGSpace(64, distances=.789),
                     ift.RGSpace([32, 32], distances=.789)],
                    ["tanh", "exp", ""],
                    [1, 1e-2, 1e2],
                    [4, 78, 23]))
    def testGaussianEnergy(self, space, nonlinearity, noise, seed):
        np.random.seed(seed)
        dim = len(space.shape)
        hspace = space.get_default_codomain()
        ht = ift.HarmonicTransformOperator(hspace, target=space)
        binbounds = ift.PowerSpace.useful_binbounds(hspace, logarithmic=False)
        pspace = ift.PowerSpace(hspace, binbounds=binbounds)
        Dist = ift.PowerDistributor(target=hspace, power_space=pspace)
        xi0 = ift.Field.from_random(domain=hspace, random_type='normal')
        xi0_var = ift.Linearization.make_var(xi0)

        def pspec(k): return 1 / (1 + k**2)**dim
        pspec = ift.PS_field(pspace, pspec)
        A = Dist(ift.sqrt(pspec))
        N = ift.ScalingOperator(noise, space)
        n = N.draw_sample()
        s = ht(ift.makeOp(A)(xi0_var))
        R = ift.ScalingOperator(10., space)
        def d_model(inp):
            if nonlinearity == "":
                return R(ht(ift.makeOp(A)(inp)))
            else:
                tmp = ht(ift.makeOp(A)(inp))
                nonlin = getattr(tmp, nonlinearity)
                return R(nonlin())
        d = d_model(xi0) + n

        if noise == 1:
            N = None

        energy = lambda inp: ift.GaussianEnergy(d, N)(d_model(inp))
        if nonlinearity == "":
            ift.extra.check_value_gradient_metric_consistency2(
                energy, xi0, ntries=10)
        else:
            ift.extra.check_value_gradient_consistency2(
                energy, xi0, ntries=10)
