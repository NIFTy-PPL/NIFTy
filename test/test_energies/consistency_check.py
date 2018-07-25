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


class Energy_Tests(unittest.TestCase):
    def make_model(self, type, **kwargs):
        if type == 'Constant':
            np.random.seed(kwargs['seed'])
            S = ift.ScalingOperator(1., kwargs['space'])
            s = S.draw_sample()
            return ift.Constant(
                ift.MultiField.from_dict({kwargs['space_key']: s}),
                ift.MultiField.from_dict({kwargs['space_key']: s}))
        elif type == 'Variable':
            np.random.seed(kwargs['seed'])
            S = ift.ScalingOperator(1., kwargs['space'])
            s = S.draw_sample()
            return ift.Variable(
                ift.MultiField.from_dict({kwargs['space_key']: s}))
        elif type == 'LinearModel':
            return ift.LinearModel(
                inp=kwargs['model'], lin_op=kwargs['lin_op'])
        else:
            raise ValueError('unknown type passed')

    @expand(product(
        ['Variable'],
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]
        ))
    def testGaussian(self, type1, space, seed):
        model = self.make_model(
            type1, space_key='s1', space=space, seed=seed)['s1']
        energy = ift.GaussianEnergy(model)
        ift.extra.check_value_gradient_consistency(energy)

    @expand(product(
        ['Variable'],
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]
        ))
    def testQuadratic(self, type1, space, seed):
        np.random.seed(seed)
        S = ift.ScalingOperator(1., space)
        s = [S.draw_sample() for _ in range(3)]
        energy = ift.QuadraticEnergy(s[0], ift.makeOp(s[1]), s[2])
        ift.extra.check_value_gradient_consistency(energy)

    @expand(product(
        ['Variable'],
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]
        ))
    def testPoissonian(self, type1, space, seed):
        model = self.make_model(
            type1, space_key='s1', space=space, seed=seed)['s1']
        model = ift.PointwiseExponential(model)
        d = np.random.poisson(120, size=space.shape)
        d = ift.Field.from_global_data(space, d)
        energy = ift.PoissonianEnergy(model, d)
        ift.extra.check_value_gradient_consistency(energy)

    @expand(product(
        ['Variable'],
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]
        ))
    def testHamiltonian_and_KL(self, type1, space, seed):
        model = self.make_model(
            type1, space_key='s1', space=space, seed=seed)['s1']
        model = ift.PointwiseExponential(model)
        lh = ift.GaussianEnergy(model)
        hamiltonian = ift.Hamiltonian(lh)
        ift.extra.check_value_gradient_consistency(hamiltonian)
        S = ift.ScalingOperator(1., space)
        samps = [S.draw_sample() for i in range(3)]
        kl = ift.SampledKullbachLeiblerDivergence(hamiltonian, samps)
        ift.extra.check_value_gradient_consistency(kl)

    @expand(product(
        ['Variable'],
        [ift.GLSpace(15),
         ift.RGSpace(64, distances=.789),
         ift.RGSpace([32, 32], distances=.789)],
        [4, 78, 23]
        ))
    def testBernoulli(self, type1, space, seed):
        model = self.make_model(
            type1, space_key='s1', space=space, seed=seed)['s1']
        model = ift.PointwisePositiveTanh(model)
        d = np.random.binomial(1, 0.1, size=space.shape)
        d = ift.Field.from_global_data(space, d)
        energy = ift.BernoulliEnergy(model, d)
        ift.extra.check_value_gradient_consistency(energy)
