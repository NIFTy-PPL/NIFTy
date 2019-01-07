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

import nifty5 as ift
from itertools import product

# Currently it is not possible to parametrize fixtures. But this will
# hopefully be fixed in the future.
# https://docs.pytest.org/en/latest/proposals/parametrize_with_fixtures.html

SPACES = [
    ift.GLSpace(15),
    ift.RGSpace(64, distances=.789),
    ift.RGSpace([32, 32], distances=.789)
]
SEEDS = [4, 78, 23]
PARAMS = product(SEEDS, SPACES)


@pytest.fixture(params=PARAMS)
def field(request):
    np.random.seed(request.param[0])
    S = ift.ScalingOperator(1., request.param[1])
    s = S.draw_sample()
    return ift.MultiField.from_dict({'s1': s})['s1']


def test_gaussian(field):
    energy = ift.GaussianEnergy(domain=field.domain)
    ift.extra.check_value_gradient_consistency(energy, field)


def test_inverse_gamma(field):
    field = field.exp()
    space = field.domain
    d = np.random.normal(10, size=space.shape)**2
    d = ift.Field.from_global_data(space, d)
    energy = ift.InverseGammaLikelihood(d)
    ift.extra.check_value_gradient_consistency(energy, field, tol=1e-7)


def testPoissonian(field):
    field = field.exp()
    space = field.domain
    d = np.random.poisson(120, size=space.shape)
    d = ift.Field.from_global_data(space, d)
    energy = ift.PoissonianEnergy(d)
    ift.extra.check_value_gradient_consistency(energy, field, tol=1e-7)


def test_hamiltonian_and_KL(field):
    field = field.exp()
    space = field.domain
    lh = ift.GaussianEnergy(domain=space)
    hamiltonian = ift.Hamiltonian(lh)
    ift.extra.check_value_gradient_consistency(hamiltonian, field)
    S = ift.ScalingOperator(1., space)
    samps = [S.draw_sample() for i in range(3)]
    kl = ift.SampledKullbachLeiblerDivergence(hamiltonian, samps)
    ift.extra.check_value_gradient_consistency(kl, field)


def test_bernoulli(field):
    field = field.positive_tanh()
    space = field.domain
    d = np.random.binomial(1, 0.1, size=space.shape)
    d = ift.Field.from_global_data(space, d)
    energy = ift.BernoulliEnergy(d)
    ift.extra.check_value_gradient_consistency(energy, field, tol=1e-6)
