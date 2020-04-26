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

import nifty6 as ift
from itertools import product
from .common import setup_function, teardown_function

# Currently it is not possible to parametrize fixtures. But this will
# hopefully be fixed in the future.
# https://docs.pytest.org/en/latest/proposals/parametrize_with_fixtures.html

SPACES = [ift.GLSpace(15),
          ift.RGSpace(64, distances=.789),
          ift.RGSpace([32, 32], distances=.789)]
for sp in SPACES[:3]:
    SPACES.append(ift.MultiDomain.make({'asdf': sp}))
SEEDS = [4, 78, 23]
PARAMS = product(SEEDS, SPACES)
pmp = pytest.mark.parametrize


@pytest.fixture(params=PARAMS)
def field(request):
    with ift.random.Context(request.param[0]):
        S = ift.ScalingOperator(request.param[1], 1.)
        return S.draw_sample(dtype=np.float64)


def test_gaussian(field):
    energy = ift.GaussianEnergy(domain=field.domain)
    ift.extra.check_jacobian_consistency(energy, field)


def test_ScaledEnergy(field):
    icov = ift.ScalingOperator(field.domain, 1.2)
    energy = ift.GaussianEnergy(inverse_covariance=icov)
    ift.extra.check_jacobian_consistency(energy.scale(0.3), field)

    lin = ift.Linearization.make_var(field, want_metric=True)
    met1 = energy(lin).metric
    met2 = energy.scale(0.3)(lin).metric
    res1 = met1(field)
    res2 = met2(field)/0.3
    ift.extra.assert_allclose(res1, res2, 0, 1e-12)
    met2.draw_sample(dtype=np.float64)


def test_QuadraticFormOperator(field):
    op = ift.ScalingOperator(field.domain, 1.2)
    endo = ift.makeOp(op.draw_sample(dtype=np.float64))
    energy = ift.QuadraticFormOperator(endo)
    ift.extra.check_jacobian_consistency(energy, field)


def test_studentt(field):
    if isinstance(field.domain, ift.MultiDomain):
        return
    energy = ift.StudentTEnergy(domain=field.domain, theta=.5)
    ift.extra.check_jacobian_consistency(energy, field, tol=1e-6)


def test_hamiltonian_and_KL(field):
    field = field.ptw("exp")
    space = field.domain
    lh = ift.GaussianEnergy(domain=space)
    hamiltonian = ift.StandardHamiltonian(lh)
    ift.extra.check_jacobian_consistency(hamiltonian, field)
    S = ift.ScalingOperator(space, 1.)
    samps = [S.draw_sample(dtype=np.float64) for i in range(3)]
    kl = ift.AveragedEnergy(hamiltonian, samps)
    ift.extra.check_jacobian_consistency(kl, field)


def test_variablecovariancegaussian(field):
    if isinstance(field.domain, ift.MultiDomain):
        return
    dc = {'a': field, 'b': field.ptw("exp")}
    mf = ift.MultiField.from_dict(dc)
    energy = ift.VariableCovarianceGaussianEnergy(field.domain, 'a', 'b')
    ift.extra.check_jacobian_consistency(energy, mf, tol=1e-6)
    energy(ift.Linearization.make_var(mf, want_metric=True)).metric.draw_sample(dtype=np.float64)


def test_inverse_gamma(field):
    if isinstance(field.domain, ift.MultiDomain):
        return
    field = field.ptw("exp")
    space = field.domain
    d = ift.random.current_rng().normal(10, size=space.shape)**2
    d = ift.Field(space, d)
    energy = ift.InverseGammaLikelihood(d)
    ift.extra.check_jacobian_consistency(energy, field, tol=1e-5)


def testPoissonian(field):
    if isinstance(field.domain, ift.MultiDomain):
        return
    field = field.ptw("exp")
    space = field.domain
    d = ift.random.current_rng().poisson(120, size=space.shape)
    d = ift.Field(space, d)
    energy = ift.PoissonianEnergy(d)
    ift.extra.check_jacobian_consistency(energy, field, tol=1e-6)


def test_bernoulli(field):
    if isinstance(field.domain, ift.MultiDomain):
        return
    field = field.ptw("sigmoid")
    space = field.domain
    d = ift.random.current_rng().binomial(1, 0.1, size=space.shape)
    d = ift.Field(space, d)
    energy = ift.BernoulliEnergy(d)
    ift.extra.check_jacobian_consistency(energy, field, tol=1e-5)
