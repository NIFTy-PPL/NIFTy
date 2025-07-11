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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest

from .common import list2fixture, setup_function, teardown_function

spaces = [ift.GLSpace(5),
          ift.MultiDomain.make({'': ift.RGSpace(5, distances=.789)}),
          (ift.RGSpace(3, distances=.789), ift.UnstructuredDomain(2))]
pmp = pytest.mark.parametrize
field = list2fixture([ift.from_random(sp, 'normal') for sp in spaces])
ntries = 10


def test_gaussian(field):
    energy = ift.GaussianEnergy(domain=field.domain, sampling_dtype=float)
    ift.extra.check_operator(energy, field)


def test_ScaledEnergy(field):
    icov = ift.ScalingOperator(field.domain, 1.2, np.float64)
    energy = ift.GaussianEnergy(inverse_covariance=icov)
    ift.extra.check_operator(energy.scale(0.3), field)

    lin = ift.Linearization.make_var(field, want_metric=True)
    met1 = energy(lin).metric
    met2 = energy.scale(0.3)(lin).metric
    res1 = met1(field)
    res2 = met2(field)/0.3
    ift.extra.assert_allclose(res1, res2, 0, 1e-12)
    met1.draw_sample()
    met2.draw_sample()


def test_QuadraticFormOperator(field):
    op = ift.ScalingOperator(field.domain, 1.2, np.float64)
    endo = ift.makeOp(op.draw_sample())
    energy = ift.QuadraticFormOperator(endo)
    ift.extra.check_operator(energy, field)


def test_studentt(field):
    if isinstance(field.domain, ift.MultiDomain):
        pytest.skip()
    energy = ift.StudentTEnergy(domain=field.domain, theta=.5)
    ift.extra.check_operator(energy, field)
    theta = ift.from_random(field.domain, 'normal').exp()
    energy = ift.StudentTEnergy(domain=field.domain, theta=theta)
    ift.extra.check_operator(energy, field, ntries=ntries)


def test_hamiltonian_and_KL(field):
    field = field.ptw("exp")
    space = field.domain
    lh = ift.GaussianEnergy(domain=space, sampling_dtype=float)
    hamiltonian = ift.StandardHamiltonian(lh)
    ift.extra.check_operator(hamiltonian, field, ntries=ntries)
    samps = [ift.from_random(space, 'normal') for i in range(2)]
    kl = ift.AveragedEnergy(hamiltonian, samps)
    ift.extra.check_operator(kl, field, ntries=ntries)


def test_variablecovariancegaussian(field):
    if isinstance(field.domain, ift.MultiDomain):
        pytest.skip()
    dc = {'a': field, 'b': field.ptw("exp")}
    mf = ift.MultiField.from_dict(dc)
    energy = ift.VariableCovarianceGaussianEnergy(field.domain, 'a', 'b', np.float64)
    ift.extra.check_operator(energy, mf, ntries=ntries)
    energy(ift.Linearization.make_var(mf, want_metric=True)).metric.draw_sample()


def test_specialgamma(field):
    if isinstance(field.domain, ift.MultiDomain):
        pytest.skip()
    energy = ift.operators.energy_operators._SpecialGammaEnergy(field)
    loc = ift.from_random(energy.domain).exp()
    ift.extra.check_operator(energy, loc, ntries=ntries)
    energy(ift.Linearization.make_var(loc, want_metric=True)).metric.draw_sample()


def test_inverse_gamma(field):
    if isinstance(field.domain, ift.MultiDomain):
        pytest.skip()
    field = field.ptw("exp")
    space = field.domain
    d = ift.random.current_rng().normal(10, size=space.shape)**2
    d = ift.Field(space, d)
    energy = ift.InverseGammaEnergy(d)
    ift.extra.check_operator(energy, field, tol=1e-10)


def testPoissonian(field):
    if isinstance(field.domain, ift.MultiDomain):
        pytest.skip()
    field = field.ptw("exp")
    space = field.domain
    d = ift.random.current_rng().poisson(120, size=space.shape)
    d = ift.Field(space, d)
    energy = ift.PoissonianEnergy(d)
    ift.extra.check_operator(energy, field)


def test_bernoulli(field):
    if isinstance(field.domain, ift.MultiDomain):
        pytest.skip()
    field = field.ptw("sigmoid")
    space = field.domain
    d = ift.random.current_rng().binomial(1, 0.1, size=space.shape)
    d = ift.Field(space, d)
    energy = ift.BernoulliEnergy(d)
    ift.extra.check_operator(energy, field, tol=1e-10)


def test_gaussian_entropy(field):
    if isinstance(field.domain, ift.MultiDomain):
        pytest.skip()
    field = field.ptw("sigmoid")
    energy = ift.library.variational_models.GaussianEntropy(field.domain)
    ift.extra.check_operator(energy, field)
