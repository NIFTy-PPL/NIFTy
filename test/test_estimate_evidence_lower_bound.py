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
# Copyright(C) 2022 Max-Planck-Society
# Author: Matteo Guardiani

import nifty8 as ift
import numpy as np
import pytest

pmp = pytest.mark.parametrize
from nifty8.operator_spectrum import _DomRemover


def _explicify(operator):
    tmp = _DomRemover(operator.domain)
    operator = operator @ tmp.adjoint
    tmp = _DomRemover(operator.target)
    operator = tmp @ operator
    identity = np.identity(operator.domain.size, dtype=np.float64)
    res = []
    for v in identity:
        res.append(operator(ift.makeField(operator.domain, v)).val)
    return np.vstack(res).T


class LinearResponse(ift.LinearOperator):
    """Calculates values of a polynomial parameterized by input at sampling
    points.

    Parameters
    ----------
    domain: UnstructuredDomain
        The domain on which the coefficients of the polynomial are defined.
    sampling_points: Numpy array
        x-values of the sampling points.
    """

    def __init__(self, intercept, slope, sampling_points):
        # FIXME: Add input checks
        self.intercept_key = intercept.domain.keys()[0]
        self.slope_key = slope.domain.keys()[0]
        domain = {self.intercept_key: intercept.domain[self.intercept_key],
                  self.slope_key: slope.domain[self.slope_key]}
        self.sampling_points = sampling_points

        self._domain = ift.MultiDomain.make(domain)
        tgt = ift.RGSpace(self.sampling_points.shape)
        self._target = ift.DomainTuple.make(tgt)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)

        if mode == self.TIMES:
            intercept = x.val[self.intercept_key]
            slope = x.val[self.slope_key]
            return ift.makeField(self._tgt(mode), intercept + slope * self.sampling_points)

        res = np.vstack((np.ones(self.sampling_points.shape[0]), self.sampling_points)).dot(x.val)
        return ift.makeField(self._tgt(mode), {'intercept': res[0], 'slope': res[1]})


def test_estimate_evidence_lower_bound():
    # Set up signal
    n_datapoints = 8
    data_space = ift.RGSpace((n_datapoints,))

    q = -1.
    m_slope = 1.5

    sigma_q = 1.5
    sigma_m = 1.8
    intercept = ift.NormalTransform(0., sigma_q, "intercept").ducktape_left("intercept")
    slope = ift.NormalTransform(0., sigma_m, "slope").ducktape_left("slope")

    x = ift.random.current_rng().random(n_datapoints) * 10
    y = q + m_slope * x
    linear_response = LinearResponse(intercept, slope, x)
    # ift.extra.check_linear_operator(linear_response)

    linear_response_on_signal = linear_response @ (intercept + slope)  # In general not a linear operator

    R = _explicify(linear_response_on_signal)
    d = ift.makeField(data_space, y)
    noise_level = 0.8
    N = ift.makeOp(ift.makeField(data_space, noise_level ** 2), sampling_dtype=float)
    noise = N.draw_sample()
    data = d + noise

    N_inv = _explicify(N.inverse)

    S = np.identity(2)
    S_inv = np.identity(2)

    D_inv = R.T @ N_inv @ R + S_inv
    D = np.linalg.inv(D_inv)

    j = R.T @ (N_inv @ data.val)
    m = D @ j
    m_dag_j = np.dot(m, j)

    det_2pi_D = np.linalg.det(2 * np.pi * D)
    det_2pi_S = np.linalg.det(2 * np.pi * S)

    H_0 = 0.5 * (data.s_vdot(N.inverse(data)) + n_datapoints * np.log(2 * np.pi * noise_level ** 2) + np.log(
        det_2pi_S) - m_dag_j)

    evidence = -H_0 + 0.5 * np.log(det_2pi_D)
    nifty_adjusted_evidence = evidence + 0.5 * n_datapoints * np.log(2 * np.pi * noise_level ** 2)
    likelihood_energy = ift.GaussianEnergy(data=data, inverse_covariance=N.inverse) @ linear_response_on_signal

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(name="Sampling (linear)", deltaE=1e-8, iteration_limit=2)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=1e-5, convergence_level=2, iteration_limit=100)

    minimizer = ift.NewtonCG(ic_newton)
    minimizer_sampling = None

    n_iterations = 2
    n_samples = 2

    samples = ift.optimize_kl(likelihood_energy, n_iterations, n_samples, minimizer, ic_sampling, minimizer_sampling)

    # Estimate the ELBO
    elbo, stats = ift.estimate_evidence_lower_bound(ift.StandardHamiltonian(lh=likelihood_energy), samples, 2)
    assert (stats['elbo_lw'].val <= nifty_adjusted_evidence <= stats['elbo_up'].val)
