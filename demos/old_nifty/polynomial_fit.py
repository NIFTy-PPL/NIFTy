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
# Author: Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import matplotlib.pyplot as plt
import numpy as np

import nifty8 as ift


def polynomial(coefficients, sampling_points):
    """Computes values of polynomial whose coefficients are stored in
    coefficients at sampling points. This is a quick version of the
    PolynomialResponse.

    Parameters
    ----------
    coefficients: Field
    sampling_points: Numpy array
    """

    if not (isinstance(coefficients, ift.Field)
            and isinstance(sampling_points, np.ndarray)):
        raise TypeError
    params = coefficients.val
    out = np.zeros_like(sampling_points)
    for ii in range(len(params)):
        out += params[ii] * sampling_points**ii
    return out


class PolynomialResponse(ift.LinearOperator):
    """Calculates values of a polynomial parameterized by input at sampling
    points.

    Parameters
    ----------
    domain: UnstructuredDomain
        The domain on which the coefficients of the polynomial are defined.
    sampling_points: Numpy array
        x-values of the sampling points.
    """

    def __init__(self, domain, sampling_points):
        if not (isinstance(domain, ift.UnstructuredDomain)
                and isinstance(sampling_points, np.ndarray)):
            raise TypeError
        self._domain = ift.DomainTuple.make(domain)
        tgt = ift.UnstructuredDomain(sampling_points.shape)
        self._target = ift.DomainTuple.make(tgt)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._mat = np.empty((self.target.size, domain.size))
        for d in range(domain.size):
            self._mat.T[d] = sampling_points**d

    def apply(self, x, mode):
        self._check_input(x, mode)
        m = self._mat
        f = m if mode == self.TIMES else m.conj().T
        return ift.makeField(self._tgt(mode), f.dot(x.val))


def mock_fct(x):
    return np.sin(x**2 / 10) * x**3


def main():
    # Generate some mock data
    N_params = 10
    N_samples = 100
    size = (12,)

    x = ift.random.current_rng().random(size) * 10
    y = mock_fct(x)

    var = np.full_like(y, y.var() / 10)
    var[-2] *= 4
    var[5] /= 2
    var[5] -= 1

    # Set up minimization problem
    p_space = ift.UnstructuredDomain(N_params)
    params = ift.full(p_space, 0.)
    R = PolynomialResponse(p_space, x)
    ift.extra.check_linear_operator(R)

    d_space = R.target
    d = ift.makeField(d_space, y)
    N = ift.makeOp(ift.makeField(d_space, var), sampling_dtype=float)

    IC = ift.DeltaEnergyController(tol_rel_deltaE=1e-12, iteration_limit=200)
    likelihood_energy = ift.GaussianEnergy(d, N.inverse) @ R
    Ham = ift.StandardHamiltonian(likelihood_energy, IC)
    H = ift.EnergyAdapter(params, Ham, want_metric=True)

    # Minimize KL
    minimizer = ift.NewtonCG(IC)
    H, _ = minimizer(H)

    # Draw posterior samples
    metric = Ham(ift.Linearization.make_var(H.position, want_metric=True)).metric
    samples = [metric.draw_sample(from_inverse=True) + H.position
               for _ in range(N_samples)]

    # Plotting
    plt.errorbar(x, y, np.sqrt(var), fmt='ko', label='Data with error bars')
    xmin, xmax = x.min(), x.max()
    xs = np.linspace(xmin, xmax, 100)

    # Ground truth
    y_true = mock_fct(xs)

    sc = ift.StatCalculator()
    for ii in range(len(samples)):
        sc.add(samples[ii])
        ys = polynomial(samples[ii], xs)
        if ii == 0:
            plt.plot(xs, ys, 'k', alpha=.05, label='Posterior samples')
            continue
        plt.plot(xs, ys, 'k', alpha=.05)
    ys = polynomial(H.position, xs)
    plt.plot(xs, ys, 'r', linewidth=2., label='Interpolation')
    plt.plot(xs, y_true, 'b', linewidth=0.5, label='$\mathrm{f}(x) = \sin({x^2}/{10}) * x^2$')
    plt.legend()
    plt.savefig('fit.png', dpi=300)
    plt.close()

    # Print parameters
    mean = sc.mean.val
    sigma = np.sqrt(sc.var.val)
    for ii in range(len(mean)):
        print('Coefficient x**{}: {:.2E} +/- {:.2E}'.format(ii, mean[ii], sigma[ii]))


if __name__ == '__main__':
    main()
