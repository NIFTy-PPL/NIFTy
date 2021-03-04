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

from .minimizer import Minimizer


class ADVIOptimizer(Minimizer):
    def __init__(self, steps, eta=1, alpha=1, tau=1, epsilon=1e-16):
        self.alpha = alpha
        self.eta = eta
        self.tau = tau
        self.epsilon = epsilon
        self.counter = 1
        self.steps = steps
        self.s = None

    def _step(self, position, gradient):
        self.s = self.alpha * gradient ** 2 + (1 - self.alpha) * self.s
        self.rho = (
            self.eta
            * self.counter ** (-0.5 + self.epsilon)
            / (self.tau + (self.s).sqrt())
        )
        new_position = position - self.rho * gradient
        self.counter += 1
        return new_position

    def __call__(self, E):
        from ..minimization.parametric_gaussian_kl import ParametricGaussianKL
        if self.s is None:
            self.s = E.gradient ** 2
        # FIXME come up with somthing how to determine convergence
        convergence = 0
        for i in range(self.steps):
            x = self._step(E.position, E.gradient)
            # FIXME maybe some KL function for resample? Should make it more generic.
            E = ParametricGaussianKL.make(
                x, E._hamiltonian, E._variational_model, E._n_samples, E._mirror_samples
            )

        return E, convergence

    def reset(self):
        self.counter = 1
        self.s = None
