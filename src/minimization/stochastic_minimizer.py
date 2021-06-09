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
from .energy import Energy


class ADVIOptimizer(Minimizer):
    """Provide an implementation of an adaptive step-size sequence optimizer,
    following https://arxiv.org/abs/1603.00788.

    Parameters
    ----------
    steps: int
        The number of concecutive steps during one call of the optimizer.
    eta: positive float
        The scale of the step-size sequence. It might have to be adapted to the
        application to increase performance. Default: 1.
    alpha: float between 0 and 1
        The fraction of how much the current gradient impacts the momentum.
    tau: positive float
        This quantity prevents division by zero.
    epsilon: positive float
        A small value guarantees Robbins and Monro conditions.
    resample: bool
        Whether the loss function is resampled for the next iteration. 
        Stochastic losses require resampleing, deterministic ones not.
    """

    def __init__(self, controller, eta=1, alpha=0.1, tau=1, epsilon=1e-16, resample=True):
        self.alpha = alpha
        self.eta = eta
        self.tau = tau
        self.epsilon = epsilon
        self.counter = 1
        self._controller = controller
        # self.steps = steps
        self.s = None
        self.resample = resample

    def _step(self, position, gradient):
        self.s = self.alpha * gradient ** 2 + (1 - self.alpha) * self.s
        self.rho = self.eta * self.counter ** (-0.5 + self.epsilon) \
                / (self.tau + (self.s).sqrt())
        new_position = position - self.rho * gradient
        self.counter += 1
        return new_position

    def __call__(self, E):
        from ..utilities import myassert

        controller = self._controller
        status = controller.start(energy)
        if status != controller.CONTINUE:
            return energy, status

        if self.s is None:
            self.s = E.gradient ** 2
        while True:
            # check if position is at a flat point
            if energy.gradient_norm == 0:
                return energy, controller.CONVERGED

            x = self._step(E.position, E.gradient)
            if self.resample:
                E = E.resample_at(x)
            myassert(isinstance(E, Energy))
            myassert(x.domain is E.position.domain)

            energy = new_energy
            status = self._controller.check(energy)
            if status != controller.CONTINUE:
                return energy, status

        return E, convergence

    def reset(self):
        self.counter = 1
        self.s = None
