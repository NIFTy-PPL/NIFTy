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

from __future__ import division

import numpy as np

from .minimizer import Minimizer
from .iteration_controlling import GradientNormController


class ConjugateGradient(Minimizer):
    """ Implementation of the Conjugate Gradient scheme.

    It is an iterative method for solving a linear system of equations:
                                    Ax = b

    Parameters
    ----------
    controller : IterationController
        Object that decides when to terminate the minimization.
    preconditioner : Operator *optional*
        This operator can be provided which transforms the variables of the
        system to improve the conditioning (default: None).

    References
    ----------
    Jorge Nocedal & Stephen Wright, "Numerical Optimization", Second Edition,
    2006, Springer-Verlag New York

    """

    def __init__(self,
                 controller=GradientNormController(iteration_limit=100),
                 preconditioner=None):
        self._preconditioner = preconditioner
        self._controller = controller

    def __call__(self, energy):
        """ Runs the conjugate gradient minimization.

        Parameters
        ----------
        energy : Energy object at the starting point of the iteration.
            Its curvature operator must be independent of position, otherwise
            linear conjugate gradient minimization will fail.

        Returns
        -------
        energy : QuadraticEnergy
            state at last point of the iteration
        status : integer
            Can be controller.CONVERGED or controller.ERROR

        """

        controller = self._controller
        controller.reset(energy)

        r = -energy.gradient
        previous_gamma = np.inf
        d = r.copy_empty()
        d.val[:] = 0.

        while True:
            if self._preconditioner is not None:
                s = self._preconditioner(r)
            else:
                s = r
            gamma = r.vdot(s).real
            if gamma < 0:
                self.logger.warn(
                    "Positive definiteness of preconditioner violated!")
            if gamma == 0:
                self.logger.info("Gamma == 0. Stopping.")
                return energy, controller.CONVERGED

            d = s + d * max(0, gamma/previous_gamma)
            previous_gamma = gamma

            status = controller.check(energy)
            if status != controller.CONTINUE:
                return energy, status

            q = energy.curvature(d)
            ddotq = d.vdot(q).real
            if ddotq == 0.:
                self.logger.error("Alpha became infinite! Stopping.")
                return energy, controller.ERROR
            alpha = previous_gamma/ddotq

            if alpha < 0:
                self.logger.error(
                        "Positive definiteness of A violated! Stopping.")
                return energy, controller.ERROR

            r -= q * alpha
            energy = energy.at(position=energy.position + d*alpha,
                               gradient=-r)
