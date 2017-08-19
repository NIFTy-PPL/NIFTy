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


class ConjugateGradient(Minimizer):
    """ Implementation of the Conjugate Gradient scheme.

    It is an iterative method for solving a linear system of equations:
                                    Ax = b

    Parameters
    ----------
    reset_count : integer *optional*
        Number of iterations after which to restart; i.e., forget previous
        conjugated directions (default: None).
    preconditioner : Operator *optional*
        This operator can be provided which transforms the variables of the
        system to improve the conditioning (default: None).

    Attributes
    ----------
    reset_count : integer
        Number of iterations after which to restart; i.e., forget previous
        conjugated directions.
    preconditioner : function
        This operator can be provided which transforms the variables of the
        system to improve the conditioning (default: None).
    controller : IterationController

    References
    ----------
    Jorge Nocedal & Stephen Wright, "Numerical Optimization", Second Edition,
    2006, Springer-Verlag New York

    """

    def __init__(self, controller, reset_count=None, preconditioner=None):
        if reset_count is not None:
            reset_count = int(reset_count)
        self.reset_count = reset_count

        self.preconditioner = preconditioner
        self._controller = controller

    def __call__(self, E):
        """ Runs the conjugate gradient minimization.

        Parameters
        ----------
        E : Energy object at the starting point of the iteration.
            E's curvature operator must be independent of position, otherwise
            linear conjugate gradient minimization will fail.

        Returns
        -------
        E : QuadraticEnergy at last point of the iteration
        convergence : integer
            Latest convergence level indicating whether the minimization
            has converged or not.

        """

        controller = self._controller
        status = controller.start(E)
        if status != controller.CONTINUE:
            return E, status

        r = -E.gradient
        if self.preconditioner is not None:
            d = self.preconditioner(r)
        else:
            d = r.copy()
        previous_gamma = (r.vdot(d)).real
        if previous_gamma == 0:
            return E, controller.CONVERGED

        while True:
            q = E.curvature(d)
            alpha = previous_gamma/(d.vdot(q).real)

            if not np.isfinite(alpha):
                self.logger.error("Alpha became infinite! Stopping.")
                return E, controller.ERROR

            E = E.at(E.position+d*alpha)
            status = self._controller.check(E)
            if status != controller.CONTINUE:
                return E, status

            reset = False
            if alpha < 0:
                self.logger.warn("Positive definiteness of A violated!")
                reset = True
            if self.reset_count is not None:
                reset += (iteration_number % self.reset_count == 0)
            if reset:
                self.logger.info("Resetting conjugate directions.")
                r = -E.gradient
            else:
                r -= q * alpha

            if self.preconditioner is not None:
                s = self.preconditioner(r)
            else:
                s = r.copy()

            gamma = r.vdot(s).real
            if gamma < 0:
                self.logger.warn("Positive definiteness of preconditioner "
                                 "violated!")
            if gamma == 0:
                return E, controller.CONVERGED

            d = s + d * max(0, gamma/previous_gamma)

            previous_gamma = gamma
