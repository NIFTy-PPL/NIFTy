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

from ..logger import logger
from .minimizer import Minimizer


class ConjugateGradient(Minimizer):
    """Implementation of the Conjugate Gradient scheme.

    It is an iterative method for solving a linear system of equations:
                                    Ax = b

    Parameters
    ----------
    controller : :py:class:`nifty8.IterationController`
        Object that decides when to terminate the minimization.
    nreset : int
        every `nreset` CG steps the residual will be recomputed accurately
        by applying the operator instead of updating the old residual

    References
    ----------
    Jorge Nocedal & Stephen Wright, "Numerical Optimization", Second Edition,
    2006, Springer-Verlag New York
    """

    def __init__(self, controller, nreset=20):
        self._controller = controller
        self._nreset = nreset

    def __call__(self, energy, preconditioner=None):
        """Runs the conjugate gradient minimization.

        Parameters
        ----------
        energy : Energy object at the starting point of the iteration.
            Its metric operator must be independent of position, otherwise
            linear conjugate gradient minimization will fail.
        preconditioner : Operator *optional*
            This operator can be provided which transforms the variables of the
            system to improve the conditioning. Default: None.

        Returns
        -------
        QuadraticEnergy
            state at last point of the iteration
        int
            Can be controller.CONVERGED or controller.ERROR
        """
        controller = self._controller
        status = controller.start(energy)
        if status != controller.CONTINUE:
            return energy, status

        r = energy.gradient
        d = r if preconditioner is None else preconditioner(r)

        previous_gamma = r.s_vdot(d).real
        if np.isnan(previous_gamma):
            logger.error("Error: ConjugateGradient: previous_gamma==NaN")
            return energy, controller.ERROR
        if previous_gamma == 0:
            return energy, controller.CONVERGED

        ii = 0
        while True:
            q = energy.apply_metric(d)
            curv = d.s_vdot(q).real
            if np.isnan(curv):
                logger.error("Error: ConjugateGradient: curv==NaN")
                return energy, controller.ERROR
            if curv == 0.:
                logger.error("Error: ConjugateGradient: curv==0.")
                return energy, controller.ERROR
            alpha = previous_gamma/curv

            if alpha < 0:
                logger.error("Error: ConjugateGradient: alpha<0.")
                return energy, controller.ERROR

            ii += 1
            if ii < self._nreset:
                r = r - q*alpha
                energy = energy.at_with_grad(energy.position - alpha*d, r)
            else:
                energy = energy.at(energy.position - alpha*d)
                r = energy.gradient
                ii = 0

            s = r if preconditioner is None else preconditioner(r)

            gamma = r.s_vdot(s).real
            if np.isnan(gamma):
                logger.error("Error: ConjugateGradient: gamma==NaN")
                return energy, controller.ERROR
            if gamma < 0:
                logger.error(
                    "Positive definiteness of preconditioner violated!")
                return energy, controller.ERROR
            if gamma == 0:
                return energy, controller.CONVERGED

            status = controller.check(energy)
            if status != controller.CONTINUE:
                return energy, status

            d = d * max(0, gamma/previous_gamma) + s

            previous_gamma = gamma
