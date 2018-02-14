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
from .minimizer import Minimizer
from .line_search_strong_wolfe import LineSearchStrongWolfe


class NonlinearCG(Minimizer):
    """ Implementation of the nonlinear Conjugate Gradient scheme according to
        Polak-Ribiere.

    Parameters
    ----------
    controller : IterationController
        Object that decides when to terminate the minimization.

    References
    ----------
    Jorge Nocedal & Stephen Wright, "Numerical Optimization", Second Edition,
    2006, Springer-Verlag New York
    """

    def __init__(self, controller, line_searcher=LineSearchStrongWolfe()):
        self._controller = controller
        self._line_searcher = line_searcher

    def __call__(self, energy):
        """ Runs the conjugate gradient minimization.
        Algorithm 5.4 from Nocedal & Wright
        Eq. (5.41a) has been replaced by eq. (5.49)

        Parameters
        ----------
        energy : Energy object at the starting point of the iteration.

        Returns
        -------
        Energy
            state at last point of the iteration
        int
            Can be controller.CONVERGED or controller.ERROR
        """
        controller = self._controller
        status = controller.start(energy)
        if status != controller.CONTINUE:
            return energy, status
        f_k_minus_1 = None

        p = -energy.gradient

        while True:
            grad_old = energy.gradient
            f_k = energy.value
            energy = self._line_searcher.perform_line_search(energy, p,
                                                             f_k_minus_1)
            f_k_minus_1 = f_k
            status = self._controller.check(energy)
            if status != controller.CONTINUE:
                return energy, status
            grad_new = energy.gradient
            gnnew = energy.gradient_norm
            beta = gnnew*gnnew/(grad_new-grad_old).vdot(p).real
            p = beta*p - grad_new
