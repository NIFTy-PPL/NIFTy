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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import division
from .minimizer import Minimizer
from .line_search_strong_wolfe import LineSearchStrongWolfe


class NonlinearCG(Minimizer):
    """ Nonlinear Conjugate Gradient scheme according to Polak-Ribiere.

    Algorithm 5.4 from Nocedal & Wright.
    Eq. (5.41a) has been replaced by eq. (5.49)

    Parameters
    ----------
    controller : IterationController
        Object that decides when to terminate the minimization.
    line_searcher : LineSearch, optional
        The line search algorithm to be used

    References
    ----------
    Jorge Nocedal & Stephen Wright, "Numerical Optimization", Second Edition,
    2006, Springer-Verlag New York
    """

    def __init__(self, controller, line_searcher=LineSearchStrongWolfe(c2=0.1), beta_heuristics = 'Polak-Ribiere'):
        if (beta_heuristics != 'Polak-Ribiere') and (beta_heuristics != 'Polak-Ribiere'):
            raise ValueError("beta heuristics must be either 'Polak-Ribiere' or 'Hestenes-Stiefel'")
        self._beta_heuristic = beta_heuristics
        self._controller = controller
        self._line_searcher = line_searcher

    def __call__(self, energy):
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
            if self._beta_heuristic == 'Hestenes-Stiefel':
                beta = grad_new.vdot(grad_new-grad_old)/(grad_new-grad_old).vdot(p).real
            else:
                beta = grad_new.vdot(grad_new-grad_old)/(grad_old.vdot(grad_old)).real
            p = beta*p - grad_new
