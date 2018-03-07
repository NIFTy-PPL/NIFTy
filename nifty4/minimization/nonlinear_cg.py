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

    Parameters
    ----------
    controller : IterationController
        Object that decides when to terminate the minimization.
    beta_heuristics : str
        One of 'Polak-Ribiere', 'Fletcher-Reeves', 'Hestenes-Stiefel'

    References
    ----------
    Jorge Nocedal & Stephen Wright, "Numerical Optimization", Second Edition,
    2006, Springer-Verlag New York
    """

    def __init__(self, controller, beta_heuristics='Polak-Ribiere'):
        valid_beta_heuristics = ['Polak-Ribiere', 'Fletcher-Reeves',
                                 'Hestenes-Stiefel', "5.49"]
        if not (beta_heuristics in valid_beta_heuristics):
            raise ValueError("beta heuristics must be either 'Polak-Ribiere', "
                             "'Fletcher-Reeves', or 'Hestenes-Stiefel'")
        self._beta_heuristic = beta_heuristics
        self._controller = controller
        self._line_searcher = LineSearchStrongWolfe(c2=0.1)

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
            energy, success = self._line_searcher.perform_line_search(
                energy, p, f_k_minus_1)
            if not success:
                return energy, controller.ERROR
            f_k_minus_1 = f_k
            status = self._controller.check(energy)
            if status != controller.CONTINUE:
                return energy, status
            grad_new = energy.gradient

            if self._beta_heuristic == 'Hestenes-Stiefel':
                # Eq. (5.46) in Nocedal & Wright.
                beta = max(0.0, (grad_new.vdot(grad_new-grad_old) /
                                 (grad_new-grad_old).vdot(p)).real)
            elif self._beta_heuristic == 'Polak-Ribiere':
                # Eq. (5.44) in Nocedal & Wright. (with (5.45) additionally)
                beta = max(0.0, (grad_new.vdot(grad_new-grad_old) /
                                 (grad_old.vdot(grad_old))).real)
            elif self._beta_heuristic == 'Fletcher-Reeves':
                # Eq. (5.41a) in Nocedal & Wright.
                beta = (grad_new.vdot(grad_new)/(grad_old.vdot(grad_old))).real
            else:
                # Eq. (5.49) in Nocedal & Wright.
                beta = (grad_new.vdot(grad_new) /
                        ((grad_new-grad_old).vdot(p))).real
            p = beta*p - grad_new
