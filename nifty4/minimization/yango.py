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


class Yango(Minimizer):
    """ Nonlinear conjugate gradient using curvature


    Parameters
    ----------
    controller : IterationController
        Object that decides when to terminate the minimization.

    Notes
    -----
    No restarting procedure has been implemented yet.

    References
    ----------
    """

    def __init__(self, controller, line_searcher = LineSearchStrongWolfe(c2=0.1)):
        self._controller = controller
        self._line_searcher = line_searcher

    def __call__(self, energy):
        controller = self._controller
        status = controller.start(energy)
        if status != controller.CONTINUE:
            return energy, status
        f_k_minus_1 = None

        p = -energy.gradient
        A_k = energy.curvature
        
        while True:
            grad_old = -energy.gradient
            f_k = energy.value
            gamma = p.vdot(grad_old)/p.vdot(A_k(p))
            if gamma < 0:
                raise ValueError("Not a descent direction?!")
            energy, success = self._line_searcher.perform_line_search(
                energy, p*gamma, f_k_minus_1)
            if not success:
                return energy, controller.ERROR
            f_k_minus_1 = f_k
            status = self._controller.check(energy)
            if status != controller.CONTINUE:
                return energy, status
            grad_new = -energy.gradient
            A_k = energy.curvature

            beta = (grad_new.vdot(A_k(p)) /
                                 p.vdot(A_k(p)))
            p = grad_new + beta*p 
