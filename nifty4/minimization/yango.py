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
from ..logger import logger
from .line_search_strong_wolfe import LineSearchStrongWolfe
import numpy as np


_default_LS = LineSearchStrongWolfe(c2=0.1, preferred_initial_step_size=1.)


class Yango(Minimizer):
    """ Nonlinear conjugate gradient using curvature
    The YANGO (Yet Another Nonlinear conjugate Gradient Optimizer)
    uses the curvature to make estimates about suitable descent
    directions. It takes the step that lets it go directly to
    the second order minimum in the subspace spanned by the last
    descent direction and the new gradient.

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

    def __init__(self, controller, line_searcher=_default_LS):
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
        energy, success = self._line_searcher.perform_line_search(
            energy, p.vdot(p)/(p.vdot(A_k(p)))*p, f_k_minus_1)
        if not success:
            return energy, controller.ERROR
        A_k = energy.curvature
        while True:
            r = -energy.gradient
            f_k = energy.value
            Ar = A_k(r)
            Ap = A_k(p)
            rAr = r.vdot(Ar)
            pAp = p.vdot(Ap)
            pAr = p.vdot(Ar)
            rAp = np.conj(pAr)
            rp = r.vdot(p)
            rr = r.vdot(r)
            if rr == 0 or rAr == 0:
                logger.warning(
                    "Warning: gradient norm 0, assuming convergence!")
                return energy, controller.CONVERGED
            det = pAp*rAr-np.abs(rAp*pAr)
            if det < 0:
                if rAr < 0:
                    logger.error(
                        "Error: negative curvature ({})".format(rAr))
                    return energy, controller.ERROR
                # Try 1D Newton Step
                energy, success = self._line_searcher.perform_line_search(
                    energy, (rr/rAr)*r, f_k_minus_1)
            else:
                a = (rAr*rp - rAp*rr)/det
                b = (pAp*rr - pAr*rp)/det
                energy, success = self._line_searcher.perform_line_search(
                    energy, p*a + r*b, f_k_minus_1)
                p = r - p*(pAr/pAp)
            if not success:
                return energy, controller.ERROR
            f_k_minus_1 = f_k
            status = self._controller.check(energy)
            if status != controller.CONTINUE:
                return energy, status
            A_k = energy.curvature
