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
from ..field import Field
from .. import dobj
from ..utilities import general_axpy


class ConjugateGradient(Minimizer):
    """ Implementation of the Conjugate Gradient scheme.

    It is an iterative method for solving a linear system of equations:
                                    Ax = b

    Parameters
    ----------
    controller : IterationController
        Object that decides when to terminate the minimization.

    References
    ----------
    Jorge Nocedal & Stephen Wright, "Numerical Optimization", Second Edition,
    2006, Springer-Verlag New York
    """

    def __init__(self, controller):
        self._controller = controller

    def __call__(self, energy, preconditioner=None):
        """ Runs the conjugate gradient minimization.

        Parameters
        ----------
        energy : Energy object at the starting point of the iteration.
            Its curvature operator must be independent of position, otherwise
            linear conjugate gradient minimization will fail.
        preconditioner : Operator *optional*
            This operator can be provided which transforms the variables of the
            system to improve the conditioning (default: None).

        Returns
        -------
        energy : QuadraticEnergy
            state at last point of the iteration
        status : integer
            Can be controller.CONVERGED or controller.ERROR
        """
        controller = self._controller
        status = controller.start(energy)
        if status != controller.CONTINUE:
            return energy, status

        r = energy.gradient
        if preconditioner is not None:
            d = preconditioner(r)
        else:
            d = r.copy()
        previous_gamma = (r.vdot(d)).real
        if previous_gamma == 0:
            return energy, controller.CONVERGED

        tpos = Field(d.domain, dtype=d.dtype)  # temporary buffer
        while True:
            q = energy.curvature(d)
            ddotq = d.vdot(q).real
            if ddotq == 0.:
                dobj.mprint("Error: ConjugateGradient: ddotq==0.")
                return energy, controller.ERROR
            alpha = previous_gamma/ddotq

            if alpha < 0:
                dobj.mprint("Error: ConjugateGradient: alpha<0.")
                return energy, controller.ERROR

            general_axpy(-alpha, q, r, out=r)

            general_axpy(-alpha, d, energy.position, out=tpos)
            energy = energy.at_with_grad(tpos, r)

            if preconditioner is not None:
                s = preconditioner(r)
            else:
                s = r

            gamma = r.vdot(s).real
            if gamma < 0:
                dobj.mprint(
                    "Positive definiteness of preconditioner violated!")
                return energy, controller.ERROR
            if gamma == 0:
                return energy, controller.CONVERGED

            status = self._controller.check(energy)
            if status != controller.CONTINUE:
                return energy, status

            general_axpy(max(0, gamma/previous_gamma), d, s, out=d)

            previous_gamma = gamma
