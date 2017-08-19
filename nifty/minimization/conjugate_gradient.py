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

from keepers import Loggable


class ConjugateGradient(Loggable, object):
    """ Implementation of the Conjugate Gradient scheme.

    It is an iterative method for solving a linear system of equations:
                                    Ax = b

    Parameters
    ----------
    convergence_tolerance : float *optional*
        Tolerance specifying the case of convergence. (default: 1E-4)
    convergence_level : integer *optional*
        Number of times the tolerance must be undershot before convergence
        is reached. (default: 3)
    iteration_limit : integer *optional*
        Maximum number of iterations performed (default: None).
    reset_count : integer *optional*
        Number of iterations after which to restart; i.e., forget previous
        conjugated directions (default: None).
    preconditioner : Operator *optional*
        This operator can be provided which transforms the variables of the
        system to improve the conditioning (default: None).
    callback : callable *optional*
        Function f(energy, iteration_number) supplied by the user to perform
        in-situ analysis at every iteration step. When being called the
        current energy and iteration_number are passed. (default: None)

    Attributes
    ----------
    convergence_tolerance : float
        Tolerance specifying the case of convergence.
    convergence_level : integer
        Number of times the tolerance must be undershot before convergence
        is reached. (default: 3)
    iteration_limit : integer
        Maximum number of iterations performed.
    reset_count : integer
        Number of iterations after which to restart; i.e., forget previous
        conjugated directions.
    preconditioner : function
        This operator can be provided which transforms the variables of the
        system to improve the conditioning (default: None).
    callback : callable
        Function f(energy, iteration_number) supplied by the user to perform
        in-situ analysis at every iteration step. When being called the
        current energy and iteration_number are passed. (default: None)

    References
    ----------
    Jorge Nocedal & Stephen Wright, "Numerical Optimization", Second Edition,
    2006, Springer-Verlag New York

    """

    def __init__(self, convergence_tolerance=1E-4, convergence_level=3,
                 iteration_limit=None, reset_count=None,
                 preconditioner=None, callback=None):

        self.convergence_tolerance = np.float(convergence_tolerance)
        self.convergence_level = np.float(convergence_level)

        if iteration_limit is not None:
            iteration_limit = int(iteration_limit)
        self.iteration_limit = iteration_limit

        if reset_count is not None:
            reset_count = int(reset_count)
        self.reset_count = reset_count

        self.preconditioner = preconditioner
        self.callback = callback

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

        r = -E.gradient
        if self.preconditioner is not None:
            d = self.preconditioner(r)
        else:
            d = r.copy()
        previous_gamma = (r.vdot(d)).real
        if previous_gamma == 0:
            self.logger.info("The starting guess is already perfect solution "
                             "for the inverse problem.")
            return E, self.convergence_level+1

        convergence = 0
        iteration_number = 1
        self.logger.info("Starting conjugate gradient.")

        while True:
            if self.callback is not None:
                self.callback(E, iteration_number)

            q = E.curvature(d)
            alpha = previous_gamma/(d.vdot(q).real)

            if not np.isfinite(alpha):
                self.logger.error("Alpha became infinite! Stopping.")
                return E, 0

            E = E.at(E.position+d*alpha)

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

            beta = max(0, gamma/previous_gamma)

            delta = r.norm()

            self.logger.debug("Iteration : %08u   alpha = %3.1E   "
                              "beta = %3.1E   delta = %3.1E" %
                              (iteration_number, alpha, beta, delta))

            if gamma == 0:
                convergence = self.convergence_level+1
                self.logger.info("Reached infinite convergence.")
                break
            elif abs(delta) < self.convergence_tolerance:
                convergence += 1
                self.logger.info("Updated convergence level to: %u" %
                                 convergence)
                if convergence == self.convergence_level:
                    self.logger.info("Reached target convergence level.")
                    break
            else:
                convergence = max(0, convergence-1)

            if self.iteration_limit is not None:
                if iteration_number == self.iteration_limit:
                    self.logger.warn("Reached iteration limit. Stopping.")
                    break

            d = s + d * beta

            iteration_number += 1
            previous_gamma = gamma

        return E, convergence
