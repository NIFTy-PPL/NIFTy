# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

from __future__ import division
import numpy as np

from keepers import Loggable


class ConjugateGradient(Loggable, object):
    """Implementation of the Conjugate Gradient scheme.
    
    It is an iterative method for solving a linear system of equations:
                                    Ax = b
    
    SUGGESTED LITERATURE:
        Thomas V. Mikosch et al., "Numerical Optimization", Second Edition, 
        2006, Springer-Verlag New York
        
    Parameters
    ----------
    convergence_tolerance : scalar
        Tolerance specifying convergence. (default: 1E-4)
    convergence_level : integer
        Number of times the tolerance should be undershot before exiting. 
        (default: 3)
    iteration_limit : integer *optional*
        Maximum number of iterations performed. (default: None)
    reset_count : integer, *optional*
        Number of iterations after which to restart; i.e., forget previous
        conjugated directions. (default: None)
    preconditioner : function *optional*
        The user can provide a function which transforms the variables of the 
        system to make the converge more favorable.(default: None)
    callback : function, *optional*
        Function f(energy, iteration_number) specified by the user to print 
        iteration number and energy value at every iteration step. It accepts 
        an Energy object(energy) and integer(iteration_number). (default: None)

    Attributes
    ----------
    convergence_tolerance : float
        Tolerance specifying convergence.
    convergence_level : float
        Number of times the tolerance should be undershot before exiting.
    iteration_limit : integer
        Maximum number of iterations performed.
    reset_count : integer
        Number of iterations after which to restart; i.e., forget previous
        conjugated directions.
    preconditioner : function
        The user can provide a function which transforms the variables of the 
        system to make the converge more favorable.
    callback : function
        Function f(energy, iteration_number) specified by the user to print 
        iteration number and energy value at every iteration step. It accepts 
        an Energy object(energy) and integer(iteration_number).
    
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

        if preconditioner is None:
            preconditioner = lambda z: z

        self.preconditioner = preconditioner
        self.callback = callback

    def __call__(self, A, b, x0):
        """Runs the conjugate gradient minimization.

        Parameters
        ----------
        A : Operator
            Operator `A` applicable to a Field.
        b : Field
            Resulting Field of the operation `A(x)`.
        x0 : Field
            Starting guess for the minimization.

        Returns
        -------
        x : Field
            Latest `x` of the minimization.
        convergence : integer
            Latest convergence level indicating whether the minimization
            has converged or not.

        """
        r = b - A(x0)
        d = self.preconditioner(r)
        previous_gamma = r.dot(d)
        if previous_gamma == 0:
            self.logger.info("The starting guess is already perfect solution "
                             "for the inverse problem.")
            return x0, self.convergence_level+1
        norm_b = np.sqrt(b.dot(b))
        x = x0
        convergence = 0
        iteration_number = 1
        self.logger.info("Starting conjugate gradient.")

        while True:
            if self.callback is not None:
                self.callback(x, iteration_number)

            q = A(d)
            alpha = previous_gamma/d.dot(q)

            if not np.isfinite(alpha):
                self.logger.error("Alpha became infinite! Stopping.")
                return x0, 0

            x += d * alpha

            reset = False
            if alpha.real < 0:
                self.logger.warn("Positive definiteness of A violated!")
                reset = True
            if self.reset_count is not None:
                reset += (iteration_number % self.reset_count == 0)
            if reset:
                self.logger.info("Resetting conjugate directions.")
                r = b - A(x)
            else:
                r -= q * alpha

            s = self.preconditioner(r)
            gamma = r.dot(s)

            if gamma.real < 0:
                self.logger.warn("Positive definitness of preconditioner "
                                 "violated!")

            beta = max(0, gamma/previous_gamma)

            delta = np.sqrt(gamma)/norm_b

            self.logger.debug("Iteration : %08u   alpha = %3.1E   "
                              "beta = %3.1E   delta = %3.1E" %
                              (iteration_number,
                               np.real(alpha),
                               np.real(beta),
                               np.real(delta)))

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

        return x, convergence
