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

import abc
from nifty.nifty_meta import NiftyMeta

import numpy as np

from keepers import Loggable

from .line_searching import LineSearchStrongWolfe


class DescentMinimizer(Loggable, object):
    """ A base class used by gradient methods to find a local minimum.

    Descent minimization methods are used to find a local minimum of a scalar
    function by following a descent direction. This class implements the
    minimization procedure once a descent direction is known. The descent
    direction has to be implemented separately.

    Parameters
    ----------
    line_searcher : callable *optional*
        Function which infers the step size in the descent direction
        (default : LineSearchStrongWolfe()).
    callback : callable *optional*
        Function f(energy, iteration_number) supplied by the user to perform
        in-situ analysis at every iteration step. When being called the
        current energy and iteration_number are passed. (default: None)
    convergence_tolerance : float *optional*
        Tolerance specifying the case of convergence. (default: 1E-4)
    convergence_level : integer *optional*
        Number of times the tolerance must be undershot before convergence
        is reached. (default: 3)
    iteration_limit : integer *optional*
        Maximum number of iterations performed (default: None).

    Attributes
    ----------
    convergence_tolerance : float
        Tolerance specifying the case of convergence.
    convergence_level : integer
        Number of times the tolerance must be undershot before convergence
        is reached. (default: 3)
    iteration_limit : integer
        Maximum number of iterations performed.
    line_searcher : LineSearch
        Function which infers the optimal step size for functional minization
        given a descent direction.
    callback : function
        Function f(energy, iteration_number) supplied by the user to perform
        in-situ analysis at every iteration step. When being called the
        current energy and iteration_number are passed.

    Notes
    ------
    The callback function can be used to externally stop the minimization by
    raising a `StopIteration` exception.
    Check `get_descent_direction` of a derived class for information on the
    concrete minization scheme.

    """

    __metaclass__ = NiftyMeta

    def __init__(self, line_searcher=LineSearchStrongWolfe(), callback=None,
                 convergence_tolerance=1E-4, convergence_level=3,
                 iteration_limit=None):

        self.convergence_tolerance = np.float(convergence_tolerance)
        self.convergence_level = np.int(convergence_level)

        if iteration_limit is not None:
            iteration_limit = int(iteration_limit)
        self.iteration_limit = iteration_limit

        self.line_searcher = line_searcher
        self.callback = callback

    def __call__(self, energy):
        """ Performs the minimization of the provided Energy functional.

        Parameters
        ----------
        energy : Energy object
           Energy object which provides value, gradient and curvature at a
           specific position in parameter space.

        Returns
        -------
        energy : Energy object
            Latest `energy` of the minimization.
        convergence : integer
            Latest convergence level indicating whether the minimization
            has converged or not.

        Note
        ----
        The minimization is stopped if
            * the callback function raises a `StopIteration` exception,
            * a perfectly flat point is reached,
            * according to the line-search the minimum is found,
            * the target convergence level is reached,
            * the iteration limit is reached.

        """

        convergence = 0
        f_k_minus_1 = None
        step_length = 0
        iteration_number = 1

        while True:
            if self.callback is not None:
                try:
                    self.callback(energy, iteration_number)
                except StopIteration:
                    self.logger.info("Minimization was stopped by callback "
                                     "function.")
                    break

            # compute the the gradient for the current location
            gradient = energy.gradient
            gradient_norm = gradient.vdot(gradient)

            # check if position is at a flat point
            if gradient_norm == 0:
                self.logger.info("Reached perfectly flat point. Stopping.")
                convergence = self.convergence_level+2
                break

            # current position is encoded in energy object
            descend_direction = self.get_descend_direction(energy)

            # compute the step length, which minimizes energy.value along the
            # search direction
            step_length, f_k, new_energy = \
                self.line_searcher.perform_line_search(
                                               energy=energy,
                                               pk=descend_direction,
                                               f_k_minus_1=f_k_minus_1)
            f_k_minus_1 = energy.value
            energy = new_energy

            # check convergence
            delta = abs(gradient).max() * (step_length/gradient_norm)
            self.logger.debug("Iteration : %08u   step_length = %3.1E   "
                              "delta = %3.1E" %
                              (iteration_number, step_length, delta))
            if delta == 0:
                convergence = self.convergence_level + 2
                self.logger.info("Found minimum according to line-search. "
                                 "Stopping.")
                break
            elif delta < self.convergence_tolerance:
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

            iteration_number += 1

        return energy, convergence

    @abc.abstractmethod
    def get_descend_direction(self, energy):
        raise NotImplementedError
