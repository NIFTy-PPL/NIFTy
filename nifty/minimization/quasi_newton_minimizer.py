# -*- coding: utf-8 -*-

import abc

import numpy as np

from keepers import Loggable

from .line_searching import LineSearchStrongWolfe


class QuasiNewtonMinimizer(object, Loggable):
    __metaclass__ = abc.ABCMeta

    def __init__(self, line_searcher=LineSearchStrongWolfe(), callback=None,
                 convergence_tolerance=1E-4, convergence_level=3,
                 iteration_limit=None):

        self.convergence_tolerance = np.float(convergence_tolerance)
        self.convergence_level = np.float(convergence_level)

        if iteration_limit is not None:
            iteration_limit = int(iteration_limit)
        self.iteration_limit = iteration_limit

        self.line_searcher = line_searcher
        self.callback = callback

    def __call__(self, energy):
        """
            Runs the steepest descent minimization.

            Parameters
            ----------
            x0 : field
                Starting guess for the minimization.
            alpha : scalar, *optional*
                Starting step width to be multiplied with normalized gradient
                (default: 1).
            tol : scalar, *optional*
                Tolerance specifying convergence; measured by maximal change in
                `x` (default: 1E-4).
            clevel : integer, *optional*
                Number of times the tolerance should be undershot before
                exiting (default: 8).
            self.iteration_limit : integer, *optional*
                Maximum number of iterations performed (default: 100,000).

            Returns
            -------
            x : field
                Latest `x` of the minimization.
            convergence : integer
                Latest convergence level indicating whether the minimization
                has converged or not.

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
            gradient_norm = gradient.dot(gradient)

            # check if position is at a flat point
            if gradient_norm == 0:
                self.logger.info("Reached perfectly flat point. Stopping.")
                convergence = self.convergence_level+2
                break

            current_position = energy.position
            descend_direction = self._get_descend_direction(current_position,
                                                            gradient)

            # compute the step length, which minimizes energy.value along the
            # search direction
            step_length, step_length = self.line_searcher.perform_line_search(
                                               energy=energy,
                                               pk=descend_direction,
                                               f_k_minus_1=f_k_minus_1)
            new_position = current_position + step_length * descend_direction
            new_energy = energy.at(new_position)

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
    def _get_descend_direction(self, gradient, gradient_norm):
        raise NotImplementedError
