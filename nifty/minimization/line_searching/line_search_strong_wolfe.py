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

import numpy as np

from .line_search import LineSearch


class LineSearchStrongWolfe(LineSearch):
    """
    Class for finding a step size that satisfies the strong Wolfe conditions.
    """

    def __init__(self, c1=1e-4, c2=0.9,
                 max_step_size=50, max_iterations=10,
                 max_zoom_iterations=10):

        """
        Parameters
        ----------

        f : callable f(x, *args)
            Objective function.

        fprime : callable f'(x, *args)
            Objective functions gradient.

        f_args : tuple (optional)
            Additional arguments passed to objective function and its
            derivation.

        c1 : float (optional)
            Parameter for Armijo condition rule.

        c2 : float (optional)
            Parameter for curvature condition rule.

        max_step_size : float (optional)
            Maximum step size
        """

        super(LineSearchStrongWolfe, self).__init__()

        self.c1 = np.float(c1)
        self.c2 = np.float(c2)
        self.max_step_size = max_step_size
        self.max_iterations = int(max_iterations)
        self.max_zoom_iterations = int(max_zoom_iterations)

    def perform_line_search(self, energy, pk, f_k_minus_1=None):
        self._set_line_energy(energy, pk, f_k_minus_1=f_k_minus_1)
        c1 = self.c1
        c2 = self.c2
        max_step_size = self.max_step_size
        max_iterations = self.max_iterations

        # initialize the zero phis
        old_phi_0 = self.f_k_minus_1
        energy_0 = self.line_energy.at(0)
        phi_0 = energy_0.value
        phiprime_0 = energy_0.gradient

        if phiprime_0 == 0:
            self.logger.warn("Flat gradient in search direction.")
            return 0., 0.

        # set alphas
        alpha0 = 0.
        if self.prefered_initial_step_size is not None:
            alpha1 = self.prefered_initial_step_size
        elif old_phi_0 is not None and phiprime_0 != 0:
            alpha1 = min(1.0, 1.01*2*(phi_0 - old_phi_0)/phiprime_0)
            if alpha1 < 0:
                alpha1 = 1.0
        else:
            alpha1 = 1.0

        # give the alpha0 phis the right init value
        phi_alpha0 = phi_0
        phiprime_alpha0 = phiprime_0

        # start the minimization loop
        for i in xrange(max_iterations):
            energy_alpha1 = self.line_energy.at(alpha1)
            phi_alpha1 = energy_alpha1.value
            if alpha1 == 0:
                self.logger.warn("Increment size became 0.")
                alpha_star = 0.
                phi_star = phi_0
                energy_star = energy_0
                break

            if (phi_alpha1 > phi_0 + c1*alpha1*phiprime_0) or \
               ((phi_alpha1 >= phi_alpha0) and (i > 1)):
                (alpha_star, phi_star, energy_star) = self._zoom(
                                                    alpha0, alpha1,
                                                    phi_0, phiprime_0,
                                                    phi_alpha0,
                                                    phiprime_alpha0,
                                                    phi_alpha1,
                                                    c1, c2)
                break

            phiprime_alpha1 = energy_alpha1.gradient
            if abs(phiprime_alpha1) <= -c2*phiprime_0:
                alpha_star = alpha1
                phi_star = phi_alpha1
                energy_star = energy_alpha1
                break

            if phiprime_alpha1 >= 0:
                (alpha_star, phi_star, energy_star) = self._zoom(
                                                    alpha1, alpha0,
                                                    phi_0, phiprime_0,
                                                    phi_alpha1,
                                                    phiprime_alpha1,
                                                    phi_alpha0,
                                                    c1, c2)
                break

            # update alphas
            alpha0, alpha1 = alpha1, min(2*alpha1, max_step_size)
            phi_alpha0 = phi_alpha1
            phiprime_alpha0 = phiprime_alpha1
            # phi_alpha1 = self._phi(alpha1)

        else:
            # max_iterations was reached
            alpha_star = alpha1
            phi_star = phi_alpha1
            energy_star = energy_alpha1
            self.logger.error("The line search algorithm did not converge.")

        # extract the full energy from the line_energy
        energy_star = energy_star.energy
        length_direction = pk.norm
        step_length = alpha_star * length_direction
        return step_length, phi_star, energy_star

    def _zoom(self, alpha_lo, alpha_hi, phi_0, phiprime_0,
              phi_lo, phiprime_lo, phi_hi, c1, c2):

        max_iterations = self.max_zoom_iterations
        # define the cubic and quadratic interpolant checks
        cubic_delta = 0.2  # cubic
        quad_delta = 0.1  # quadratic

        # initialize the most recent versions (j-1) of phi and alpha
        alpha_recent = 0
        phi_recent = phi_0

        for i in xrange(max_iterations):
            delta_alpha = alpha_hi - alpha_lo
            if delta_alpha < 0:
                a, b = alpha_hi, alpha_lo
            else:
                a, b = alpha_lo, alpha_hi

            # Try cubic interpolation
            if i > 0:
                cubic_check = cubic_delta * delta_alpha
                alpha_j = self._cubicmin(alpha_lo, phi_lo, phiprime_lo,
                                         alpha_hi, phi_hi,
                                         alpha_recent, phi_recent)
            # If cubic was not successful or not available, try quadratic
            if (i == 0) or (alpha_j is None) or (alpha_j > b - cubic_check) or\
               (alpha_j < a + cubic_check):
                quad_check = quad_delta * delta_alpha
                alpha_j = self._quadmin(alpha_lo, phi_lo, phiprime_lo,
                                        alpha_hi, phi_hi)
                # If quadratic was not successfull, try bisection
                if (alpha_j is None) or (alpha_j > b - quad_check) or \
                   (alpha_j < a + quad_check):
                    alpha_j = alpha_lo + 0.5*delta_alpha

            # Check if the current value of alpha_j is already sufficient
            energy_alphaj = self.line_energy.at(alpha_j)
            phi_alphaj = energy_alphaj.value

            # If the first Wolfe condition is not met replace alpha_hi
            # by alpha_j
            if (phi_alphaj > phi_0 + c1*alpha_j*phiprime_0) or\
               (phi_alphaj >= phi_lo):
                alpha_recent, phi_recent = alpha_hi, phi_hi
                alpha_hi, phi_hi = alpha_j, phi_alphaj
            else:
                phiprime_alphaj = energy_alphaj.gradient
                # If the second Wolfe condition is met, return the result
                if abs(phiprime_alphaj) <= -c2*phiprime_0:
                    alpha_star = alpha_j
                    phi_star = phi_alphaj
                    energy_star = energy_alphaj
                    break
                # If not, check the sign of the slope
                if phiprime_alphaj*delta_alpha >= 0:
                    alpha_recent, phi_recent = alpha_hi, phi_hi
                    alpha_hi, phi_hi = alpha_lo, phi_lo
                else:
                    alpha_recent, phi_recent = alpha_lo, phi_lo
                # Replace alpha_lo by alpha_j
                (alpha_lo, phi_lo, phiprime_lo) = (alpha_j, phi_alphaj,
                                                   phiprime_alphaj)

        else:
            alpha_star, phi_star, energy_star = \
                alpha_j, phi_alphaj, energy_alphaj
            self.logger.error("The line search algorithm (zoom) did not "
                              "converge.")

        return alpha_star, phi_star, energy_star

    def _cubicmin(self, a, fa, fpa, b, fb, c, fc):
        """
        Finds the minimizer for a cubic polynomial that goes through the
        points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
        If no minimizer can be found return None
        """
        # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

        with np.errstate(divide='raise', over='raise', invalid='raise'):
            try:
                C = fpa
                db = b - a
                dc = c - a
                denom = (db * dc) ** 2 * (db - dc)
                d1 = np.empty((2, 2))
                d1[0, 0] = dc ** 2
                d1[0, 1] = -db ** 2
                d1[1, 0] = -dc ** 3
                d1[1, 1] = db ** 3
                [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                                fc - fa - C * dc]).flatten())
                A /= denom
                B /= denom
                radical = B * B - 3 * A * C
                xmin = a + (-B + np.sqrt(radical)) / (3 * A)
            except ArithmeticError:
                return None
        if not np.isfinite(xmin):
            return None
        return xmin

    def _quadmin(self, a, fa, fpa, b, fb):
        """
        Finds the minimizer for a quadratic polynomial that goes through
        the points (a,fa), (b,fb) with derivative at a of fpa,
        """
        # f(x) = B*(x-a)^2 + C*(x-a) + D
        with np.errstate(divide='raise', over='raise', invalid='raise'):
            try:
                D = fa
                C = fpa
                db = b - a * 1.0
                B = (fb - D - C * db) / (db * db)
                xmin = a - C / (2.0 * B)
            except ArithmeticError:
                return None
        if not np.isfinite(xmin):
            return None
        return xmin
