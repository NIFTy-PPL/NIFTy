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
from builtins import range
import numpy as np
from .line_search import LineSearch
from .line_energy import LineEnergy
from .. import dobj


class LineSearchStrongWolfe(LineSearch):
    """Class for finding a step size that satisfies the strong Wolfe conditions.

    Algorithm contains two stages. It begins with a trial step length and
    keeps increasing it until it finds an acceptable step length or an
    interval. If it does not satisfy the Wolfe conditions, it performs the Zoom
    algorithm (second stage). By interpolating it decreases the size of the
    interval until an acceptable step length is found.

    Parameters
    ----------
    c1 : float
        Parameter for Armijo condition rule. (Default: 1e-4)
    c2 : float
        Parameter for curvature condition rule. (Default: 0.9)
    max_step_size : float
        Maximum step allowed in to be made in the descent direction.
        (Default: 50)
    max_iterations : integer
        Maximum number of iterations performed by the line search algorithm.
        (Default: 10)
    max_zoom_iterations : integer
        Maximum number of iterations performed by the zoom algorithm.
        (Default: 10)

    Attributes
    ----------
    c1 : float
        Parameter for Armijo condition rule.
    c2 : float
        Parameter for curvature condition rule.
    max_step_size : float
        Maximum step allowed in to be made in the descent direction.
    max_iterations : integer
        Maximum number of iterations performed by the line search algorithm.
    max_zoom_iterations : integer
        Maximum number of iterations performed by the zoom algorithm.
    """

    def __init__(self, c1=1e-4, c2=0.9,
                 max_step_size=1000000000, max_iterations=100,
                 max_zoom_iterations=100):

        super(LineSearchStrongWolfe, self).__init__()

        self.c1 = np.float(c1)
        self.c2 = np.float(c2)
        self.max_step_size = max_step_size
        self.max_iterations = int(max_iterations)
        self.max_zoom_iterations = int(max_zoom_iterations)

    def perform_line_search(self, energy, pk, f_k_minus_1=None):
        """Performs the first stage of the algorithm.

        It starts with a trial step size and it keeps increasing it until it
        satisfies the strong Wolf conditions. It also performs the descent and
        returns the optimal step length and the new energy.

        Parameters
        ----------
        energy : Energy object
            Energy object from which we will calculate the energy and the
            gradient at a specific point.
        pk : Field
            Vector pointing into the search direction.
        f_k_minus_1 : float
            Value of the fuction (which is being minimized) at the k-1
            iteration of the line search procedure. (Default: None)

        Returns
        -------
        energy_star : Energy object
            The new Energy object on the new position.
        """
        le_0 = LineEnergy(0., energy, pk, 0.)

        # initialize the zero phis
        old_phi_0 = f_k_minus_1
        phi_0 = le_0.value
        phiprime_0 = le_0.directional_derivative
        if phiprime_0 == 0:
            dobj.mprint("Directional derivative is zero; assuming convergence")
            return energy
        if phiprime_0 > 0:
            dobj.mprint("Error: search direction is not a descent direction")
            raise ValueError("search direction must be a descent direction")

        # set alphas
        alpha0 = 0.
        phi_alpha0 = phi_0
        phiprime_alpha0 = phiprime_0

        if self.preferred_initial_step_size is not None:
            alpha1 = self.preferred_initial_step_size
        elif old_phi_0 is not None:
            alpha1 = min(1.0, 1.01*2*(phi_0 - old_phi_0)/phiprime_0)
            if alpha1 < 0:
                alpha1 = 1.0
        else:
            alpha1 = 1.0/pk.norm()

        # start the minimization loop
        iteration_number = 0
        while iteration_number < self.max_iterations:
            iteration_number += 1
            if alpha1 == 0:
                result_energy = le_0.energy
                break

            le_alpha1 = le_0.at(alpha1)
            phi_alpha1 = le_alpha1.value

            if (phi_alpha1 > phi_0 + self.c1*alpha1*phiprime_0) or \
               ((phi_alpha1 >= phi_alpha0) and (iteration_number > 1)):
                le_star = self._zoom(alpha0, alpha1, phi_0, phiprime_0,
                                     phi_alpha0, phiprime_alpha0, phi_alpha1,
                                     le_0)
                result_energy = le_star.energy
                break

            phiprime_alpha1 = le_alpha1.directional_derivative
            if abs(phiprime_alpha1) <= -self.c2*phiprime_0:
                result_energy = le_alpha1.energy
                break

            if phiprime_alpha1 >= 0:
                le_star = self._zoom(alpha1, alpha0, phi_0, phiprime_0,
                                     phi_alpha1, phiprime_alpha1, phi_alpha0,
                                     le_0)
                result_energy = le_star.energy
                break

            # update alphas
            alpha0, alpha1 = alpha1, min(2*alpha1, self.max_step_size)
            if alpha1 == self.max_step_size:
                return le_alpha1.energy

            phi_alpha0 = phi_alpha1
            phiprime_alpha0 = phiprime_alpha1
        else:
            dobj.mprint("max iterations reached")
            return le_alpha1.energy
        return result_energy

    def _zoom(self, alpha_lo, alpha_hi, phi_0, phiprime_0,
              phi_lo, phiprime_lo, phi_hi, le_0):
        """Performs the second stage of the line search algorithm.

        If the first stage was not successful then the Zoom algorithm tries to
        find a suitable step length by using bisection, quadratic, cubic
        interpolation.

        Parameters
        ----------
        alpha_lo : float
            A boundary for the step length interval.
            Fulfills Wolfe condition 1.
        alpha_hi : float
            The other boundary for the step length interval.
        phi_0 : float
            Value of the energy at the starting point of the line search
            algorithm.
        phiprime_0 : float
            directional derivative at the starting point of the line search
            algorithm.
        phi_lo : float
            Value of the energy if we perform a step of length alpha_lo in
            descent direction.
        phiprime_lo : float
            directional derivative at the new position if we perform a step of
            length alpha_lo in descent direction.
        phi_hi : float
            Value of the energy if we perform a step of length alpha_hi in
            descent direction.

        Returns
        -------
        energy_star : Energy object
            The new Energy object on the new position.
        """
        cubic_delta = 0.2  # cubic interpolant checks
        quad_delta = 0.1  # quadratic interpolant checks
        alpha_recent = None
        phi_recent = None

        assert phi_lo <= phi_0 + self.c1*alpha_lo*phiprime_0
        assert phiprime_lo*(alpha_hi-alpha_lo) < 0.
        for i in range(self.max_zoom_iterations):
            # assert phi_lo <= phi_0 + self.c1*alpha_lo*phiprime_0
            # assert phiprime_lo*(alpha_hi-alpha_lo)<0.
            delta_alpha = alpha_hi - alpha_lo
            a, b = min(alpha_lo, alpha_hi), max(alpha_lo, alpha_hi)

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
                # If quadratic was not successful, try bisection
                if (alpha_j is None) or (alpha_j > b - quad_check) or \
                   (alpha_j < a + quad_check):
                    alpha_j = alpha_lo + 0.5*delta_alpha

            # Check if the current value of alpha_j is already sufficient
            le_alphaj = le_0.at(alpha_j)
            phi_alphaj = le_alphaj.value

            # If the first Wolfe condition is not met replace alpha_hi
            # by alpha_j
            if (phi_alphaj > phi_0 + self.c1*alpha_j*phiprime_0) or \
               (phi_alphaj >= phi_lo):
                alpha_recent, phi_recent = alpha_hi, phi_hi
                alpha_hi, phi_hi = alpha_j, phi_alphaj
            else:
                phiprime_alphaj = le_alphaj.directional_derivative
                # If the second Wolfe condition is met, return the result
                if abs(phiprime_alphaj) <= -self.c2*phiprime_0:
                    return le_alphaj
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
            dobj.mprint("The line search algorithm (zoom) did not converge.")
            return le_alphaj

    def _cubicmin(self, a, fa, fpa, b, fb, c, fc):
        """Estimating the minimum with cubic interpolation.

        Finds the minimizer for a cubic polynomial that goes through the
        points (a,a), (b,fb), and (c,fc) with derivative at point a of fpa.
        If no minimizer can be found return None

        Parameters
        ----------
        a, fa, fpa : float
            abscissa, function value and derivative at first point
        b, fb : float
            abscissa and function value at second point
        c, fc : float
            abscissa and function value at third point

        Returns
        -------
        xmin : float
            Position of the approximated minimum.
        """
        with np.errstate(divide='raise', over='raise', invalid='raise'):
            try:
                C = fpa
                db = b - a
                dc = c - a
                denom = db * db * dc * dc * (db - dc)
                d1 = np.empty((2, 2))
                d1[0, 0] = dc * dc
                d1[0, 1] = -(db*db)
                d1[1, 0] = -(dc*dc*dc)
                d1[1, 1] = db*db*db
                [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                                fc - fa - C * dc]).ravel())
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
        """Estimating the minimum with quadratic interpolation.

        Finds the minimizer for a quadratic polynomial that goes through
        the points (a,fa), (b,fb) with derivative at point a of fpa.

        Parameters
        ----------
        a, fa, fpa : float
            abscissa, function value and derivative at first point
        b, fb : float
            abscissa and function value at second point

        Returns
        -------
        xmin : float
            Position of the approximated minimum.
        """
        with np.errstate(divide='raise', over='raise', invalid='raise'):
            try:
                db = b - a * 1.0
                B = (fb - fa - fpa * db) / (db * db)
                xmin = a - fpa / (2.0 * B)
            except ArithmeticError:
                return None
        if not np.isfinite(xmin):
            return None
        return xmin
