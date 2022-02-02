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
# Copyright(C) 2013-2019, 2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..logger import logger
from ..utilities import NiftyMeta


class LineEnergy:
    """Evaluates an underlying Energy along a certain line direction.

    Given an Energy class and a line direction, its position is parametrized by
    a scalar step size along the descent direction relative to a zero point.

    Parameters
    ----------
    line_position : float
        Defines the full spatial position of this energy via
        self.energy.position = zero_point + line_position*line_direction
    energy : Energy
        The Energy object which will be evaluated along the given direction.
    line_direction : :class:`nifty8.field.Field`
        Direction used for line evaluation. Does not have to be normalized.
    offset :  float *optional*
        Indirectly defines the zero point of the line via the equation
        energy.position = zero_point + offset*line_direction
        (default : 0.).

    Notes
    -----
    The LineEnergy is used in minimization schemes in order perform line
    searches. It describes an underlying Energy which is restricted along one
    direction, only requiring the step size parameter to determine a new
    position.
    """

    def __init__(self, line_position, energy, line_direction, offset=0.):
        self._line_position = float(line_position)
        self._line_direction = line_direction

        if self._line_position == float(offset):
            self._energy = energy
        else:
            pos = energy.position \
                + (self._line_position-float(offset))*self._line_direction
            self._energy = energy.at(position=pos)

    def at(self, line_position):
        """Returns LineEnergy at new position, memorizing the zero point.

        Parameters
        ----------
        line_position : float
            Parameter for the new position on the line direction.

        Returns
        -------
            LineEnergy object at new position with same zero point as `self`.

        """

        return LineEnergy(line_position, self._energy, self._line_direction,
                          offset=self._line_position)

    @property
    def energy(self):
        """
        Energy : The underlying Energy object
        """
        return self._energy

    @property
    def value(self):
        """
        float : The value of the energy functional at given `position`.
        """
        return self._energy.value

    @property
    def directional_derivative(self):
        """
        float : The directional derivative at the given `position`.
        """
        res = self._energy.gradient.s_vdot(self._line_direction)
        if abs(res.imag) / max(abs(res.real), 1.) > 1e-12:
            from ..logger import logger
            logger.warning("directional derivative has non-negligible "
                           "imaginary part: {}".format(res))
        return res.real


class LineSearch(metaclass=NiftyMeta):
    """Class for finding a step size that satisfies the strong Wolfe
    conditions.

    Algorithm contains two stages. It begins with a trial step length and
    keeps increasing it until it finds an acceptable step length or an
    interval. If it does not satisfy the Wolfe conditions, it performs the Zoom
    algorithm (second stage). By interpolating it decreases the size of the
    interval until an acceptable step length is found.

    Parameters
    ----------
    preferred_initial_step_size : float, optional
        Newton-based methods should intialize this to 1.
    c1 : float
        Parameter for Armijo condition rule. Default: 1e-4.
    c2 : float
        Parameter for curvature condition rule. Default: 0.9.
    max_step_size : float
        Maximum step allowed in to be made in the descent direction.
        Default: 1e30.
    max_iterations : int, optional
        Maximum number of iterations performed by the line search algorithm.
        Default: 100.
    max_zoom_iterations : int, optional
        Maximum number of iterations performed by the zoom algorithm.
        Default: 100.
    """

    def __init__(self, preferred_initial_step_size=None, c1=1e-4, c2=0.9,
                 max_step_size=1e30, max_iterations=100,
                 max_zoom_iterations=100):

        self.preferred_initial_step_size = preferred_initial_step_size
        self.c1 = float(c1)
        self.c2 = float(c2)
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
        energy : Energy
            Energy object from which we will calculate the energy and the
            gradient at a specific point.
        pk : :class:`nifty8.field.Field`
            Vector pointing into the search direction.
        f_k_minus_1 : float, optional
            Value of the fuction (which is being minimized) at the k-1
            iteration of the line search procedure. Default: None.

        Returns
        -------
        Energy
            The new Energy object on the new position.
        bool
            whether the line search was considered successful or not
        """
        le_0 = LineEnergy(0., energy, pk, 0.)

        maxstepsize = energy.longest_step(pk)
        if maxstepsize is None:
            maxstepsize = self.max_step_size
        maxstepsize = min(maxstepsize, self.max_step_size)

        # initialize the zero phis
        old_phi_0 = f_k_minus_1
        phi_0 = le_0.value
        phiprime_0 = le_0.directional_derivative
        if phiprime_0 == 0:
            logger.warning(
                "Directional derivative is zero; assuming convergence")
            return energy, False
        if phiprime_0 > 0:
            logger.error("Error: search direction is not a descent direction")
            return energy, False

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
        alpha1 = min(alpha1, 0.99*maxstepsize)

        # start the minimization loop
        iteration_number = 0
        while iteration_number < self.max_iterations:
            iteration_number += 1
            if alpha1 == 0:
                return le_0.energy, False

            try:
                le_alpha1 = le_0.at(alpha1)
                phi_alpha1 = le_alpha1.value
            except FloatingPointError:  # backtrack
                alpha1 = (alpha0+alpha1)/2
                continue  # next iteration

            if np.isnan(phi_alpha1) or np.abs(phi_alpha1) > 1e100:  # also backtrack
                alpha1 = (alpha0+alpha1)/2
                continue  # next iteration

            if (phi_alpha1 > phi_0 + self.c1*alpha1*phiprime_0) or \
               ((phi_alpha1 >= phi_alpha0) and (iteration_number > 1)):
                return self._zoom(alpha0, alpha1, phi_0, phiprime_0,
                                  phi_alpha0, phiprime_alpha0, phi_alpha1,
                                  le_0)

            phiprime_alpha1 = le_alpha1.directional_derivative
            if abs(phiprime_alpha1) <= -self.c2*phiprime_0:
                return le_alpha1.energy, True

            if phiprime_alpha1 >= 0:
                return self._zoom(alpha1, alpha0, phi_0, phiprime_0,
                                  phi_alpha1, phiprime_alpha1, phi_alpha0,
                                  le_0)

            # update alphas
            alpha0, alpha1 = alpha1, min(2*alpha1, maxstepsize)
            if alpha1 == maxstepsize:
                logger.warning("max step size reached")
                return le_alpha1.energy, False

            phi_alpha0 = phi_alpha1
            phiprime_alpha0 = phiprime_alpha1

        logger.warning("max iterations reached")
        return le_alpha1.energy, False

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
        Energy
            The new Energy object on the new position.
        """
        cubic_delta = 0.2  # cubic interpolant checks
        quad_delta = 0.1  # quadratic interpolant checks
        alpha_recent = None
        phi_recent = None

        if phi_lo > phi_0 + self.c1*alpha_lo*phiprime_0:
            raise ValueError("inconsistent data")
        if phiprime_lo*(alpha_hi-alpha_lo) >= 0.:
            raise ValueError("inconsistent data")
        for i in range(self.max_zoom_iterations):
            # myassert(phi_lo <= phi_0 + self.c1*alpha_lo*phiprime_0)
            # myassert(phiprime_lo*(alpha_hi-alpha_lo)<0.)
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
                    return le_alphaj.energy, True
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
            logger.warning(
                "The line search algorithm (zoom) did not converge.")
            return le_alphaj.energy, False

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
