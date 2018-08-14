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

from __future__ import absolute_import, division, print_function

import numpy as np
from ..compat import *
from ..logger import logger
from .line_search_strong_wolfe import LineSearchStrongWolfe
from .minimizer import Minimizer


class DescentMinimizer(Minimizer):
    """ A base class used by gradient methods to find a local minimum.

    Descent minimization methods are used to find a local minimum of a scalar
    function by following a descent direction. This class implements the
    minimization procedure once a descent direction is known. The descent
    direction has to be implemented separately.

    Parameters
    ----------
    controller : IterationController
        Object that decides when to terminate the minimization.
    line_searcher : callable *optional*
        Function which infers the step size in the descent direction
        (default : LineSearchStrongWolfe()).
    """

    def __init__(self, controller, line_searcher=LineSearchStrongWolfe()):
        self._controller = controller
        self.line_searcher = line_searcher

    def __call__(self, energy):
        """ Performs the minimization of the provided Energy functional.

        Parameters
        ----------
        energy : Energy
           Energy object which provides value, gradient and metric at a
           specific position in parameter space.

        Returns
        -------
        Energy
            Latest `energy` of the minimization.
        int
            Can be controller.CONVERGED or controller.ERROR

        Notes
        -----
        The minimization is stopped if
            * the controller returns controller.CONVERGED or controller.ERROR,
            * a perfectly flat point is reached,
            * according to the line-search the minimum is found,
        """
        f_k_minus_1 = None
        controller = self._controller
        status = controller.start(energy)
        if status != controller.CONTINUE:
            return energy, status

        while True:
            # check if position is at a flat point
            if energy.gradient_norm == 0:
                return energy, controller.CONVERGED

            # compute a step length that reduces energy.value sufficiently
            new_energy, success = self.line_searcher.perform_line_search(
                energy=energy, pk=self.get_descent_direction(energy),
                f_k_minus_1=f_k_minus_1)
            if not success:
                self.reset()

            f_k_minus_1 = energy.value

            if new_energy.value > energy.value:
                logger.error("Error: Energy has increased")
                return energy, controller.ERROR

            if new_energy.value == energy.value:
                logger.warning(
                    "Warning: Energy has not changed. Assuming convergence...")
                return new_energy, controller.CONVERGED

            energy = new_energy
            status = self._controller.check(energy)
            if status != controller.CONTINUE:
                return energy, status

    def reset(self):
        pass

    def get_descent_direction(self, energy):
        """ Calculates the next descent direction.

        Parameters
        ----------
        energy : Energy
            An instance of the Energy class which shall be minimized. The
            position of `energy` is used as the starting point of minimization.

        Returns
        -------
        Field
           The descent direction.
        """
        raise NotImplementedError


class SteepestDescent(DescentMinimizer):
    """ Implementation of the steepest descent minimization scheme.

    Also known as 'gradient descent'. This algorithm simply follows the
    functional's gradient for minimization.
    """

    def get_descent_direction(self, energy):
        return -energy.gradient


class RelaxedNewton(DescentMinimizer):
    """ Calculates the descent direction according to a Newton scheme.

    The descent direction is determined by weighting the gradient at the
    current parameter position with the inverse local metric.
    """

    def __init__(self, controller, line_searcher=None):
        if line_searcher is None:
            line_searcher = LineSearchStrongWolfe(
                preferred_initial_step_size=1.)
        super(RelaxedNewton, self).__init__(controller=controller,
                                            line_searcher=line_searcher)

    def get_descent_direction(self, energy):
        return -energy.metric.inverse_times(energy.gradient)


class L_BFGS(DescentMinimizer):
    def __init__(self, controller, line_searcher=LineSearchStrongWolfe(),
                 max_history_length=5):
        super(L_BFGS, self).__init__(controller=controller,
                                     line_searcher=line_searcher)
        self.max_history_length = max_history_length

    def __call__(self, energy):
        self.reset()
        return super(L_BFGS, self).__call__(energy)

    def reset(self):
        self._k = 0
        self._s = [None]*self.max_history_length
        self._y = [None]*self.max_history_length

    def get_descent_direction(self, energy):
        x = energy.position
        s = self._s
        y = self._y
        k = self._k
        maxhist = self.max_history_length
        gradient = energy.gradient

        nhist = min(k, maxhist)
        alpha = [None]*maxhist
        p = -gradient
        if k > 0:
            idx = (k-1) % maxhist
            s[idx] = x-self._lastx
            y[idx] = gradient-self._lastgrad
        if nhist > 0:
            for i in range(k-1, k-nhist-1, -1):
                idx = i % maxhist
                alpha[idx] = s[idx].vdot(p)/s[idx].vdot(y[idx])
                p = p - alpha[idx]*y[idx]
            idx = (k-1) % maxhist
            fact = s[idx].vdot(y[idx]) / y[idx].vdot(y[idx])
            if fact <= 0.:
                logger.error("L-BFGS curvature not positive definite!")
            p = p*fact
            for i in range(k-nhist, k):
                idx = i % maxhist
                beta = y[idx].vdot(p) / s[idx].vdot(y[idx])
                p = p + (alpha[idx]-beta)*s[idx]
        self._lastx = x
        self._lastgrad = gradient
        self._k += 1
        return p


class VL_BFGS(DescentMinimizer):
    """Implementation of the Vector-free L-BFGS minimization scheme.

    Find the descent direction by using the inverse Hessian.
    Instead of storing the whole matrix, it stores only the last few
    updates, which are used to do operations requiring the inverse
    Hessian product. The updates are represented in a new basis to optimize
    the algorithm.

    References
    ----------
    W. Chen, Z. Wang, J. Zhou, "Large-scale L-BFGS using MapReduce", 2014,
    Microsoft
    """

    def __init__(self, controller, line_searcher=LineSearchStrongWolfe(),
                 max_history_length=5):
        super(VL_BFGS, self).__init__(controller=controller,
                                      line_searcher=line_searcher)
        self.max_history_length = max_history_length

    def __call__(self, energy):
        self._information_store = None
        return super(VL_BFGS, self).__call__(energy)

    def reset(self):
        self._information_store = None

    def get_descent_direction(self, energy):
        x = energy.position
        gradient = energy.gradient
        # initialize the information store if it doesn't already exist
        try:
            self._information_store.add_new_point(x, gradient)
        except AttributeError:
            self._information_store = _InformationStore(
                self.max_history_length, x0=x, gradient=gradient)

        b = self._information_store.b
        delta = self._information_store.delta

        descent_direction = delta[0] * b[0]
        for i in range(1, len(delta)):
            descent_direction = descent_direction + delta[i]*b[i]

        return descent_direction


class _InformationStore(object):
    """Class for storing a list of past updates.

    Parameters
    ----------
    max_history_length : int
        Maximum number of stored past updates.
    x0 : Field
        Initial position in variable space.
    gradient : Field
        Gradient at position x0.

    Attributes
    ----------
    max_history_length : int
        Maximum number of stored past updates.
    s : List
        Circular buffer of past position differences, which are Fields.
    y : List
        Circular buffer of past gradient differences, which are Fields.
    last_x : Field
        Latest position in variable space.
    last_gradient : Field
        Gradient at latest position.
    k : int
        Number of updates that have taken place
    ss : numpy.ndarray
        2D circular buffer of scalar products between different elements of s.
    sy : numpy.ndarray
        2D circular buffer of scalar products between elements of s and y.
    yy : numpy.ndarray
        2D circular buffer of scalar products between different elements of y.
    """

    def __init__(self, max_history_length, x0, gradient):
        self.max_history_length = max_history_length
        self.s = [None]*max_history_length
        self.y = [None]*max_history_length
        self.last_x = x0
        self.last_gradient = gradient
        self.k = 0

        mmax = max_history_length
        self.ss = np.empty((mmax, mmax), dtype=np.float64)
        self.sy = np.empty((mmax, mmax), dtype=np.float64)
        self.yy = np.empty((mmax, mmax), dtype=np.float64)

    @property
    def history_length(self):
        """Returns the number of currently stored updates."""
        return min(self.k, self.max_history_length)

    @property
    def b(self):
        """Combines s, y and gradient to form the new base vectors b.

        Returns
        -------
        List
            List of new basis vectors.
        """
        result = []
        m = self.history_length
        mmax = self.max_history_length

        for i in range(m):
            result.append(self.s[(self.k-m+i) % mmax])

        for i in range(m):
            result.append(self.y[(self.k-m+i) % mmax])

        result.append(self.last_gradient)

        return result

    @property
    def b_dot_b(self):
        """Generates the (2m+1) * (2m+1) scalar matrix.

        The i,j-th element of the matrix is a scalar product between the i-th
        and j-th base vector.

        Returns
        -------
        numpy.ndarray
            Scalar matrix.
        """
        m = self.history_length
        mmax = self.max_history_length
        k = self.k
        result = np.empty((2*m+1, 2*m+1), dtype=np.float)

        # update the stores
        k1 = (k-1) % mmax
        for i in range(m):
            kmi = (k-m+i) % mmax
            self.ss[kmi, k1] = self.ss[k1, kmi] = self.s[kmi].vdot(self.s[k1])
            self.yy[kmi, k1] = self.yy[k1, kmi] = self.y[kmi].vdot(self.y[k1])
            self.sy[kmi, k1] = self.s[kmi].vdot(self.y[k1])
        for j in range(m-1):
            kmj = (k-m+j) % mmax
            self.sy[k1, kmj] = self.s[k1].vdot(self.y[kmj])

        for i in range(m):
            kmi = (k-m+i) % mmax
            for j in range(m):
                kmj = (k-m+j) % mmax
                result[i, j] = self.ss[kmi, kmj]
                result[i, m+j] = result[m+j, i] = self.sy[kmi, kmj]
                result[m+i, m+j] = self.yy[kmi, kmj]

            sgrad_i = self.s[kmi].vdot(self.last_gradient)
            result[2*m, i] = result[i, 2*m] = sgrad_i

            ygrad_i = self.y[kmi].vdot(self.last_gradient)
            result[2*m, m+i] = result[m+i, 2*m] = ygrad_i

        result[2*m, 2*m] = self.last_gradient.norm()
        return result

    @property
    def delta(self):
        """Calculates the new scalar coefficients (deltas).

        Returns
        -------
        List
            List of the new scalar coefficients (deltas).
        """
        m = self.history_length
        b_dot_b = self.b_dot_b

        delta = np.zeros(2*m+1, dtype=np.float)
        delta[2*m] = -1

        alpha = np.empty(m, dtype=np.float)

        for j in range(m-1, -1, -1):
            delta_b_b = sum([delta[l] * b_dot_b[l, j] for l in range(2*m+1)])
            alpha[j] = delta_b_b/b_dot_b[j, m+j]
            delta[m+j] -= alpha[j]

        for i in range(2*m+1):
            delta[i] *= b_dot_b[m-1, 2*m-1]/b_dot_b[2*m-1, 2*m-1]

        for j in range(m):
            delta_b_b = sum([delta[l]*b_dot_b[m+j, l] for l in range(2*m+1)])
            beta = delta_b_b/b_dot_b[j, m+j]
            delta[j] += (alpha[j] - beta)

        return delta

    def add_new_point(self, x, gradient):
        """Updates the s list and y list.

        Calculates the new position and gradient differences and enters them
        into the respective list.
        """
        mmax = self.max_history_length
        self.s[self.k % mmax] = x - self.last_x
        self.y[self.k % mmax] = gradient - self.last_gradient

        self.last_x = x
        self.last_gradient = gradient

        self.k += 1
