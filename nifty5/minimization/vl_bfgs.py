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

from __future__ import (absolute_import, division, print_function)
from builtins import *
import numpy as np
from .descent_minimizer import DescentMinimizer
from .line_search_strong_wolfe import LineSearchStrongWolfe


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
