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

import numpy as np

from .descent_minimizer import DescentMinimizer
from .line_searching import LineSearchStrongWolfe


class VL_BFGS(DescentMinimizer):
    def __init__(self, line_searcher=LineSearchStrongWolfe(), callback=None,
                 convergence_tolerance=1E-4, convergence_level=3,
                 iteration_limit=None, max_history_length=5):

        super(VL_BFGS, self).__init__(
                                line_searcher=line_searcher,
                                callback=callback,
                                convergence_tolerance=convergence_tolerance,
                                convergence_level=convergence_level,
                                iteration_limit=iteration_limit)

        self.max_history_length = max_history_length

    def __call__(self, energy):
        self._information_store = None
        return super(VL_BFGS, self).__call__(energy)

    def get_descend_direction(self, energy):
        """Implementation of the Vector-free L-BFGS minimization scheme.

        Find the descent direction by using the inverse Hessian.
        Instead of storing the whole matrix, it stores only the last few
        updates, which are used to do operations requiring the inverse
        Hessian product. The updates are represented in a new basis to optimize
        the algorithm.

        Parameters
        ----------
        energy : Energy
            An instance of the Energy class which shall be minized. The
            position of `energy` is used as the starting point of minization.

        Returns
        -------
        descend_direction : Field
            Returns the descent direction.

        References
        ----------
        W. Chen, Z. Wang, J. Zhou, "Large-scale L-BFGS using MapReduce", 2014,
        Microsoft

        """

        x = energy.position
        gradient = energy.gradient
        # initialize the information store if it doesn't already exist
        try:
            self._information_store.add_new_point(x, gradient)
        except AttributeError:
            self._information_store = InformationStore(self.max_history_length,
                                                       x0=x,
                                                       gradient=gradient)

        b = self._information_store.b
        delta = self._information_store.delta

        descend_direction = delta[0] * b[0]
        for i in xrange(1, len(delta)):
            descend_direction += delta[i] * b[i]

        return descend_direction


class InformationStore(object):
    """Class for storing a list of past updates.

    Parameters
    ----------
    max_history_length : integer
        Maximum number of stored past updates.
    x0 : Field
        Initial position in variable space.
    gradient : Field
        Gradient at position x0.

    Attributes
    ----------
    max_history_length : integer
        Maximum number of stored past updates.
    s : List
        List of past position differences, which are Fields.
    y : List
        List of past gradient differences, which are Fields.
    last_x : Field
        Latest position in variable space.
    last_gradient : Field
        Gradient at latest position.
    k : integer
        Number of updates that have taken place
    _ss_store : dictionary
        Dictionary of scalar products between different elements of s.
    _sy_store : dictionary
        Dictionary of scalar products between elements of s and y.
    _yy_store : dictionary
        Dictionary of scalar products between different elements of y.

    """
    def __init__(self, max_history_length, x0, gradient):
        self.max_history_length = max_history_length
        self.s = [None]*max_history_length
        self.y = [None]*max_history_length
        self.last_x = x0.copy()
        self.last_gradient = gradient.copy()
        self.k = 0

        hl = max_history_length
        self._ss_store = np.empty((hl, hl), dtype=np.float64)
        self._sy_store = np.empty((hl, hl), dtype=np.float64)
        self._yy_store = np.empty((hl, hl), dtype=np.float64)

    @property
    def history_length(self):
        """Returns the number of currently stored updates.

        """
        return min(self.k, self.max_history_length)

    @property
    def b(self):
        """Combines s, y and gradient to form the new base vectors b.

        Returns
        -------
        result : List
            List of new basis vectors.

        """
        result = []
        m = self.history_length
        k = self.k

        s = self.s
        for i in xrange(m):
            result.append(s[(k-m+i) % m])

        y = self.y
        for i in xrange(m):
            result.append(y[(k-m+i) % m])

        result.append(self.last_gradient)

        return result

    @property
    def b_dot_b(self):
        """Generates the (2m+1) * (2m+1) scalar matrix.

        The i,j-th element of the matrix is a scalar product between the i-th
        and j-th base vector.

        Returns
        -------
        result : numpy.ndarray
            Scalar matrix.

        """
        m = self.history_length
        k = self.k
        result = np.empty((2*m+1, 2*m+1), dtype=np.float)

        # update the stores
        if (m > 0):
            k1 = (k-1) % m

        for i in xrange(m):
            kmi = (k-m+i) % m
            self._ss_store[kmi, k1] = self._ss_store[k1, kmi] \
                = self.s[kmi].vdot(self.s[k1])
            self._yy_store[kmi, k1] = self._yy_store[k1, kmi] \
                = self.y[kmi].vdot(self.y[k1])
            self._sy_store[kmi, k1] = self.s[kmi].vdot(self.y[k1])
        for j in xrange(m-1):
            kmj = (k-m+j) % m
            self._sy_store[k1, kmj] = self.s[k1].vdot(self.y[kmj])

        for i in xrange(m):
            kmi = (k-m+i) % m
            for j in xrange(m):
                kmj = (k-m+j) % m
                result[i, j] = self._ss_store[kmi, kmj]
                result[i, m+j] = result[m+j, i] = self._sy_store[kmi, kmj]
                result[m+i, m+j] = self._yy_store[kmi, kmj]

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
        delta : List
            List of the new scalar coefficients (deltas).

        """
        m = self.history_length
        b_dot_b = self.b_dot_b

        delta = np.zeros(2*m+1, dtype=np.float)
        delta[2*m] = -1

        alpha = np.empty(m, dtype=np.float)

        for j in xrange(m-1, -1, -1):
            delta_b_b = sum([delta[l] * b_dot_b[l, j] for l in xrange(2*m+1)])
            alpha[j] = delta_b_b/b_dot_b[j, m+j]
            delta[m+j] -= alpha[j]

        for i in xrange(2*m+1):
            delta[i] *= b_dot_b[m-1, 2*m-1]/b_dot_b[2*m-1, 2*m-1]

        for j in xrange(m):
            delta_b_b = sum([delta[l]*b_dot_b[m+j, l] for l in xrange(2*m+1)])
            beta = delta_b_b/b_dot_b[j, m+j]
            delta[j] += (alpha[j] - beta)

        return delta

    def add_new_point(self, x, gradient):
        """Updates the s list and y list.

        Calculates the new position and gradient differences and adds them to
        the respective list.

        """
        m = self.max_history_length
        self.s[self.k % m] = x - self.last_x
        self.y[self.k % m] = gradient - self.last_gradient

        self.last_x = x.copy()
        self.last_gradient = gradient.copy()

        self.k += 1
