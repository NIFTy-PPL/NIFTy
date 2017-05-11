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

from .descent_minimizer import DescentMinimizer
from .line_searching import LineSearchStrongWolfe


class VL_BFGS(DescentMinimizer):
    def __init__(self, line_searcher=LineSearchStrongWolfe(), callback=None,
                 convergence_tolerance=1E-4, convergence_level=3,
                 iteration_limit=None, max_history_length=10):

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

    def _get_descend_direction(self, energy):
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

        norm = descend_direction.norm()
        if norm != 1:
            descend_direction /= norm
        return descend_direction


class InformationStore(object):
    def __init__(self, max_history_length, x0, gradient):
        self.max_history_length = max_history_length
        self.s = LimitedList(max_history_length)
        self.y = LimitedList(max_history_length)
        self.last_x = x0.copy()
        self.last_gradient = gradient.copy()
        self.k = 0

        self._ss_store = {}
        self._sy_store = {}
        self._yy_store = {}

    @property
    def history_length(self):
        return min(self.k, self.max_history_length)

    @property
    def b(self):
        result = []
        m = self.history_length
        k = self.k

        s = self.s
        for i in xrange(m):
            result.append(s[k-m+i])

        y = self.y
        for i in xrange(m):
            result.append(y[k-m+i])

        result.append(self.last_gradient)

        return result

    @property
    def b_dot_b(self):
        m = self.history_length
        k = self.k
        result = np.empty((2*m+1, 2*m+1), dtype=np.float)

        for i in xrange(m):
            for j in xrange(m):
                result[i, j] = self.ss_store(k-m+i, k-m+j)

                sy_ij = self.sy_store(k-m+i, k-m+j)
                result[i, m+j] = sy_ij
                result[m+j, i] = sy_ij

                result[m+i, m+j] = self.yy_store(k-m+i, k-m+j)

            sgrad_i = self.sgrad_store(k-m+i)
            result[2*m, i] = sgrad_i
            result[i, 2*m] = sgrad_i

            ygrad_i = self.ygrad_store(k-m+i)
            result[2*m, m+i] = ygrad_i
            result[m+i, 2*m] = ygrad_i

        result[2*m, 2*m] = self.gradgrad_store()

        return result

    @property
    def delta(self):
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

        for j in xrange(m-1, -1, -1):
            delta_b_b = sum([delta[l]*b_dot_b[m+j, l] for l in xrange(2*m+1)])
            beta = delta_b_b/b_dot_b[j, m+j]
            delta[j] += (alpha[j] - beta)

        return delta

    def ss_store(self, i, j):
        key = tuple(sorted((i, j)))
        if key not in self._ss_store:
            self._ss_store[key] = self.s[i].dot(self.s[j])
        return self._ss_store[key]

    def sy_store(self, i, j):
        key = (i, j)
        if key not in self._sy_store:
            self._sy_store[key] = self.s[i].dot(self.y[j])
        return self._sy_store[key]

    def yy_store(self, i, j):
        key = tuple(sorted((i, j)))
        if key not in self._yy_store:
            self._yy_store[key] = self.y[i].dot(self.y[j])
        return self._yy_store[key]

    def sgrad_store(self, i):
        return self.s[i].dot(self.last_gradient)

    def ygrad_store(self, i):
        return self.y[i].dot(self.last_gradient)

    def gradgrad_store(self):
        return self.last_gradient.dot(self.last_gradient)

    def add_new_point(self, x, gradient):
        self.k += 1

        new_s = x - self.last_x
        self.s.add(new_s)

        new_y = gradient - self.last_gradient
        self.y.add(new_y)

        self.last_x = x.copy()
        self.last_gradient = gradient.copy()


class LimitedList(object):
    def __init__(self, history_length):
        self.history_length = int(history_length)
        self._offset = 0
        self._storage = []

    def __getitem__(self, index):
        return self._storage[index-self._offset]

    def add(self, value):
        if len(self._storage) == self.history_length:
            self._storage.pop(0)
            self._offset += 1
        self._storage.append(value)
