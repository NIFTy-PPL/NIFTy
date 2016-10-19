# -*- coding: utf-8 -*-

from .quasi_newton_minimizer import QuasiNewtonMinimizer


class VL_BFGS(QuasiNewtonMinimizer):
    def _get_descend_direction(self, x, gradient):
        pass


class InformationStore(object):
    def __init__(self, history_length, x0, gradient):
        self.history_length = history_length
        self.s = LimitedList(history_length)
        self.y = LimitedList(history_length)
        self.last_x = x0
        self.last_gradient = gradient
        self.k = 0
        self.dot_matrix = {}

    @property
    def actual_history_length(self):
        return min(self.k, self.history_length)

    def add_new_point(self, x, gradient):
        self.k += 1

        new_s = x - self.last_x
        self.s.add(new_s)

        new_y = gradient - self.last_gradient
        self.y.add(new_y)

        k = self.k
        m = self.actual_history_length
        big_m = self.history_length

        # compute dot products
        for i in xrange(k-1, k-m-1, -1):
            # new_s with s
            key = (big_m+m, big_m+1+i)
            self.dot_matrix[key] = new_s.dot(self.s[i])

            # new_s with y
            key = (big_m+m, i+1)
            self.dot_matrix[key] = new_s.dot(self.y[i])

            # new_y with s
            if i != k-1:
                key = (big_m+1+i, k)
                self.dot_matrix[key] = new_y.dot(self.s[i])

            # new_y with y
            # actually key = (i+1, k) but the convention is that the first
            # index is larger than the second one
            key = (k, i+1)
            self.dot_matrix[key] = new_y.dot(self.y[i])

            # gradient with s
            key = (big_m+1+i, 0)
            self.dot_matrix[key] = gradient.dot(self.s[i])

            # gradient with y
            key = (i+1, 0)
            self.dot_matrix[key] = gradient.dot(self.y[i])

        # gradient with gradient
        key = (0, 0)
        self.dot_matrix[key] = gradient.dot(gradient)

        self.last_x = x
        self.last_gradient = gradient

        # TODO: remove old entries from dictionary


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
