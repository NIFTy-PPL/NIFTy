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

from __future__ import division
from builtins import range
import numpy as np
from .descent_minimizer import DescentMinimizer
from .line_search_strong_wolfe import LineSearchStrongWolfe
from .. import dobj


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
                p -= alpha[idx]*y[idx]
            idx = (k-1) % maxhist
            fact = s[idx].vdot(y[idx]) / y[idx].vdot(y[idx])
            if fact <= 0.:
                dobj.mprint("L-BFGS curvature not positive definite!")
            p *= fact
            for i in range(k-nhist, k):
                idx = i % maxhist
                beta = y[idx].vdot(p) / s[idx].vdot(y[idx])
                p += (alpha[idx]-beta)*s[idx]
        self._lastx = x
        self._lastgrad = gradient
        self._k += 1
        return p
