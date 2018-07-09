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
from ..compat import *
import numpy as np
from scipy.stats import invgamma, norm
from ..field import Field
from ..sugar import makeOp
from ..multi.multi_field import MultiField
from ..models.model import Model

from ..operators.selection_operator import SelectionOperator
from ..utilities import memo


class PointSources(Model):
    def __init__(self, position, alpha, q):
        super(PointSources, self).__init__(position)
        self._alpha = alpha
        self._q = q

    def at(self, position):
        return self.__class__(position, self._alpha, self._q)

    @property
    @memo
    def value(self):
        points = self.position['points'].local_data
        # MR FIXME?!
        points = np.clip(points, None, 8.2)
        points = Field.from_local_data(self.position['points'].domain, points)
        return self.IG(points, self._alpha, self._q)

    @property
    @memo
    def jacobian(self):
        u = self.position['points'].local_data
        inner = norm.pdf(u)
        outer_inv = invgamma.pdf(invgamma.ppf(norm.cdf(u),
                                              self._alpha,
                                              scale=self._q),
                                 self._alpha, scale=self._q)
        # FIXME
        outer_inv = np.clip(outer_inv, 1e-20, None)
        outer = 1/outer_inv
        grad = Field.from_local_data(self.position['points'].domain,
                                     inner*outer)
        grad = makeOp(MultiField.from_dict({"points": grad},
                                           self.position._domain))
        return SelectionOperator(grad.target, 'points')*grad

    @staticmethod
    def IG(field, alpha, q):
        foo = invgamma.ppf(norm.cdf(field.local_data), alpha, scale=q)
        return Field.from_local_data(field.domain, foo)

    # MR FIXME: is this function needed?
    @staticmethod
    def IG_prime(field, alpha, q):
        inner = norm.pdf(field.local_data)
        outer = invgamma.pdf(invgamma.ppf(norm.cdf(field.local_data), alpha,
                                          scale=q), alpha, scale=q)
        # # FIXME
        # outer = np.clip(outer, 1e-20, None)
        return Field.from_local_data(field.domain, inner/outer)

    # MR FIXME: why does this take an np.ndarray instead of a Field?
    @staticmethod
    def inverseIG(u, alpha, q):
        res = norm.ppf(invgamma.cdf(u, alpha, scale=q))
        # # FIXME
        # res = np.clip(res, 0, None)
        return res
