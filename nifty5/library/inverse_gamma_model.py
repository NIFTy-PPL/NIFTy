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
from scipy.stats import invgamma, norm

from ..compat import *
from ..field import Field
from ..models.model import Model
from ..multi.multi_field import MultiField
from ..operators.selection_operator import SelectionOperator
from ..sugar import makeOp
from ..utilities import memo


class InverseGammaModel(Model):
    def __init__(self, position, alpha, q, key):
        super(InverseGammaModel, self).__init__(position)
        self._alpha = alpha
        self._q = q
        self._key = key

    @classmethod
    def make(cls, actual_position, alpha, q, key):
        pos = cls.inverseIG(actual_position, alpha, q)
        mf = MultiField.from_dict({key: pos})
        return cls(mf, alpha, q, key)

    def at(self, position):
        return self.__class__(position, self._alpha, self._q, self._key)

    @property
    @memo
    def value(self):
        points = self.position[self._key].local_data
        # MR FIXME?!
        points = np.clip(points, None, 8.2)
        points = Field.from_local_data(self.position[self._key].domain, points)
        return self.IG(points, self._alpha, self._q)

    @property
    @memo
    def jacobian(self):
        u = self.position[self._key].local_data
        inner = norm.pdf(u)
        outer_inv = invgamma.pdf(invgamma.ppf(norm.cdf(u),
                                              self._alpha,
                                              scale=self._q),
                                 self._alpha, scale=self._q)
        # FIXME
        outer_inv = np.clip(outer_inv, 1e-20, None)
        outer = 1/outer_inv
        grad = Field.from_local_data(self.position[self._key].domain,
                                     inner*outer)
        grad = makeOp(MultiField.from_dict({self._key: grad},
                                           self.position._domain))
        return SelectionOperator(grad.target, self._key)*grad

    @staticmethod
    def IG(field, alpha, q):
        foo = invgamma.ppf(norm.cdf(field.local_data), alpha, scale=q)
        return Field.from_local_data(field.domain, foo)

    @staticmethod
    def inverseIG(u, alpha, q):
        res = norm.ppf(invgamma.cdf(u.local_data, alpha, scale=q))
        return Field.from_local_data(u.domain, res)
