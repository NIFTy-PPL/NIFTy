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

from ..field import Field, exp
from ..minimization.energy import Energy
from ..operators.diagonal_operator import DiagonalOperator
import numpy as np


class NoiseEnergy(Energy):
    def __init__(self, position, alpha, q, res_sample_list):
        super(NoiseEnergy, self).__init__(position)

        self.N = DiagonalOperator(exp(position))
        self.alpha = alpha
        self.q = q
        self.res_sample_list = res_sample_list

        self._gradient = None

        for s in res_sample_list:
            lh = .5 * s.vdot(self.N.inverse_times(s))
            grad = -.5 * self.N.inverse_times(s.conjugate()*s)
            if self._gradient is None:
                self._value = lh
                self._gradient = grad.copy()
            else:
                self._value += lh
                self._gradient += grad

        expmpos = exp(-position)
        self._value *= 1./len(res_sample_list)
#        possum = position.sum()
#        s1 = (alpha-1.)*possum if np.isscalar(alpha) \
#            else (alpha-1.).vdot(position)
#        s2 = q*expmpos.sum() if np.isscalar(q) else q.vdot(expmpos)
#        self._value += .5*possum + s1 + s2

#        self._gradient *= 1./len(res_sample_list)
#        self._gradient += (alpha-0.5) - q*expmpos
        alpha_field = Field(position.domain, val=alpha)
        q_field = Field(position.domain, val=q)
        self._value += .5 * self.position.sum()
        self._value += (alpha_field-1.).vdot(self.position) + \
            q_field.vdot(expmpos)

        self._gradient *= 1./len(self.res_sample_list)
        self._gradient += (alpha_field-0.5) - q_field*expmpos
        self._gradient.lock()

    def at(self, position):
        return self.__class__(position, self.alpha, self.q,
                              self.res_sample_list)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient
