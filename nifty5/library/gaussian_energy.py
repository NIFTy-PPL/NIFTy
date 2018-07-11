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
from ..minimization.energy import Energy
from ..operators.sandwich_operator import SandwichOperator
from ..utilities import memo


# MR FIXME documentation incomplete
class GaussianEnergy(Energy):
    def __init__(self, inp, mean=None, covariance=None):
        """
        inp: Model object

        value = 0.5 * s.vdot(s), i.e. a log-Gauss distribution with unit
        covariance
        """
        super(GaussianEnergy, self).__init__(inp._position)
        self._inp = inp
        self._mean = mean
        self._cov = covariance

    def at(self, position):
        return self.__class__(self._inp.at(position), self._mean, self._cov)

    @property
    @memo
    def _residual(self):
        return self._inp.value if self._mean is None else \
            self._inp.value - self._mean

    @property
    @memo
    def _icovres(self):
        return self._residual if self._cov is None else \
            self._cov.inverse_times(self._residual)

    @property
    @memo
    def value(self):
        return .5 * self._residual.vdot(self._icovres).real

    @property
    @memo
    def gradient(self):
        return self._inp.jacobian.adjoint_times(self._icovres)

    @property
    @memo
    def metric(self):
        return SandwichOperator.make(
            self._inp.jacobian,
            None if self._cov is None else self._cov.inverse)
