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

from ..minimization.energy import Energy
from ..operators.sandwich_operator import SandwichOperator
from ..utilities import memo


class GaussianEnergy(Energy):
    def __init__(self, inp, mean=None, covariance=None):
        """
        inp: Model object

        value = 0.5 * s.vdot(s), i.e. a log-Gauss distribution with unit
        covariance
        """
        super(GaussianEnergy, self).__init__(inp.position)
        self._inp = inp
        self._mean = mean
        self._cov = covariance

    def at(self, position):
        return self.__class__(self._inp.at(position), self._mean, self._cov)

    @property
    @memo
    def residual(self):
        if self._mean is not None:
            return self._inp.value - self._mean
        return self._inp.value

    @property
    @memo
    def value(self):
        if self._cov is None:
            return .5 * self.residual.vdot(self.residual).real
        return .5 * self.residual.vdot(
            self._cov.inverse_times(self.residual)).real

    @property
    @memo
    def gradient(self):
        if self._cov is None:
            return self._inp.jacobian.adjoint_times(self.residual)
        return self._inp.jacobian.adjoint_times(
            self._cov.inverse_times(self.residual))

    @property
    @memo
    def curvature(self):
        if self._cov is None:
            return SandwichOperator.make(self._inp.jacobian, None)
        return SandwichOperator.make(self._inp.jacobian, self._cov.inverse)
