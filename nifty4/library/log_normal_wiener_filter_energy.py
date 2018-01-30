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
from ..utilities import memo
from .log_normal_wiener_filter_curvature import LogNormalWienerFilterCurvature
from ..sugar import create_composed_ht_operator
from ..field import exp


class LogNormalWienerFilterEnergy(Energy):
    """The Energy for the log-normal Wiener filter.

    It covers the case of linear measurement with
    Gaussian noise and Gaussain signal prior with known covariance.

    Parameters
    ----------
    position: Field,
       The current position.
    d: Field,
       the data.
    R: Operator,
       The response operator, describtion of the measurement process.
    N: EndomorphicOperator,
       The noise covariance in data space.
    S: EndomorphicOperator,
       The prior signal covariance in harmonic space.
    """

    def __init__(self, position, d, R, N, S, inverter, ht=None):
        super(LogNormalWienerFilterEnergy, self).__init__(position=position)
        self.d = d
        self.R = R
        self.N = N
        self.S = S
        self._inverter = inverter

        if ht is None:
            self._ht = create_composed_ht_operator(self.S.domain)
        else:
            self._ht = ht

        self._expp_sspace = exp(self._ht(self.position))

        Sp = self.S.inverse_times(self.position)
        expp = self._ht.adjoint_times(self._expp_sspace)
        Rexppd = self.R(expp) - self.d
        NRexppd = self.N.inverse_times(Rexppd)
        self._value = 0.5*(self.position.vdot(Sp) + Rexppd.vdot(NRexppd))
        exppRNRexppd = self._ht.adjoint_times(
            self._expp_sspace * self._ht(self.R.adjoint_times(NRexppd)))
        self._gradient = Sp + exppRNRexppd

    def at(self, position):
        return self.__class__(position, self.d, self.R, self.N, self.S,
                              self._inverter, self._ht)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @property
    @memo
    def curvature(self):
        return LogNormalWienerFilterCurvature(
            R=self.R, N=self.N, S=self.S, ht=self._ht,
            expp_sspace=self._expp_sspace, inverter=self._inverter)
