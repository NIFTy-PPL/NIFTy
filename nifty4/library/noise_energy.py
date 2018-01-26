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

from .. import Field, exp
from ..operators.diagonal_operator import DiagonalOperator
from ..minimization.energy import Energy


class NoiseEnergy(Energy):
    def __init__(self, position, d, m, D, t, FFT, Instrument, nonlinearity,
                 alpha, q, Projection, munit=1., sunit=1., dunit=1., samples=3, sample_list=None,
                 inverter=None):
        super(NoiseEnergy, self).__init__(position=position)
        self.m = m
        self.D = D
        self.d = d
        self.N = DiagonalOperator(diagonal=dunit**2 * exp(self.position))
        self.t = t
        self.samples = samples
        self.FFT = FFT
        self.Instrument = Instrument
        self.nonlinearity = nonlinearity
        self.munit = munit
        self.sunit = sunit
        self.dunit = dunit

        self.alpha = alpha
        self.q = q
        self.Projection = Projection
        self.power = self.Projection.adjoint_times(munit * exp(0.5 * self.t))
        self.one = Field(self.position.domain, val=1.)
        if sample_list is None:
            if samples is None or samples == 0:
                sample_list = [m]
            else:
                sample_list = [D.generate_posterior_sample() + m
                               for _ in range(samples)]
        self.sample_list = sample_list
        self.inverter = inverter

        A = Projection.adjoint_times(munit * exp(.5*self.t)) # unit: munit
        map_s = FFT.inverse_times(A * m)

        self._gradient = None
        for sample in self.sample_list:
            map_s = FFT.inverse_times(A * sample)

            residual = self.d - self.Instrument(sunit * self.nonlinearity(map_s))
            lh = .5 * residual.vdot(self.N.inverse_times(residual))
            grad = -.5 * self.N.inverse_times(residual.conjugate() * residual)

            if self._gradient is None:
                self._value = lh
                self._gradient = grad.copy()
            else:
                self._value += lh
                self._gradient += grad

        self._value *= 1./len(self.sample_list)
        self._value += .5 * self.position.integrate() + (self.alpha - 1.).vdot(self.position) + self.q.vdot(exp(-self.position))

        self._gradient *= 1./len(self.sample_list)
        self._gradient += (self.alpha-0.5) - self.q * (exp(-self.position))

    def at(self, position):
        return self.__class__(
            position, self.d, self.m, self.D, self.t, self.FFT,
            self.Instrument, self.nonlinearity, self.alpha, self.q,
            self.Projection, munit=self.munit, sunit=self.sunit,
            dunit=self.dunit, sample_list=self.sample_list,
            samples=self.samples, inverter=self.inverter)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient
