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
                 alpha, q, Projection, samples=3, sample_list=None,
                 inverter=None):
        super(NoiseEnergy, self).__init__(position=position)
        self.m = m
        self.D = D
        self.d = d
        self.N = DiagonalOperator(diagonal=exp(self.position))
        self.t = t
        self.samples = samples
        self.FFT = FFT
        self.Instrument = Instrument
        self.nonlinearity = nonlinearity

        self.alpha = alpha
        self.q = q
        self.Projection = Projection
        self.power = self.Projection.adjoint_times(exp(0.5 * self.t))
        self.one = Field(self.position.domain, val=1.)
        if sample_list is None:
            if samples is None or samples == 0:
                sample_list = [m]
            else:
                sample_list = [D.generate_posterior_sample() + m
                               for _ in range(samples)]
        self.sample_list = sample_list
        self.inverter = inverter
        self._value, self._gradient = self._value_and_grad()

    def at(self, position):
        return self.__class__(
            position, self.d, self.m, self.D, self.t, self.FFT,
            self.Instrument, self.nonlinearity, self.alpha, self.q,
            self.Projection, sample_list=self.sample_list,
            samples=self.samples, inverter=self.inverter)

    def _value_and_grad(self):
        likelihood_gradient = None
        for sample in self.sample_list:
            residual = self.d - \
                self.Instrument(self.nonlinearity(
                    self.FFT.adjoint_times(self.power*sample)))
            lh = 0.5 * residual.vdot(self.N.inverse_times(residual))
            grad = -0.5 * self.N.inverse_times(residual.conjugate() * residual)
            if likelihood_gradient is None:
                likelihood = lh
                likelihood_gradient = grad.copy()
            else:
                likelihood += lh
                likelihood_gradient += grad

        likelihood = ((likelihood / float(len(self.sample_list))) +
                      0.5 * self.position.integrate() +
                      (self.alpha - 1.).vdot(self.position) +
                      self.q.vdot(exp(-self.position)))
        likelihood_gradient = (
            likelihood_gradient / float(len(self.sample_list)) +
            (self.alpha-0.5) -
            self.q * (exp(-self.position)))
        return likelihood, likelihood_gradient

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient
