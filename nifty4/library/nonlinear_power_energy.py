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

from ..sugar import exp
from ..minimization.energy import Energy
from ..operators.smoothness_operator import SmoothnessOperator
from ..operators.inversion_enabler import InversionEnabler
from ..utilities import memo


def _LinearizedPowerResponse(Instrument, nonlinearity, ht, Distributor, tau,
                             xi):
    power = exp(0.5*tau)
    position = ht(Distributor(power)*xi)
    linearization = nonlinearity.derivative(position)
    return 0.5*Instrument*linearization*ht*xi*Distributor*power


class NonlinearPowerEnergy(Energy):
    """The Energy of the power spectrum according to the critical filter.

    It describes the energy of the logarithmic amplitudes of the power spectrum
    of a Gaussian random field under reconstruction uncertainty with smoothness
    and inverse gamma prior. It is used to infer the correlation structure of a
    correlated signal. The smoothness prior operates on logarithmic scale, i.e.
    it prefers power-law-like power spectra.

    Parameters
    ----------
    position : Field
        The current position of this energy.
    xi : Field
        The excitation field.
    D : EndomorphicOperator
        The curvature of the Gaussian encoding the posterior covariance.
        If not specified, the map is assumed to be no reconstruction.
        default : None
    sigma : float
        The parameter of the smoothness prior. Needs to be positive. A bigger
        number means a stronger smoothness prior.
        default : 0
    samples : int
        Number of samples used for the estimation of the uncertainty
        corrections.
        default : 3
    """
    # MR FIXME: docstring incomplete and outdated
    def __init__(self, position, d, N, xi, D, ht, Instrument, nonlinearity,
                 Distributor, sigma=0., samples=3, xi_sample_list=None,
                 inverter=None):
        super(NonlinearPowerEnergy, self).__init__(position)
        self.xi = xi
        self.D = D
        self.d = d
        self.N = N
        self.T = SmoothnessOperator(domain=position.domain[0],
                                    strength=sigma, logarithmic=True)
        self.ht = ht
        self.Instrument = Instrument
        self.nonlinearity = nonlinearity
        self.Distributor = Distributor
        self.sigma = sigma
        if xi_sample_list is None:
            if samples is None or samples == 0:
                xi_sample_list = [xi]
            else:
                xi_sample_list = [D.draw_sample(from_inverse=True) + xi
                                  for _ in range(samples)]
        self.xi_sample_list = xi_sample_list
        self.inverter = inverter

        A = Distributor(exp(.5 * position))

        self._gradient = None
        for xi_sample in self.xi_sample_list:
            map_s = ht(A*xi_sample)
            LinR = _LinearizedPowerResponse(Instrument, nonlinearity, ht,
                                            Distributor, position, xi_sample)

            residual = d - Instrument(nonlinearity(map_s))
            tmp = N.inverse_times(residual)
            lh = 0.5 * residual.vdot(tmp)
            grad = LinR.adjoint_times(tmp)

            if self._gradient is None:
                self._value = lh
                self._gradient = grad.copy()
            else:
                self._value += lh
                self._gradient += grad

        self._value *= 1. / len(self.xi_sample_list)
        Tpos = self.T(position)
        self._value += 0.5 * position.vdot(Tpos)
        self._gradient *= -1. / len(self.xi_sample_list)
        self._gradient += Tpos
        self._gradient.lock()

    def at(self, position):
        return self.__class__(position, self.d, self.N, self.xi, self.D,
                              self.ht, self.Instrument, self.nonlinearity,
                              self.Distributor, sigma=self.sigma,
                              samples=len(self.xi_sample_list),
                              xi_sample_list=self.xi_sample_list,
                              inverter=self.inverter)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @property
    @memo
    def curvature(self):
        result = None
        for xi_sample in self.xi_sample_list:
            LinearizedResponse = _LinearizedPowerResponse(
                self.Instrument, self.nonlinearity, self.ht, self.Distributor,
                self.position, xi_sample)
            op = LinearizedResponse.adjoint*self.N.inverse*LinearizedResponse
            result = op if result is None else result + op
        result = result*(1./len(self.xi_sample_list)) + self.T
        return InversionEnabler(result, self.inverter)
