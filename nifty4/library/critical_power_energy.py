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
from ..operators.smoothness_operator import SmoothnessOperator
from ..operators.power_projection_operator import PowerProjectionOperator
from .critical_power_curvature import CriticalPowerCurvature
from ..utilities import memo
from .. import Field, exp


class CriticalPowerEnergy(Energy):
    """The Energy of the power spectrum according to the critical filter.

    It describes the energy of the logarithmic amplitudes of the power spectrum
    of a Gaussian random field under reconstruction uncertainty with smoothness
    and inverse gamma prior. It is used to infer the correlation structure of a
    correlated signal.

    Parameters
    ----------
    position : Field,
        The current position of this energy. (Logarithm of power spectrum)
    m : Field,
        The map whose power spectrum has to be inferred
    D : EndomorphicOperator,
        The curvature of the Gaussian encoding the posterior covariance.
        If not specified, the map is assumed to be no reconstruction.
        default : None
    alpha : float
        The spectral prior of the inverse gamma distribution. 1.0 corresponds
        to non-informative.
        default : 1.0
    q : float
        The cutoff parameter of the inverse gamma distribution. 0.0 corresponds
        to non-informative.
        default : 0.0
    smoothness_prior : float
        Controls the strength of the smoothness prior
        default : 0.0
    logarithmic : boolean
        Whether smoothness acts on linear or logarithmic scale.
    samples : integer
        Number of samples used for the estimation of the uncertainty
        corrections.
        default : 3
    w : Field
        The contribution from the map with or without uncertainty. It is used
        to pass on the result of the costly sampling during the minimization.
        default : None
    inverter : ConjugateGradient
        The inversion strategy to invert the curvature and to generate samples.
        default : None
    """

    def __init__(self, position, m, D=None, alpha=1.0, q=0.,
                 smoothness_prior=0., logarithmic=True, samples=3, w=None,
                 inverter=None):
        super(CriticalPowerEnergy, self).__init__(position=position)
        self.m = m
        self.D = D
        self.samples = samples
        self.alpha = float(alpha)
        self.q = float(q)
        self._smoothness_prior = smoothness_prior
        self._logarithmic = logarithmic
        self.T = SmoothnessOperator(domain=self.position.domain[0],
                                    strength=smoothness_prior,
                                    logarithmic=logarithmic)
        self._inverter = inverter

        if w is None:
            P = PowerProjectionOperator(domain=self.m.domain,
                                        power_space=self.position.domain[0])
            if self.D is not None:
                w = Field.zeros(self.position.domain, dtype=self.m.dtype)
                for i in range(self.samples):
                    sample = self.D.generate_posterior_sample() + self.m
                    w += P(abs(sample)**2)

                w *= 1./self.samples
            else:
                w = P(abs(self.m)**2)
        self._w = w

        self._theta = exp(-self.position) * (self.q + self._w*0.5)
        Tt = self.T(self.position)

        energy = self._theta.sum()
        energy += self.position.sum() * (self.alpha-0.5)
        energy += 0.5*self.position.vdot(Tt)
        self._value = energy.real

        gradient = -self._theta
        gradient += self.alpha-0.5
        gradient += Tt
        self._gradient = gradient

    def at(self, position):
        return self.__class__(position, self.m, D=self.D, alpha=self.alpha,
                              q=self.q,
                              smoothness_prior=self._smoothness_prior,
                              logarithmic=self._logarithmic,
                              samples=self.samples, w=self._w,
                              inverter=self._inverter)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @property
    @memo
    def curvature(self):
        return CriticalPowerCurvature(theta=self._theta, T=self.T,
                                      inverter=self._inverter)
