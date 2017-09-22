from ...energies.energy import Energy
from ...operators.smoothness_operator import SmoothnessOperator
from . import CriticalPowerCurvature
from ...energies.memoization import memo

from ...sugar import generate_posterior_sample
from ... import Field, exp


class CriticalPowerEnergy(Energy):
    """The Energy of the power spectrum according to the critical filter.

    It describes the energy of the logarithmic amplitudes of the power spectrum
    of a Gaussian random field under reconstruction uncertainty with smoothness
    and inverse gamma prior. It is used to infer the correlation structure of a
    correlated signal.

    Parameters
    ----------
    position : Field,
        The current position of this energy.
    m : Field,
        The map whichs power spectrum has to be inferred
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

    # ---Overwritten properties and methods---

    def __init__(self, position, m, D=None, alpha=1.0, q=0.,
                 smoothness_prior=0., logarithmic=True, samples=3, w=None,
                 inverter=None):
        super(CriticalPowerEnergy, self).__init__(position=position)
        self.m = m
        self.D = D
        self.samples = samples
        self.alpha = Field(self.position.domain, val=alpha)
        self.q = Field(self.position.domain, val=q)
        self.T = SmoothnessOperator(domain=self.position.domain[0],
                                    strength=smoothness_prior,
                                    logarithmic=logarithmic)
        self.rho = self.position.domain[0].rho
        self._w = w
        self._inverter = inverter

    # ---Mandatory properties and methods---

    def at(self, position):
        return self.__class__(position, self.m, D=self.D, alpha=self.alpha,
                              q=self.q, smoothness_prior=self.smoothness_prior,
                              logarithmic=self.logarithmic,
                              w=self.w, samples=self.samples,
                              inverter=self._inverter)

    @property
    def value(self):
        energy = self._theta.sum()
        energy += self.position.weight(-1).vdot(self._rho_prime)
        energy += 0.5 * self.position.vdot(self._Tt)
        return energy.real

    @property
    def gradient(self):
        gradient = -self._theta.weight(-1)
        gradient += (self._rho_prime).weight(-1)
        gradient += self._Tt
        gradient = gradient.real
        return gradient

    @property
    def curvature(self):
        curvature = CriticalPowerCurvature(theta=self._theta.weight(-1),
                                           T=self.T, inverter=self._inverter)
        return curvature

    # ---Added properties and methods---

    @property
    def logarithmic(self):
        return self.T.logarithmic

    @property
    def smoothness_prior(self):
        return self.T.strength

    @property
    def w(self):
        if self._w is None:
            # self.logger.info("Initializing w")
            w = Field(domain=self.position.domain, val=0., dtype=self.m.dtype)
            if self.D is not None:
                for i in range(self.samples):
                    # self.logger.info("Drawing sample %i" % i)
                    posterior_sample = generate_posterior_sample(
                                                            self.m, self.D)
                    projected_sample = posterior_sample.power_analyze(
                     binbounds=self.position.domain[0].binbounds)
                    w += (projected_sample) * self.rho
                w /= float(self.samples)
            else:
                w = self.m.power_analyze(
                     binbounds=self.position.domain[0].binbounds)
                w *= self.rho
            self._w = w
        return self._w

    @property
    @memo
    def _theta(self):
        return exp(-self.position) * (self.q + self.w / 2.)

    @property
    @memo
    def _rho_prime(self):
        return self.alpha - 1. + self.rho / 2.

    @property
    @memo
    def _Tt(self):
        return self.T(self.position)
