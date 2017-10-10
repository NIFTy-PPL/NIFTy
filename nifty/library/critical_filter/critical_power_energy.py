from ...energies.energy import Energy
from ...operators.smoothness_operator import SmoothnessOperator
from ...operators.diagonal_operator import DiagonalOperator
from ...operators.linear_operator import LinearOperator
from ...operators.power_projection_operator import PowerProjection
from . import CriticalPowerCurvature
from ...memoization import memo
from ...minimization import ConjugateGradient

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
                 inverter=None, gradient=None, curvature=None):
        super(CriticalPowerEnergy, self).__init__(position=position,
                                                  gradient=gradient,
                                                  curvature=curvature)
        self.m = m
        self.D = D
        self.samples = samples
        self.alpha = Field(self.position.domain, val=alpha)
        self.q = Field(self.position.domain, val=q)
        self.T = SmoothnessOperator(domain=self.position.domain[0],
                                    strength=smoothness_prior,
                                    logarithmic=logarithmic)
        self.rho = self.position.domain[0].rho
        self.P = PowerProjection(domain=self.m.domain,target=self.position.domain)
        self._w = w if w is not None else None
        if inverter is None:
            preconditioner = DiagonalOperator(self._theta.domain,
                                              diagonal=self._theta,
                                              copy=False)
            inverter = ConjugateGradient(preconditioner=preconditioner)
        self._inverter = inverter
        self.one = Field(self.position.domain,val=1.)
        self.constants = self.one/2. + self.alpha - 1

    @property
    def inverter(self):
        return self._inverter

    # ---Mandatory properties and methods---

    def at(self, position):
        return self.__class__(position, self.m, D=self.D, alpha=self.alpha,
                              q=self.q, smoothness_prior=self.smoothness_prior,
                              logarithmic=self.logarithmic,
                              w=self.w, samples=self.samples,
                              inverter=self.inverter)

    @property
    @memo
    def value(self):
        energy = self.one.vdot(self._theta)
        energy += self.position.vdot(self.constants)
        energy += 0.5 * self.position.vdot(self._Tt)
        return energy.real

    @property
    @memo
    def gradient(self):
        gradient = -self._theta
        gradient += (self.constants)
        gradient += self._Tt
        gradient.val = gradient.val.real
        return gradient

    @property
    @memo
    def curvature(self):
        return CriticalPowerCurvature(theta=self._theta, T=self.T,
                                      inverter=self.inverter)

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
            self.logger.info("Initializing w")
            w = Field(domain=self.position.domain, val=0., dtype=self.m.dtype)
            if self.D is not None:
                for i in range(self.samples):
                    self.logger.info("Drawing sample %i" % i)
                    posterior_sample = generate_posterior_sample(
                                                            self.m, self.D)
                    w += self.P(abs(posterior_sample) ** 2)

                w /= float(self.samples)
            else:
                w = self.P(abs(self.m)**2)
            self._w = w
        return self._w

    @property
    @memo
    def _theta(self):
        return exp(-self.position) * (self.q + self.w / 2.)


    @property
    @memo
    def _Tt(self):
        return self.T(self.position)
