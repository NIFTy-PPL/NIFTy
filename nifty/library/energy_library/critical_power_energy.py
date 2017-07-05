from nifty.energies.energy import Energy
from nifty.library.operator_library import CriticalPowerCurvature,\
                                            SmoothnessOperator

from nifty.sugar import generate_posterior_sample
from nifty import Field, exp

class CriticalPowerEnergy(Energy):
    """The Energy of the power spectrum according to the critical filter.

    It describes the energy of the logarithmic amplitudes of the power spectrum of
    a Gaussian random field under reconstruction uncertainty with smoothness and
    inverse gamma prior. It is used to infer the correlation structure of a correlated signal.

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
        The spectral prior of the inverse gamma distribution. 1.0 corresponds to
        non-informative.
        default : 1.0
    q : float
        The cutoff parameter of the inverse gamma distribution. 0.0 corresponds to
        non-informative.
        default : 0.0
    sigma : float
        The parameter of the smoothness prior.
        default : ??? None? ???????
    logarithmic : boolean
        Whether smoothness acts on linear or logarithmic scale.
    samples : integer
        Number of samples used for the estimation of the uncertainty corrections.
        default : 3
    w : Field
        The contribution from the map with or without uncertainty. It is used
        to pass on the result of the costly sampling during the minimization.
        default : None
    inverter : ConjugateGradient
        The inversion strategy to invert the curvature and to generate samples.
        default : None
    """

    def __init__(self, position, m, D=None, alpha =1.0, q=0., sigma=0.,
                 logarithmic = True, samples=3, w=None, inverter=None):
        super(CriticalPowerEnergy, self).__init__(position = position)
        self.m = m
        self.D = D
        self.samples = samples
        self.sigma = sigma
        self.alpha = Field(self.position.domain, val=alpha)
        self.q = Field(self.position.domain, val=q)
        self.T = SmoothnessOperator(domain=self.position.domain[0], sigma=self.sigma,
                                    logarithmic=logarithmic)
        self.rho = self.position.domain[0].rho
        self.inverter = inverter
        self.w = w
        if self.w is None:
            self.w = self._calculate_w(self.m, self.D, self.samples)
        self.theta = (exp(-self.position) * (self.q + self.w / 2.))


    def at(self, position):
        return self.__class__(position, self.m, D=self.D,
                              alpha =self.alpha,
                              q=self.q,
                              sigma=self.sigma, w=self.w,
                              samples=self.samples)

    @property
    def value(self):
        energy = exp(-self.position).dot(self.q + self.w / 2., bare= True)
        energy += self.position.dot(self.alpha - 1. + self.rho / 2., bare=True)
        energy += 0.5 * self.position.dot(self.T(self.position))
        return energy.real

    @property
    def gradient(self):
        gradient = - self.theta.weight(-1)
        gradient += (self.alpha - 1. + self.rho / 2.).weight(-1)
        gradient +=  self.T(self.position)
        gradient.val = gradient.val.real
        return gradient

    @property
    def curvature(self):
        curvature = CriticalPowerCurvature(theta=self.theta.weight(-1), T=self.T, inverter=self.inverter)
        return curvature

    def _calculate_w(self, m, D, samples):
        w = Field(domain=self.position.domain, val=0. , dtype=m.dtype)
        if D is not None:
            for i in range(samples):
                posterior_sample = generate_posterior_sample(m, D, inverter = self.inverter)
                projected_sample = posterior_sample.power_analyze(
                    logarithmic=self.position.domain[0].config["logarithmic"],
                    nbin= self.position.domain[0].config["nbin"])
                w += (projected_sample) * self.rho
            w /= float(samples)
        else:
            w = m.power_analyze(
                    logarithmic=self.position.domain[0].config["logarithmic"],
                        nbin=self.position.domain[0].config["nbin"])
            w *= self.rho

        return w



