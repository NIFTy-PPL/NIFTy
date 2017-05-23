from nifty.energies.energy import Energy
from nifty.library.operator_library import CriticalPowerCurvature,\
                                            SmoothnessOperator
from nifty.sugar import generate_posterior_sample
from nifty import Field, exp

class CriticalPowerEnergy(Energy):
    """The Energy for the Gaussian lognormal case.

    It describes the situation of linear measurement  of a
    lognormal signal with Gaussian noise and Gaussain signal prior.

    Parameters
    ----------
    d : Field,
        the data.
    R : Operator,
        The nonlinear response operator, describtion of the measurement process.
    N : EndomorphicOperator,
        The noise covariance in data space.
    S : EndomorphicOperator,
        The prior signal covariance in harmonic space.
    """

    def __init__(self, position, m, D=None, alpha =1.0, q=0., sigma=0., w=None, samples=3):
        super(CriticalPowerEnergy, self).__init__(position = position)
        self.m = m
        self.D = D
        self.samples = samples
        self.sigma = sigma
        self.alpha = Field(self.position.domain, val=alpha)
        self.q = Field(self.position.domain, val=q)
        self.T = SmoothnessOperator(domain=self.position.domain[0], sigma=self.sigma)
        self.rho = self.position.domain[0].rho
        self.w = w
        if self.w is None:
            self.w = self._calculate_w(self.m, self.D, self.samples)
        self.theta = exp(-self.position) * (self.q + self.w / 2.)

    def at(self, position):
        return self.__class__(position, self.m, D=self.D,
                              alpha =self.alpha,
                              q=self.q,
                              sigma=self.sigma, w=self.w,
                              samples=self.samples)

    @property
    def value(self):
        energy = exp(-self.position).dot(self.q + self.w / 2.)
        energy += self.position.dot(self.alpha - 1 + self.rho / 2.)
        energy += 0.5 * self.position.dot(self.T(self.position))
        return energy.real

    @property
    def gradient(self):
        gradient = - self.theta
        gradient += self.alpha - 1 + self.rho / 2.
        gradient += self.T(self.position)
        gradient.val[0] = 0.
        return gradient

    @property
    def curvature(self):
        curvature = CriticalPowerCurvature(theta=self.theta, T = self.T)
        return curvature

    def _calculate_w(self, m, D, samples):
        w = Field(domain=self.position.domain, val=0. , dtype=m.dtype)
        if D is not None:
            for i in range(samples):
                posterior_sample = generate_posterior_sample(m, D)
                projected_sample = posterior_sample.project_power(
                    logarithmic=self.position.domain[0].config["logarithmic"],
                        decompose_power=False)
                w += projected_sample
            w /= float(samples)
        else:
            w = m.project_power(
                    logarithmic=self.position.domain[0].config["logarithmic"],
                        decompose_power=False)

        return w



