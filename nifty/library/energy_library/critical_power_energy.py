from nifty.energies.energy import Energy
from nifty.library.operator_library import CriticalPowerCurvature
from nifty.sugar import generate_posterior_sample
from nifty import Field

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

    def __init__(self, position, m, D, alpha, beta, w = None, samples=3):
        super(CriticalPowerEnergy, self).__init__(position = position)
        self.al = d
        self.R = R
        self.N = N
        self.S = S


    def at(self, position):
        return self.__class__(position, self.d, self.R, self.N, self.S)

    @property
    def value(self):
        energy = 0.5 * self.position.dot(self.S.inverse_times(self.position))
        energy += 0.5 * (self.d - self.R(self.position)).dot(
            self.N.inverse_times(self.d - self.R(self.position)))
        return energy.real

    @property
    def gradient(self):
        gradient = self.S.inverse_times(self.position)
        gradient -= self.R.derived_adjoint_times(
                    self.N.inverse_times(self.d - self.R(self.position)), self.position)
        return gradient

    @property
    def curvature(self):
        curvature =CriticalPowerCurvature(R=self.R,
                                                   N=self.N,
                                                   S=self.S,
                                                   position=self.position)
        return curvature

    def _calculate_w(self, m, D, samples):
        w = Field(domain=self.position.domain, val=0)
        for i in range(samples):
            posterior_sample = generate_posterior_sample(m, D)
            projected_sample =posterior_sample.power_analyze()**2
            w += projected_sample
        return w / float(samples)



