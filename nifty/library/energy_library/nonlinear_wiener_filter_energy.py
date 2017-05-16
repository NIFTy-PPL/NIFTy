from nifty.energies.energy import Energy
from nifty.library.operator_library import NonlinearWienerFilterCurvature

class NonlinearWienerFilterEnergy(Energy):
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

    def __init__(self, position, d, R, N, S):
        super(NonlinearWienerFilterEnergy, self).__init__(position = position)
        self.d = d
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
        curvature = NonlinearWienerFilterCurvature(R=self.R,
                                                   N=self.N,
                                                   S=self.S,
                                                   position=self.position)
        return curvature

