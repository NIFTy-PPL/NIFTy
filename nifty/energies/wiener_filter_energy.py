from .energy import Energy
from nifty.operators.curvature_operators import WienerFilterCurvature

class WienerFilterEnergy(Energy):
    """The Energy for the Wiener filter.

    It describes the situation of linear measurement with
    Gaussian noise and Gaussain signal prior.

    Parameters
    ----------
    d : Field,
        the data.
    R : Operator,
        The response operator, describtion of the measurement process.
    N : EndomorphicOperator,
        The noise covariance in data space.
    S : EndomorphicOperator,
        The prior signal covariance in harmonic space.
    """

    def __init__(self, position, d, R, N, S):
        super(WienerFilterEnergy, self).__init__(position)
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
        return energy

    @property
    def gradient(self):
        gradient = self.S.inverse_times(self.position)
        gradient -= self.N.inverse_times(self.d - self.R(self.position))
        return gradient

    @property
    def curvature(self):
        curvature = WienerFilterCurvature(R=self.R, N=self.N, S=self.S)
        return curvature

    def analytic_solution(self):
        D_inverse = self.curvature()
        j = self.R.adjoint_times(self.N.inverse_times(self.d))
        new_position = D_inverse.inverse_times(j)
        return self.at(new_position)

