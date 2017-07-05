from nifty.energies.energy import Energy
from nifty.energies.memoization import memo
from nifty.library.operator_library import WienerFilterCurvature

class WienerFilterEnergy(Energy):
    """The Energy for the Wiener filter.

    It describes the situation of linear measurement with
    Gaussian noise and Gaussain signal prior with known covariance.

    Parameters
    ----------
    position: Field,
        The current position.
    d : Field,
        the data.
    R : Operator,
        The response operator, describtion of the measurement process.
    N : EndomorphicOperator,
        The noise covariance in data space.
    S : EndomorphicOperator,
        The prior signal covariance in harmonic space.
    """

    def __init__(self, position, d, R, N, S, inverter=None):
        super(WienerFilterEnergy, self).__init__(position = position)
        self.d = d
        self.R = R
        self.N = N
        self.S = S
        self.inverter = inverter

    def at(self, position):
        return self.__class__(position, self.d, self.R, self.N, self.S)

    @property
    def value(self):
        residual = self._residual()
        energy = 0.5 * self.position.dot(self.S.inverse_times(self.position))
        energy += 0.5 * (residual).dot(self.N.inverse_times(residual))
        return energy.real

    @property
    def gradient(self):
        residual = self._residual()
        gradient = self.S.inverse_times(self.position)
        gradient -= self.R.adjoint_times(
                    self.N.inverse_times(residual))
        return gradient

    @property
    def curvature(self):
        curvature = WienerFilterCurvature(R=self.R, N=self.N, S=self.S, inverter=self.inverter)
        return curvature

    @memo
    def _residual(self):
        residual = self.d - self.R(self.position)
        return residual

