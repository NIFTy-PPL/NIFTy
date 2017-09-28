from ...energies.energy import Energy
from ...energies.memoization import memo
from ...minimization import ConjugateGradient

from . import WienerFilterCurvature


class WienerFilterEnergy(Energy):
    """The Energy for the Wiener filter.

    It covers the case of linear measurement with
    Gaussian noise and Gaussian signal prior with known covariance.

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

    def __init__(self, position, d, R, N, S, inverter=None,
                 gradient=None, curvature=None):
        super(WienerFilterEnergy, self).__init__(position=position,
                                                 gradient=gradient,
                                                 curvature=curvature)
        self.d = d
        self.R = R
        self.N = N
        self.S = S
        if inverter is None:
            inverter = ConjugateGradient(preconditioner=self.S.times)
        self._inverter = inverter

    @property
    def inverter(self):
        return self._inverter

    def at(self, position, gradient=None, curvature=None):
        return self.__class__(position=position, d=self.d, R=self.R, N=self.N,
                              S=self.S, inverter=self.inverter)

    @property
    @memo
    def value(self):
        energy = 0.5*self.position.vdot(self._Dx)
        energy -= self._j.vdot(self.position)
        return energy.real

    @property
    @memo
    def gradient(self):
        return self._Dx - self._j

    @property
    @memo
    def curvature(self):
        return WienerFilterCurvature(R=self.R, N=self.N, S=self.S,
                                     inverter=self.inverter)

    @property
    @memo
    def _Dx(self):
        return self.curvature(self.position)

    @property
    @memo
    def _j(self):
        return self.R.adjoint_times(self.N.inverse_times(self.d))
