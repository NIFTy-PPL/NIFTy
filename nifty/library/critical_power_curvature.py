from ..operators.endomorphic_operator import EndomorphicOperator
from ..operators.diagonal_operator import DiagonalOperator


class CriticalPowerCurvature(EndomorphicOperator):
    """The curvature of the CriticalPowerEnergy.

    This operator implements the second derivative of the
    CriticalPowerEnergy used in some minimization algorithms or
    for error estimates of the power spectrum.

    Parameters
    ----------
    theta: Field,
        The map and inverse gamma prior contribution to the curvature.
    T : SmoothnessOperator,
        The smoothness prior contribution to the curvature.
    """

    def __init__(self, theta, T):
        super(CriticalPowerCurvature, self).__init__()
        self._theta = DiagonalOperator(theta)
        self._T = T

    @property
    def preconditioner(self):
        return self._theta.inverse_times

    def _times(self, x):
        return self._T(x) + self._theta(x)

    @property
    def domain(self):
        return self._theta.domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False
