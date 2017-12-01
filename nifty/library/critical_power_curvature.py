from ..operators import EndomorphicOperator, InversionEnabler, DiagonalOperator


class CriticalPowerCurvature(InversionEnabler, EndomorphicOperator):
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

    def __init__(self, theta, T, inverter):
        EndomorphicOperator.__init__(self)
        self._theta = DiagonalOperator(theta)
        InversionEnabler.__init__(self, inverter, self._theta.inverse_times)
        self._T = T

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
