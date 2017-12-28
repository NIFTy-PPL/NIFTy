from ..operators import EndomorphicOperator, InversionEnabler, DiagonalOperator


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

    def __init__(self, theta, T, inverter):
        super(CriticalPowerCurvature, self).__init__()
        theta = DiagonalOperator(theta)
        self._op = InversionEnabler(T+theta, inverter, theta.inverse_times)

    @property
    def capability(self):
        return self._op.capability

    def apply(self, x, mode):
        return self._op.apply(x, mode)

    @property
    def domain(self):
        return self._op.domain
