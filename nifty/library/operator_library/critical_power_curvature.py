from nifty.operators import EndomorphicOperator,\
                            InvertibleOperatorMixin,\
                            DiagonalOperator


class CriticalPowerCurvature(InvertibleOperatorMixin, EndomorphicOperator):
    """The curvature of the CriticalPowerEnergy.

    This operator implements the second derivative of the
    CriticalPowerEnergy used in some minimization algorithms or
    for error estimates of the powerspectrum.


    Parameters
    ----------
    theta: Field,
        The map and inverse gamma prior contribution to the curvature.
    T : SmoothnessOperator,
        The smoothness prior contribution to the curvature.
    """
    def __init__(self, theta, T, inverter=None, preconditioner=None):

        self.theta = DiagonalOperator(theta.domain, diagonal=theta)
        self.T = T
        if preconditioner is None:
            preconditioner = self.theta.inverse_times
        self._domain = self.theta.domain
        super(CriticalPowerCurvature, self).__init__(inverter=inverter,
                                                 preconditioner=preconditioner)
    @property
    def domain(self):
        return self._domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False

    # ---Added properties and methods---

    def _times(self, x, spaces):
        return self.T(x) + self.theta(x)
