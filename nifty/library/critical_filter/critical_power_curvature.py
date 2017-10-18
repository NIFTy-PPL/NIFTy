from ...operators.endomorphic_operator import EndomorphicOperator
from ...operators.diagonal_operator import DiagonalOperator


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

    # ---Overwritten properties and methods---

    def __init__(self, theta, T):
        self.theta = DiagonalOperator(theta.weight(1))
        self.T = T
        super(CriticalPowerCurvature, self).__init__()

    @property
    def preconditioner(self):
        return self.theta.inverse_times

    def _times(self, x):
        return self.T(x) + self.theta(x)

    # ---Mandatory properties and methods---

    @property
    def domain(self):
        return self.theta.domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False
