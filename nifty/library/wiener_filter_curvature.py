from ..operators import EndomorphicOperator


class WienerFilterCurvature(EndomorphicOperator):
    """The curvature of the WienerFilterEnergy.

    This operator implements the second derivative of the
    WienerFilterEnergy used in some minimization algorithms or
    for error estimates of the posterior maps. It is the
    inverse of the propagator operator.

    Parameters
    ----------
    R: LinearOperator,
       The response operator of the Wiener filter measurement.
    N: EndomorphicOperator
       The noise covariance.
    S: DiagonalOperator,
       The prior signal covariance
    """

    def __init__(self, R, N, S):
        super(WienerFilterCurvature, self).__init__()
        self.R = R
        self.N = N
        self.S = S

    @property
    def preconditioner(self):
        return self.S.times

    @property
    def domain(self):
        return self.S.domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False

    def _times(self, x):
        res = self.R.adjoint_times(self.N.inverse_times(self.R(x)))
        res += self.S.inverse_times(x)
        return res
