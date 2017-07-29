from nifty.operators import EndomorphicOperator,\
                            InvertibleOperatorMixin
from nifty.basic_arithmetics import exp

class WienerFilterCurvature(InvertibleOperatorMixin, EndomorphicOperator):
    """The curvature of the LogNormalWienerFilterEnergy.

    This operator implements the second derivative of the
    LogNormalWienerFilterEnergy used in some minimization algorithms or for
    error estimates of the posterior maps. It is the inverse of the propagator
    operator.

    Parameters
    ----------
    R: LinearOperator,
        The response operator of the Wiener filter measurement.
    N : EndomorphicOperator
        The noise covariance.
    S: DiagonalOperator,
        The prior signal covariance

    """

    def __init__(self, R, N, S, inverter=None, preconditioner=None, **kwargs):

        self.R = R
        self.N = N
        self.S = S
        if preconditioner is None:
            preconditioner = self.S.times
        self._domain = self.S.domain
        super(WienerFilterCurvature, self).__init__(
                                                 inverter=inverter,
                                                 preconditioner=preconditioner,
                                                 **kwargs)

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
        expx = exp(x)
        expxx = expx*x
        part1 = self.S.inverse_times(x)
        part2 = (expx *   # is an adjoint necessary here?
                 self.R.adjoint_times(self.N.inverse_times(self.R(expxx))))
        part3 = (expxx *  # is an adjoint necessary here?
                 self.R.adjoint_times(self.N.inverse_times(self.R(expx))))
        return part1 + part2 + part3
