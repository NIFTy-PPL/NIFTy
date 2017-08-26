from ...operators import EndomorphicOperator,\
                         InvertibleOperatorMixin


class WienerFilterCurvature(InvertibleOperatorMixin, EndomorphicOperator):
    """The curvature of the WienerFilterEnergy.

    This operator implements the second derivative of the
    WienerFilterEnergy used in some minimization algorithms or
    for error estimates of the posterior maps. It is the
    inverse of the propagator operator.


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

    def _add_attributes_to_copy(self, copy, **kwargs):
        copy._domain = self._domain
        copy.R = self.R.copy()
        copy.N = self.N.copy()
        copy.S = self.S.copy()
        copy = super(WienerFilterCurvature, self)._add_attributes_to_copy(
                                                                copy, **kwargs)
        return copy

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
        return (self.R.adjoint_times(self.N.inverse_times(self.R(x))) +
                self.S.inverse_times(x))
