from nifty.operators import EndomorphicOperator,\
                            InvertibleOperatorMixin


class CriticalPowerCurvature(InvertibleOperatorMixin, EndomorphicOperator):
    def __init__(self, theta, T, inverter=None, preconditioner=None):

        self.theta = theta
        self.T = T
        # if preconditioner is None:
        #     preconditioner = self.T.times
        self._domain = self.T.domain
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
        return self.T(x) + self.theta * x
