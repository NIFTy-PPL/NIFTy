from nifty.operators import EndomorphicOperator,\
                            InvertibleOperatorMixin


class CriticalPowerCurvature(InvertibleOperatorMixin, EndomorphicOperator):
    def __init__(self, theta, Laplace, sigma, inverter=None, preconditioner=None):

        self.theta = theta
        self.Laplace = Laplace
        self.sigma = sigma
        # if preconditioner is None:
        #     preconditioner = self.T.times
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
        return self.Laplace.adjoint_times(self.Laplace(x)) / self.sigma ** 2 \
               + self.theta * x
