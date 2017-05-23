from nifty.operators import EndomorphicOperator,\
                            InvertibleOperatorMixin


class NonlinearWienerFilterCurvature(InvertibleOperatorMixin, EndomorphicOperator):
    def __init__(self, R, N, S, position, inverter=None, preconditioner=None):

        self.R = R
        self.N = N
        self.S = S
        self.position = position
        # if preconditioner is None:
        #     preconditioner = self.S.times
        self._domain = self.S.domain
        super(NonlinearWienerFilterCurvature, self).__init__(inverter=inverter,
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
        return self.R.derived_adjoint_times(
                self.N.inverse_times(self.R.derived_times(
                    x, self.position)), self.position)\
               + self.S.inverse_times(x)
