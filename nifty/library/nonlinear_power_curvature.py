from .. import Field
from ..operators.endomorphic_operator import EndomorphicOperator
from .response_operators import LinearizedPowerResponse


class NonlinearPowerCurvature(EndomorphicOperator):

    def __init__(self, position, Instrument, nonlinearity,
                 Projection, N, T, sample_list):
        self.N = N
        self.Instrument = Instrument
        self.T = T
        self.sample_list = sample_list
        self.position = position
        self.Projection = Projection
        self.nonlinearity = nonlinearity

        # if preconditioner is None:
        #     preconditioner = self.theta.inverse_times
        super(NonlinearPowerCurvature, self).__init__()

    @property
    def domain(self):
        return self.position.domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False

    def _times(self, x):
        result = Field.zeros_like(self.position, dtype=np.float64)
        for sample in self.sample_list:
            result += self._sample_times(x, sample)
        result *= 1./len(self.sample_list)
        return result + self.T(x)

    def _sample_times(self, x, sample):
        LinearizedResponse = LinearizedPowerResponse(self.Instrument, self.nonlinearity,
                                                     self.Projection, self.position, sample)
        return LinearizedResponse.adjoint_times(
            self.N.inverse_times(LinearizedResponse(x)))
