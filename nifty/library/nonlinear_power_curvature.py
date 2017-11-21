from .. import Field
from ..operators.endomorphic_operator import EndomorphicOperator
from .response_operators import LinearizedPowerResponse


class NonlinearPowerCurvature(EndomorphicOperator):

    def __init__(self, position, FFT, Instrument, nonlinearity,
                 Projection, N, T, sample_list, inverter=None):
        self.N = N
        self.FFT = FFT
        self.Instrument = Instrument
        self.T = T
        self.sample_list = sample_list
        self.samples = len(sample_list)
        self.position = position
        self.Projection = Projection
        self.nonlinearity = nonlinearity

        # if preconditioner is None:
        #     preconditioner = self.theta.inverse_times
        self._domain = self.position.domain
        super(NonlinearPowerCurvature, self).__init__(inverter=inverter)

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

        result = Field(self.domain, val=0.)
        for i in range(self.samples):
            sample = self.sample_list[i]
            result += self._sample_times(x, sample)
        result /= self.samples

        return (result + self.T(x))

    def _sample_times(self, x, sample):
        LinearizedResponse = LinearizedPowerResponse(self.Instrument, self.nonlinearity,
                                                     self.FFT, self.Projection, self.position, sample)
        result = LinearizedResponse.adjoint_times(
            self.N.inverse_times(LinearizedResponse(x)))

        return result
