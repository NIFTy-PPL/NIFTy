from ..operators import EndomorphicOperator, InversionEnabler
from .response_operators import LinearizedPowerResponse


class NonlinearPowerCurvature(InversionEnabler, EndomorphicOperator):

    def __init__(self, position, FFT, Instrument, nonlinearity,
                 Projection, N, T, sample_list, inverter):
        InversionEnabler.__init__(self, inverter)
        EndomorphicOperator.__init__(self)
        self.N = N
        self.FFT = FFT
        self.Instrument = Instrument
        self.T = T
        self.sample_list = sample_list
        self.position = position
        self.Projection = Projection
        self.nonlinearity = nonlinearity

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
        result = None
        for sample in self.sample_list:
            if result is None:
                result = self._sample_times(x, sample)
            else:
                result += self._sample_times(x, sample)
        result *= 1./len(self.sample_list)
        return result + self.T(x)

    def _sample_times(self, x, sample):
        LinearizedResponse = LinearizedPowerResponse(
            self.Instrument, self.nonlinearity, self.FFT, self.Projection,
            self.position, sample)
        return LinearizedResponse.adjoint_times(
            self.N.inverse_times(LinearizedResponse(x)))
