from ..operators import EndomorphicOperator, InversionEnabler
from .response_operators import LinearizedPowerResponse


class NonlinearPowerCurvature(EndomorphicOperator):
    class _Helper(EndomorphicOperator):
        def __init__(self, position, FFT, Instrument, nonlinearity,
                     Projection, N, T, sample_list):
            super(NonlinearPowerCurvature._Helper, self).__init__()
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
        def capability(self):
            return self.TIMES

        def apply(self, x, mode):
            self._check_input(x, mode)
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

    def __init__(self, position, FFT, Instrument, nonlinearity,
                 Projection, N, T, sample_list, inverter):
        super(NonlinearPowerCurvature, self).__init__()
        self._op = self._Helper(position, FFT, Instrument, nonlinearity,
                                Projection, N, T, sample_list)
        self._op = InversionEnabler(self._op, inverter)

    @property
    def domain(self):
        return self._op.domain

    @property
    def capability(self):
        return self._op.capability

    def apply(self, x, mode):
        return self._op.apply(x, mode)
