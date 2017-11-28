from .. import exp
from ..operators.linear_operator import LinearOperator


class LinearizedSignalResponse(LinearOperator):
    def __init__(self, Instrument, nonlinearity, FFT, power, m, default_spaces=None):
        super(LinearizedSignalResponse, self).__init__(default_spaces)
        self._target = Instrument.target
        self._domain = FFT.target
        self.Instrument = Instrument
        self.FFT = FFT
        self.power = power
        position = FFT.adjoint_times(self.power * m)
        self.linearization = nonlinearity.derivative(position)

    def _times(self, x, spaces):
        return self.Instrument(self.linearization * self.FFT.adjoint_times(self.power * x))

    def _adjoint_times(self, x, spaces):
        return self.power * self.FFT(self.linearization * self.Instrument.adjoint_times(x))

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def unitary(self):
        return False


class LinearizedPowerResponse(LinearOperator):
    def __init__(self, Instrument, nonlinearity, FFT, Projection, t, m, default_spaces=None):
        super(LinearizedPowerResponse, self).__init__(default_spaces)
        self._target = Instrument.target
        self._domain = t.domain
        self.Instrument = Instrument
        self.FFT = FFT
        self.Projection = Projection
        self.power = exp(0.5 * t)
        self.m = m
        position = FFT.adjoint_times(
            self.Projection.adjoint_times(self.power) * self.m)
        self.linearization = nonlinearity.derivative(position)

    def _times(self, x, spaces):
        return 0.5 * self.Instrument(self.linearization
                                     * self.FFT.adjoint_times(self.m
                                                              * self.Projection.adjoint_times(self.power * x)))

    def _adjoint_times(self, x, spaces):
        return 0.5 * self.power * self.Projection(self.m.conjugate()
                                                  * self.FFT(self.linearization
                                                             * self.Instrument.adjoint_times(x)))  # .weight(-1)

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def unitary(self):
        return False
