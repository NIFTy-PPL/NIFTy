from ..field import exp
from ..operators.linear_operator import LinearOperator


class LinearizedSignalResponse(LinearOperator):
    def __init__(self, Instrument, nonlinearity, FFT, power, m):
        super(LinearizedSignalResponse, self).__init__()
        position = FFT.adjoint_times(power*m)
        self._op = (Instrument * nonlinearity.derivative(position) *
                    FFT.adjoint * power)

    @property
    def domain(self):
        return self._op.domain

    @property
    def target(self):
        return self._op.target

    @property
    def capability(self):
        return self._op.capability

    def apply(self, x, mode):
        return self._op.apply(x, mode)


class LinearizedPowerResponse(LinearOperator):
    def __init__(self, Instrument, nonlinearity, FFT, Projection, t, m):
        super(LinearizedPowerResponse, self).__init__()
        power = exp(0.5*t)
        position = FFT.adjoint_times(Projection.adjoint_times(power) * m)
        linearization = nonlinearity.derivative(position)
        self._op = (0.5 * Instrument * linearization * FFT.adjoint * m *
                    Projection.adjoint * power)

    @property
    def domain(self):
        return self._op.domain

    @property
    def target(self):
        return self._op.target

    @property
    def capability(self):
        return self._op.capability

    def apply(self, x, mode):
        return self._op.apply(x, mode)
