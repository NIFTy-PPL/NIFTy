from ..field import exp
from ..operators.linear_operator import LinearOperator


class LinearizedSignalResponse(LinearOperator):
    def __init__(self, Instrument, nonlinearity, FFT, power, m):
        super(LinearizedSignalResponse, self).__init__()
        self.Instrument = Instrument
        self.FFT = FFT
        self.power = power
        position = FFT.adjoint_times(self.power*m)
        self.linearization = nonlinearity.derivative(position)

    def _times(self, x):
        tmp = self.FFT.adjoint_times(self.power*x)
        tmp *= self.linearization
        return self.Instrument(tmp)

    def _adjoint_times(self, x):
        tmp = self.Instrument.adjoint_times(x)
        tmp *= self.linearization
        tmp = self.FFT(tmp)
        tmp *= self.power
        return tmp

    @property
    def domain(self):
        return self.FFT.target

    @property
    def target(self):
        return self.Instrument.target

    @property
    def unitary(self):
        return False


class LinearizedPowerResponse(LinearOperator):
    def __init__(self, Instrument, nonlinearity, FFT, Projection, t, m):
        super(LinearizedPowerResponse, self).__init__()
        self.Instrument = Instrument
        self.FFT = FFT
        self.Projection = Projection
        self.power = exp(0.5 * t)
        self.m = m
        position = FFT.adjoint_times(
            self.Projection.adjoint_times(self.power) * self.m)
        self.linearization = nonlinearity.derivative(position)

    def _times(self, x):
        tmp = self.Projection.adjoint_times(self.power*x)
        tmp *= self.m
        tmp = self.FFT.adjoint_times(tmp)
        tmp *= self.linearization
        tmp = self.Instrument(tmp)
        tmp *= 0.5
        return tmp

    def _adjoint_times(self, x):
        tmp = self.Instrument.adjoint_times(x)
        tmp *= self.linearization
        tmp = self.FFT(tmp)
        tmp *= self.m.conjugate()
        tmp = self.Projection(tmp)
        tmp *= self.power
        tmp *= 0.5
        return tmp

    @property
    def domain(self):
        return self.power.domain

    @property
    def target(self):
        return self.Instrument.target

    @property
    def unitary(self):
        return False
