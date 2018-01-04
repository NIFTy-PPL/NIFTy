from ..field import exp


def LinearizedSignalResponse(Instrument, nonlinearity, FFT, power, m):
    position = FFT.adjoint_times(power*m)
    return (Instrument * nonlinearity.derivative(position) *
            FFT.adjoint * power)

def LinearizedPowerResponse(Instrument, nonlinearity, FFT, Projection, t, m):
    power = exp(0.5*t)
    position = FFT.adjoint_times(Projection.adjoint_times(power) * m)
    linearization = nonlinearity.derivative(position)
    return (0.5 * Instrument * linearization * FFT.adjoint * m *
            Projection.adjoint * power)
