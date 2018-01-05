from ..operators import InversionEnabler
from .response_operators import LinearizedPowerResponse


def NonlinearPowerCurvature(position, FFT, Instrument, nonlinearity,
                            Projection, N, T, sample_list, inverter):
    result = None
    for sample in sample_list:
        LinearizedResponse = LinearizedPowerResponse(
            Instrument, nonlinearity, FFT, Projection, position, sample)
        op = LinearizedResponse.adjoint*N.inverse*LinearizedResponse
        result = op if result is None else result + op
    result = result*(1./len(sample_list)) + T
    return InversionEnabler(result, inverter)
