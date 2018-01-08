from ..operators import InversionEnabler

def LogNormalWienerFilterCurvature(R, N, S, fft, expp_sspace, inverter):
    part1 = S.inverse
    part3 = (fft.adjoint * expp_sspace * fft *
             R. adjoint * N.inverse * R *
             fft.adjoint * expp_sspace * fft)
    return InversionEnabler(part1 + part3, inverter)
