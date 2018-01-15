from .scaling_operator import ScalingOperator
from .fft_operator import FFTOperator
from ..utilities import infer_space
from .diagonal_operator import DiagonalOperator
from .. import DomainTuple


def FFTSmoothingOperator(domain, sigma, space=None):
    sigma = float(sigma)
    if sigma < 0.:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0.:
        return ScalingOperator(1., domain)

    domain = DomainTuple.make(domain)
    space = infer_space(domain, space)
    if domain[space].harmonic:
        raise TypeError("domain must not be harmonic")
    fftdom = list(domain)
    codomain = domain[space].get_default_codomain()
    fftdom[space] = codomain
    fftdom = DomainTuple.make(fftdom)
    FFT = FFTOperator(fftdom, domain[space], space=space)
    kernel = codomain.get_k_length_array()
    smoother = codomain.get_fft_smoothing_kernel_function(sigma)
    kernel = smoother(kernel)
    diag = DiagonalOperator(kernel, fftdom, space)
    return FFT*diag*FFT.inverse
