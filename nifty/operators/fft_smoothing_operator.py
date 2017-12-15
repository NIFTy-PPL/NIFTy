from .endomorphic_operator import EndomorphicOperator
from .fft_operator import FFTOperator
from ..utilities import infer_space
from .diagonal_operator import DiagonalOperator
from .. import DomainTuple


class FFTSmoothingOperator(EndomorphicOperator):
    def __init__(self, domain, sigma, space=None):
        super(FFTSmoothingOperator, self).__init__()

        dom = DomainTuple.make(domain)
        self._sigma = float(sigma)
        self._space = infer_space(dom, space)

        self._FFT = FFTOperator(dom, space=self._space)
        codomain = self._FFT.domain[self._space].get_default_codomain()
        kernel = codomain.get_k_length_array()
        smoother = codomain.get_fft_smoothing_kernel_function(self._sigma)
        kernel = smoother(kernel)
        ddom = list(dom)
        ddom[self._space] = codomain
        self._diag = DiagonalOperator(kernel, ddom, self._space)

    def _times(self, x):
        if self._sigma == 0:
            return x.copy()

        return self._FFT.adjoint_times(self._diag(self._FFT(x)))

    @property
    def domain(self):
        return self._FFT.domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False
