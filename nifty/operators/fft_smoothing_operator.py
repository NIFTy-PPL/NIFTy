from builtins import range
import numpy as np

from .endomorphic_operator import EndomorphicOperator
from .fft_operator import FFTOperator
from .. import DomainTuple

class FFTSmoothingOperator(EndomorphicOperator):
    def __init__(self, domain, sigma, space=None):
        super(FFTSmoothingOperator, self).__init__()

        dom = DomainTuple.make(domain)
        self._sigma = float(sigma)
        if space is None:
            if len(dom.domains) != 1:
                raise ValueError("need a Field with exactly one domain")
            space = 0
        space = int(space)
        if (space<0) or space>=len(dom.domains):
            raise ValueError("space index out of range")
        self._space = space

        self._transformator = FFTOperator(dom, space=space)
        codomain = self._transformator.domain[space].get_default_codomain()
        self._kernel = codomain.get_k_length_array()
        smoother = codomain.get_fft_smoothing_kernel_function(self._sigma)
        self._kernel = smoother(self._kernel)

    def _times(self, x):
        if self._sigma == 0:
            return x.copy()

        return self._smooth(x)

    @property
    def domain(self):
        return self._transformator.domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False

    def _smooth(self, x):
        # transform to the (global-)default codomain and perform all remaining
        # steps therein
        transformed_x = self._transformator(x)
        coaxes = transformed_x.domain.axes[self._space]

        # now, apply the kernel to transformed_x
        # this is done node-locally utilizing numpy's reshaping in order to
        # apply the kernel to the correct axes

        reshaper = [transformed_x.shape[i] if i in coaxes else 1
                    for i in range(len(transformed_x.shape))]

        transformed_x *= np.reshape(self._kernel, reshaper)

        return self._transformator.adjoint_times(transformed_x)
