# -*- coding: utf-8 -*-

from builtins import range
import numpy as np

from ..endomorphic_operator import EndomorphicOperator
from ..fft_operator import FFTOperator


class FFTSmoothingOperator(EndomorphicOperator):

    def __init__(self, domain, sigma,
                 default_spaces=None):
        super(FFTSmoothingOperator, self).__init__(default_spaces)

        self._domain = self._parse_domain(domain)
        if len(self._domain) != 1:
            raise ValueError("SmoothingOperator only accepts exactly one "
                             "space as input domain.")

        self._sigma = float(sigma)
        if self._sigma == 0.:
            return

        self._transformator = FFTOperator(self._domain)
        codomain = self._domain[0].get_default_codomain()
        self._kernel = codomain.get_k_length_array()
        smoother = codomain.get_fft_smoothing_kernel_function(self._sigma)
        self._kernel = smoother(self._kernel)

    def _times(self, x, spaces):
        if self._sigma == 0:
            return x.copy()

        # the domain of the smoothing operator contains exactly one space.
        # Hence, if spaces is None, but we passed LinearOperator's
        # _check_input_compatibility, we know that x is also solely defined
        # on that space
        return self._smooth(x, (0,) if spaces is None else spaces)

    # ---Mandatory properties and methods---
    @property
    def domain(self):
        return self._domain

    @property
    def self_adjoint(self):
        return True

    @property
    def unitary(self):
        return False

    # ---Added properties and methods---

    def _smooth(self, x, spaces):
        # transform to the (global-)default codomain and perform all remaining
        # steps therein
        transformed_x = self._transformator(x, spaces=spaces)
        coaxes = transformed_x.domain_axes[spaces[0]]

        # now, apply the kernel to transformed_x
        # this is done node-locally utilizing numpy's reshaping in order to
        # apply the kernel to the correct axes

        reshaper = [transformed_x.shape[i] if i in coaxes else 1
                    for i in range(len(transformed_x.shape))]

        transformed_x *= np.reshape(self._kernel, reshaper)

        return self._transformator.adjoint_times(transformed_x, spaces=spaces)
