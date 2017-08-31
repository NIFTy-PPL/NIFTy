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

        self._sigma = sigma
        self._transformator_cache = {}

    def _times(self, x, spaces):
        if self.sigma == 0:
            return x.copy()

        # the domain of the smoothing operator contains exactly one space.
        # Hence, if spaces is None, but we passed LinearOperator's
        # _check_input_compatibility, we know that x is also solely defined
        # on that space
        if spaces is None:
            spaces = (0,)

        return self._smooth(x, spaces)

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

    @property
    def sigma(self):
        return self._sigma

    def _smooth(self, x, spaces):
        # transform to the (global-)default codomain and perform all remaining
        # steps therein
        transformator = self._get_transformator(x.dtype)
        transformed_x = transformator(x, spaces=spaces)
        codomain = transformed_x.domain[spaces[0]]
        coaxes = transformed_x.domain_axes[spaces[0]]

        kernel = codomain.get_distance_array()

        kernel = codomain.get_fft_smoothing_kernel_function(self.sigma)(kernel)

        # now, apply the kernel to transformed_x
        # this is done node-locally utilizing numpy's reshaping in order to
        # apply the kernel to the correct axes
        local_transformed_x = transformed_x.val
        local_kernel = kernel

        reshaper = [local_transformed_x.shape[i] if i in coaxes else 1
                    for i in range(len(transformed_x.shape))]
        local_kernel = np.reshape(local_kernel, reshaper)

        local_transformed_x *= local_kernel

        transformed_x.val=local_transformed_x

        smoothed_x = transformator.adjoint_times(transformed_x,
                                                 spaces=spaces)

        result = x.copy_empty()
        result=smoothed_x

        return result

    def _get_transformator(self, dtype):
        if dtype not in self._transformator_cache:
            self._transformator_cache[dtype] = FFTOperator(self.domain)
        return self._transformator_cache[dtype]
