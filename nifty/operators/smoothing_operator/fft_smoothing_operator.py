# -*- coding: utf-8 -*-

from builtins import range
import numpy as np

from nifty.operators.fft_operator import FFTOperator

from .smoothing_operator import SmoothingOperator


class FFTSmoothingOperator(SmoothingOperator):

    def __init__(self, domain, sigma, log_distances=False,
                 default_spaces=None):
        super(FFTSmoothingOperator, self).__init__(
                                                domain=domain,
                                                sigma=sigma,
                                                log_distances=log_distances,
                                                default_spaces=default_spaces)
        self._transformator_cache = {}

    def _add_attributes_to_copy(self, copy, **kwargs):
        copy._transformator_cache = self._transformator_cache

        copy = super(FFTSmoothingOperator, self)._add_attributes_to_copy(
                                                                copy, **kwargs)
        return copy

    def _smooth(self, x, spaces, inverse):
        # transform to the (global-)default codomain and perform all remaining
        # steps therein
        transformator = self._get_transformator(x.dtype)
        transformed_x = transformator(x, spaces=spaces)
        codomain = transformed_x.domain[spaces[0]]
        coaxes = transformed_x.domain_axes[spaces[0]]

        kernel = codomain.get_distance_array()

        #MR FIXME: this causes calls of log(0.) which should probably be avoided
        if self.log_distances:
            kernel=np.log(kernel)

        #kernel.apply_scalar_function(
        #    codomain.get_fft_smoothing_kernel_function(self.sigma),
        #    inplace=True)
        kernel = codomain.get_fft_smoothing_kernel_function(self.sigma)(kernel)

        # now, apply the kernel to transformed_x
        # this is done node-locally utilizing numpys reshaping in order to
        # apply the kernel to the correct axes
        local_transformed_x = transformed_x.val
        local_kernel = kernel

        reshaper = [local_transformed_x.shape[i] if i in coaxes else 1
                    for i in range(len(transformed_x.shape))]
        local_kernel = np.reshape(local_kernel, reshaper)

        # apply the kernel
        if inverse:
            #MR FIXME: danger of having division by zero or overflows
            local_transformed_x /= local_kernel
        else:
            local_transformed_x *= local_kernel

        transformed_x.val=local_transformed_x

        smoothed_x = transformator.adjoint_times(transformed_x,
                                                 spaces=spaces)

        result = x.copy_empty()
        result=smoothed_x

        return result

    def _get_transformator(self, dtype):
        if dtype not in self._transformator_cache:
            self._transformator_cache[dtype] = FFTOperator(
                                                    self.domain,
                                                    domain_dtype=dtype,
                                                    target_dtype=np.complex)
        return self._transformator_cache[dtype]
