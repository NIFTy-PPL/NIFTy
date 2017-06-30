# -*- coding: utf-8 -*-

from builtins import range
import numpy as np

from nifty.operators.fft_operator import FFTOperator

from .smoothing_operator import SmoothingOperator


class FFTSmoothingOperator(SmoothingOperator):

    def _smooth(self, x, spaces, inverse):
        Transformator = FFTOperator(x.domain[spaces[0]])

        # transform to the (global-)default codomain and perform all remaining
        # steps therein
        transformed_x = Transformator(x, spaces=spaces)
        codomain = transformed_x.domain[spaces[0]]
        coaxes = transformed_x.domain_axes[spaces[0]]

        # create the kernel using the knowledge of codomain about itself
        axes_local_distribution_strategy = \
            transformed_x.val.get_axes_local_distribution_strategy(axes=coaxes)

        kernel = codomain.get_distance_array(
            distribution_strategy=axes_local_distribution_strategy)

        #MR FIXME: this causes calls of log(0.) which should probably be avoided
        if self.log_distances:
            kernel.apply_scalar_function(np.log, inplace=True)

        kernel.apply_scalar_function(
            codomain.get_fft_smoothing_kernel_function(self.sigma),
            inplace=True)

        # now, apply the kernel to transformed_x
        # this is done node-locally utilizing numpys reshaping in order to
        # apply the kernel to the correct axes
        local_transformed_x = transformed_x.val.get_local_data(copy=False)
        local_kernel = kernel.get_local_data(copy=False)

        reshaper = [transformed_x.shape[i] if i in coaxes else 1
                    for i in range(len(transformed_x.shape))]
        local_kernel = np.reshape(local_kernel, reshaper)

        # apply the kernel
        if inverse:
            #MR FIXME: danger of having division by zero or overflows
            local_transformed_x /= local_kernel
        else:
            local_transformed_x *= local_kernel

        transformed_x.val.set_local_data(local_transformed_x, copy=False)

        smoothed_x = Transformator.adjoint_times(transformed_x, spaces=spaces)

        result = x.copy_empty()
        result.set_val(smoothed_x, copy=False)

        return result
