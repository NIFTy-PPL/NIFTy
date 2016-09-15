import numpy as np

from nifty.config import about
import nifty.nifty_utilities as utilities
from nifty import RGSpace, LMSpace
from nifty.operators.endomorphic_operator import EndomorphicOperator
from nifty.operators.fft_operator import FFTOperator

class SmoothOperator(EndomorphicOperator):

    # ---Overwritten properties and methods---
    def __init__(self, domain=(), field_type=(), sigma=None):
        super(SmoothOperator, self).__init__(domain=domain,
                                             field_type=field_type)

        if len(self.domain) != 1:
            raise ValueError(
                about._errors.cstring(
                    'ERROR: SmoothOperator accepts only exactly one '
                    'space as input domain.')
            )

        if self.field_type != ():
            raise ValueError(about._errors.cstring(
                'ERROR: SmoothOperator field-type must be an '
                'empty tuple.'
            ))

        self._sigma = sigma

    def _inverse_times(self, x, spaces, types):
        return self._smooth_helper(x, spaces, types, inverse=True)

    def _times(self, x, spaces, types):
        return self._smooth_helper(x, spaces, types)

    # ---Mandatory properties and methods---
    @property
    def implemented(self):
        return True

    @property
    def symmetric(self):
        return False

    @property
    def unitary(self):
        return False

    # ---Added properties and methods---
    @property
    def sigma(self):
        return self._sigma

    def _smooth_helper(self, x, spaces, types, inverse=False):
        if self.sigma == 0:
            return x.copy()

        # the domain of the smoothing operator contains exactly one space.
        # Hence, if spaces is None, but we passed LinearOperator's
        # _check_input_compatibility, we know that x is also solely defined
        # on that space
        if spaces is None:
            spaces = (0,)
        else:
            spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))

        Transformator = FFTOperator(x.domain[spaces[0]])

        # transform to the (global-)default codomain and perform all remaining
        # steps therein
        transformed_x = Transformator(x, spaces=spaces)
        codomain = transformed_x.domain[spaces[0]]
        coaxes = transformed_x.domain_axes[spaces[0]]

        # create the kernel using the knowledge of codomain about itself
        axes_local_distribution_strategy = \
            transformed_x.val.get_axes_local_distribution_strategy(axes=coaxes)

        kernel = codomain.distance_array(
                        distribution_strategy=axes_local_distribution_strategy)
        kernel.apply_scalar_function(
            codomain.get_smoothing_kernel_function(self.sigma),
            inplace=True)

        # now, apply the kernel to transformed_x
        # this is done node-locally utilizing numpys reshaping in order to
        # apply the kernel to the correct axes
        local_transformed_x = transformed_x.val.get_local_data(copy=False)
        local_kernel = kernel.get_local_data(copy=False)

        reshaper = [transformed_x.shape[i] if i in coaxes else 1
                    for i in xrange(len(transformed_x.shape))]
        local_kernel = np.reshape(local_kernel, reshaper)

        # apply the kernel
        if inverse:
            local_transformed_x /= local_kernel
        else:
            local_transformed_x *= local_kernel

        transformed_x.val.set_local_data(local_transformed_x, copy=False)

        result = Transformator.inverse_times(transformed_x, spaces=spaces)

        return result
