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
        # copy for doing the actual smoothing
        smooth_out = x.copy()

        if spaces is not None and self.sigma != 0:
            spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))

            space_obj = x.domain[spaces[0]]
            axes = x.domain_axes[spaces[0]]

            transform = FFTOperator(space_obj)

            # create the kernel
            kernel = space_obj.distance_array(
                x.val.get_axes_local_distribution_strategy(axes=axes))
            kernel = kernel.apply_scalar_function(
                space_obj.get_codomain_smoothing_function(self.sigma))

            # transform
            smooth_out = transform(smooth_out, spaces=spaces[0])

            # local data
            local_val = smooth_out.val.get_local_data(copy=False)

            # extract local kernel and reshape
            local_kernel = kernel.get_local_data(copy=False)
            new_shape = np.ones(len(local_val.shape), dtype=np.int)
            for space_axis, val_axis in zip(range(len(space_obj.shape)), axes):
                new_shape[val_axis] = local_kernel.shape[space_axis]
            local_kernel = local_kernel.reshape(new_shape)

            # multiply kernel
            if inverse:
                local_val /= kernel
            else:
                local_val *= kernel

            smooth_out.val.set_local_data(local_val, copy=False)

            # inverse transform
            smooth_out = transform.inverse_times(smooth_out, spaces=spaces[0])

        return smooth_out
