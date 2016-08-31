import numpy as np

from nifty.config import about
import nifty.nifty_utilities as utilities
from nifty import RGSpace, LMSpace
from nifty.operators.endomorphic_operator import EndomorphicOperator
from nifty.operators.fft_operator import FFTOperator

class SmoothOperator(EndomorphicOperator):

    # ---Overwritten properties and methods---
    def __init__(self, domain=(), field_type=(), inplace=False, sigma=None):
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
        self._inplace = bool(inplace)

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

    @property
    def inplace(self):
        return self._inplace

    def _smooth_helper(self, x, spaces, types, inverse=False):
        if self.sigma == 0:
            return x if self.inplace else x.copy()

        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))

        if spaces is None:
            return x if self.inplace else x.copy()

        # copy for doing the actual smoothing
        smooth_out = x.copy()

        space_obj = x.domain[spaces[0]]
        axes = x.domain_axes[spaces[0]]
        for space_axis, val_axis in zip(range(len(space_obj.shape)), axes):
            transform = FFTOperator(space_obj)
            kernel = space_obj.get_codomain_smoothing_kernel(
                self.sigma, space_axis
            )

            if isinstance(space_obj, RGSpace):
                new_shape = np.ones(len(x.shape), dtype=np.int)
                new_shape[val_axis] = len(kernel)
                kernel = kernel.reshape(new_shape)

                # transform
                smooth_out = transform(smooth_out, spaces=spaces[0])

                # multiply kernel
                if inverse:
                    smooth_out.val /= kernel
                else:
                    smooth_out.val *= kernel

                # inverse transform
                smooth_out = transform.inverse_times(smooth_out,
                                                     spaces=spaces[0])
            elif isinstance(space_obj, LMSpace):
                pass
            else:
                raise ValueError(about._errors.cstring(
                    'ERROR: SmoothOperator cannot smooth space ' +
                    str(space_obj)))

        if self.inplace:
            x.set_val(val=smooth_out.val)
            return x
        else:
            return smooth_out
