import numpy as np

from nifty.config import about
import nifty.nifty_utilities as utilities
from nifty import RGSpace, LMSpace
from nifty.operators.endomorphic_operator import EndomorphicOperator
from nifty.operators.fft_operator import FFTOperator

class SmoothOperator(EndomorphicOperator):

    # ---Overwritten properties and methods---
    def __init__(self, domain=(), field_type=(), inplace=False,
                 sigma = None, implemented=False):
        super(SmoothOperator, self).__init__(domain=domain,
                                             field_type=field_type,
                                             implemented=implemented)

        if self.field_type != ():
            raise ValueError(about._errors.cstring(
                'ERROR: TransformationOperator field-type must be an '
                'empty tuple.'
            ))

        self._sigma = sigma
        self._inplace = inplace
        self._implemented = bool(implemented)

    def _times(self, x, spaces, types):
        if sigma == 0:
            return x if self.inplace else x.copy()

        spaces = utilities.cast_axis_to_tuple(spaces, len(x.domain))

        if spaces is None:
            return x if self.inplace else x.copy()

        for space in spaces:
            axes = x.domain_axes[space]
            for space_axis, val_axis in zip(
                    range(len(x.domain[space].shape)), axes):
                transform = FFTOperator(x.domain[space])
                kernel = x.domain[space].get_codomain_mask(
                    self.sigma, space_axis)
                if isinstance(x.domain[space], RGSpace):
                    new_shape = np.ones(len(x.shape), dtype=np.int)
                    new_shape[val_axis] = len(kernel)
                    kernel = kernel.reshape(new_shape)

                    # transform
                    transformed_inp = transform(x)
                    transformed_inp *= kernel
                elif isinstance(x.domain[space], LMSpace):
                    pass
                else:
                    raise ValueError(about._errors.cstring(
                        'ERROR: SmoothOperator cannot smooth space ' +
                        str(x.domain[space]))

    # ---Added properties and methods---
    @property
    def sigma(self):
        return self._sigma

    @property
    def inplace(self):
        return self._inplace
