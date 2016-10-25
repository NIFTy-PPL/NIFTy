import numpy as np

import nifty.nifty_utilities as utilities
from nifty.operators.endomorphic_operator import EndomorphicOperator
from nifty.operators.fft_operator import FFTOperator
import smooth_util as su
from d2o import STRATEGIES


class SmoothingOperator(EndomorphicOperator):
    # ---Overwritten properties and methods---
    def __init__(self, domain=(), field_type=(), sigma=0,
                 log_distances=False):

        self._domain = self._parse_domain(domain)
        self._field_type = self._parse_field_type(field_type)

        if len(self.domain) != 1:
            raise ValueError(

                'ERROR: SmoothOperator accepts only exactly one '
                'space as input domain.'
            )

        if self.field_type != ():
            raise ValueError(
                'ERROR: SmoothOperator field-type must be an '
                'empty tuple.'
            )

        self.sigma = sigma
        self.log_distances = log_distances

        self._direct_smoothing_width = 2.

    def _inverse_times(self, x, spaces, types):
        return self._smoothing_helper(x, spaces, inverse=True)

    def _times(self, x, spaces, types):
        return self._smoothing_helper(x, spaces, inverse=False)

    # ---Mandatory properties and methods---
    @property
    def domain(self):
        return self._domain

    @property
    def field_type(self):
        return self._field_type

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

    @sigma.setter
    def sigma(self, sigma):
        self._sigma = np.float(sigma)

    @property
    def log_distances(self):
        return self._log_distances

    @log_distances.setter
    def log_distances(self, log_distances):
        self._log_distances = bool(log_distances)

    def _smoothing_helper(self, x, spaces, inverse):
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

        try:
            result = self._fft_smoothing(x, spaces, inverse)
        except ValueError:
            result = self._direct_smoothing(x, spaces, inverse)
        return result

    def _fft_smoothing(self, x, spaces, inverse):
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

    def _direct_smoothing(self, x, spaces, inverse):
        # infer affected axes
        # we rely on the knowledge, that `spaces` is a tuple with length 1.
        affected_axes = x.domain_axes[spaces[0]]

        axes_local_distribution_strategy = \
            x.val.get_axes_local_distribution_strategy(axes=affected_axes)

        distance_array = x.domain[spaces[0]].get_distance_array(
            distribution_strategy=axes_local_distribution_strategy)

        if self.log_distances:
            distance_array.apply_scalar_function(np.log, inplace=True)

        # collect the local data + ghost cells
        local_data_Q = False

        if x.distribution_strategy == 'not':
            local_data_Q = True
        elif x.distribution_strategy in STRATEGIES['slicing']:
            # infer the local start/end based on the slicing information of
            # x's d2o. Only gets non-trivial for axis==0.
            if 0 not in affected_axes:
                local_data_Q = True
            else:
                # we rely on the fact, that the content of x.domain_axes is
                # sorted
                true_starts = [x.val.distributor.local_start]
                true_starts += [0] * (len(affected_axes) - 1)
                true_ends = [x.val.distributor.local_end]
                true_ends += [x.shape[i] for i in affected_axes[1:]]

                augmented_start = max(0,
                                      true_starts[0] -
                                      self._direct_smoothing_width * self.sigma)
                augmented_end = min(x.shape[affected_axes[0]],
                                    true_ends[0] +
                                    self._direct_smoothing_width * self.sigma)
                augmented_slice = slice(augmented_start, augmented_end)
                augmented_data = x.val.get_data(augmented_slice,
                                                local_keys=True,
                                                copy=False)
                augmented_data = augmented_data.get_local_data(copy=False)

                augmented_distance_array = distance_array.get_data(
                    augmented_slice,
                    local_keys=True,
                    copy=False)
                augmented_distance_array = \
                    augmented_distance_array.get_local_data(copy=False)

        else:
            raise ValueError(about._errors.cstring(
                "ERROR: Direct smoothing not implemented for given"
                "distribution strategy."))

        if local_data_Q:
            # if the needed data resides on the nodes already, the necessary
            # are the same; no matter what the distribution strategy was.
            augmented_data = x.val.get_local_data(copy=False)
            augmented_distance_array = \
                distance_array.get_local_data(copy=False)
            true_starts = [0] * len(affected_axes)
            true_ends = [x.shape[i] for i in affected_axes]

        # perform the convolution along the affected axes
        local_result = augmented_data
        for index in range(len(affected_axes)):
            data_axis = affected_axes[index]
            distances_axis = index
            true_start = true_starts[index]
            true_end = true_ends[index]

            local_result = self._direct_smoothing_single_axis(
                local_result,
                data_axis,
                augmented_distance_array,
                distances_axis,
                true_start,
                true_end,
                inverse)

        result = x.copy_empty()
        result.val.set_local_data(local_result, copy=False)
        return result

    def _direct_smoothing_single_axis(self, data, data_axis, distances,
                                      distances_axis, true_start, true_end,
                                      inverse):
        if inverse:
            true_sigma = 1 / self.sigma
        else:
            true_sigma = self.sigma

        if (data.dtype == np.dtype('float32')):
            smoothed_data = su.apply_along_axis_f(data_axis, data,
                                                  startindex=true_start,
                                                  endindex=true_end,
                                                  distances=distances,
                                                  smooth_length=true_sigma)
        else:
            smoothed_data = su.apply_along_axis(data_axis, data,
                                                startindex=true_start,
                                                endindex=true_end,
                                                distances=distances,
                                                smooth_length=true_sigma)
        return smoothed_data
