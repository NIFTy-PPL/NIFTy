# -*- coding: utf-8 -*-
import numpy as np

from nifty.field import Field

from nifty.operators.endomorphic_operator import EndomorphicOperator


class ProjectionOperator(EndomorphicOperator):

    # ---Overwritten properties and methods---

    def __init__(self, projection_field):
        if not isinstance(projection_field, Field):
            raise TypeError("The projection_field must be a NIFTy-Field"
                            "instance.")
        self._projection_field = projection_field
        self._unitary = None

    def _times(self, x, spaces):
        # if the domain matches directly
        # -> multiply the fields directly
        if x.domain == self.domain:
            # here the actual multiplication takes place
            dotted = (self._projection_field * x).sum()
            return self._projection_field * dotted

        # if the distribution_strategy of self is sub-slice compatible to
        # the one of x, reshape the local data of self and apply it directly
        active_axes = []
        if spaces is None:
            active_axes = range(len(x.shape))
        else:
            for space_index in spaces:
                active_axes += x.domain_axes[space_index]

        axes_local_distribution_strategy = \
            x.val.get_axes_local_distribution_strategy(active_axes)
        if axes_local_distribution_strategy == \
           self._projection_field.distribution_strategy:
            local_projection_vector = \
                self._projection_field.val.get_local_data(copy=False)
        else:
            # create an array that is sub-slice compatible
            self.logger.warn("The input field is not sub-slice compatible to "
                             "the distribution strategy of the operator. "
                             "Performing an probably expensive "
                             "redistribution.")
            redistr_projection_val = self._projection_field.val.copy(
                distribution_strategy=axes_local_distribution_strategy)
            local_projection_vector = \
                redistr_projection_val.get_local_data(copy=False)

        local_x = x.val.get_local_data(copy=False)

        l = len(local_projection_vector.shape)
        sublist_projector = range(l)
        sublist_x = np.arange(len(local_x.shape)) + l

        for i in xrange(l):
            a = active_axes[i]
            sublist_x[a] = i

        dotted = np.einsum(local_projection_vector, sublist_projector,
                           local_x, sublist_x)

        # get those elements from sublist_x that haven't got contracted
        sublist_dotted = sublist_x[sublist_x >= l]

        remultiplied = np.einsum(local_projection_vector, sublist_projector,
                                 dotted, sublist_dotted,
                                 sublist_x)
        result_field = x.copy_empty(dtype=remultiplied.dtype)
        result_field.val.set_local_data(remultiplied, copy=False)
        return result_field

    def _inverse_times(self, x, spaces):
        raise NotImplementedError("The ProjectionOperator is a singular "
                                  "operator and therefore has no inverse.")

    # ---Mandatory properties and methods---

    @property
    def domain(self):
        return self._projection_field.domain

    @property
    def implemented(self):
        return True

    @property
    def unitary(self):
        if self._unitary is None:
            self._unitary = (self._projection_field.val == 1).all()
        return self._unitary

    @property
    def symmetric(self):
        return True
