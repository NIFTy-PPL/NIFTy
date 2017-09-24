# Data object module for NIFTy that uses simple numpy ndarrays.

import numpy as np
from numpy import ndarray as data_object
from numpy import full, empty, sqrt, ones, zeros, vdot, abs, bincount, exp, log
from ..nifty_utilities import cast_iseq_to_tuple, get_slice_list
from functools import reduce

def from_object(object, dtype=None, copy=True):
    return np.array(object, dtype=dtype, copy=copy)

def bincount_axis(obj, minlength=None, weights=None, axis=None):
    if minlength is not None:
        length = max(np.amax(obj) + 1, minlength)
    else:
        length = np.amax(obj) + 1

    if obj.shape == ():
        raise ValueError("object of too small depth for desired array")
    data = obj

    # if present, parse the axis keyword and transpose/reorder self.data
    # such that all affected axes follow each other. Only if they are in a
    # sequence flattening will be possible
    if axis is not None:
        # do the reordering
        ndim = len(obj.shape)
        axis = sorted(cast_iseq_to_tuple(axis))
        reordering = [x for x in range(ndim) if x not in axis]
        reordering += axis

        data = np.transpose(data, reordering)
        if weights is not None:
            weights = np.transpose(weights, reordering)

        reord_axis = list(range(ndim-len(axis), ndim))

        # semi-flatten the dimensions in `axis`, i.e. after reordering
        # the last ones.
        semi_flat_dim = reduce(lambda x, y: x*y,
                               data.shape[ndim-len(reord_axis):])
        flat_shape = data.shape[:ndim-len(reord_axis)] + (semi_flat_dim, )
    else:
        flat_shape = (reduce(lambda x, y: x*y, data.shape), )

    data = np.ascontiguousarray(data.reshape(flat_shape))
    if weights is not None:
        weights = np.ascontiguousarray(weights.reshape(flat_shape))

    # compute the local bincount results
    # -> prepare the local result array
    result_dtype = np.int if weights is None else np.float
    local_counts = np.empty(flat_shape[:-1] + (length, ), dtype=result_dtype)
    # iterate over all entries in the surviving axes and compute the local
    # bincounts
    for slice_list in get_slice_list(flat_shape, axes=(len(flat_shape)-1,)):
        current_weights = None if weights is None else weights[slice_list]
        local_counts[slice_list] = np.bincount(data[slice_list],
                                               weights=current_weights,
                                               minlength=length)

    # restore the original ordering
    # place the bincount stuff at the location of the first `axis` entry
    if axis is not None:
        # axis has been sorted above
        insert_position = axis[0]
        new_ndim = len(local_counts.shape)
        return_order = (list(range(0, insert_position)) +
                        [new_ndim-1, ] +
                        list(range(insert_position, new_ndim-1)))
        local_counts = np.ascontiguousarray(
                            local_counts.transpose(return_order))
    return local_counts
