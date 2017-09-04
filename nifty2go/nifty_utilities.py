# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from builtins import next
from builtins import range
import numpy as np
from itertools import product
import itertools
from functools import reduce


def get_slice_list(shape, axes):
    """
    Helper function which generates slice list(s) to traverse over all
    combinations of axes, other than the selected axes.

    Parameters
    ----------
    shape: tuple
        Shape of the data array to traverse over.
    axes: tuple
        Axes which should not be iterated over.

    Yields
    -------
    list
        The next list of indices and/or slice objects for each dimension.

    Raises
    ------
    ValueError
        If shape is empty.
    ValueError
        If axes(axis) does not match shape.
    """

    if not shape:
        raise ValueError("shape cannot be None.")

    if axes:
        if not all(axis < len(shape) for axis in axes):
            raise ValueError("axes(axis) does not match shape.")
        axes_select = [0 if x in axes else 1 for x, y in enumerate(shape)]
        axes_iterables = \
            [list(range(y)) for x, y in enumerate(shape) if x not in axes]
        for index in product(*axes_iterables):
            it_iter = iter(index)
            slice_list = [
                next(it_iter)
                if axis else slice(None, None) for axis in axes_select
                ]
            yield slice_list
    else:
        yield [slice(None, None)]
        return


def cast_axis_to_tuple(axis, length=None):
    if axis is None:
        return None
    try:
        axis = tuple(int(item) for item in axis)
    except(TypeError):
        if np.isscalar(axis):
            axis = (int(axis),)
        else:
            raise TypeError("Could not convert axis-input to tuple of ints")

    if length is not None:
        # shift negative indices to positive ones
        axis = tuple(item if (item >= 0) else (item + length) for item in axis)

        # Deactivated this, in order to allow for the ComposedOperator
        # remove duplicate entries
        # axis = tuple(set(axis))

        # assert that all entries are elements in [0, length]
        for elem in axis:
            assert (0 <= elem < length)

    return axis


def parse_domain(domain):
    from .domain_object import DomainObject
    if domain is None:
        domain = ()
    elif isinstance(domain, DomainObject):
        domain = (domain,)
    elif not isinstance(domain, tuple):
        domain = tuple(domain)

    for d in domain:
        if not isinstance(d, DomainObject):
            raise TypeError(
                "Given object contains something that is not an "
                "instance of DomainObject-class.")
    return domain


def slicing_generator(shape, axes):
    """
    Helper function which generates slice list(s) to traverse over all
    combinations of axes, other than the selected axes.

    Parameters
    ----------
    shape: tuple
        Shape of the data array to traverse over.
    axes: tuple
        Axes which should not be iterated over.

    Yields
    -------
    list
        The next list of indices and/or slice objects for each dimension.

    Raises
    ------
    ValueError
        If shape is empty.
    ValueError
        If axes(axis) does not match shape.
    """

    if not shape:
        raise ValueError("ERROR: shape cannot be None.")

    if axes:
        if not all(axis < len(shape) for axis in axes):
            raise ValueError("ERROR: axes(axis) does not match shape.")
        axes_select = [0 if x in axes else 1 for x, y in enumerate(shape)]
        axes_iterables =\
            [list(range(y)) for x, y in enumerate(shape) if x not in axes]
        for current_index in itertools.product(*axes_iterables):
            it_iter = iter(current_index)
            slice_list = [next(it_iter) if use_axis else
                          slice(None, None) for use_axis in axes_select]
            yield slice_list
    else:
        yield [slice(None, None)]
        return


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
        axis = sorted(cast_axis_to_tuple(axis, length=ndim))
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
        weights = np.ascontiguousarray(
                            weights.reshape(flat_shape))

    # compute the local bincount results
    # -> prepare the local result array
    if weights is None:
        result_dtype = np.int
    else:
        result_dtype = np.float
    local_counts = np.empty(flat_shape[:-1] + (length, ),
                            dtype=result_dtype)
    # iterate over all entries in the surviving axes and compute the local
    # bincounts
    for slice_list in slicing_generator(flat_shape,
                                        axes=(len(flat_shape)-1, )):
        if weights is not None:
            current_weights = weights[slice_list]
        else:
            current_weights = None
        local_counts[slice_list] = np.bincount(
                                        data[slice_list],
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
