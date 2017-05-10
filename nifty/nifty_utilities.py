# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

import numpy as np
from itertools import product


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
            [range(y) for x, y in enumerate(shape) if x not in axes]
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
            raise TypeError(
                "Could not convert axis-input to tuple of ints")

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
    from nifty.domain_object import DomainObject
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
