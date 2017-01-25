# -*- coding: utf-8 -*-

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


def hermitianize_gaussian(x, axes=None):
    # make the point inversions
    flipped_x = _hermitianize_inverter(x, axes=axes)
    flipped_x = flipped_x.conjugate()
    # check if x was already hermitian
    if (x == flipped_x).all():
        return x
    # average x and flipped_x.
    # Correct the variance by multiplying sqrt(0.5)
    x = (x + flipped_x) * np.sqrt(0.5)
    # The fixed points of the point inversion must not be avaraged.
    # Hence one must multiply them again with sqrt(0.5)
    # -> Get the middle index of the array
    mid_index = np.array(x.shape, dtype=np.int) // 2
    dimensions = mid_index.size
    # Use ndindex to iterate over all combinations of zeros and the
    # mid_index in order to correct all fixed points.
    if axes is None:
        axes = xrange(dimensions)

    ndlist = [2 if i in axes else 1 for i in xrange(dimensions)]
    ndlist = tuple(ndlist)
    for i in np.ndindex(ndlist):
        temp_index = tuple(i * mid_index)
        x[temp_index] *= np.sqrt(0.5)
    try:
        x.hermitian = True
    except(AttributeError):
        pass

    return x


def hermitianize(x, axes=None):
    # make the point inversions
    flipped_x = _hermitianize_inverter(x, axes=axes)
    flipped_x = flipped_x.conjugate()

    # average x and flipped_x.
    # x = (x + flipped_x) / 2.
    result_x = x + flipped_x
    result_x /= 2.

#    try:
#        x.hermitian = True
#    except(AttributeError):
#        pass

    return result_x


def _hermitianize_inverter(x, axes):
    # calculate the number of dimensions the input array has
    dimensions = len(x.shape)
    # prepare the slicing object which will be used for mirroring
    slice_primitive = [slice(None), ] * dimensions
    # copy the input data
    y = x.copy()

    if axes is None:
        axes = xrange(dimensions)

    # flip in the desired directions
    for i in axes:
        slice_picker = slice_primitive[:]
        slice_picker[i] = slice(1, None, None)
        slice_picker = tuple(slice_picker)

        slice_inverter = slice_primitive[:]
        slice_inverter[i] = slice(None, 0, -1)
        slice_inverter = tuple(slice_inverter)

        try:
            y.set_data(to_key=slice_picker, data=y,
                       from_key=slice_inverter)
        except(AttributeError):
            y[slice_picker] = y[slice_inverter]
    return y


def direct_vdot(x, y):
    # the input could be fields. Try to extract the data
    try:
        x = x.get_val()
    except(AttributeError):
        pass
    try:
        y = y.get_val()
    except(AttributeError):
        pass
    # try to make a direct vdot
    try:
        return x.vdot(y)
    except(AttributeError):
        pass

    try:
        return y.vdot(x)
    except(AttributeError):
        pass

    # fallback to numpy
    return np.vdot(x, y)


def convert_nested_list_to_object_array(x):
    # if x is a nested_list full of ndarrays all having the same size,
    # np.shape returns the shape of the ndarrays, too, i.e. too many
    # dimensions
    possible_shape = np.shape(x)
    # Check if possible_shape goes too deep.
    dimension_counter = 0
    current_extract = x
    for i in xrange(len(possible_shape)):
        if not isinstance(current_extract, list) and \
                not isinstance(current_extract, tuple):
            break
        current_extract = current_extract[0]
        dimension_counter += 1
    real_shape = possible_shape[:dimension_counter]
    # if the numpy array was not encapsulated at all, return x directly
    if real_shape == ():
        return x
    # Prepare the carrier-object
    carrier = np.empty(real_shape, dtype=np.object)
    for i in xrange(reduce(lambda x, y: x * y, real_shape)):
        ii = np.unravel_index(i, real_shape)
        try:
            carrier[ii] = x[ii]
        except(TypeError):
            extracted = x
            for j in xrange(len(ii)):
                extracted = extracted[ii[j]]
            carrier[ii] = extracted
    return carrier


def field_map(ishape, function, *args):
    if ishape == ():
        return function(*args)
    else:
        if args == ():
            result = np.empty(ishape, dtype=np.object)
            for i in xrange(reduce(lambda x, y: x * y, ishape)):
                ii = np.unravel_index(i, ishape)
                result[ii] = function()
            return result
        else:
            # define a helper function in order to clip the get-indices
            # to be suitable for the foreign arrays in args.
            # This allows you to do operations, like adding to fields
            # with ishape (3,4,3) and (3,4,1)
            def get_clipped(w, ind):
                w_shape = np.array(np.shape(w))
                get_tuple = tuple(np.clip(ind, 0, w_shape - 1))
                return w[get_tuple]

            result = np.empty_like(args[0])
            for i in xrange(reduce(lambda x, y: x * y, result.shape)):
                ii = np.unravel_index(i, result.shape)
                result[ii] = function(
                    *map(
                        lambda z: get_clipped(z, ii), args
                    )
                )
                # result[ii] = function(*map(lambda z: z[ii], args))
            return result


def cast_axis_to_tuple(axis, length):
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

    # shift negative indices to positive ones
    axis = tuple(item if (item >= 0) else (item + length) for item in axis)

    # Deactivated this, in order to allow for the ComposedOperator
    # remove duplicate entries
    # axis = tuple(set(axis))

    # assert that all entries are elements in [0, length]
    for elem in axis:
        assert (0 <= elem < length)

    return axis


def complex_bincount(x, weights=None, minlength=None):
    try:
        complex_weights_Q = issubclass(weights.dtype.type,
                                       np.complexfloating)
    except AttributeError:
        complex_weights_Q = False

    if complex_weights_Q:
        real_bincount = x.bincount(weights=weights.real,
                                   minlength=minlength)
        imag_bincount = x.bincount(weights=weights.imag,
                                   minlength=minlength)
        return real_bincount + imag_bincount
    else:
        return x.bincount(weights=weights, minlength=minlength)


def get_default_codomain(domain):
    from nifty.spaces import RGSpace, HPSpace, GLSpace, LMSpace
    from nifty.operators.fft_operator.transformations import RGRGTransformation, \
        HPLMTransformation, GLLMTransformation, LMGLTransformation

    if isinstance(domain, RGSpace):
        return RGRGTransformation.get_codomain(domain)
    elif isinstance(domain, HPSpace):
        return HPLMTransformation.get_codomain(domain)
    elif isinstance(domain, GLSpace):
        return GLLMTransformation.get_codomain(domain)
    elif isinstance(domain, LMSpace):
        # TODO: get the preferred transformation path from config
        return LMGLTransformation.get_codomain(domain)
    else:
        raise TypeError('ERROR: unknown domain')


def parse_domain(domain):
    from nifty.spaces.space import Space
    if domain is None:
        domain = ()
    elif isinstance(domain, Space):
        domain = (domain,)
    elif not isinstance(domain, tuple):
        domain = tuple(domain)

    for d in domain:
        if not isinstance(d, Space):
            raise TypeError(
                "Given object contains something that is not a "
                "nifty.space.")
    return domain


def parse_field_type(field_type):
    from nifty.field_types import FieldType
    if field_type is None:
        field_type = ()
    elif isinstance(field_type, FieldType):
        field_type = (field_type,)
    elif not isinstance(field_type, tuple):
        field_type = tuple(field_type)

    for ft in field_type:
        if not isinstance(ft, FieldType):
            raise TypeError(
                "Given object is not a nifty.FieldType.")
    return field_type
