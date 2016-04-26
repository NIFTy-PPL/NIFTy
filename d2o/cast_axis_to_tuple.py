# -*- coding: utf-8 -*-

import numpy as np
from nifty import about


def cast_axis_to_tuple(axis):
    if axis is None:
        return None
    try:
        axis = tuple([int(item) for item in axis])
    except(TypeError):
        if np.isscalar(axis):
            axis = (int(axis), )
        else:
            raise TypeError(about._errors.cstring(
              "ERROR: Could not convert axis-input to tuple of ints"))
    return axis
