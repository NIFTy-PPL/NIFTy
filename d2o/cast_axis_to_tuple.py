# -*- coding: utf-8 -*-

import numbers
import numpy as np
from nifty import about


def cast_axis_to_tuple(axis):
    if axis is None:
        return None
    try:
        axis = tuple([int(item) for item in axis])
    except(TypeError):
        if isinstance(axis, numbers.Number):
            axis = (int(axis), )
        else:
            raise TypeError(about._errors.cstring(
              "ERROR: Could not convert axis-input to tuple of ints"))
    return axis
