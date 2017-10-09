# Data object module for NIFTy that uses simple numpy ndarrays.

import numpy as np
from numpy import ndarray as data_object
from numpy import full, empty, sqrt, ones, zeros, vdot, abs, bincount, exp, log
from ..nifty_utilities import cast_iseq_to_tuple, get_slice_list
from functools import reduce

def from_object(object, dtype=None, copy=True):
    return np.array(object, dtype=dtype, copy=copy)
