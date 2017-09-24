# Data object module for NIFTy that uses simple numpy ndarrays.

import numpy as np
from numpy import ndarray as data_object
from numpy import full, empty, sqrt, ones, zeros, vdot, abs

def from_object(object, dtype=None, copy=True):
    return np.array(object, dtype=dtype, copy=copy)
