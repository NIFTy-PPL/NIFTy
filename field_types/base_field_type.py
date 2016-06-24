# -*- coding: utf-8 -*-

import numpy as np


class Field_type(object):
    def __init__(self, shape, dtype):
        try:
            new_shape = tuple([int(i) for i in shape])
        except TypeError:
            new_shape = (int(shape), )
        self._shape = new_shape

        self._dtype = np.dtype(dtype)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def dof(self):
        return self._dof

    @property
    def dof_split(self):
        return self._dof_split

    def _get_dof(self, split=False):
        if issubclass(self.dtype.type, np.complexfloating):
            multiplicator = 2
        else:
            multiplicator = 1

        if split:
            dof = tuple(multiplicator*np.array(self.shape))
        else:
            dof = multiplicator*reduce(lambda x, y: x*y, self.shape)

        return dof

    def process(self, method_name, array, inplace=True, **kwargs):
        try:
            result_array = self.__getattr__(method_name)(array,
                                                         inplace,
                                                         **kwargs)
        except AttributeError:
            if inplace:
                result_array = array
            else:
                result_array = array.copy()

        return result_array

    def complement_cast(self, x, axis=None):
        return x

    def dot_contraction(self, x, axes):
        raise NotImplementedError
