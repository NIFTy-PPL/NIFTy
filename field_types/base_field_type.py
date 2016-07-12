# -*- coding: utf-8 -*-

import numpy as np


class FieldType(object):
    def __init__(self, shape, dtype):
        try:
            new_shape = tuple([int(i) for i in shape])
        except TypeError:
            new_shape = (int(shape), )
        self._shape = new_shape

        self._dtype = np.dtype(dtype)

        self._dof = self._get_dof()

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def dof(self):
        return self._dof

    def _get_dof(self):
        if issubclass(self.dtype.type, np.complexfloating):
            multiplicator = 2
        else:
            multiplicator = 1

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
