# -*- coding: utf-8 -*-

import pickle

import numpy as np

from field_type import FieldType


class FieldArray(FieldType):

    def __init__(self, shape, dtype=np.float):
        try:
            new_shape = tuple([int(i) for i in shape])
        except TypeError:
            new_shape = (int(shape), )
        self._shape = new_shape
        super(FieldArray, self).__init__(dtype=dtype)

    @property
    def shape(self):
        return self._shape

    @property
    def dim(self):
        return reduce(lambda x, y: x*y, self.shape)

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group['shape'] = self.shape
        hdf5_group['dtype'] = pickle.dumps(self.dtype)

        return None

    @classmethod
    def _from_hdf5(cls, hdf5_group, loopback_get):
        result = cls(
            hdf5_group['shape'][:],
            pickle.loads(hdf5_group['dtype'][()])
            )
        return result
