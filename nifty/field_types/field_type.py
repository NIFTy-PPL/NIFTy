# -*- coding: utf-8 -*-

import pickle
import numpy as np
from keepers import Versionable


class FieldType(Versionable, object):
    def __init__(self, shape, dtype):
        try:
            new_shape = tuple([int(i) for i in shape])
        except TypeError:
            new_shape = (int(shape), )
        self._shape = new_shape

        self._dtype = np.dtype(dtype)

    def __hash__(self):
        # Extract the identifying parts from the vars(self) dict.
        result_hash = 0
        for (key, item) in vars(self).items():
            result_hash ^= item.__hash__() ^ int(hash(key)/117)
        return result_hash

    def __eq__(self, x):
        if isinstance(x, type(self)):
            return hash(self) == hash(x)
        else:
            return False

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def dim(self):
        raise NotImplementedError

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

    def pre_cast(self, x, axes=None):
        return x

    def post_cast(self, x, axes=None):
        return x

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
