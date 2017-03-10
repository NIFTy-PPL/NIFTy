# -*- coding: utf-8 -*-

import abc

import numpy as np

from keepers import Loggable,\
                    Versionable


class DomainObject(Versionable, Loggable, object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, dtype):
        self._dtype = np.dtype(dtype)
        self._ignore_for_hash = []

    def __hash__(self):
        # Extract the identifying parts from the vars(self) dict.
        result_hash = 0
        for key in sorted(vars(self).keys()):
            item = vars(self)[key]
            if key in self._ignore_for_hash or key == '_ignore_for_hash':
                continue
            result_hash ^= item.__hash__() ^ int(hash(key)/117)
        return result_hash

    def __eq__(self, x):
        if isinstance(x, type(self)):
            return hash(self) == hash(x)
        else:
            return False

    def __ne__(self, x):
        return not self.__eq__(x)

    @property
    def dtype(self):
        return self._dtype

    @abc.abstractproperty
    def shape(self):
        raise NotImplementedError(
            "There is no generic shape for DomainObject.")

    @abc.abstractproperty
    def dim(self):
        raise NotImplementedError(
            "There is no generic dim for DomainObject.")

    @abc.abstractmethod
    def weight(self, x, power=1, axes=None, inplace=False):
        raise NotImplementedError(
            "There is no generic weight-method for DomainObject.")

    def pre_cast(self, x, axes=None):
        return x

    def post_cast(self, x, axes=None):
        return x

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group.attrs['dtype'] = self.dtype.name
        return None

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        result = cls(dtype=np.dtype(hdf5_group.attrs['dtype']))
        return result
