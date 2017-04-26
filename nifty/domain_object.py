# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

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
            for key in vars(self).keys():
                item1 = vars(self)[key]
                if key in self._ignore_for_hash or key == '_ignore_for_hash':
                    continue
                item2 = vars(x)[key]
                if item1 != item2:
                    return False
            return True
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
