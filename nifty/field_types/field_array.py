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
#
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from field_type import FieldType


class FieldArray(FieldType):

    def __init__(self, shape):
        super(FieldArray, self).__init__()
        try:
            self._shape = tuple([int(i) for i in shape])
        except TypeError:
            self._shape = (int(shape), )

    @property
    def shape(self):
        return self._shape

    @property
    def dim(self):
        return reduce(lambda x, y: x*y, self.shape)

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group['shape'] = self.shape
        return None

    @classmethod
    def _from_hdf5(cls, hdf5_group, loopback_get):
        result = cls(shape=hdf5_group['shape'][:])
        return result
