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

import abc
import numpy as np

from .... import nifty_utilities as utilities
from .transformation import Transformation


class SlicingTransformation(Transformation):

    def transform(self, val, axes=None):
        return_shape = np.array(val.shape)
        return_shape[list(axes)] = self.codomain.shape
        return_shape = tuple(return_shape)
        return_val = np.empty(return_shape,dtype=val.dtype)

        for slice_list in utilities.get_slice_list(val.shape, axes):
            return_val[slice_list] = self._transformation_of_slice(
                                                               val[slice_list])
        return return_val

    @abc.abstractmethod
    def _transformation_of_slice(self, inp):
        raise NotImplementedError
