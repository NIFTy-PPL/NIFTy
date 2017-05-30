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

import nifty.nifty_utilities as utilities
from transformation import Transformation


class SlicingTransformation(Transformation):

    def transform(self, val, axes=None, **kwargs):

        return_shape = np.array(val.shape)
        return_shape[list(axes)] = self.codomain.shape
        return_shape = tuple(return_shape)

        return_val = None

        for slice_list in utilities.get_slice_list(val.shape, axes):
            if return_val is None:
                return_val = val.copy_empty(global_shape=return_shape)

            data = val.get_data(slice_list, copy=False)
            data = data.get_full_data()

            data = self._transformation_of_slice(data, **kwargs)

            return_val.set_data(data=data, to_key=slice_list, copy=False)

        return return_val

    def _combine_complex_result(self, resultReal, resultImag):
        # construct correct complex dtype
        one = resultReal.dtype.type(1)
        result_dtype = np.dtype(type(one + 1j))

        result = np.empty_like(resultReal, dtype=result_dtype)
        result.real = resultReal
        result.imag = resultImag

        return result

    @abc.abstractmethod
    def _transformation_of_slice(self, inp):
        raise NotImplementedError
