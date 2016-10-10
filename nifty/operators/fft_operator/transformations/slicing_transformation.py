# -*- coding: utf-8 -*-

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
