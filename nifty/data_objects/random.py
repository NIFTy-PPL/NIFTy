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

from builtins import object
import numpy as np


class Random(object):
    @staticmethod
    def pm1(dtype, shape):
        if issubclass(dtype, (complex, np.complexfloating)):
            x = np.array([1+0j, 0+1j, -1+0j, 0-1j], dtype=dtype)
            x = x[np.random.randint(4, size=shape)]
        else:
            x = 2*np.random.randint(2, size=shape) - 1
        return x.astype(dtype, copy=False)

    @staticmethod
    def normal(dtype, shape, mean=0., std=1.):
        if issubclass(dtype, (complex, np.complexfloating)):
            x = np.empty(shape, dtype=dtype)
            x.real = np.random.normal(mean.real, std*np.sqrt(0.5), shape)
            x.imag = np.random.normal(mean.imag, std*np.sqrt(0.5), shape)
        else:
            x = np.random.normal(mean, std, shape).astype(dtype, copy=False)
        return x

    @staticmethod
    def uniform(dtype, shape, low=0., high=1.):
        if issubclass(dtype, (complex, np.complexfloating)):
            x = np.empty(shape, dtype=dtype)
            x.real = np.random.uniform(low, high, shape)
            x.imag = np.random.uniform(low, high, shape)
        elif dtype in [np.dtype('int8'), np.dtype('int16'), np.dtype('int32'),
                       np.dtype('int64')]:
            x = np.random.random.randint(low, high+1, shape)
        else:
            x = np.random.uniform(low, high, shape)
        return x.astype(dtype, copy=False)
