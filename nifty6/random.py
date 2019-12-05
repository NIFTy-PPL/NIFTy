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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np


class Random(object):
    @staticmethod
    def pm1(dtype, shape):
        if np.issubdtype(dtype, np.complexfloating):
            x = np.array([1+0j, 0+1j, -1+0j, 0-1j], dtype=dtype)
            x = x[np.random.randint(4, size=shape)]
        else:
            x = 2*np.random.randint(2, size=shape) - 1
        return x.astype(dtype, copy=False)

    @staticmethod
    def normal(dtype, shape, mean=0., std=1.):
        if not (np.issubdtype(dtype, np.floating) or
                np.issubdtype(dtype, np.complexfloating)):
            raise TypeError("dtype must be float or complex")
        if not np.isscalar(mean) or not np.isscalar(std):
            raise TypeError("mean and std must be scalars")
        if np.issubdtype(type(std), np.complexfloating):
            raise TypeError("std must not be complex")
        if ((not np.issubdtype(dtype, np.complexfloating)) and
                np.issubdtype(type(mean), np.complexfloating)):
            raise TypeError("mean must not be complex for a real result field")
        if np.issubdtype(dtype, np.complexfloating):
            x = np.empty(shape, dtype=dtype)
            x.real = np.random.normal(mean.real, std*np.sqrt(0.5), shape)
            x.imag = np.random.normal(mean.imag, std*np.sqrt(0.5), shape)
        else:
            x = np.random.normal(mean, std, shape).astype(dtype, copy=False)
        return x

    @staticmethod
    def uniform(dtype, shape, low=0., high=1.):
        if not np.isscalar(low) or not np.isscalar(high):
            raise TypeError("low and high must be scalars")
        if (np.issubdtype(type(low), np.complexfloating) or
                np.issubdtype(type(high), np.complexfloating)):
            raise TypeError("low and high must not be complex")
        if np.issubdtype(dtype, np.complexfloating):
            x = np.empty(shape, dtype=dtype)
            x.real = np.random.uniform(low, high, shape)
            x.imag = np.random.uniform(low, high, shape)
        elif np.issubdtype(dtype, np.integer):
            if not (np.issubdtype(type(low), np.integer) and
                    np.issubdtype(type(high), np.integer)):
                raise TypeError("low and high must be integer")
            x = np.random.randint(low, high+1, shape)
        else:
            x = np.random.uniform(low, high, shape)
        return x.astype(dtype, copy=False)
