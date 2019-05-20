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

from .utilities import iscomplextype
import numpy as np
import pypocketfft

_nthreads = 1


def nthreads():
    return _nthreads


def set_nthreads(nthr):
    global _nthreads
    _nthreads = nthr


# FIXME this should not be necessary ... no one should call a complex FFT
# with a float array.
def _make_complex(a):
    if a.dtype in (np.complex64, np.complex128):
        return a
    if a.dtype == np.float64:
        return a.astype(np.complex128)
    if a.dtype == np.float32:
        return a.astype(np.complex64)
    raise NotImplementedError


def fftn(a, axes=None):
    return pypocketfft.fftn(_make_complex(a), axes=axes, nthreads=_nthreads)


def rfftn(a, axes=None):
    return pypocketfft.rfftn(a, axes=axes, nthreads=_nthreads)


def ifftn(a, axes=None):
    # FIXME this is a temporary fix and can be done more elegantly
    if axes is None:
        fct = 1./a.size
    else:
        fct = 1./np.prod(np.take(a.shape, axes))
    return pypocketfft.ifftn(_make_complex(a), axes=axes, fct=fct,
                             nthreads=_nthreads)


def hartley(a, axes=None):
    return pypocketfft.hartley2(a, axes=axes, nthreads=_nthreads)


# Do a real-to-complex forward FFT and return the _full_ output array
def my_fftn_r2c(a, axes=None):
    # Check if the axes provided are valid given the shape
    if axes is not None and \
            not all(axis < len(a.shape) for axis in axes):
        raise ValueError("Provided axes do not match array shape")
    if iscomplextype(a.dtype):
        raise TypeError("Transform requires real-valued input arrays.")

    tmp = rfftn(a, axes=axes)

    def _fill_complex_array(tmp, res, axes):
        if axes is None:
            axes = tuple(range(tmp.ndim))
        lastaxis = axes[-1]
        ntmplast = tmp.shape[lastaxis]
        slice1 = [slice(None)]*lastaxis + [slice(0, ntmplast)]
        res[tuple(slice1)] = tmp

        def _fill_upper_half_complex(tmp, res, axes):
            lastaxis = axes[-1]
            nlast = res.shape[lastaxis]
            ntmplast = tmp.shape[lastaxis]
            nrem = nlast - ntmplast
            slice1 = [slice(None)]*lastaxis + [slice(ntmplast, None)]
            slice2 = [slice(None)]*lastaxis + [slice(nrem, 0, -1)]
            for i in axes[:-1]:
                slice1[i] = slice(1, None)
                slice2[i] = slice(None, 0, -1)
            # np.conjugate(tmp[slice2], out=res[slice1])
            res[tuple(slice1)] = np.conjugate(tmp[tuple(slice2)])
            for i, ax in enumerate(axes[:-1]):
                dim1 = tuple([slice(None)]*ax + [slice(0, 1)])
                axes2 = axes[:i] + axes[i+1:]
                _fill_upper_half_complex(tmp[dim1], res[dim1], axes2)

        _fill_upper_half_complex(tmp, res, axes)
        return res

    return _fill_complex_array(tmp, np.empty_like(a, dtype=tmp.dtype), axes)


def my_fftn(a, axes=None):
    return fftn(a, axes=axes)
