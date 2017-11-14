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

import numpy as np

__all__ = ["hartley", "general_axpy"]

special_hartley = False
special_fill_array = False

use_numba = False

if special_hartley or special_fill_array:
    import hartley as extmod

if not special_hartley:

    def _fill_upper_half(tmp, res, axes):
        lastaxis = axes[-1]
        nlast = res.shape[lastaxis]
        ntmplast = tmp.shape[lastaxis]
        nrem = nlast - ntmplast
        slice1 = [slice(None)]*lastaxis + [slice(ntmplast, None)]
        slice2 = [slice(None)]*lastaxis + [slice(nrem, 0, -1)]
        for i in axes[:-1]:
            slice1[i] = slice(1, None)
            slice2[i] = slice(None, 0, -1)
        np.subtract(tmp[slice2].real, tmp[slice2].imag, out=res[slice1])
        for i, ax in enumerate(axes[:-1]):
            dim1 = [slice(None)]*ax + [slice(0, 1)]
            axes2 = axes[:i] + axes[i+1:]
            _fill_upper_half(tmp[dim1], res[dim1], axes2)

    def _fill_array(tmp, res, axes):
        if axes is None:
            axes = range(tmp.ndim)
        lastaxis = axes[-1]
        ntmplast = tmp.shape[lastaxis]
        slice1 = [slice(None)]*lastaxis + [slice(0, ntmplast)]
        np.add(tmp.real, tmp.imag, out=res[slice1])
        _fill_upper_half(tmp, res, axes)
        return res


def hartley(a, axes=None):
    # Check if the axes provided are valid given the shape
    if axes is not None and \
            not all(axis < len(a.shape) for axis in axes):
        raise ValueError("Provided axes do not match array shape")
    if issubclass(a.dtype.type, np.complexfloating):
        raise TypeError("Hartley tansform requires real-valued arrays.")

    if special_hartley:
        return extmod.hartley(a, np.empty_like(a), axes)
    else:
        from pyfftw.interfaces.numpy_fft import rfftn
        tmp = rfftn(a, axes=axes)
        if special_fill_array:
            return extmod.fill_array(tmp, np.empty_like(a), axes)
        else:
            return _fill_array(tmp, np.empty_like(a), axes)

if use_numba:
    from numba import complex128 as ncplx, float64 as nflt, vectorize as nvct

    @nvct([nflt(nflt, nflt, nflt), ncplx(nflt, ncplx, ncplx)], nopython=True,
          target="cpu")
    def _general_axpy(a, x, y):
        return a*x + y

    def general_axpy(a, x, y, out):
        if x.domain != y.domain or x.domain != out.domain:
            raise ValueError("Incompatible domains")
        return _general_axpy(a, x.val, y.val, out.val)

else:

    def general_axpy(a, x, y, out):
        if x.domain != y.domain or x.domain != out.domain:
            raise ValueError("Incompatible domains")

        if out is x:
            if a != 1.:
                out *= a
            out += y
        elif out is y:
            if a != 1.:
                out += a*x
            else:
                out += x
        else:
            out.copy_content_from(y)
            if a != 1.:
                out += a*x
            else:
                out += x
        return out
