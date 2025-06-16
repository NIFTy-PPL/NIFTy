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
# Copyright(C) 2013-2021 Max-Planck-Society
# Copyright(C) 2025 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.


import operator
from warnings import warn

import numpy as np
import scipy.fft

from ..config import _config

_nthreads = 1


def nthreads():
    return _nthreads


def set_nthreads(nthr):
    global _nthreads
    _nthreads = int(nthr)


try:
    from pyvkfft.fft import fftn as vkfftn
    from pyvkfft.fft import ifftn as vkifftn
    def _force_complex(a):
        # Vkfft only works on complex inputs
        if a.dtype == np.float32:
            return a.astype(np.complex64)
        if a.dtype == np.float64:
            return a.astype(np.complex128)
        return a
    def cufftn(a, *args, **kwargs):
        return vkfftn(_force_complex(a), *args, **kwargs)
    def cuifftn(a, *args, **kwargs):
        return vkifftn(_force_complex(a), *args, **kwargs)

except ImportError:
    try:
        import cupy

        warn("Using cupy FFT. Consider to install the significantly faster pyvkfft")
        cufftn = cupy.fft.fftn
        cuifftn = cupy.fft.ifftn
    except ImportError:
        pass

def _scipy_fftn(a, axes=None):
    from .any_array import AnyArray
    if a.device_id > -1:
        return AnyArray(cufftn(a._val, axes=axes))
    return AnyArray(scipy.fft.fftn(a._val, axes=axes, workers=_nthreads))


def _scipy_ifftn(a, axes=None):
    from .any_array import AnyArray
    if a.device_id > -1:
        return AnyArray(cuifftn(a._val, axes=axes))
    return AnyArray(scipy.fft.ifftn(a._val, axes=axes, workers=_nthreads))


def _scipy_hartley(a, axes=None):
    from .any_array import AnyArray
    if a.device_id > -1:
        tmp = AnyArray(cufftn(a._val, axes=axes))
    else:
        tmp = AnyArray(scipy.fft.fftn(a._val, axes=axes, workers=_nthreads))
    assert isinstance(tmp, AnyArray)
    c = _config.get("hartley_convention")
    add_or_sub = operator.add if c == "non_canonical_hartley" else operator.sub
    assert isinstance(tmp.real, AnyArray)
    assert isinstance(tmp.imag, AnyArray)
    assert isinstance(add_or_sub(tmp.real, tmp.imag), AnyArray)
    return add_or_sub(tmp.real, tmp.imag)


def _scipy_vdot(a, b):
    from .logger import logger
    if (isinstance(a, np.ndarray) and a.dtype == np.float32) or \
    (isinstance(b, np.ndarray) and b.dtype == np.float32):
        logger.warning("Calling np.vdot in single precision may lead to inaccurate results")
    return np.vdot(a, b)


try:
    import ducc0.fft as my_fft
    import ducc0.misc


    def fftn(a, axes=None):
        from .any_array import AnyArray
        if a.device_id > -1:
            a = cufftn(a._val, axes=axes)
        else:
            a = my_fft.c2c(a._val, axes=axes, nthreads=max(_nthreads, 0))
        return AnyArray(a)


    def ifftn(a, axes=None):
        from .any_array import AnyArray
        if a.device_id > -1:
            a = cuifftn(a._val, axes=axes)
        else:
            a = my_fft.c2c(a._val, axes=axes, inorm=2, forward=False,
                           nthreads=max(_nthreads, 0))
        return AnyArray(a)


    def hartley(a, axes=None):
        from .any_array import AnyArray
        if a.device_id > -1:
            return _scipy_hartley(a, axes)
        c = _config.get("hartley_convention")
        ht = my_fft.genuine_hartley if c == "non_canonical_hartley" else my_fft.genuine_fht
        a = ht(a._val, axes=axes, nthreads=max(_nthreads, 0))
        return AnyArray(a)


    def vdot(a, b):
        if isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.integer):
            a = a.astype(np.float64)
        if isinstance(b, np.ndarray) and np.issubdtype(b.dtype, np.integer):
            b = b.astype(np.float64)
        return ducc0.misc.vdot(a, b)

except ImportError:
    fftn = _scipy_fftn
    ifftn = _scipy_ifftn
    hartley = _scipy_hartley
    vdot = _scipy_vdot
