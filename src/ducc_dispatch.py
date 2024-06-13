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
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.


import operator

import numpy as np
import scipy.fft

from .config import _config

_nthreads = 1


def nthreads():
    return _nthreads


def set_nthreads(nthr):
    global _nthreads
    _nthreads = int(nthr)


def _scipy_fftn(a, axes=None):
    return scipy.fft.fftn(a, axes=axes, workers=_nthreads)


def _scipy_ifftn(a, axes=None):
    return scipy.fft.ifftn(a, axes=axes, workers=_nthreads)


def _scipy_hartley(a, axes=None):
    tmp = scipy.fft.fftn(a, axes=axes, workers=_nthreads)
    c = _config.get("hartley_convention")
    add_or_sub = operator.add if c == "non_canonical_hartley" else operator.sub
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
        return my_fft.c2c(a, axes=axes, nthreads=max(_nthreads, 0))


    def ifftn(a, axes=None):
        return my_fft.c2c(a, axes=axes, inorm=2, forward=False,
                          nthreads=max(_nthreads, 0))


    def hartley(a, axes=None):
        c = _config.get("hartley_convention")
        ht = my_fft.genuine_hartley if c == "non_canonical_hartley" else my_fft.genuine_fht
        return ht(a, axes=axes, nthreads=max(_nthreads, 0))


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
