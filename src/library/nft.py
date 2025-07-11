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
# Copyright(C) 2019-2021 Max-Planck-Society
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np
from scipy.constants import speed_of_light

from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..ducc_dispatch import nthreads
from ..operators.linear_operator import LinearOperator
from ..sugar import makeDomain, makeField


class Gridder(LinearOperator):
    """Compute non-uniform 2D FFT with ducc.

    Parameters
    ----------
    target : Domain, tuple of domains or DomainTuple.
        Target domain, must be a single two-dimensional RGSpace.
    uv : np.ndarray
        Coordinates of the data-points, shape (n, 2).
    eps : float
        Requested precision, defaults to 2e-10.
    """
    def __init__(self, target, uv, eps=2e-10):
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._target = makeDomain(target)
        for ii in [0, 1]:
            if target.shape[ii] % 2 != 0:
                raise ValueError("even number of pixels is required for gridding operation")
        if (len(self._target) != 1 or not isinstance(self._target[0], RGSpace)
                or not len(self._target.shape) == 2):
            raise ValueError("need target with exactly one 2D RGSpace")
        if uv.ndim != 2:
            raise ValueError("uv must be a 2D array")
        if uv.shape[1] != 2:
            raise ValueError("second dimension of uv must have length 2")
        self._domain = DomainTuple.make(UnstructuredDomain((uv.shape[0])))
        # wasteful hack to adjust to shape required by ducc0.wgridder
        self._uvw = np.empty((uv.shape[0], 3), dtype=np.float64)
        self._uvw[:, 0:2] = uv
        self._uvw[:, 2] = 0.
        self._eps = float(eps)

    def apply(self, x, mode):
        from ducc0.wgridder import dirty2ms, ms2dirty
        self._check_input(x, mode)
        freq = np.array([speed_of_light])
        x = x.val
        nxdirty, nydirty = self._target[0].shape
        dstx, dsty = self._target[0].distances
        if mode == self.TIMES:
            res = ms2dirty(self._uvw, freq, x.reshape((-1,1)), None, nxdirty,
                           nydirty, dstx, dsty, 0, 0,
                           self._eps, False, nthreads(), 0)
        else:
            res = dirty2ms(self._uvw, freq, x, None, dstx, dsty, 0, 0,
                           self._eps, False, nthreads(), 0)
            res = res.reshape((-1,))
        return makeField(self._tgt(mode), res)


class Nufft(LinearOperator):
    """Compute non-uniform 1D, 2D and 3D FFTs.

    Parameters
    ----------
    target : Domain, tuple of domains or DomainTuple.
        Target domain, must be an RGSpace with one to three dimensions.
    pos : np.ndarray
        Coordinates of the data-points, shape (n, ndim).
    eps: float
        Requested precision, defaults to 2e-10.
    """
    def __init__(self, target, pos, eps=2e-10):
        try:
            from ducc0.nufft import nu2u, u2nu
        except ImportError:
            raise ImportError("ducc0 needs to be installed for nifty.Nufft()")
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._target = makeDomain(target)
        if not isinstance(self._target[0], RGSpace):
            raise TypeError("target needs to be an RGSpace")
        if len(self._target.shape) > 3:
            raise ValueError("Only 1D, 2D and 3D FFTs are supported")
        if pos.ndim != 2:
            raise TypeError(f"pos needs to be 2d array (got shape {pos.shape})")
        self._domain = DomainTuple.make(UnstructuredDomain((pos.shape[0])))
        dst = np.array(self._target[0].distances)
        pos = (2*np.pi*pos*dst) % (2*np.pi)
        self._args = dict(nthreads=1, epsilon=float(eps), coord=pos)

    def apply(self, x, mode):
        from ducc0.nufft import nu2u, u2nu

        self._check_input(x, mode)
        if mode == self.TIMES:
            res = np.empty(self.target.shape, dtype=x.dtype)
            nu2u(points=x.val, out=res, forward=False, **self._args)
            res = res.real
        else:
            res = u2nu(grid=x.val.astype('complex128'), forward=True, **self._args)
            #if res.ndim == 0:
                #res = np.array([res])
        return makeField(self._tgt(mode), res)


def FinuFFT(*args, **kwargs):
    from warnings import warn

    warn("This operator has been renamed to Nufft. "
         "FinuFFT won't be present in the upcoming release.", DeprecationWarning)
    return Nufft(*args, **kwargs)
