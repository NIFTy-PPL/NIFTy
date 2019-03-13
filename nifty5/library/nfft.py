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
# Copyright(C) 2018-2019 Max-Planck-Society
#
# Resolve is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

import nifty5 as ift


class NFFTOperator(ift.LinearOperator):
    """Performs a non-equidistant Fourier transform, i.e. a Fourier transform
    followed by a degridding operation.

    Parameters
    ----------
    domain : RGSpace
        Domain of the operator. It has to be two-dimensional and have shape
        `(2N, 2N)`. The coordinates of the lower left pixel of the dirty image
        are `(-N,-N)`, and of the upper right pixel `(N-1,N-1)`.
    uv : numpy.ndarray
        2D numpy array of type float64 and shape (M,2), where M is the number
        of measurements. uv[i,0] and uv[i,1] contain the u and v coordinates
        of measurement #i, respectively. All coordinates must lie in the range
        `[-0.5; 0,5[`.
    """
    def __init__(self, domain, uv):
        from pynfft.nfft import NFFT
        npix = domain.shape[0]
        assert npix == domain.shape[1]
        assert len(domain.shape) == 2
        assert type(npix) == int, "npix must be integer"
        assert npix > 0 and (
            npix % 2) == 0, "npix must be an even, positive integer"
        assert isinstance(uv, np.ndarray), "uv must be a Numpy array"
        assert uv.dtype == np.float64, "uv must be an array of float64"
        assert uv.ndim == 2, "uv must be a 2D array"
        assert uv.shape[0] > 0, "at least one point needed"
        assert uv.shape[1] == 2, "the second dimension of uv must be 2"
        assert np.all(uv >= -0.5) and np.all(uv <= 0.5),\
            "all coordinates must lie between -0.5 and 0.5"

        self._domain = ift.DomainTuple.make(domain)
        self._target = ift.DomainTuple.make(
            ift.UnstructuredDomain(uv.shape[0]))
        self._capability = self.TIMES | self.ADJOINT_TIMES

        self.npt = uv.shape[0]
        self.plan = NFFT(self.domain.shape, self.npt, m=6)
        self.plan.x = uv
        self.plan.precompute()

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            self.plan.f_hat = x.to_global_data()
            res = self.plan.trafo().copy()
        else:
            self.plan.f = x.to_global_data()
            res = self.plan.adjoint().copy()
        return ift.Field.from_global_data(self._tgt(mode), res)
