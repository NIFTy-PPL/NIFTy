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
# Authors: Vincent Eberle und Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..library.gridder import Gridder
from ..sugar import makeDomain, makeField
from .harmonic_operators import HartleyOperator
from ..domains.rg_space import RGSpace
from .linear_operator import LinearOperator


class FFTInterpolator(LinearOperator):
    """FFT Interpolation using Gridder and HartleyOperator

    Parameters
    ---------
    domain : RGSpace
    sampling_points : numpy.ndarray
        Positions at which to interpolate, shape (dim, ndata)
    eps :
    nthreads :

    Notes
    ----
    #FIXME Documentation from Philipp ? PBCs? / Torus?
    """
    def __init__(self, domain, pos, eps=2e-10, nthreads=1):
        self._domain = makeDomain(domain)
        if not isinstance(pos, np.ndarray):
            raise TypeError("sampling_points need to be a numpy.ndarray")
        if pos.ndim != 2:
            raise ValueError("sampling_points must be a 2D array")
        if pos.shape[0] != 2:
            raise ValueError("first dimension of sampling_points must have length 2")
        for ii in [0, 1]:
            if domain.shape[ii] % 2 != 0:
                raise ValueError("even number of samples is required for gridding operation")
        dist = [list(dom.distances) for dom in self.domain]
        dist = np.array(dist).reshape(-1, 1)
        pos = pos / dist
        gridderdom = RGSpace(self.domain.shape)
        self._gridder = Gridder(gridderdom, pos.T, eps, nthreads)
        self._ht = HartleyOperator(self._domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._target = self._gridder.domain

    def apply(self, x, mode):
        self._check_input(x, mode)
        ht = self._ht
        gridder = self._gridder
        nx, ny = ht.target.shape
        if mode == self.TIMES:
            x = ht(x)
            x = makeField(gridder.target, np.fft.fftshift(x.val))
            x = gridder.adjoint(x)
            x = x.real + x.imag
        else:
            x = gridder(x + 1j*x)
            x = makeField(ht.target, np.fft.fftshift(x.val))
            x = ht.adjoint(x)
        return x/self.domain.total_volume()
