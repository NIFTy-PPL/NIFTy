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

from ..library.gridder import FinuFFT
from ..sugar import makeDomain, makeField
from .harmonic_operators import HartleyOperator
from ..domains.rg_space import RGSpace
from .linear_operator import LinearOperator


class FFTInterpolator(LinearOperator):
    """FFT Interpolation using FinuFFT and HartleyOperator

    Parameters
    ---------
    domain : RGSpace
    sampling_points : numpy.ndarray
        Positions at which to interpolate, shape (dim, ndata)
    eps :
    nthreads :

    Notes
    ----
    #FIXME Documentation from Philipp
    """
    def __init__(self, domain, pos, eps=2e-10):
        self._domain = makeDomain(domain)
        if not isinstance(pos, np.ndarray):
            raise TypeError("sampling_points need to be a numpy.ndarray")
        if pos.ndim != 2:
            raise ValueError("sampling_points must be a 2D array")
        dist = [list(dom.distances) for dom in self.domain]
        dist = np.array(dist).reshape(-1, 1)
        pos = pos / dist
        finudom = RGSpace(self.domain.shape)
        self._finu = FinuFFT(finudom, pos.T, eps)
        self._ht = HartleyOperator(self._domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._target = self._finu.domain

    def apply(self, x, mode):
        self._check_input(x, mode)
        ht = self._ht
        finu = self._finu
        nx, ny = ht.target.shape
        if mode == self.TIMES:
            x = ht(x)
            x = makeField(finu.target, np.fft.fftshift(x.val))
            x = finu.adjoint(x)
            x = x.real + x.imag
        else:
            x = finu(x + 1j*x)
            x = makeField(ht.target, np.fft.ifftshift(x.val))
            x = ht.adjoint(x)
        return x/self.domain.total_volume()
