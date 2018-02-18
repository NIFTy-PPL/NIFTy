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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import division
import numpy as np
from .structured_domain import StructuredDomain
from ..field import Field, exp
from .. import dobj


class LMSpace(StructuredDomain):
    """NIFTy subclass for sets of spherical harmonic coefficients.

    Its harmonic partner spaces are :class:`~nifty4.domains.hp_space.HPSpace`
    and :class:`~nifty4.domains.gl_space.GLSpace`.

    Parameters
    ----------
    lmax : int
        The maximum :math:`l` value of any spherical harmonic coefficient
        :math:`a_{lm}` that is represented by this object.
        Must be :math:`\ge 0`.

    mmax : int, optional
        The maximum :math:`m` value of any spherical harmonic coefficient
        :math:`a_{lm}` that is represented by this object.
        If not supplied, it is set to `lmax`.
        Must be :math:`\ge 0` and :math:`\le` `lmax`.
    """

    _needed_for_hash = ["_lmax", "_mmax"]

    def __init__(self, lmax, mmax=None):
        super(LMSpace, self).__init__()
        self._lmax = np.int(lmax)
        if self._lmax < 0:
            raise ValueError("lmax must be >=0.")
        if mmax is None:
            mmax = self._lmax
        self._mmax = np.int(mmax)
        if self._mmax < 0 or self._mmax > self._lmax:
            raise ValueError("mmax must be >=0 and <=lmax.")

    def __repr__(self):
        return ("LMSpace(lmax=%r, mmax=%r)" % (self.lmax, self.mmax))

    @property
    def harmonic(self):
        return True

    @property
    def shape(self):
        return (self.size, )

    @property
    def size(self):
        l = self._lmax
        m = self._mmax
        # the LMSpace consists of the full triangle (including -m's!),
        # minus two little triangles if mmax < lmax
        return (l+1)**2 - (l-m)*(l-m+1)

    @property
    def scalar_dvol(self):
        return 1.

    def get_k_length_array(self):
        lmax = self._lmax
        mmax = self._mmax
        ldist = np.empty((self.size,), dtype=np.float64)
        ldist[0:lmax+1] = np.arange(lmax+1, dtype=np.float64)
        tmp = np.empty((2*lmax+2), dtype=np.float64)
        tmp[0::2] = np.arange(lmax+1)
        tmp[1::2] = np.arange(lmax+1)
        idx = lmax+1
        for m in range(1, mmax+1):
            ldist[idx:idx+2*(lmax+1-m)] = tmp[2*m:]
            idx += 2*(lmax+1-m)
        return Field.from_global_data(self, ldist)

    def get_unique_k_lengths(self):
        return np.arange(self.lmax+1, dtype=np.float64)

    @staticmethod
    def _kernel(x, sigma):
        # cf. "All-sky convolution for polarimetry experiments"
        # by Challinor et al.
        # http://arxiv.org/abs/astro-ph/0008228
        res = x+1.
        res *= x
        res *= -0.5*sigma*sigma
        exp(res, out=res)
        return res

    def get_fft_smoothing_kernel_function(self, sigma):
        return lambda x: self._kernel(x, sigma)

    @property
    def lmax(self):
        """int : maximum allowed :math:`l`

        The maximum :math:`l` value of any spherical harmonic
        coefficient :math:`a_{lm}` that is represented in this domain.
        """
        return self._lmax

    @property
    def mmax(self):
        """int : maximum allowed :math:`m`

        The maximum :math:`m` value of any spherical harmonic
        coefficient :math:`a_{lm}` that is represented in this domain.
        """
        return self._mmax

    def get_default_codomain(self):
        """Returns a :class:`~nifty4.domains.gl_space.GLSpace` object, which is
        capable of storing an accurate representation of data residing on
        `self`.

        Returns
        -------
        GLSpace
            The partner domain
        """
        from .. import GLSpace
        return GLSpace(self.lmax+1, self.mmax*2+1)

    def check_codomain(self, codomain):
        """Raises `TypeError` if `codomain` is not a matching partner domain
        for `self`.

        Notes
        -----
        This function only checks whether `codomain` is of type
        :class:`GLSpace` or :class:`HPSpace`.
        """
        from .. import GLSpace, HPSpace
        if not isinstance(codomain, (GLSpace, HPSpace)):
            raise TypeError("codomain must be a GLSpace or HPSpace.")
