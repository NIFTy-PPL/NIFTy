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

from __future__ import division
import numpy as np
from .space import Space
from ..field import Field, exp
from .. import dobj


class LMSpace(Space):
    """NIFTY subclass for spherical harmonics components, for representations
    of fields on the two-sphere.

    Parameters
    ----------
    lmax : int
        The maximum :math:`l` value of any spherical harmonics
        :math:`Y_{lm}` that is represented in this Space.
        Must be >=0.

    mmax : int *optional*
        The maximum :math:`m` value of any spherical harmonics
        :math:`Y_{lm}` that is represented in this Space.
        If not supplied, it is set to lmax.
        Must be >=0 and <=lmax.

    See Also
    --------
    HPSpace : A class for the HEALPix discretization of the sphere [#]_.
    GLSpace : A class for the Gauss-Legendre discretization of the
        sphere [#]_.

    References
    ----------
    .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
           High-Resolution Discretization and Fast Analysis of Data
           Distributed on the Sphere", *ApJ* 622..759G.
    .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
           harmonic transforms revisited";
           `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_
    """

    def __init__(self, lmax, mmax=None):
        super(LMSpace, self).__init__()
        self._needed_for_hash += ["_lmax", "_mmax"]
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
        return (self.dim, )

    @property
    def dim(self):
        l = self._lmax
        m = self._mmax
        # the LMSpace consists of the full triangle (including -m's!),
        # minus two little triangles if mmax < lmax
        return (l+1)**2 - (l-m)*(l-m+1)

    def scalar_dvol(self):
        return 1.

    def get_k_length_array(self):
        lmax = self._lmax
        mmax = self._mmax
        ldist = np.empty((self.dim,), dtype=np.float64)
        ldist[0:lmax+1] = np.arange(lmax+1, dtype=np.float64)
        tmp = np.empty((2*lmax+2), dtype=np.float64)
        tmp[0::2] = np.arange(lmax+1)
        tmp[1::2] = np.arange(lmax+1)
        idx = lmax+1
        for m in range(1, mmax+1):
            ldist[idx:idx+2*(lmax+1-m)] = tmp[2*m:]
            idx += 2*(lmax+1-m)
        return Field((self,), dobj.from_global_data(ldist))

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
        """ Returns the maximum :math:`l` value of any spherical harmonic
        :math:`Y_{lm}` that is represented in this Space.
        """
        return self._lmax

    @property
    def mmax(self):
        """ Returns the maximum :math:`m` value of any spherical harmonic
        :math:`Y_{lm}` that is represented in this Space.
        """
        return self._mmax

    def get_default_codomain(self):
        from .. import GLSpace
        return GLSpace(self.lmax+1, self.mmax*2+1)

    def check_codomain(self, codomain):
        from .. import GLSpace, HPSpace
        if not isinstance(codomain, (GLSpace, HPSpace)):
            raise TypeError("codomain must be a GLSpace or HPSpace.")
