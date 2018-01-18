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


class GLSpace(Space):
    """NIFTy subclass for Gauss-Legendre pixelizations [#]_ of the two-sphere.

    Parameters
    ----------
    nlat : int
        Number of latitudinal bins (or rings) that are used for this
        pixelization.
    nlon : int, *optional*
        Number of longitudinal bins that are used for this pixelization.
        Default value is 2*nlat + 1.

    Raises
    ------
    ValueError
        If input `nlat` or `nlon` is invalid.

    See Also
    --------
    HPSpace, LMSpace

    References
    ----------
    .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
           harmonic transforms revisited";
           `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_
    .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
           High-Resolution Discretization and Fast Analysis of Data
           Distributed on the Sphere", *ApJ* 622..759G.
    """

    def __init__(self, nlat, nlon=None):
        super(GLSpace, self).__init__()
        self._needed_for_hash += ["_nlat", "_nlon"]

        self._nlat = int(nlat)
        if self._nlat < 1:
            raise ValueError("nlat must be a positive number.")
        if nlon is None:
            self._nlon = 2*self._nlat - 1
        else:
            self._nlon = int(nlon)
            if self._nlon < 1:
                raise ValueError("nlon must be a positive number.")
        self._dvol = None

    def __repr__(self):
        return ("GLSpace(nlat=%r, nlon=%r)" % (self.nlat, self.nlon))

    @property
    def harmonic(self):
        return False

    @property
    def shape(self):
        return (np.int((self.nlat * self.nlon)),)

    @property
    def dim(self):
        return np.int((self.nlat * self.nlon))

    def scalar_dvol(self):
        return None

    # MR FIXME: this is potentially wasteful, since the return array is
    #           blown up by a factor of self.nlon
    def dvol(self):
        from pyHealpix import GL_weights
        if self._dvol is None:
            self._dvol = GL_weights(self.nlat, self.nlon)
        return np.repeat(self._dvol, self.nlon)

    @property
    def nlat(self):
        """ Number of latitudinal bins (or rings) that are used for this
        pixelization.
        """
        return self._nlat

    @property
    def nlon(self):
        """Number of longitudinal bins that are used for this pixelization."""
        return self._nlon

    def get_default_codomain(self):
        from .. import LMSpace
        return LMSpace(lmax=self._nlat-1, mmax=self._nlon//2)

    def check_codomain(self, codomain):
        from .. import LMSpace
        if not isinstance(codomain, LMSpace):
            raise TypeError("codomain must be a LMSpace.")
