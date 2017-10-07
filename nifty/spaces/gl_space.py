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
    """
        ..                 __
        ..               /  /
        ..     ____ __  /  /
        ..   /   _   / /  /
        ..  /  /_/  / /  /_
        ..  \___   /  \___/  space class
        .. /______/

        NIFTY subclass for Gauss-Legendre pixelizations [#]_ of the two-sphere.

        Parameters
        ----------
        nlat : int
            Number of latitudinal bins (or rings) that are used for this
            pixelization.
        nlon : int, *optional*
            Number of longitudinal bins that are used for this pixelization.

        Attributes
        ----------
        dim : np.int
            Total number of dimensionality, i.e. the number of pixels.
        harmonic : bool
            Specifies whether the space is a signal or harmonic space.
        nlat : int
            Number of latitudinal bins (or rings) that are used for this
            pixelization.
        nlon : int
            Number of longitudinal bins that are used for this pixelization.
        shape : tuple of np.ints
            The shape of the space's data array.

        Raises
        ------
        ValueError
            If input `nlat` or `nlon` is invalid.

        See Also
        --------
        hp_space : A class for the HEALPix discretization of the sphere [#]_.
        lm_space : A class for spherical harmonic components.

        References
        ----------
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.

    """

    # ---Overwritten properties and methods---

    def __init__(self, nlat, nlon=None):
        super(GLSpace, self).__init__()
        self._needed_for_hash += ["_nlat", "_nlon"]

        self._nlat = self._parse_nlat(nlat)
        self._nlon = self._parse_nlon(nlon)
        self._dvol = None

    # ---Mandatory properties and methods---

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

    # ---Added properties and methods---

    @property
    def nlat(self):
        """ Number of latitudinal bins (or rings) that are used for this
        pixelization.
        """
        return self._nlat

    @property
    def nlon(self):
        """ Number of longitudinal bins that are used for this pixelization.
        """
        return self._nlon

    def _parse_nlat(self, nlat):
        nlat = int(nlat)
        if nlat < 1:
            raise ValueError("nlat must be a positive number.")
        return nlat

    def _parse_nlon(self, nlon):
        if nlon is None:
            nlon = 2 * self.nlat - 1
        else:
            nlon = int(nlon)
            if nlon < 1:
                raise ValueError("nlon must be a positive number.")
        return nlon

    def get_default_codomain(self):
        from .. import LMSpace
        return LMSpace(lmax=self.nlat-1, mmax=(self.nlon-1)//2)

    def check_codomain(self, codomain):
        from .. import LMSpace
        if not isinstance(codomain, LMSpace):
            raise TypeError("codomain must be a LMSpace.")
