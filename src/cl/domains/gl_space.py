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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .structured_domain import StructuredDomain


class GLSpace(StructuredDomain):
    """Represents a 2-sphere with Gauss-Legendre pixelization.

    Its harmonic partner domain is the
    :class:`~nifty8.domains.lm_space.LMSpace`.

    Parameters
    ----------
    nlat : int
        Number of latitudinal bins (or rings) that are used for this
        pixelization.
    nlon : int, optional
        Number of longitudinal bins that are used for this pixelization.
        Default value is 2*nlat - 1.
    """

    _needed_for_hash = ["_nlat", "_nlon"]

    def __init__(self, nlat, nlon=None):
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
        return "GLSpace(nlat={}, nlon={})".format(self.nlat, self.nlon)

    @property
    def harmonic(self):
        return False

    @property
    def shape(self):
        return (int(self.nlat * self.nlon),)

    @property
    def size(self):
        return int(self.nlat * self.nlon)

    @property
    def scalar_dvol(self):
        return None

    # MR FIXME: this is potentially wasteful, since the return array is
    #           blown up by a factor of self.nlon
    @property
    def dvol(self):
        from ducc0.misc import GL_weights
        if self._dvol is None:
            self._dvol = GL_weights(self.nlat, self.nlon)
        return np.repeat(self._dvol, self.nlon)

    @property
    def total_volume(self):
        return 4*np.pi

    @property
    def nlat(self):
        """int : number of rings in this domain"""
        return self._nlat

    @property
    def nlon(self):
        """int : pixels per ring in this domain"""
        return self._nlon

    def get_default_codomain(self):
        """Returns a :class:`LMSpace` object, which is capable of storing an
        accurate representation of data residing on `self` (if this data is
        band-limited).

        Returns
        -------
        LMSpace
            The partner domain
        """
        from ..domains.lm_space import LMSpace
        mmax = self._nlon//2
        lmax = max(mmax, self._nlat-1)
        return LMSpace(lmax=lmax, mmax=mmax)

    def check_codomain(self, codomain):
        """Raises `TypeError` if `codomain` is not a matching partner domain
        for `self`.

        Notes
        -----
        This function only checks whether `codomain` is of type
        :class:`LMSpace`.
        """
        from ..domains.lm_space import LMSpace
        if not isinstance(codomain, LMSpace):
            raise TypeError("codomain must be a LMSpace.")
