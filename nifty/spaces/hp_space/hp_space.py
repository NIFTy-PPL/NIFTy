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

from ..space import Space


class HPSpace(Space):
    """
        ..        __
        ..      /  /
        ..     /  /___    ______
        ..    /   _   | /   _   |
        ..   /  / /  / /  /_/  /
        ..  /__/ /__/ /   ____/  space class
        ..           /__/

        NIFTY subclass for HEALPix discretizations of the two-sphere [#]_.

        Parameters
        ----------
        nside : int
            The corresponding HEALPix pixelization. The total number of pixels
            is 12*nside**2.

        Attributes
        ----------
        dim : np.int
            Total number of dimensionality, i.e. the number of pixels.
        harmonic : bool
            Specifies whether the space is a signal or harmonic space.
        nside : int
            The corresponding HEALPix pixelization. The total number of pixels
            is 12*nside**2.
        shape : tuple of np.ints
            The shape of the space's data array.

        Raises
        ------
        ValueError
            If given `nside` < 1.

        See Also
        --------
        gl_space : A class for the Gauss-Legendre discretization of the
            sphere [#]_.
        lm_space : A class for spherical harmonic components.

        References
        ----------
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_

    """

    # ---Overwritten properties and methods---

    def __init__(self, nside):
        super(HPSpace, self).__init__()
        self._needed_for_hash += ["_nside"]
        self._nside = self._parse_nside(nside)

    # ---Mandatory properties and methods---

    def __repr__(self):
        return ("HPSpace(nside=%r)" % self.nside)

    @property
    def harmonic(self):
        return False

    @property
    def shape(self):
        return (self.dim,)

    @property
    def dim(self):
        return np.int(12 * self.nside * self.nside)

    def scalar_dvol(self):
        return np.pi / (3*self._nside*self._nside)

    # ---Added properties and methods---

    @property
    def nside(self):
        """ Returns the nside of the corresponding HEALPix pixelization.
        The total number of pixels is 12*nside**2
        """
        return self._nside

    def _parse_nside(self, nside):
        nside = int(nside)
        if nside < 1:
            raise ValueError("nside must be >=1.")
        return nside

    def get_default_codomain(self):
        from .. import LMSpace
        return LMSpace(lmax=2*self.nside)

    def check_codomain(self, codomain):
        from .. import LMSpace
        if not isinstance(codomain, LMSpace):
            raise TypeError("codomain must be a LMSpace.")
