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


class LMSpace(Space):
    """
        ..       __
        ..     /  /
        ..    /  /    __ ____ ___
        ..   /  /   /   _    _   |
        ..  /  /_  /  / /  / /  /
        ..  \___/ /__/ /__/ /__/  space class

        NIFTY subclass for spherical harmonics components, for representations
        of fields on the two-sphere.

        Parameters
        ----------
        lmax : int
            The maximal :math:`l` value of any spherical harmonics
            :math:`Y_{lm}` that is represented in this Space.

        Attributes
        ----------
        lmax : int
            The maximal :math:`l` value of any spherical harmonics
            :math:`Y_{lm}` that is represented in this Space.
        mmax : int
            The maximal :math:`m` value of any spherical harmonic
            :math:`Y_{lm}` that is represented in this Space.
        dim : np.int
            Total number of dimensionality, i.e. the number of pixels.
        harmonic : bool
            Specifies whether the space is a signal or harmonic space.
        total_volume : np.float
            The total volume of the space.
        shape : tuple of np.ints
            The shape of the space's data array.

        See Also
        --------
        hp_space : A class for the HEALPix discretization of the sphere [#]_.
        gl_space : A class for the Gauss-Legendre discretization of the
            sphere [#]_.

        Raises
        ------
        ValueError
            If given lmax is negative.

        Notes
        -----
            This implementation implicitly sets the mmax parameter to lmax.

        References
        ----------
        .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
               High-Resolution Discretization and Fast Analysis of Data
               Distributed on the Sphere", *ApJ* 622..759G.
        .. [#] M. Reinecke and D. Sverre Seljebotn, 2013, "Libsharp - spherical
               harmonic transforms revisited";
               `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_
    """

    def __init__(self, lmax):
        super(LMSpace, self).__init__()
        self._needed_for_hash += ["_lmax"]
        self._lmax = self._parse_lmax(lmax)

    def __repr__(self):
        return ("LMSpace(lmax=%r)" % self.lmax)

    @property
    def harmonic(self):
        return True

    @property
    def shape(self):
        return (self.dim, )

    @property
    def dim(self):
        l = self.lmax
        # the LMSpace consists of the full triangle (including -m's!),
        # minus two little triangles if mmax < lmax
        # dim = (((2*(l+1)-1)+1)**2/4 - 2 * (l-m)(l-m+1)/2
        # dim = np.int((l+1)**2 - (l-m)*(l-m+1.))
        # We fix l == m
        return np.int((l+1)*(l+1))

    @property
    def total_volume(self):
        # the individual pixels have a fixed volume of 1.
        return np.float64(self.dim)

    def scalar_dvol(self):
        return 1.

    def get_k_length_array(self):
        lmax = self.lmax
        ldist = np.empty((self.dim,), dtype=np.float64)
        ldist[0:lmax+1] = np.arange(lmax+1, dtype=np.float64)
        tmp = np.empty((2*lmax+2), dtype=np.float64)
        tmp[0::2] = np.arange(lmax+1)
        tmp[1::2] = np.arange(lmax+1)
        idx = lmax+1
        for l in range(1, lmax+1):
            ldist[idx:idx+2*(lmax+1-l)] = tmp[2*l:]
            idx += 2*(lmax+1-l)
        return ldist

    def get_unique_k_lengths(self):
        return np.arange(self.lmax+1, dtype=np.float64)

    @staticmethod
    def _kernel(x, sigma):
        res = x+1.
        res *= x
        res *= -0.5*sigma*sigma
        np.exp(res, out=res)
        return res

    def get_fft_smoothing_kernel_function(self, sigma):
        # cf. "All-sky convolution for polarimetry experiments"
        # by Challinor et al.
        # http://arxiv.org/abs/astro-ph/0008228
        return lambda x: self._kernel(x, sigma)

    @property
    def lmax(self):
        """ Returns the maximal :math:`l` value of any spherical harmonics
        :math:`Y_{lm}` that is represented in this Space.
        """
        return self._lmax

    @property
    def mmax(self):
        """ Returns the maximal :math:`m` value of any spherical harmonic
        :math:`Y_{lm}` that is represented in this Space. As :math:`m` goes
        from :math:`-l` to :math:`l` for every :math:`l` this just returns the
        same as lmax.

        See Also
        --------
        lmax : Returns the maximal :math:`l`-value of the spherical harmonics
            being used.

        """

        return self._lmax

    def _parse_lmax(self, lmax):
        lmax = np.int(lmax)
        if lmax < 0:
            raise ValueError("lmax must be >=0.")
        return lmax

    def get_default_codomain(self):
        from .. import GLSpace
        return GLSpace(self.lmax+1, self.mmax*2+1)

    def check_codomain(self, codomain):
        from .. import GLSpace, HPSpace
        if not isinstance(codomain, (GLSpace, HPSpace)):
            raise TypeError("codomain must be a GLSpace.")
