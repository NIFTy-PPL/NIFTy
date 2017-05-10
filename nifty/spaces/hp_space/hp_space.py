# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

import numpy as np

from nifty.spaces.space import Space


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
        nside :
            Resolution parameter for the HEALPix discretization, resulting in
            ``12*nside**2`` pixels. Must be positive.

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
        """
            Sets the attributes for a hp_space class instance.

            Parameters
            ----------
            nside : int
                Resolution parameter for the HEALPix discretization, resulting
                in ``12*nside**2`` pixels. Must be positive.

            Returns
            -------
            None

            Raises
            ------
            ValueError
                If input `nside` is invalid.

        """

        super(HPSpace, self).__init__()

        self._nside = self._parse_nside(nside)

    # ---Mandatory properties and methods---

    @property
    def harmonic(self):
        return False

    @property
    def shape(self):
        return (np.int(12 * self.nside ** 2),)

    @property
    def dim(self):
        return np.int(12 * self.nside ** 2)

    @property
    def total_volume(self):
        return 4 * np.pi

    def copy(self):
        """Returns a copied version of this HPSpace.
			
        Returns
        -------
		HPSpace : A copy of this object.
        """
        return self.__class__(nside=self.nside)

    def weight(self, x, power=1, axes=None, inplace=False):
        """ Weights a field living on this space with a specified amount of volume-weights.

		Weights hereby refer to integration weights, as they appear in discretized integrals.
		Per default, this function mutliplies each bin of the field x by its volume, which lets
		it behave like a density (top form). However, different powers of the volume can be applied
		with the power parameter.
        Parameters
        ----------
        x : Field
            A field with this space as domain to be weighted.
        power : int, *optional*
            The power to which the volume-weight is raised.
            (default: 1).
        axes : {int, tuple}, *optional*
            This should not be used. It does nothing.
        inplace : bool, *optional*
            If this is True, the weighting is done on the values of x,
			if it is False, x is not modified and this method returns a 
			weighted copy of x
            (default: False).

        Returns
        -------
		Field
			A weighted version of x, with volume-weights raised to power.
            
        """    

        weight = ((4 * np.pi) / (12 * self.nside**2))**power

        if inplace:
            x *= weight
            result_x = x
        else:
            result_x = x * weight

        return result_x

    def get_distance_array(self, distribution_strategy):
        """This should not be used, it just raises an error when called.
        
        Raises
        ------
        NotImplementedError
            Always when called.
        """
        raise NotImplementedError

    def get_fft_smoothing_kernel_function(self, sigma):
        """This should not be used, it just raises an error when called.
        
        Raises
        ------
        NotImplementedError
            Always when called.
        """
        raise NotImplementedError

    # ---Added properties and methods---

    @property
    def nside(self):
        return self._nside

    def _parse_nside(self, nside):
        nside = int(nside)
        if nside < 1:
            raise ValueError("nside must be >=1.")
        return nside

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group['nside'] = self.nside
        return None

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        result = cls(
            nside=hdf5_group['nside'][()],
            )
        return result
