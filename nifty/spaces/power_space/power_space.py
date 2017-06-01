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

import numpy as np

import d2o

from nifty.spaces.space import Space
from power_indices import PowerIndices


class PowerSpace(Space):
    """ NIFTY class for spaces of power spectra.

    Parameters
    ----------
    harmonic_partner : Space
        The harmonic Space of which this is the power space.
    distribution_strategy : str *optional*
        The distribution strategy used for the distributed_data_objects
        derived from this PowerSpace, e.g. the pindex.
        (default : 'not')
    logarithmic : bool *optional*
        True if logarithmic binning should be used (default : False).
    nbin : {int, None} *optional*
        The number of bins that should be used for power spectrum binning
        (default : None).
        if nbin == None, then nbin is set to the length of kindex.
    binbounds :  {list, array-like} *optional*
        Array-like inner boundaries of the used bins of the default
        indices.
        (default : None)
        if binbounds == None :
            Calculates the bounds from the kindex while applying the
            logarithmic and nbin keywords.

    Attributes
    ----------
    pindex : distributed_data_object
        This holds the information which pixel of the harmonic partner gets
        mapped to which power bin
    kindex : numpy.ndarray
        Sorted array of all k-modes.
    rho : numpy.ndarray
        The amount of k-modes that get mapped to one power bin is given by
        rho.
    dim : np.int
        Total number of dimensionality, i.e. the number of pixels.
    harmonic : bool
        Specifies whether the space is a signal or harmonic space.
    total_volume : np.float
        The total volume of the space.
    shape : tuple of np.ints
        The shape of the space's data array.
    config : {logarithmic, nbin, binbounds}
        Dictionary storing the values for `logarithmic`, `nbin`, and
        `binbounds` that were used during initialization.

    Notes
    -----
    A power space is the result of a projection of a harmonic space where
    k-modes of equal length get mapped to one power index.

    """

    # ---Overwritten properties and methods---

    def __init__(self, harmonic_partner,
                 distribution_strategy='not',
                 logarithmic=False, nbin=None, binbounds=None):
        super(PowerSpace, self).__init__()
        self._ignore_for_hash += ['_pindex', '_kindex', '_rho', '_k_array']

        if not isinstance(harmonic_partner, Space):
            raise ValueError(
                "harmonic_partner must be a Space.")
        if not harmonic_partner.harmonic:
            raise ValueError(
                "harmonic_partner must be a harmonic space.")
        self._harmonic_partner = harmonic_partner

        self._logarithmic = logarithmic
        self._nbin = nbin
        self._binbounds = binbounds
        tmp = PowerIndices(self.harmonic_partner, distribution_strategy)
        self._pindex, self._kindex, self._rho, self._k_array = tmp.get_index_dict(logarithmic=logarithmic,
                                                   nbin=nbin,
                                                   binbounds=binbounds)

        if nbin is not None:
            if nbin > len(self.kindex):
                self.logger.warn("nbin was set to a value being larger than "
                                 "the length of kindex!")

    def pre_cast(self, x, axes):
        """ Casts power spectrum functions to discretized power spectra.

        This function takes an array or a function. If it is an array it does
        nothing, otherwise it interpretes the function as power spectrum and
        evaluates it at every k-mode.

        Parameters
        ----------
        x : {array-like, function array-like -> array-like}
            power spectrum given either in discretized form or implicitly as a
            function
        axes : tuple of ints
            Specifies the axes of x which correspond to this space. For
            explicifying the power spectrum function, this is ignored.

        Returns
        -------
        array-like
            discretized power spectrum

        """

        if callable(x):
            return x(self.kindex)
        else:
            return x

    # ---Mandatory properties and methods---

    def __repr__(self):
        return ("PowerSpace(harmonic_partner=%r, distribution_strategy=%r, "
                "logarithmic=%r, nbin=%r, binbounds=%r)"
                % (self.harmonic_partner, self.pindex.distribution_strategy,
                   self._logarithmic, self._nbin, self._binbounds))

    @property
    def harmonic(self):
        return True

    @property
    def shape(self):
        return self.kindex.shape

    @property
    def dim(self):
        return self.shape[0]

    @property
    def total_volume(self):
        # every power-pixel has a volume of 1
        return float(reduce(lambda x, y: x*y, self.pindex.shape))

    def copy(self):
        distribution_strategy = self.pindex.distribution_strategy
        return self.__class__(harmonic_partner=self.harmonic_partner,
                              distribution_strategy=distribution_strategy,
                              logarithmic=self._logarithmic,
                              nbin=self._nbin,
                              binbounds=self._binbounds)

    def weight(self, x, power=1, axes=None, inplace=False):
        reshaper = [1, ] * len(x.shape)
        # we know len(axes) is always 1
        reshaper[axes[0]] = self.shape[0]

        weight = self.rho.reshape(reshaper)
        if power != 1:
            weight = weight ** np.float(power)

        if inplace:
            x *= weight
            result_x = x
        else:
            result_x = x*weight

        return result_x

    def get_distance_array(self, distribution_strategy):
        result = d2o.distributed_data_object(
                                self.kindex, dtype=np.float64,
                                distribution_strategy=distribution_strategy)
        return result

    def get_fft_smoothing_kernel_function(self, sigma):
        raise NotImplementedError(
            "There is no fft smoothing function for PowerSpace.")

    # ---Added properties and methods---

    @property
    def harmonic_partner(self):
        """ Returns the Space of which this is the power space.
        """
        return self._harmonic_partner

    @property
    def logarithmic(self):
        return self._logarithmic

    @property
    def nbin(self):
        return self._nbin

    @property
    def binbounds(self):
        return self._binbounds

    @property
    def pindex(self):
        """ A distributed_data_object having the shape of the harmonic partner
        space containing the indices of the power bin a pixel belongs to.
        """
        return self._pindex

    @property
    def kindex(self):
        """ Sorted array of all k-modes.
        """
        return self._kindex

    @property
    def rho(self):
        """Degeneracy factor of the individual k-vectors.
        """
        return self._rho

    @property
    def k_array(self):
        """ An array containing distances to the grid center (i.e. zero-mode)
        for every k-mode in the grid of the harmonic partner space.
        """
        return self._k_array

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group['kindex'] = self.kindex
        hdf5_group['rho'] = self.rho
        hdf5_group['logarithmic'] = self._logarithmic
        # Store nbin as string, since it can be None
        hdf5_group.attrs['nbin'] = str(self._nbin)
        hdf5_group.attrs['binbounds'] = str(self._binbounds)

        #MR FIXME: why not "return None" as happens everywhere else?
        return {
            'harmonic_partner': self.harmonic_partner,
            'pindex': self.pindex,
            'k_array': self.k_array
        }

    @classmethod
    def _from_hdf5(cls, hdf5_group, repository):
        # make an empty PowerSpace object
        new_ps = EmptyPowerSpace()
        # reset class
        new_ps.__class__ = cls
        # call instructor so that classes are properly setup
        super(PowerSpace, new_ps).__init__()
        # set all values
        new_ps._harmonic_partner = repository.get('harmonic_partner',
                                                  hdf5_group)

        new_ps._logarithmic = hdf5_group['logarithmic'][()]
        exec("new_ps._nbin = " + hdf5_group.attrs['nbin'])
        exec("new_ps._binbounds = " + hdf5_group.attrs['binbounds'])

        new_ps._pindex = repository.get('pindex', hdf5_group)
        new_ps._kindex = hdf5_group['kindex'][:]
        new_ps._rho = hdf5_group['rho'][:]
        new_ps._k_array = repository.get('k_array', hdf5_group)
        new_ps._ignore_for_hash += ['_pindex', '_kindex', '_rho', '_k_array']

        return new_ps


class EmptyPowerSpace(PowerSpace):
    def __init__(self):
        pass
