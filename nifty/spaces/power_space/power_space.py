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

import d2o

from power_index_factory import PowerIndexFactory

from nifty.spaces.space import Space
from nifty.spaces.rg_space import RGSpace


class PowerSpace(Space):

    # ---Overwritten properties and methods---

    def __init__(self, harmonic_partner=RGSpace((1,)),
                 distribution_strategy='not',
                 logarithmic=False, nbin=None, binbounds=None):
        """Sets the attributes for a PowerSpace class instance.

            Parameters
            ----------
            harmonic_partner : Space
                The harmonic Space of which this is the power space.
            distribution_strategy : str *optional*
                The distribution strategy of a d2o-object represeting a field over this PowerSpace.
                (default : 'not')
            logarithmic : bool *optional*
                True if logarithmic binning should be used.
                (default : False)
            nbin : {int, None} *optional*
                The number of bins this space has.
                (default : None) if nbin == None : It takes the nbin from its harmonic_partner
            binbounds :  {list, array} *optional*
                Array-like inner boundaries of the used bins of the default
                indices.
                (default : None) if binbounds == None : Calculates the bounds from the kindex and corrects for logartihmic scale 
            Notes
            -----
            A power space is the result of a projection of a harmonic space where multiple k-modes get mapped to one power index.
            This can be regarded as a response operator :math:`R` going from harmonic space to power space. 
            An array giving this map is stored in pindex (array which says in which power box a k-mode gets projected)
            An array for the adjoint of :math:`R` is given by kindex, which is an array of arrays stating which k-mode got mapped to a power index
            The a right-inverse to :math:`R` is given by the pundex which is an array giving one k-mode that maps to a power bin for every power bin.
            The amount of k-modes that get mapped to one power bin is given by rho. This is :math:`RR^\dagger` in the language of this projection operator            
            Returns
            -------
            None.

        """
        #FIXME: default probably not working for log and normal scale
        super(PowerSpace, self).__init__()
        self._ignore_for_hash += ['_pindex', '_kindex', '_rho', '_pundex',
                                  '_k_array']

        if not isinstance(harmonic_partner, Space):
            raise ValueError(
                "harmonic_partner must be a Space.")
        if not harmonic_partner.harmonic:
            raise ValueError(
                "harmonic_partner must be a harmonic space.")
        self._harmonic_partner = harmonic_partner

        power_index = PowerIndexFactory.get_power_index(
                        domain=self.harmonic_partner,
                        distribution_strategy=distribution_strategy,
                        logarithmic=logarithmic,
                        nbin=nbin,
                        binbounds=binbounds)

        config = power_index['config']
        self._logarithmic = config['logarithmic']
        self._nbin = config['nbin']
        self._binbounds = config['binbounds']

        self._pindex = power_index['pindex']
        self._kindex = power_index['kindex']
        self._rho = power_index['rho']
        self._pundex = power_index['pundex']
        self._k_array = power_index['k_array']

    def pre_cast(self, x, axes=None):
        """Casts power spectra to discretized power spectra.
        
        This function takes an array or a function. If it is an array it does nothing,
        otherwise it intepretes the function as power spectrum and evaluates it at every
        k-mode.
        Parameters
        ----------
        x : {array-like, function array-like -> array-like}
            power spectrum given either in discretized form or implicitly as a function
        axes : {tuple, int} *optional*
            does nothing
            (default : None)
        Returns
        -------
        array-like : discretized power spectrum
        """
        if callable(x):
            return x(self.kindex)
        else:
            return x

    # ---Mandatory properties and methods---

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
                              logarithmic=self.logarithmic,
                              nbin=self.nbin,
                              binbounds=self.binbounds)

    def weight(self, x, power=1, axes=None, inplace=False):
        reshaper = [1, ] * len(x.shape)
        # we know len(axes) is always 1
        reshaper[axes[0]] = self.shape[0]

        weight = self.rho.reshape(reshaper)
        if power != 1:
            weight = weight ** power

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
        """Returns the Space of which this is the power space.
        Returns
        -------
        Space : The harmonic Space of which this is the power space.
        """
        return self._harmonic_partner

    @property
    def logarithmic(self):
        """Returns a True if logarithmic binning is used.
        Returns
        -------
        Bool : True if for this PowerSpace logarithmic binning is used.
        """
        return self._logarithmic

    @property
    def nbin(self):
        """Returns the number of power bins.
        Returns
        -------
        int : The number of bins this space has.
        """
        return self._nbin

    @property
    def binbounds(self):
        """ Inner boundaries of the used bins of the default
                indices.
        Returns
        -------
        {list, array} : the inner boundaries of the used bins in the used scale, as they were
        set in __init__ or computed.
        """
        # FIXME check wether this returns something sensible if 'None' was set in __init__
        return self._binbounds

    @property
    def pindex(self):
    """Index of the Fourier grid points that belong to a specific power index
    Returns
    -------
        distributed_data_object : Index of the Fourier grid points in a distributed_data_object.
    """
        return self._pindex

    @property
    def kindex(self):
    """Array of all k-vector lengths.
    Returns
    -------
        ndarray : Array which states for each k-mode which power index it maps to (adjoint to pindex)
    """
        return self._kindex

    @property
    def rho(self):
    """Degeneracy factor of the individual k-vectors.
    
    ndarray : Array stating how many k-modes are mapped to one power index for every power index
    """
        return self._rho

    @property
    def pundex(self):
    """List of one k-mode per power bin which is in the bin.
    Returns
    -------
    array-like : An array for which the n-th entry is an example one k-mode which belongs to the n-th power bin
    """
        return self._pundex

    @property
    def k_array(self):
    """This contains distances to zero for every k-mode of the harmonic partner.
    
    Returns
    -------
    array-like : An array containing distances to the zero mode for every k-mode of the harmonic partner.
    """
        return self._k_array

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group['kindex'] = self.kindex
        hdf5_group['rho'] = self.rho
        hdf5_group['pundex'] = self.pundex
        hdf5_group['logarithmic'] = self.logarithmic
        # Store nbin as string, since it can be None
        hdf5_group.attrs['nbin'] = str(self.nbin)
        hdf5_group.attrs['binbounds'] = str(self.binbounds)

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
        new_ps._harmonic_partner = repository.get('harmonic_partner', hdf5_group)
        new_ps._logarithmic = hdf5_group['logarithmic'][()]
        exec('new_ps._nbin = ' + hdf5_group.attrs['nbin'])
        exec('new_ps._binbounds = ' + hdf5_group.attrs['binbounds'])

        new_ps._pindex = repository.get('pindex', hdf5_group)
        new_ps._kindex = hdf5_group['kindex'][:]
        new_ps._rho = hdf5_group['rho'][:]
        new_ps._pundex = hdf5_group['pundex'][:]
        new_ps._k_array = repository.get('k_array', hdf5_group)
        new_ps._ignore_for_hash += ['_pindex', '_kindex', '_rho', '_pundex',
                                    '_k_array']

        return new_ps


class EmptyPowerSpace(PowerSpace):
    def __init__(self):
        pass
