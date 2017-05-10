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

    def __init__(self, harmonic_domain,
                 distribution_strategy='not',
                 log=False, nbin=None, binbounds=None):

        super(PowerSpace, self).__init__()
        self._ignore_for_hash += ['_pindex', '_kindex', '_rho', '_pundex',
                                  '_k_array']

        if not isinstance(harmonic_domain, Space):
            raise ValueError(
                "harmonic_domain must be a Space.")
        if not harmonic_domain.harmonic:
            raise ValueError(
                "harmonic_domain must be a harmonic space.")
        self._harmonic_domain = harmonic_domain

        power_index = PowerIndexFactory.get_power_index(
                        domain=self.harmonic_domain,
                        distribution_strategy=distribution_strategy,
                        log=log,
                        nbin=nbin,
                        binbounds=binbounds)

        config = power_index['config']
        self._log = config['log']
        self._nbin = config['nbin']
        self._binbounds = config['binbounds']

        self._pindex = power_index['pindex']
        self._kindex = power_index['kindex']
        self._rho = power_index['rho']
        self._pundex = power_index['pundex']
        self._k_array = power_index['k_array']

    def pre_cast(self, x, axes=None):
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
        return self.__class__(harmonic_domain=self.harmonic_domain,
                              distribution_strategy=distribution_strategy,
                              log=self.log,
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
    def harmonic_domain(self):
        return self._harmonic_domain

    @property
    def log(self):
        return self._log

    @property
    def nbin(self):
        return self._nbin

    @property
    def binbounds(self):
        return self._binbounds

    @property
    def pindex(self):
        return self._pindex

    @property
    def kindex(self):
        return self._kindex

    @property
    def rho(self):
        return self._rho

    @property
    def pundex(self):
        return self._pundex

    @property
    def k_array(self):
        return self._k_array

    # ---Serialization---

    def _to_hdf5(self, hdf5_group):
        hdf5_group['kindex'] = self.kindex
        hdf5_group['rho'] = self.rho
        hdf5_group['pundex'] = self.pundex
        hdf5_group['log'] = self.log
        # Store nbin as string, since it can be None
        hdf5_group.attrs['nbin'] = str(self.nbin)
        hdf5_group.attrs['binbounds'] = str(self.binbounds)

        return {
            'harmonic_domain': self.harmonic_domain,
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
        new_ps._harmonic_domain = repository.get('harmonic_domain', hdf5_group)
        new_ps._log = hdf5_group['log'][()]
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
