# -*- coding: utf-8 -*-

import numpy as np

from power_index_factory import PowerIndexFactory

from nifty.config import about
from nifty.spaces.space import Space
from nifty.spaces.rg_space import RGSpace
from nifty.nifty_utilities import cast_axis_to_tuple


class PowerSpace(Space):

    # ---Overwritten properties and methods---

    def __init__(self, harmonic_domain=RGSpace((1,)), datamodel='not',
                 log=False, nbin=None, binbounds=None,
                 dtype=np.dtype('float')):

        super(PowerSpace, self).__init__(dtype)
        self._ignore_for_hash += ['_pindex', '_kindex', '_rho']

        if not isinstance(harmonic_domain, Space):
            raise ValueError(about._errors.cstring(
                "ERROR: harmonic_domain must be a Space."))
        if not harmonic_domain.harmonic:
            raise ValueError(about._errors.cstring(
                "ERROR: harmonic_domain must be a harmonic space."))
        self._harmonic_domain = harmonic_domain

        power_index = PowerIndexFactory.get_power_index(
                        domain=self.harmonic_domain,
                        distribution_strategy=datamodel,
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

    def compute_k_array(self, distribution_strategy):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: There is no k_array implementation for PowerSpace."))

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
        return reduce(lambda x, y: x*y, self.pindex.shape)

    def copy(self):
        datamodel = self.pindex.distribution_strategy
        return self.__class__(harmonic_domain=self.harmonic_domain,
                              datamodel=datamodel,
                              log=self.log,
                              nbin=self.nbin,
                              binbounds=self.binbounds,
                              dtype=self.dtype)

    def weight(self, x, power=1, axes=None, inplace=False):
        total_shape = x.shape

        axes = cast_axis_to_tuple(axes, len(total_shape))
        if len(axes) != 1:
            raise ValueError(about._errors.cstring(
                "ERROR: axes must be of length 1."))

        reshaper = [1, ] * len(total_shape)
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
