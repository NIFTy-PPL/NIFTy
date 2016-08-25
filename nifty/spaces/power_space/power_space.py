# -*- coding: utf-8 -*-

import numpy as np
from d2o import STRATEGIES

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

    def calculate_power_spectrum(self, x, axes=None):
        fieldabs = abs(x)
        fieldabs **= 2

        pindex = self.pindex
        if axes is not None:
            pindex = self._shape_up_pindex(
                                    pindex=pindex,
                                    target_shape=x.shape,
                                    target_strategy=x.distribution_strategy,
                                    axes=axes)
        power_spectrum = pindex.bincount(weights=fieldabs,
                                         axis=axes)

        rho = self.rho
        if axes is not None:
            new_rho_shape = [1, ] * len(power_spectrum.shape)
            new_rho_shape[axes[0]] = len(rho)
            rho = rho.reshape(new_rho_shape)
        power_spectrum /= rho

        power_spectrum **= 0.5
        return power_spectrum

    def _shape_up_pindex(self, pindex, target_shape, target_strategy, axes):
        if pindex.distribution_strategy not in STRATEGIES['global']:
            raise ValueError("ERROR: pindex's distribution strategy must be "
                             "global-type")

        if pindex.distribution_strategy in STRATEGIES['slicing']:
            if ((0 not in axes) or
                    (target_strategy is not pindex.distribution_strategy)):
                raise ValueError(
                    "ERROR: A slicing distributor shall not be reshaped to "
                    "something non-sliced.")

        semiscaled_shape = [1, ] * len(target_shape)
        for i in axes:
            semiscaled_shape[i] = target_shape[i]
        local_data = pindex.get_local_data(copy=False)
        semiscaled_local_data = local_data.reshape(semiscaled_shape)
        result_obj = pindex.copy_empty(global_shape=target_shape,
                                       distribution_strategy=target_strategy)
        result_obj.set_full_data(semiscaled_local_data, copy=False)

        return result_obj
