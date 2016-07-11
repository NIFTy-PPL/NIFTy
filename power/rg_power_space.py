# -*- coding: utf-8 -*-

import numpy as np
from d2o import STRATEGIES

from nifty.power import PowerSpace
from nifty.nifty_paradict import rg_power_space_paradict

from power_index_factory import RGPowerIndexFactory


class RGPowerSpace(PowerSpace):
    def __init__(self, shape, dgrid, distribution_strategy,
                 dtype=np.dtype('float'), zerocenter=False,
                 log=False, nbin=None, binbounds=None):
        self.dtype = np.dtype(dtype)
        self.paradict = rg_power_space_paradict(
                                shape=shape,
                                dgrid=dgrid,
                                zerocenter=zerocenter,
                                distribution_strategy=distribution_strategy,
                                log=log,
                                nbin=nbin,
                                binbounds=binbounds)

        temp_dict = self.paradict.parameters.copy()
        del temp_dict['complexity']
        self.power_indices = RGPowerIndexFactory.get_power_indices(**temp_dict)

        self.distances = (tuple(self.power_indices['rho']),)

        self.harmonic = True

    @property
    def shape(self):
        return tuple(self.paradict['shape'])

    def calculate_power_spectrum(self, x, axes=None):
        fieldabs = abs(x)**2
        pindex = self.power_indices['pindex']
        if axes is not None:
            pindex = self._shape_up_pindex(
                                    pindex=pindex,
                                    target_shape=x.shape,
                                    target_strategy=x.distribution_strategy,
                                    axes=axes)
        power_spectrum = pindex.bincount(weights=fieldabs,
                                         axis=axes)

        rho = self.power_indices['rho']
        if axes is not None:
            new_rho_shape = [1, ] * len(power_spectrum.shape)
            new_rho_shape[axes[0]] = len(rho)
            rho = rho.reshape(new_rho_shape)
        power_spectrum /= rho

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

