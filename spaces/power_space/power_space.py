# -*- coding: utf-8 -*-

import numpy as np
from d2o import STRATEGIES

from nifty.config import about
from nifty.space import Space
from power_space_paradict import PowerSpaceParadict
from nifty.nifty_utilities import cast_axis_to_tuple


class PowerSpace(Space):
    def __init__(self, pindex, kindex, rho, config,
                 harmonic_domain, dtype=np.dtype('float'), **kwargs):
        # the **kwargs is in the __init__ in order to enable a
        # PowerSpace(**power_index) initialization
        self.dtype = np.dtype(dtype)
        self.paradict = PowerSpaceParadict(pindex=pindex,
                                           kindex=kindex,
                                           rho=rho,
                                           config=config,
                                           harmonic_domain=harmonic_domain)
        self._harmonic = True

    @property
    def shape(self):
        return self.paradict['kindex'].shape

    @property
    def dim(self):
        return self.shape[0]

    @property
    def total_volume(self):
        # every power-pixel has a volume of 1
        return reduce(lambda x, y: x*y, self.paradict['pindex'].shape)

    def weight(self, x, power=1, axes=None):
        total_shape = x.shape

        axes = cast_axis_to_tuple(axes, len(total_shape))
        if len(axes) != 1:
            raise ValueError(about._errors.cstring(
                "ERROR: axes must be of length 1."))

        reshaper = [1, ] * len(total_shape)
        reshaper[axes[0]] = self.shape[0]

        weight = self.paradict['rho'].reshape(reshaper)
        if power != 1:
            weight = weight ** power
        result_x = x * weight

        return result_x

    def compute_k_array(self, distribution_strategy):
        raise NotImplementedError(about._errors.cstring(
            "ERROR: There is no k_array implementation for PowerSpace."))

    def calculate_power_spectrum(self, x, axes=None):
        fieldabs = abs(x)**2
        pindex = self.paradict['pindex']
        if axes is not None:
            pindex = self._shape_up_pindex(
                                    pindex=pindex,
                                    target_shape=x.shape,
                                    target_strategy=x.distribution_strategy,
                                    axes=axes)
        power_spectrum = pindex.bincount(weights=fieldabs,
                                         axis=axes)

        rho = self.paradict['rho']
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
