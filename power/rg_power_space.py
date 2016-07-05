# -*- coding: utf-8 -*-

import numpy as np

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
        self.discrete = False

    def calculate_power_spectrum(self, x, axes=None):
        fieldabs = abs(x)**2
        # need a bincount with axes function here.
