# -*- coding: utf-8 -*-

import numpy as np

from nifty.power import PowerSpace
from nifty.nifty_paradict import rg_power_space_paradict
# from nifty.power.power_index_factory import RGPowerIndexFactory


class RGPowerSpace(PowerSpace):
    def __init__(self, shape, dgrid, distribution_strategy, zerocentered=False,
                 dtype=np.dtype('float'), log=False, nbin=None,
                 binbounds=None):
        self.dtype = np.dtype(dtype)
        self.paradict = rg_power_space_paradict(
                                    shape=shape,
                                    dgrid=dgrid,
                                    zerocentered=zerocentered,
                                    distribution_strategy=distribution_strategy,
                                    log=log,
                                    nbin=nbin,
                                    binbounds=binbounds)

        # self.power_indices = RGPowerIndexFactory.get_power_indices(
        #                         **self.paradict.parameters)
