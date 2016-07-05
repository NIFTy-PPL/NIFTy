# -*- coding: utf-8 -*-

import numpy as np

from nifty.space import Space
from nifty.nifty_paradict import power_space_paradict


class PowerSpace(Space):
    def __init__(self, distribution_strategy, dtype=np.dtype('float'),
                 log=False, nbin=None, binbounds=None):
        self.dtype = np.dtype(dtype)
        self.paradict = power_space_paradict(
                                distribution_strategy=distribution_strategy,
                                log=log,
                                nbin=nbin,
                                binbounds=binbounds)
        # Here it would be time to initialize the power indices
        raise NotImplementedError

        self.distances = None

        self.harmonic = True

    def calculate_power_spectrum(self):
        raise NotImplementedError

    def cast_power_spectrum(self):
        raise NotImplementedError

    def get_weight(self, power=1):
        raise NotImplementedError

    def smooth(self):
        raise NotImplementedError
