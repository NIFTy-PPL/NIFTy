# -*- coding: utf-8 -*-
from power_indices import PowerIndices,\
                          RGPowerIndices,\
                          LMPowerIndices


class _PowerIndexFactory(object):
    def __init__(self):
        self.power_indices_storage = {}

    def _get_power_index_class(self):
        return PowerIndices

    def hash_arguments(self, **kwargs):
        return frozenset(kwargs.items())

    def get_power_indices(self, log, nbin, binbounds, **kwargs):
        current_hash = self.hash_arguments(**kwargs)
        if current_hash not in self.power_indices_storage:
            power_class = self._get_power_index_class()
            self.power_indices_storage[current_hash] = power_class(
                                                        log=log,
                                                        nbin=nbin,
                                                        binbounds=binbounds,
                                                        **kwargs)
        power_indices = self.power_indices_storage[current_hash]
        power_index = power_indices.get_index_dict(log=log,
                                                   nbin=nbin,
                                                   binbounds=binbounds)
        return power_index


class _RGPowerIndexFactory(_PowerIndexFactory):
    def _get_power_index_class(self):
        return RGPowerIndices


class _LMPowerIndexFactory(_PowerIndexFactory):
    def _get_power_index_class(self):
        return LMPowerIndices

PowerIndexFactory = _PowerIndexFactory()
RGPowerIndexFactory = _RGPowerIndexFactory()
LMPowerIndexFactory = _LMPowerIndexFactory()
