# -*- coding: utf-8 -*-
from power_indices import PowerIndices


class _PowerIndexFactory(object):
    def __init__(self):
        self.power_indices_storage = {}

    def get_power_index(self, domain, distribution_strategy,
                        log=False, nbin=None, binbounds=None):
        current_hash = domain.__hash__() ^ (111*hash(distribution_strategy))

        if current_hash not in self.power_indices_storage:
            self.power_indices_storage[current_hash] = \
                PowerIndices(domain, distribution_strategy,
                             log=log, nbin=nbin, binbounds=binbounds)
        power_indices = self.power_indices_storage[current_hash]
        power_index = power_indices.get_index_dict(log=log,
                                                   nbin=nbin,
                                                   binbounds=binbounds)
        return power_index


PowerIndexFactory = _PowerIndexFactory()
