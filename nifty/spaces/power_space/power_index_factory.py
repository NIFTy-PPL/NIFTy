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

from power_indices import PowerIndices


class _PowerIndexFactory(object):
    def __init__(self):
        self.power_indices_storage = {}

    def get_power_index(self, domain, distribution_strategy,
                        log=False, nbin=None, binbounds=None):
        key=(domain,distribution_strategy)

        if key not in self.power_indices_storage:
            self.power_indices_storage[key] = \
                PowerIndices(domain, distribution_strategy,
                             log=log, nbin=nbin, binbounds=binbounds)
        power_indices = self.power_indices_storage[key]
        power_index = power_indices.get_index_dict(log=log,
                                                   nbin=nbin,
                                                   binbounds=binbounds)
        return power_index


PowerIndexFactory = _PowerIndexFactory()
