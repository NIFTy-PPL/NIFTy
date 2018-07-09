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
#
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import absolute_import, division, print_function
from ..compat import *
from ..minimization.energy import Energy
from ..utilities import memo, my_sum


class SampledKullbachLeiblerDivergence(Energy):
    def __init__(self, h, res_samples):
        """
        # MR FIXME: does h have to be a Hamiltonian? Couldn't it be any energy?
        h: Hamiltonian
        N: Number of samples to be used
        """
        super(SampledKullbachLeiblerDivergence, self).__init__(h.position)
        self._h = h
        self._res_samples = res_samples

        self._energy_list = tuple(h.at(self.position+ss)
                                  for ss in res_samples)

    def at(self, position):
        return self.__class__(self._h.at(position), self._res_samples)

    @property
    @memo
    def value(self):
        return (my_sum(map(lambda v: v.value, self._energy_list)) /
                len(self._energy_list))

    @property
    @memo
    def gradient(self):
        return (my_sum(map(lambda v: v.gradient, self._energy_list)) /
                len(self._energy_list))

    @property
    @memo
    def metric(self):
        return (my_sum(map(lambda v: v.metric, self._energy_list)) *
                (1./len(self._energy_list)))
