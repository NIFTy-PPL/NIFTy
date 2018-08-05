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
from ..operators.operator import Operator
from ..utilities import my_sum


class SampledKullbachLeiblerDivergence(Operator):
    def __init__(self, h, res_samples):
        """
        # MR FIXME: does h have to be a Hamiltonian? Couldn't it be any energy?
        h: Hamiltonian
        N: Number of samples to be used
        """
        super(SampledKullbachLeiblerDivergence, self).__init__()
        self._h = h
        self._res_samples = tuple(res_samples)

    @property
    def domain(self):
        return self._h.domain

    @property
    def target(self):
        return DomainTuple.scalar_domain()

    def apply(self, x):
        return (my_sum(map(lambda v: self._h(x+v), self._res_samples)) *
                (1./len(self._res_samples)))
