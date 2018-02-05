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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from builtins import object


class StatCalculator(object):
    def __init__(self):
        self._count = 0

    def add(self, value):
        self._count += 1
        if self._count == 1:
            self._mean = 1.*value
            self._M2 = 0.*value
        else:
            delta = value - self._mean
            self._mean += delta*(1./self._count)
            delta2 = value - self._mean
            self._M2 += delta*delta2

    @property
    def mean(self):
        if self._count == 0:
            raise RuntimeError
        return 1.*self._mean

    @property
    def var(self):
        if self._count < 2:
            raise RuntimeError
        return self._M2 * (1./(self._count-1))


def probe_with_posterior_samples(op, post_op, nprobes):
    sc = StatCalculator()
    for i in range(nprobes):
        sample = post_op(op.generate_posterior_sample())
        sc.add(sample)
    return sc.mean, sc.var