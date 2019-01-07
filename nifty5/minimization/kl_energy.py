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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from .energy import Energy
from ..linearization import Linearization
from .. import utilities


class KL_Energy(Energy):
    def __init__(self, position, h, nsamp, constants=[],
                 constants_samples=None, gen_mirrored_samples=False,
                 _samples=None):
        super(KL_Energy, self).__init__(position)
        if h.domain is not position.domain:
            raise TypeError
        self._h = h
        self._constants = constants
        if constants_samples is None:
            constants_samples = constants
        self._constants_samples = constants_samples
        if _samples is None:
            met = h(Linearization.make_partial_var(
                position, constants_samples, True)).metric
            _samples = tuple(met.draw_sample(from_inverse=True)
                             for _ in range(nsamp))
            if gen_mirrored_samples:
                _samples += tuple(-s for s in _samples)
        self._samples = _samples

        self._lin = Linearization.make_partial_var(position, constants)
        v, g = None, None
        for s in self._samples:
            tmp = self._h(self._lin+s)
            if v is None:
                v = tmp.val.local_data[()]
                g = tmp.gradient
            else:
                v += tmp.val.local_data[()]
                g = g + tmp.gradient
        self._val = v / len(self._samples)
        self._grad = g * (1./len(self._samples))
        self._metric = None

    def at(self, position):
        return KL_Energy(position, self._h, 0,
                         self._constants, self._constants_samples,
                         _samples=self._samples)

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def _get_metric(self):
        if self._metric is None:
            lin = self._lin.with_want_metric()
            mymap = map(lambda v: self._h(lin+v).metric, self._samples)
            self._metric = utilities.my_sum(mymap)
            self._metric = self._metric.scale(1./len(self._samples))

    def apply_metric(self, x):
        self._get_metric()
        return self._metric(x)

    @property
    def metric(self):
        self._get_metric()
        return self._metric

    @property
    def samples(self):
        return self._samples

    def __repr__(self):
        return 'KL ({} samples):\n'.format(len(
            self._samples)) + utilities.indent(self._ham.__repr__())
