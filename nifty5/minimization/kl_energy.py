from __future__ import absolute_import, division, print_function

from ..compat import *
from .energy import Energy
from ..linearization import Linearization
from .. import utilities


class KL_Energy(Energy):
    def __init__(self, position, h, nsamp, constants=[],
                 constants_samples=None, mirror_samples=False,
                 _samples=None):
        super(KL_Energy, self).__init__(position)
        if h.domain is not position.domain:
            raise TypeError
        self._h = h
        self._constants = constants
        if constants_samples is None:
            constants_samples = constants
        self._constants_samples = constants_samples
        self._mirror_samples = mirror_samples
        if _samples is None:
            met = h(Linearization.make_partial_var(
                position, constants_samples, True)).metric
            _samples = tuple(met.draw_sample(from_inverse=True)
                             for _ in range(nsamp))
            if mirror_samples:
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
        return KL_Energy(position, self._h, 0, self._constants,
                         self._constants_samples, self._mirror_samples,
                         self._samples)

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
