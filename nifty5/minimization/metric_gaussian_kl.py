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


class MetricGaussianKL(Energy):
    """Provides the sampled Kullback-Leibler divergence between a distribution and a Metric Gaussian.

        A Metric Gaussian is used to approximate some other distribution.
        It is a Gaussian distribution that uses the Fisher Information Metric
        of the other distribution at the location of its mean to approximate the variance.
        In order to infer the mean, the a stochastic estimate of the Kullback-Leibler divergence
        is minimized. This estimate is obtained by drawing samples from the Metric Gaussian at the current mean.
        During minimization these samples are kept constant, updating only the mean. Due to the typically nonlinear
        structure of the true distribution these samples have to be updated by re-initializing this class at some point.
        Here standard parametrization of the true distribution is assumed.

    Parameters
    ----------
    mean : Field
        The current mean of the Gaussian.
    hamiltonian : StandardHamiltonian
        The StandardHamiltonian of the approximated probability distribution.
    n_samples : integer
        The number of samples used to stochastically estimate the KL.
    constants : list
        A list of parameter keys that are kept constant during optimization.
    point_estimates : list
        A list of parameter keys for which no samples are drawn, but that are
        optimized for, corresponding to point estimates of these.
    mirror_samples : boolean
        Whether the negative of the drawn samples are also used,
        as they are equaly legitimate samples. If true, the number of used
        samples doubles. Mirroring samples stabilizes the KL estimate as
        extreme sample variation is counterbalanced. (default : False)

    Notes
    -----
    For further details see: Metric Gaussian Variational Inference (in preparation)
    """

    def __init__(self, mean, hamiltonian, n_sampels, constants=[],
                 point_estimates=None, mirror_samples=False,
                 _samples=None):
        super(MetricGaussianKL, self).__init__(mean)
        if hamiltonian.domain is not mean.domain:
            raise TypeError
        self._hamiltonian = hamiltonian
        self._constants = constants
        if point_estimates is None:
            point_estimates = constants
        self._constants_samples = point_estimates
        if _samples is None:
            met = hamiltonian(Linearization.make_partial_var(
                mean, point_estimates, True)).metric
            _samples = tuple(met.draw_sample(from_inverse=True)
                             for _ in range(n_sampels))
            if mirror_samples:
                _samples += tuple(-s for s in _samples)
        self._samples = _samples

        self._lin = Linearization.make_partial_var(mean, constants)
        v, g = None, None
        for s in self._samples:
            tmp = self._hamiltonian(self._lin+s)
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
        return MetricGaussianKL(position, self._hamiltonian, 0,
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
            mymap = map(lambda v: self._hamiltonian(lin+v).metric, self._samples)
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
