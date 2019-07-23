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

from .. import utilities
from ..linearization import Linearization
from ..operators.energy_operators import StandardHamiltonian
from ..probing import approximation2endo
from ..sugar import makeOp
from .energy import Energy


class MetricGaussianKL(Energy):
    """Provides the sampled Kullback-Leibler divergence between a distribution
    and a Metric Gaussian.

    A Metric Gaussian is used to approximate another probability distribution.
    It is a Gaussian distribution that uses the Fisher information metric of
    the other distribution at the location of its mean to approximate the
    variance. In order to infer the mean, a stochastic estimate of the
    Kullback-Leibler divergence is minimized. This estimate is obtained by
    sampling the Metric Gaussian at the current mean. During minimization
    these samples are kept constant; only the mean is updated. Due to the
    typically nonlinear structure of the true distribution these samples have
    to be updated eventually by intantiating `MetricGaussianKL` again. For the
    true probability distribution the standard parametrization is assumed.

    Parameters
    ----------
    mean : Field
        Mean of the Gaussian probability distribution.
    hamiltonian : StandardHamiltonian
        Hamiltonian of the approximated probability distribution.
    n_samples : integer
        Number of samples used to stochastically estimate the KL.
    constants : list
        List of parameter keys that are kept constant during optimization.
        Default is no constants.
    point_estimates : list
        List of parameter keys for which no samples are drawn, but that are
        (possibly) optimized for, corresponding to point estimates of these.
        Default is to draw samples for the complete domain.
    mirror_samples : boolean
        Whether the negative of the drawn samples are also used,
        as they are equally legitimate samples. If true, the number of used
        samples doubles. Mirroring samples stabilizes the KL estimate as
        extreme sample variation is counterbalanced. Default is False.
    _samples : None
        Only a parameter for internal uses. Typically not to be set by users.

    Note
    ----
    The two lists `constants` and `point_estimates` are independent from each
    other. It is possible to sample along domains which are kept constant
    during minimization and vice versa.

    See also
    --------
    Metric Gaussian Variational Inference (FIXME in preparation)
    """

    def __init__(self, mean, hamiltonian, n_samples, constants=[],
                 point_estimates=[], mirror_samples=False,
                 _samples=None, napprox=0):
        super(MetricGaussianKL, self).__init__(mean)

        if not isinstance(hamiltonian, StandardHamiltonian):
            raise TypeError
        if hamiltonian.domain is not mean.domain:
            raise ValueError
        if not isinstance(n_samples, int):
            raise TypeError
        self._constants = list(constants)
        self._point_estimates = list(point_estimates)
        if not isinstance(mirror_samples, bool):
            raise TypeError

        self._hamiltonian = hamiltonian

        if _samples is None:
            met = hamiltonian(Linearization.make_partial_var(
                mean, point_estimates, True)).metric
            if napprox > 1:
                print('Calculate preconditioner for sampling')
                met._approximation = makeOp(approximation2endo(met, napprox))
                print('Done')
            _samples = tuple(met.draw_sample(from_inverse=True)
                             for _ in range(n_samples))
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
        self._napprox = napprox

    def at(self, position):
        return MetricGaussianKL(position, self._hamiltonian, 0,
                                self._constants, self._point_estimates,
                                _samples=self._samples, napprox=self._napprox)

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def _get_metric(self):
        if self._metric is None:
            lin = self._lin.with_want_metric()
            mymap = map(lambda v: self._hamiltonian(lin+v).metric,
                        self._samples)
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
            self._samples)) + utilities.indent(self._hamiltonian.__repr__())
