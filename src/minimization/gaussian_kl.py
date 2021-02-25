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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .. import random, utilities
from ..linearization import Linearization
from ..logger import logger
from ..multi_field import MultiField
from ..operators.endomorphic_operator import EndomorphicOperator
from ..operators.energy_operators import StandardHamiltonian
from ..operators.multifield_flattener import MultifieldFlattener
from ..probing import approximation2endo
from ..sugar import makeOp, full, from_random
from ..utilities import myassert
from .energy import Energy



class _KLMetric(EndomorphicOperator):
    def __init__(self, KL):
        self._KL = KL
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._domain = KL.position.domain

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._KL.apply_metric(x)


def _get_lo_hi(comm, n_samples):
    ntask, rank, _ = utilities.get_MPI_params_from_comm(comm)
    return utilities.shareRange(n_samples, ntask, rank)


def _modify_sample_domain(sample, domain):
    """Takes only keys from sample which are also in domain and inserts zeros
    for keys which are not in sample.domain."""
    from ..domain_tuple import DomainTuple
    from ..field import Field
    from ..multi_domain import MultiDomain
    from ..sugar import makeDomain
    domain = makeDomain(domain)
    if isinstance(domain, DomainTuple) and isinstance(sample, Field):
        if sample.domain is not domain:
            raise TypeError
        return sample
    elif isinstance(domain, MultiDomain) and isinstance(sample, MultiField):
        if sample.domain is domain:
            return sample
        out = {kk: vv for kk, vv in sample.items() if kk in domain.keys()}
        out = MultiField.from_dict(out, domain)
        return out
    raise TypeError


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
    The samples of this class can be distributed among MPI tasks.

    Notes
    -----

    DomainTuples should never be created using the constructor, but rather
    via the factory function :attr:`make`!
    See also
    --------
    `Metric Gaussian Variational Inference`, Jakob Knollmüller,
    Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
    """
    def __init__(self, mean, hamiltonian, n_samples, mirror_samples, comm,
                 local_samples, nanisinf, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        super(MetricGaussianKL, self).__init__(mean)
        myassert(mean.domain is hamiltonian.domain)
        self._hamiltonian = hamiltonian
        self._n_samples = int(n_samples)
        self._mirror_samples = bool(mirror_samples)
        self._comm = comm
        self._local_samples = local_samples
        self._nanisinf = bool(nanisinf)

        lin = Linearization.make_var(mean)
        v, g = [], []
        for s in self._local_samples:
            s = _modify_sample_domain(s, mean.domain)
            tmp = hamiltonian(lin+s)
            tv = tmp.val.val
            tg = tmp.gradient
            if mirror_samples:
                tmp = hamiltonian(lin-s)
                tv = tv + tmp.val.val
                tg = tg + tmp.gradient
            v.append(tv)
            g.append(tg)
        self._val = utilities.allreduce_sum(v, self._comm)[()]/self.n_eff_samples
        if np.isnan(self._val) and self._nanisinf:
            self._val = np.inf
        self._grad = utilities.allreduce_sum(g, self._comm)/self.n_eff_samples

    @staticmethod
    def make(mean, hamiltonian, n_samples, mirror_samples, constants=[],
             point_estimates=[], napprox=0, comm=None, nanisinf=False):
        """Return instance of :class:`MetricGaussianKL`.

        Parameters
        ----------
        mean : Field
            Mean of the Gaussian probability distribution.
        hamiltonian : StandardHamiltonian
            Hamiltonian of the approximated probability distribution.
        n_samples : integer
            Number of samples used to stochastically estimate the KL.
        mirror_samples : boolean
            Whether the negative of the drawn samples are also used, as they are
            equally legitimate samples. If true, the number of used samples
            doubles. Mirroring samples stabilizes the KL estimate as extreme
            sample variation is counterbalanced. Since it improves stability in
            many cases, it is recommended to set `mirror_samples` to `True`.
        constants : list
            List of parameter keys that are kept constant during optimization.
            Default is no constants.
        point_estimates : list
            List of parameter keys for which no samples are drawn, but that are
            (possibly) optimized for, corresponding to point estimates of these.
            Default is to draw samples for the complete domain.
        napprox : int
            Number of samples for computing preconditioner for sampling. No
            preconditioning is done by default.
        comm : MPI communicator or None
            If not None, samples will be distributed as evenly as possible
            across this communicator. If `mirror_samples` is set, then a sample and
            its mirror image will always reside on the same task.
        nanisinf : bool
            If true, nan energies which can happen due to overflows in the forward
            model are interpreted as inf. Thereby, the code does not crash on
            these occaisions but rather the minimizer is told that the position it
            has tried is not sensible.

        Note
        ----
        The two lists `constants` and `point_estimates` are independent from each
        other. It is possible to sample along domains which are kept constant
        during minimization and vice versa.
        """

        if not isinstance(hamiltonian, StandardHamiltonian):
            raise TypeError
        if hamiltonian.domain is not mean.domain:
            raise ValueError
        if not isinstance(n_samples, int):
            raise TypeError
        if not isinstance(mirror_samples, bool):
            raise TypeError
        if isinstance(mean, MultiField) and set(point_estimates) == set(mean.keys()):
            raise RuntimeError(
                'Point estimates for whole domain. Use EnergyAdapter instead.')
        n_samples = int(n_samples)
        mirror_samples = bool(mirror_samples)

        if isinstance(mean, MultiField):
            cstpos = mean.extract_by_keys(point_estimates)
            _, ham_sampling = hamiltonian.simplify_for_constant_input(cstpos)
        else:
            ham_sampling = hamiltonian
        lin = Linearization.make_var(mean.extract(ham_sampling.domain), True)
        met = ham_sampling(lin).metric
        if napprox >= 1:
            met._approximation = makeOp(approximation2endo(met, napprox))
        local_samples = []
        sseq = random.spawn_sseq(n_samples)
        for i in range(*_get_lo_hi(comm, n_samples)):
            with random.Context(sseq[i]):
                local_samples.append(met.draw_sample(from_inverse=True))
        local_samples = tuple(local_samples)

        if isinstance(mean, MultiField):
            _, hamiltonian = hamiltonian.simplify_for_constant_input(mean.extract_by_keys(constants))
            mean = mean.extract_by_keys(set(mean.keys()) - set(constants))
        return MetricGaussianKL(
            mean, hamiltonian, n_samples, mirror_samples, comm, local_samples,
            nanisinf, _callingfrommake=True)

    def at(self, position):
        return MetricGaussianKL(
            position, self._hamiltonian, self._n_samples, self._mirror_samples,
            self._comm, self._local_samples, self._nanisinf, True)

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def apply_metric(self, x):
        lin = Linearization.make_var(self.position, want_metric=True)
        res = []
        for s in self._local_samples:
            s = _modify_sample_domain(s, self._hamiltonian.domain)
            tmp = self._hamiltonian(lin+s).metric(x)
            if self._mirror_samples:
                tmp = tmp + self._hamiltonian(lin-s).metric(x)
            res.append(tmp)
        return utilities.allreduce_sum(res, self._comm)/self.n_eff_samples

    @property
    def n_eff_samples(self):
        if self._mirror_samples:
            return 2*self._n_samples
        return self._n_samples

    @property
    def metric(self):
        return _KLMetric(self)

    @property
    def samples(self):
        ntask, rank, _ = utilities.get_MPI_params_from_comm(self._comm)
        if ntask == 1:
            for s in self._local_samples:
                yield s
                if self._mirror_samples:
                    yield -s
        else:
            rank_lo_hi = [utilities.shareRange(self._n_samples, ntask, i) for i in range(ntask)]
            lo, _ = _get_lo_hi(self._comm, self._n_samples)
            for itask, (l, h) in enumerate(rank_lo_hi):
                for i in range(l, h):
                    data = self._local_samples[i-lo] if rank == itask else None
                    s = self._comm.bcast(data, root=itask)
                    yield s
                    if self._mirror_samples:
                        yield -s


class ParametricGaussianKL(Energy):
    """Provides the sampled Kullback-Leibler divergence between a distribution
    and a Parametric Gaussian.
    Notes
    -----

    See also

    """
    def __init__(self, variational_parameters, hamiltonian, variational_model, 
                    n_samples, mirror_samples, comm,
                        local_samples, nanisinf, _callingfrommake=False):
        if not _callingfrommake:
            raise NotImplementedError
        super(ParametricGaussianKL, self).__init__(variational_parameters)
        assert variational_model.generator.target is hamiltonian.domain
        self._hamiltonian = hamiltonian
        self._variational_model = variational_model
        self._full_model = hamiltonian(variational_model.generator) + variational_model.entropy

        self._n_samples = int(n_samples)
        self._mirror_samples = bool(mirror_samples)
        self._comm = comm
        self._local_samples = local_samples
        self._nanisinf = bool(nanisinf)

        lin = Linearization.make_partial_var(variational_parameters, ['latent'])
        v, g = [], []
        for s in self._local_samples:
            # s = _modify_sample_domain(s, variational_parameters.domain)
            tmp = self._full_model(lin+s)
            tv = tmp.val.val
            tg = tmp.gradient
            if mirror_samples:
                tmp = self._full_model(lin-s)
                tv = tv + tmp.val.val
                tg = tg + tmp.gradient
            v.append(tv)
            g.append(tg)
        self._val = utilities.allreduce_sum(v, self._comm)[()]/self.n_eff_samples
        if np.isnan(self._val) and self._nanisinf:
            self._val = np.inf
        self._grad = utilities.allreduce_sum(g, self._comm)/self.n_eff_samples

    @staticmethod
    def make(variational_parameters, hamiltonian, variational_model, n_samples, mirror_samples,
                    comm=None, nanisinf=False):
        """Return instance of :class:`MetricGaussianKL`.

        Parameters
        ----------
        mean : Field
            Mean of the Gaussian probability distribution.
        hamiltonian : StandardHamiltonian
            Hamiltonian of the approximated probability distribution.
        n_samples : integer
            Number of samples used to stochastically estimate the KL.
        mirror_samples : boolean
            Whether the negative of the drawn samples are also used, as they are
            equally legitimate samples. If true, the number of used samples
            doubles. Mirroring samples stabilizes the KL estimate as extreme
            sample variation is counterbalanced. Since it improves stability in
            many cases, it is recommended to set `mirror_samples` to `True`.
        constants : list
            List of parameter keys that are kept constant during optimization.
            Default is no constants.
        napprox : int
            Number of samples for computing preconditioner for sampling. No
            preconditioning is done by default.
        comm : MPI communicator or None
            If not None, samples will be distributed as evenly as possible
            across this communicator. If `mirror_samples` is set, then a sample and
            its mirror image will always reside on the same task.
        nanisinf : bool
            If true, nan energies which can happen due to overflows in the forward
            model are interpreted as inf. Thereby, the code does not crash on
            these occaisions but rather the minimizer is told that the position it
            has tried is not sensible.

        Note
        ----
        The two lists `constants` and `point_estimates` are independent from each
        other. It is possible to sample along domains which are kept constant
        during minimization and vice versa.
        """

        if not isinstance(hamiltonian, StandardHamiltonian):
            raise TypeError
        if hamiltonian.domain is not variational_model.generator.target:
            raise ValueError
        if not isinstance(n_samples, int):
            raise TypeError
        if not isinstance(mirror_samples, bool):
            raise TypeError

        n_samples = int(n_samples)
        mirror_samples = bool(mirror_samples)
        local_samples = []
        sseq = random.spawn_sseq(n_samples)

        #FIXME dirty trick, many multiplications with zero
        DirtyMaskDict = full(variational_model.generator.domain,0.).to_dict()
        DirtyMaskDict['latent'] = full(variational_model.generator.domain['latent'], 1.)
        DirtyMask = MultiField.from_dict(DirtyMaskDict)

        for i in range(*_get_lo_hi(comm, n_samples)):
            with random.Context(sseq[i]):
                local_samples.append(DirtyMask * from_random(variational_model.generator.domain))
        local_samples = tuple(local_samples)

        return ParametricGaussianKL(
            variational_parameters, hamiltonian,variational_model,n_samples, mirror_samples, comm, local_samples,
            nanisinf, _callingfrommake=True)
    
    @property
    def n_eff_samples(self):
        if self._mirror_samples:
            return 2*self._n_samples
        return self._n_samples

    def at(self, position):
        return ParametricGaussianKL(
            position, self._hamiltonian, self._variational_model, self._n_samples, self._mirror_samples,
            self._comm, self._local_samples, self._nanisinf, True)
            

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    @property
    def samples(self):
        ntask, rank, _ = utilities.get_MPI_params_from_comm(self._comm)
        if ntask == 1:
            for s in self._local_samples:
                yield s
                if self._mirror_samples:
                    yield -s
        else:
            rank_lo_hi = [utilities.shareRange(self._n_samples, ntask, i) for i in range(ntask)]
            lo, _ = _get_lo_hi(self._comm, self._n_samples)
            for itask, (l, h) in enumerate(rank_lo_hi):
                for i in range(l, h):
                    data = self._local_samples[i-lo] if rank == itask else None
                    s = self._comm.bcast(data, root=itask)
                    yield s
                    if self._mirror_samples:
                        yield -s
