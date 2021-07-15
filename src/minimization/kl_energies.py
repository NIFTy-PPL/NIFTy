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
# Authors: Philipp Frank, Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce

import numpy as np

from .. import random, utilities
from ..domain_tuple import DomainTuple
from ..linearization import Linearization
from ..multi_field import MultiField
from ..operators.adder import Adder
from ..operators.endomorphic_operator import EndomorphicOperator
from ..operators.energy_operators import GaussianEnergy, StandardHamiltonian
from ..operators.inversion_enabler import InversionEnabler
from ..operators.sampling_enabler import SamplingDtypeSetter
from ..operators.sandwich_operator import SandwichOperator
from ..operators.scaling_operator import ScalingOperator
from ..probing import approximation2endo
from ..sugar import makeOp
from ..utilities import myassert
from .descent_minimizers import ConjugateGradient, DescentMinimizer
from .energy import Energy
from .energy_adapter import EnergyAdapter
from .quadratic_energy import QuadraticEnergy


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


def _reduce_by_keys(field, operator, keys):
    """Partially insert a field into an operator

    If the domain of the operator is an instance of `DomainTuple`

    Parameters
    ----------
    field : Field or MultiField
        Potentially partially constant input field.
    operator : Operator
        Operator into which `field` is partially inserted.
    keys : list
        List of constant `MultiDomain` entries.

    Returns
    -------
    list
        The variable part of the field and the contracted operator.
    """
    from ..sugar import is_fieldlike, is_operator
    myassert(is_fieldlike(field))
    myassert(is_operator(operator))
    if isinstance(field, MultiField):
        cst_field = field.extract_by_keys(keys)
        var_field = field.extract_by_keys(set(field.keys()) - set(keys))
        _, new_ham = operator.simplify_for_constant_input(cst_field)
        return var_field, new_ham
    myassert(len(keys) == 0)
    return field, operator


class _SelfAdjointOperatorWrapper(EndomorphicOperator):
    def __init__(self, domain, func):
        from ..sugar import makeDomain
        self._func = func
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._domain = makeDomain(domain)

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._func(x)


class _SampledKLEnergy(Energy):
    """Base class for Energies representing a sampled Kullback-Leibler
    divergence for the variational approximation of a distribution with another
    distribution.

    Supports the samples to be distributed across MPI tasks."""
    def __init__(self, mean, hamiltonian, n_samples, mirror_samples, comm,
                 local_samples, nanisinf):
        super(_SampledKLEnergy, self).__init__(mean)
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

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def at(self, position):
        return _SampledKLEnergy(
            position, self._hamiltonian, self._n_samples, self._mirror_samples,
            self._comm, self._local_samples, self._nanisinf)

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
        return _SelfAdjointOperatorWrapper(self.position.domain,
                                           self.apply_metric)

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


class _MetricGaussianSampler:
    def __init__(self, position, H, n_samples, mirror_samples, napprox=0):
        if not isinstance(H, StandardHamiltonian):
            raise NotImplementedError
        lin = Linearization.make_var(position.extract(H.domain), True)
        self._met = H(lin).metric
        if napprox >= 1:
            self._met._approximation = makeOp(approximation2endo(self._met, napprox))
        self._n = int(n_samples)

    def draw_samples(self, comm):
        local_samples = []
        utilities.check_MPI_synced_random_state(comm)
        sseq = random.spawn_sseq(self._n)
        for i in range(*_get_lo_hi(comm, self._n)):
            with random.Context(sseq[i]):
                local_samples.append(self._met.draw_sample(from_inverse=True))
        return tuple(local_samples)


class _GeoMetricSampler:
    def __init__(self, position, H, minimizer, start_from_lin,
                 n_samples, mirror_samples, napprox=0, want_error=False):
        if not isinstance(H, StandardHamiltonian):
            raise NotImplementedError

        # Check domain dtype
        dts = H._prior._met._dtype
        if isinstance(H.domain, DomainTuple):
            real = np.issubdtype(dts, np.floating)
        else:
            real = all([np.issubdtype(dts[kk], np.floating) for kk in dts.keys()])
        if not real:
            raise ValueError("_GeoMetricSampler only supports real valued latent DOFs.")
        # /Check domain dtype

        if isinstance(position, MultiField):
            self._position = position.extract(H.domain)
        else:
            self._position = position
        tr = H._lh.get_transformation()
        if tr is None:
            raise ValueError("_GeoMetricSampler only works for likelihoods")
        dtype, f_lh = tr
        scale = ScalingOperator(f_lh.target, 1.)
        if isinstance(dtype, dict):
            sampling = reduce((lambda a,b: a*b),
                              [dtype[k] is not None for k in dtype.keys()])
        else:
            sampling = dtype is not None
        scale = SamplingDtypeSetter(scale, dtype) if sampling else scale

        fl = f_lh(Linearization.make_var(self._position))
        self._g = (Adder(-self._position) + fl.jac.adjoint@Adder(-fl.val)@f_lh)
        self._likelihood = SandwichOperator.make(fl.jac, scale)
        self._prior = SamplingDtypeSetter(ScalingOperator(fl.domain,1.), np.float64)
        self._met = self._likelihood + self._prior
        if napprox >= 1:
            self._approximation = makeOp(approximation2endo(self._met, napprox)).inverse
        else:
            self._approximation = None
        self._ic = H._ic_samp
        self._minimizer = minimizer
        self._start_from_lin = start_from_lin
        self._want_error = want_error

        sseq = random.spawn_sseq(n_samples)
        if mirror_samples:
            mysseq = []
            for seq in sseq:
                mysseq += [seq, seq]
        else:
            mysseq = sseq
        self._sseq = mysseq
        self._neg = (False, True)*n_samples if mirror_samples else (False, )*n_samples
        self._n_samples = n_samples
        self._mirror_samples = mirror_samples

    @property
    def n_eff_samples(self):
        return 2*self._n_samples if self._mirror_samples else self._n_samples

    @property
    def position(self):
        return self._position

    def _draw_lin(self, neg):
        s = self._prior.draw_sample(from_inverse=True)
        s = -s if neg else s
        nj = self._likelihood.draw_sample()
        nj = -nj if neg else nj
        y = self._prior(s) + nj
        if self._start_from_lin:
            energy = QuadraticEnergy(s, self._met, y,
                                     _grad=self._likelihood(s) - nj)
            inverter = ConjugateGradient(self._ic)
            energy, convergence = inverter(energy,
                                           preconditioner=self._approximation)
            yi = energy.position
        else:
            yi = s
        return y, yi

    def _draw_nonlin(self, y, yi):
        en = EnergyAdapter(self._position+yi, GaussianEnergy(mean=y)@self._g,
                           nanisinf=True, want_metric=True)
        en, _ = self._minimizer(en)
        sam = en.position - self._position
        if self._want_error:
            er = y - self._g(sam)
            er = er.s_vdot(InversionEnabler(self._met, self._ic).inverse(er))
            return sam, er
        return sam

    def draw_samples(self, comm):
        local_samples = []
        prev = None
        utilities.check_MPI_synced_random_state(comm)
        utilities.check_MPI_equality(self._sseq, comm)
        for i in range(*_get_lo_hi(comm, self.n_eff_samples)):
            with random.Context(self._sseq[i]):
                neg = self._neg[i]
                if (prev is None) or not self._mirror_samples:
                    y, yi = self._draw_lin(neg)
                    if not neg:
                        prev = (-y, -yi)
                else:
                    (y, yi) = prev
                    prev = None
                local_samples.append(self._draw_nonlin(y, yi))
        return tuple(local_samples)


def MetricGaussianKL(mean, hamiltonian, n_samples, mirror_samples, constants=[],
                     point_estimates=[], napprox=0, comm=None, nanisinf=False):
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
        these occasions but rather the minimizer is told that the position it
        has tried is not sensible.

    Note
    ----
    The two lists `constants` and `point_estimates` are independent from each
    other. It is possible to sample along domains which are kept constant
    during minimization and vice versa.

    See also
    --------
    `Metric Gaussian Variational Inference`, Jakob Knollmüller,
    Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
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

    _, ham_sampling = _reduce_by_keys(mean, hamiltonian, point_estimates)
    sampler = _MetricGaussianSampler(mean, ham_sampling, n_samples,
                                     mirror_samples, napprox)
    local_samples = sampler.draw_samples(comm)

    mean, hamiltonian = _reduce_by_keys(mean, hamiltonian, constants)
    return _SampledKLEnergy(mean, hamiltonian, n_samples, mirror_samples, comm,
                            local_samples, nanisinf)


def GeoMetricKL(mean, hamiltonian, n_samples, minimizer_samp, mirror_samples,
                start_from_lin=True, constants=[], point_estimates=[],
                napprox=0, comm=None, nanisinf=True):
    """Provides the sampled Kullback-Leibler used in geometric Variational
    Inference (geoVI).

    In geoVI a probability distribution is approximated with a standard normal
    distribution in the canonical coordinate system of the Riemannian manifold
    associated with the metric of the other distribution. The coordinate
    transformation is approximated by expanding around a point. In order to
    infer the expansion point, a stochastic estimate of the Kullback-Leibler
    divergence is minimized. This estimate is obtained by sampling from the
    approximation using the current expansion point. During minimization these
    samples are kept constant; only the expansion point is updated. Due to the
    typically nonlinear structure of the true distribution these samples have
    to be updated eventually by instantiating `GeoMetricKL` again. For the true
    probability distribution the standard parametrization is assumed.
    The samples of this class can be distributed among MPI tasks.

    Parameters
    ----------
    mean : Field
        Expansion point of the coordinate transformation.
    hamiltonian : StandardHamiltonian
        Hamiltonian of the approximated probability distribution.
    n_samples : integer
        Number of samples used to stochastically estimate the KL.
    minimizer_samp : DescentMinimizer
        Minimizer used to draw samples.
    mirror_samples : boolean
        Whether the mirrored version of the drawn samples are also used.
        If true, the number of used samples doubles.
        Mirroring samples stabilizes the KL estimate as extreme
        sample variation is counterbalanced.
    start_from_lin : boolean
        Whether the non-linear sampling should start using the inverse
        linearized transformation (i.e. the corresponding MGVI sample).
        If False, the minimization starts from the prior sample.
        Default is True.
    constants : list
        List of parameter keys that are kept constant during optimization.
        Default is no constants.
    point_estimates : list
        List of parameter keys for which no samples are drawn, but that are
        (possibly) optimized for, corresponding to point estimates of these.
        Default is to draw samples for the complete domain.
    napprox : int
        Number of samples for computing preconditioner for linear sampling.
        No preconditioning is done by default.
    comm : MPI communicator or None
        If not None, samples will be distributed as evenly as possible
        across this communicator. If `mirror_samples` is set, then a sample and
        its mirror image will preferably reside on the same task if necessary.
    nanisinf : bool
        If true, nan energies which can happen due to overflows in the forward
        model are interpreted as inf. Thereby, the code does not crash on
        these occasions but rather the minimizer is told that the position it
        has tried is not sensible.

    Note
    ----
    The two lists `constants` and `point_estimates` are independent from each
    other. It is possible to sample along domains which are kept constant
    during minimization and vice versa.
    DomainTuples should never be created using the constructor, but rather
    via the factory function :attr:`make`!

    Note
    ----
    As in MGVI, mirroring samples can help to stabilize the latent mean as it
    reduces sampling noise. But unlike MGVI a mirrored sample involves an
    additional solve of the non-linear transformation. Therefore, when using
    MPI, the mirrored samples also get distributed if enough tasks are
    available.  If there are more total samples than tasks, the mirrored
    counterparts try to reside on the same task as their non mirrored partners.
    This ensures that at least the starting position can be re-used.

    See also
    --------
    `Geometric Variational Inference`, Philipp Frank, Reimar Leike,
    Torsten A. Enßlin, `<https://arxiv.org/abs/2105.10470>`_
    `<https://doi.org/10.3390/e23070853>`_
    """
    if not isinstance(hamiltonian, StandardHamiltonian):
        raise TypeError
    if hamiltonian.domain is not mean.domain:
        raise ValueError
    if not isinstance(n_samples, int):
        raise TypeError
    if not isinstance(mirror_samples, bool):
        raise TypeError
    if not isinstance(minimizer_samp, DescentMinimizer):
        raise TypeError
    if isinstance(mean, MultiField) and set(point_estimates) == set(mean.keys()):
        s = 'Point estimates for whole domain. Use EnergyAdapter instead.'
        raise RuntimeError(s)

    n_samples = int(n_samples)
    mirror_samples = bool(mirror_samples)

    _, ham_sampling = _reduce_by_keys(mean, hamiltonian, point_estimates)
    sampler = _GeoMetricSampler(mean, ham_sampling, minimizer_samp,
                                start_from_lin, n_samples, mirror_samples,
                                napprox)
    local_samples = sampler.draw_samples(comm)
    mean, hamiltonian = _reduce_by_keys(mean, hamiltonian, constants)
    return _SampledKLEnergy(mean, hamiltonian, sampler.n_eff_samples, False,
                            comm, local_samples, nanisinf)
