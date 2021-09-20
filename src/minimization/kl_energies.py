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
# Authors: Philipp Arras, Philipp Frank
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce

import numpy as np

from .. import random, utilities
from ..domain_tuple import DomainTuple
from ..linearization import Linearization
from ..multi_field import MultiField
from ..operators.endomorphic_operator import EndomorphicOperator
from ..operators.energy_operators import GaussianEnergy, StandardHamiltonian
from ..operators.sampling_enabler import SamplingDtypeSetter, SamplingEnabler
from ..operators.sandwich_operator import SandwichOperator
from ..operators.scaling_operator import ScalingOperator
from ..probing import approximation2endo
from ..sugar import domain_union, is_fieldlike, makeOp
from ..utilities import myassert
from .descent_minimizers import DescentMinimizer
from .energy import Energy
from .energy_adapter import EnergyAdapter
from .sample_list import ResidualSampleList, SampleList


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


def _insert_missing(field, insert):
    """Inserts the parts of `insert` that are not part of `field` into field.
    
    Parameters
    ----------
    field : Field or MultiField
        The original (smaller) field.
    insert : Field or MultiField
        The field which yields the remaining values not in `field`.
    """
    myassert(is_fieldlike(field))
    myassert(is_fieldlike(insert))
    domain = domain_union((field.domain, insert.domain))
    if domain == field.domain:
        return field
    myassert(isinstance(field, MultiField))
    myassert(isinstance(insert, MultiField))
    subkeys = []
    for k in insert.domain.keys():
        if k not in field.domain.keys():
            subkeys.append(k)
    field = field.to_dict()
    for k in subkeys:
        field[k] = insert[k]
    return MultiField.from_dict(field, domain=domain)


def _build_neg(keys, lin_keys):
    myassert(all(kk in keys for kk in lin_keys))
    neg = {kk : kk in lin_keys for kk in keys}
    return neg

class _SelfAdjointOperatorWrapper(EndomorphicOperator):
    def __init__(self, domain, func):
        from ..sugar import makeDomain
        self._func = func
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._domain = makeDomain(domain)

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._func(x)


def draw_samples(position, H, minimizer, n_samples, mirror_samples, linear_keys,
                napprox=0, want_error=False, comm=None):
    if not isinstance(H, StandardHamiltonian):
            raise NotImplementedError
    if isinstance(position, MultiField):
        sam_position = position.extract(H.domain)
    else:
        sam_position = position

    # Check domain dtype
    dts = H._prior._met._dtype
    if isinstance(H.domain, DomainTuple):
        real = np.issubdtype(dts, np.floating)
    else:
        real = all([np.issubdtype(dts[kk], np.floating) for kk in dts.keys()])
    if not real:
        raise ValueError("_GeoMetricSampler only supports real valued latent DOFs.")
    # /Check domain dtype

    # Construct transformation
    tr = H._lh.get_transformation()
    if tr is None:
        raise ValueError("_GeoMetricSampler only works for likelihoods")
    dtype, f_lh = tr
    if isinstance(dtype, dict):
        myassert(all([dtype[k] is not None for k in dtype.keys()]))
    else:
        myassert(dtype is not None)
    scale = SamplingDtypeSetter(ScalingOperator(f_lh.target, 1.), dtype)

    fl = f_lh(Linearization.make_var(sam_position))
    transformation = ScalingOperator(sam_position.domain, 1.) + fl.jac.adjoint@f_lh
    transformation_mean = sam_position + fl.jac.adjoint(fl.val)
    # /Construct transformation

    # Draw samples
    sseq = random.spawn_sseq(n_samples)
    if mirror_samples:
        sseq = reduce((lambda a,b: a+b), [[ss, ]*2 for ss in sseq])

    met = SamplingEnabler(SandwichOperator.make(fl.jac, scale),
        SamplingDtypeSetter(ScalingOperator(fl.domain,1.), np.float64),
        H._ic_samp)
    if napprox >= 1:
        met._approximation = makeOp(approximation2endo(met, napprox))

    local_samples = []
    local_neg = []
    utilities.check_MPI_synced_random_state(comm)
    utilities.check_MPI_equality(sseq, comm)
    y, yi = None, None
    for i in SampleList.indices_from_comm(len(sseq), comm):
        with random.Context(sseq[i]):
            neg = mirror_samples and (i%2 != 0)
            if not neg or y is None:  # we really need to draw a sample
                y, yi = met.draw_sample(True, True)

            
            if minimizer is not None:
                m = transformation_mean - y if neg else transformation_mean + y
                en = GaussianEnergy(mean=m)@transformation
                pos = sam_position - yi if neg else sam_position + yi
                pos, en = _reduce_by_keys(pos, en, linear_keys)
                en = EnergyAdapter(pos, en, nanisinf=True, want_metric=True)
                en, _ = minimizer(en)
                local_samples.append(_insert_missing(en.position - sam_position, yi))
                local_neg.append(False if (not neg or len(linear_keys) == 0)
                    else {kk : kk in linear_keys for kk in yi.domain.keys()})
            else:
                local_samples.append(yi)
                local_neg.append(neg)
    return ResidualSampleList(position, local_samples, local_neg, comm)


class SampledKLEnergy(Energy):
    """Base class for Energies representing a sampled Kullback-Leibler
    divergence for the variational approximation of a distribution with another
    distribution.

    Supports the samples to be distributed across MPI tasks."""
    def __init__(self, sample_list, hamiltonian, nanisinf,
        _callingfrommake = False):
        if not _callingfrommake:
            raise NotImplementedError
        myassert(isinstance(sample_list, ResidualSampleList))
        self._sample_list = sample_list
        super(SampledKLEnergy, self).__init__(self._sample_list.mean)
        myassert(self._sample_list.domain is hamiltonian.domain)
        self._hamiltonian = hamiltonian
        self._nanisinf = bool(nanisinf)

        def _func(inp):
            tmp = hamiltonian(Linearization.make_var(inp))
            return tmp.val.val[()], tmp.gradient

        self._val, self._grad = sample_list.global_average(_func)
        if np.isnan(self._val) and self._nanisinf:
            self._val = np.inf

    @property
    def value(self):
        return self._val

    @property
    def gradient(self):
        return self._grad

    def at(self, position):
        return SampledKLEnergy(self._sample_list.at(position),
            self._hamiltonian, self._nanisinf, _callingfrommake = True)

    def apply_metric(self, x):
        def _func(inp):
            tmp = self._hamiltonian(
                Linearization.make_var(inp, want_metric=True))
            return tmp.metric(x)
        return self._sample_list.global_average(_func)

    @property
    def metric(self):
        return _SelfAdjointOperatorWrapper(self.position.domain,
                                           self.apply_metric)

    @property
    def samples(self):
        return self._sample_list

    @staticmethod
    def make(position, hamiltonian, n_samples, minimizer_sampling, mirror_samples,
    sampling_types='geometric', napprox=0, comm=None, nanisinf=True):
        """Provides the sampled Kullback-Leibler used for Variational Inference,
        specifically for geometric Variational Inference (geoVI) and Metric 
        Gaussian VI (MGVI).

        In geoVI a probability distribution is approximated with a standard
        normal distribution in the canonical coordinate system of the Riemannian
        manifold associated with the metric of the other distribution. The
        coordinate transformation is approximated by expanding around a point.
        The MGVI simplification occurs in case this transformation can be
        approximated using a linear expansion. In order to infer the optimal 
        expansion point, a stochastic estimate of the Kullback-Leibler
        divergence is minimized. This estimate is obtained by sampling from the
        approximation using the current expansion point. During minimization
        these samples are kept constant; only the expansion point is updated.
        Due to the typically nonlinear structure of the true distribution these
        samples have to be updated eventually by instantiating a 
        `SampledKLEnergy` again. For the true probability distribution the
        standard parametrization is assumed. The samples of this class can be
        distributed among MPI tasks.

        Parameters
        ----------
        position : Field
            Expansion point of the coordinate transformation.
        hamiltonian : StandardHamiltonian
            Hamiltonian of the approximated probability distribution.
        n_samples : integer
            Number of samples used to stochastically estimate the KL.
        minimizer_samp : DescentMinimizer or None
            Minimizer used to draw samples. Can only be None in case no
            `geometric` samples are drawn.
        mirror_samples : boolean
            Whether the mirrored version of the drawn samples are also used.
            If true, the number of used samples doubles.
            Mirroring samples stabilizes the KL estimate as extreme
            sample variation is counterbalanced.
        sampling_types : String or dict(String)
            The type of sampling used to perform variational approximation.
            There are three supported modes: `point`; A point (aka maximum a 
            posterior) estimate. No sampling is performed. `linear`; Linear
            (MGVI) sample assuming that the linear approximation of the
            transformation is sufficiently accurate. `geometric`; Full (geoVI)
            sample. Internally, first a `linear` sample is drawn and then
            further optimized non-linearly. Also supports different sampling
            types for different parts of the domain. In this case `sampling_type`
            must be a dict that contains one of the three sampling types for
            each key of the domain.
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
        Mirroring samples can help to stabilize the latent mean as it
        reduces sampling noise. But a mirrored sample involves an additional
        solve of the non-linear part of the transformation. When using MPI, the
        samples get distributed as evenly as possible over all tasks. If the
        number of tasks is smaller then the total number of samples (including
        mirrored ones), the mirrored pairs try to reside on the same task as
        their non mirrored partners. This ensures that at least the linear part
        of the sampling is re-used.

        See also
        --------
        `Geometric Variational Inference`, Philipp Frank, Reimar Leike,
        Torsten A. Enßlin, `<https://arxiv.org/abs/2105.10470>`_
        `<https://doi.org/10.3390/e23070853>`_

        `Metric Gaussian Variational Inference`, Jakob Knollmüller,
        Torsten A. Enßlin, `<https://arxiv.org/abs/1901.11033>`_
        """
        if not isinstance(hamiltonian, StandardHamiltonian):
            raise TypeError
        if hamiltonian.domain is not position.domain:
            raise ValueError
        if not isinstance(n_samples, int):
            raise TypeError
        if not isinstance(mirror_samples, bool):
            raise TypeError
        if not (minimizer_sampling is None or
            isinstance(minimizer_sampling, DescentMinimizer)):
            raise TypeError

        types = ['geometric', 'linear', 'point']
        lists = {tt : [] for tt in types}
        if isinstance(position, MultiField):
            if isinstance(sampling_types, str):
                sampling = {k:sampling_types for k in position.domain.keys()}
                sampling_types = sampling
            else:
                myassert(set(position.domain.keys()) == set(sampling_types.keys()))
            for k in sampling_types.keys():
                tt = sampling_types[k]
                if tt not in types:
                    raise ValueError(f'Sampling type {tt} for key {k} not understood')
                lists[tt].append(k)
        if len(lists['geometric']) != 0 and minimizer_sampling is None:
            raise ValueError("Cannot draw geometric samples without a Minimizer")
        elif len(lists['geometric']) == 0:
            minimizer_sampling = None

        n_samples = int(n_samples)
        mirror_samples = bool(mirror_samples)
        _, ham_sampling = _reduce_by_keys(position, hamiltonian, lists['point'])
        sample_list = draw_samples(position, ham_sampling, minimizer_sampling,
            n_samples, mirror_samples, lists['linear'], napprox=napprox, comm=comm)

        return SampledKLEnergy(sample_list, hamiltonian, nanisinf,
                            _callingfrommake = True)