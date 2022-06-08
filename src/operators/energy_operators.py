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
# Copyright(C) 2013-2022 Max-Planck-Society, Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce
from operator import add

import numpy as np

from .. import utilities
from ..domain_tuple import DomainTuple
from ..field import Field
from ..linearization import Linearization
from ..multi_domain import MultiDomain
from ..multi_field import MultiField
from ..sugar import makeDomain, makeOp
from ..utilities import iscomplextype, myassert
from .adder import Adder
from .linear_operator import LinearOperator
from .operator import Operator, _OpChain, _OpSum
from .sampling_enabler import SamplingEnabler
from .sandwich_operator import SandwichOperator
from .scaling_operator import ScalingOperator
from .simple_linear_operators import FieldAdapter, VdotOperator


def _check_sampling_dtype(domain, dtypes):
    if dtypes is None:
        return
    if isinstance(domain, DomainTuple):
        np.dtype(dtypes)
        return
    elif isinstance(domain, MultiDomain):
        if isinstance(dtypes, dict):
            for dt in dtypes.values():
                np.dtype(dt)
            if set(domain.keys()) == set(dtypes.keys()):
                return
        else:
            np.dtype(dtypes)
            return
    raise TypeError


def _field_to_dtype(field):
    if isinstance(field, Field):
        dt = field.dtype
    elif isinstance(field, MultiField):
        dt = {kk: ff.dtype for kk, ff in field.items()}
    else:
        raise TypeError
    _check_sampling_dtype(field.domain, dt)
    return dt


class EnergyOperator(Operator):
    """Operator which has a scalar domain as target domain.

    It is intended as an objective function for field inference.  It can
    implement a positive definite, symmetric form (called `metric`) that is
    used as curvature for second-order minimizations.

    Examples
    --------
     - Information Hamiltonian, i.e. negative-log-probabilities.
     - Gibbs free energy, i.e. an averaged Hamiltonian, aka Kullback-Leibler
       divergence.
    """
    _target = DomainTuple.scalar_domain()


class LikelihoodEnergyOperator(EnergyOperator):
    """Represent a negative log-likelihood.

    The input to the Operator are the parameters of the negative log-likelihood.
    Unlike a general `EnergyOperator`, the metric of a
    `LikelihoodEnergyOperator` is the Fisher information metric of the
    likelihood.
    """
    def __init__(self, data_residual, sqrt_data_metric_at):
        from ..extra import is_operator
        if data_residual is not None and not is_operator(data_residual):
            raise TypeError(f"{data_residual} is not an operator")
        self._res = data_residual
        self._sqrt_data_metric_at = sqrt_data_metric_at
        self._name = None

    def normalized_residual(self, x):
        return (self._sqrt_data_metric_at(x) @ self._res).force(x)

    @property
    def data_domain(self):
        if self._res is None:
            return None
        return self._res.target

    def get_transformation(self):
        """The coordinate transformation that maps into a coordinate system in
        which the metric of a likelihood is the Euclidean metric.

        Returns
        -------
        np.dtype, or dict of np.dtype : The dtype(s) of the target space of the
        transformation.

        Operator : The transformation that maps from `domain` into the
        Euclidean target space.

        Note
        ----
        This Euclidean target space is the disjoint union of the Euclidean
        target spaces of all summands. Therefore, the keys of `MultiDomains`
        are prefixed with an index and `DomainTuples` are converted to
        `MultiDomains` with the index as the key.
        """
        raise NotImplementedError

    def __matmul__(self, other):
        return _LikelihoodChain(self, other)

    def __rmatmul__(self, other):
        return _LikelihoodChain(other, self)

    def __add__(self, other):
        return _LikelihoodSum.make([self, other])

    def __radd__(self, other):
        return _LikelihoodSum.make([other, self])

    def get_metric_at(self, x):
        """Compute the Fisher information metric for a `LikelihoodEnergyOperator`
        at `x` using the Jacobian of the coordinate transformation given by
        :func:`~nifty8.operators.operator.Operator.get_transformation`. """
        dtp, f = self.get_transformation()
        bun = f(Linearization.make_var(x)).jac
        return SandwichOperator.make(bun, sampling_dtype=dtp)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, x):
        if isinstance(self, _LikelihoodSum):
            raise RuntimeError("The name of a LikelihoodSum cannot be set. "
                               "Set the name of each individual LikelihoodEnergy separately.")
        self._name = x


class _LikelihoodChain(LikelihoodEnergyOperator):
    def __init__(self, op1, op2):
        from .simple_linear_operators import PartialExtractor
        self._op = _OpChain.make((op1, op2))
        self._domain = self._op.domain

        if isinstance(op1, ScalingOperator):
            res = op2._res
            sqrt_data_metric_at = op2._sqrt_data_metric_at
        elif op1._res is None:
            res = None
            sqrt_data_metric_at = None
        else:
            if isinstance(op2.target, MultiDomain):
                extract = PartialExtractor(op2.target, op1._res.domain)
            else:
                extract = Operator.identity_operator(op2.target)
            res = op1._res @ extract @ op2
            sqrt_data_metric_at = lambda x: op1._sqrt_data_metric_at(op2.force(x))

        super(_LikelihoodChain, self).__init__(res, sqrt_data_metric_at)
        self.name = (op2 if isinstance(op1, ScalingOperator) else op1).name

    def get_transformation(self):
        scaled_lh = isinstance(self._op._ops[0], ScalingOperator)
        ii = 1 if scaled_lh else 0
        tr = self._op._ops[ii].get_transformation()
        if tr is None:
            return tr
        dtype, trafo = tr
        if scaled_lh:
            trafo = trafo.scale(np.sqrt(self._op._ops[0]._factor))
        return dtype, _OpChain.make((trafo,)+self._op._ops[ii+1:])

    def apply(self, x):
        self._check_input(x)
        return self._op(x)

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        return self._op._simplify_for_constant_input_nontrivial(c_inp)

    def __repr__(self):
        return self._op.__repr__()


class _LikelihoodSum(LikelihoodEnergyOperator):
    def __init__(self, ops, _callingfrommake=False):
        from .simple_linear_operators import PrependKey

        if not _callingfrommake:
            raise NotImplementedError
        if len(set([isinstance(oo.domain, DomainTuple) for oo in ops])) > 1:
            raise RuntimeError("Some operators have DomainTuple and others have "
                    "MultiDomain as domain. This should not happen.")
        self._ops = ops
        res, prep, data_ops = [], [], []
        for ii, oo in enumerate(ops):
            rr = oo._res
            if rr is None:
                continue
            tgt = oo.data_domain
            lprep = Operator.identity_operator(tgt)
            key = self._get_name(ii)
            if isinstance(lprep.target, DomainTuple):
                lprep = lprep.ducktape_left("")
            else:
                key = key + ": "
            lprep = PrependKey(lprep.target, key) @ lprep
            prep.append(lprep)
            res.append(lprep @ rr)
            data_ops.append(oo)

        def sqrt_data_metric_at(x):
            return reduce(add, (pp @ oo._sqrt_data_metric_at(x) @ pp.adjoint
                                for pp, oo in zip(prep, data_ops)))

        lst = self._all_names()
        if len(lst) != len(set(lst)):
            raise ValueError(f"Name collision in likelihoods detected: {lst}")

        data_residuals = reduce(add, res)
        super(_LikelihoodSum, self).__init__(data_residuals, sqrt_data_metric_at)
        self._domain = data_residuals.domain

    @classmethod
    def unpack(cls, ops, res):
        for op in ops:
            if isinstance(op, cls):
                res = cls.unpack(op._ops, res)
            else:
                res = res + [op]
        return res

    @classmethod
    def make(cls, ops):
        res = cls.unpack(ops, [])
        if len(res) == 1:
            return res[0]
        return cls(res, _callingfrommake=True)

    def apply(self, x):
        from ..linearization import Linearization
        self._check_input(x)
        return _OpSum._apply_operator_sum(x, self._ops)

    def get_transformation(self):
        from .simple_linear_operators import PrependKey

        if any(oo.get_transformation() is None for oo in self._ops):
            return None
        dtype, trafo = {}, []
        for ii, lh in enumerate(self._ops):
            dtp, tr = lh.get_transformation()
            key = self._get_name(ii)
            if isinstance(tr.target, MultiDomain):
                key = key + ": "
                dtype.update({key+d: dtp[d] for d in dtp.keys()})
                tr = PrependKey(tr.target, key) @ tr
            else:
                dtype[key] = dtp
                tr = tr.ducktape_left(key)
            trafo.append(tr)
        return dtype, reduce(add, trafo)

    def _all_names(self):
        return [self._get_name(ii) for ii in range(len(self._ops))]

    def _get_name(self, i):
        res = self._ops[i].name
        if res is None:
            return f"Likelihood {i}"
        return res

    def __repr__(self):
        subs = []
        for ii, oo in enumerate(self._ops):
            subs.append(f"*{self._get_name(ii)}*")
            subs.append(oo.__repr__())
        return "_LikelihoodSum:\n" + utilities.indent("\n".join(subs))


class Squared2NormOperator(EnergyOperator):
    """Computes the square of the L2-norm of the output of an operator.

    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        Domain of the operator in which the L2-norm shall be computed.
    """

    def __init__(self, domain):
        self._domain = domain

    def apply(self, x):
        self._check_input(x)
        if x.jac is None:
            return x.vdot(x)
        res = x.val.vdot(x.val)
        return x.new(res, VdotOperator(2*x.val))


class QuadraticFormOperator(EnergyOperator):
    """Computes the L2-norm of a Field or MultiField with respect to a
    specific kernel given by `endo`.

    .. math ::
        E(f) = \\frac12 f^\\dagger \\text{endo}(f)

    Parameters
    ----------
    endo : EndomorphicOperator
         Kernel of the quadratic form
    """

    def __init__(self, endo):
        from .endomorphic_operator import EndomorphicOperator
        if not isinstance(endo, EndomorphicOperator):
            raise TypeError("op must be an EndomorphicOperator")
        self._op = endo
        self._domain = endo.domain

    def apply(self, x):
        self._check_input(x)
        if x.jac is None:
            return 0.5*x.vdot(self._op(x))
        res = 0.5*x.val.vdot(self._op(x.val))
        return x.new(res, VdotOperator(self._op(x.val)))


class VariableCovarianceGaussianEnergy(LikelihoodEnergyOperator):
    """Computes the negative log pdf of a Gaussian with unknown covariance.

    The covariance is assumed to be diagonal.

    .. math ::
        E(s,D) = - \\log G(s, C) = 0.5 (s)^\\dagger C (s) - 0.5 tr log(C),

    an information energy for a Gaussian distribution with residual s and
    inverse diagonal covariance C.
    The domain of this energy will be a MultiDomain with two keys,
    the target will be the scalar domain.

    Parameters
    ----------
    domain : Domain, DomainTuple, tuple of Domain
        domain of the residual and domain of the covariance diagonal.

    residual_key : key
        Residual key of the Gaussian.

    inverse_covariance_key : key
        Inverse covariance diagonal key of the Gaussian.

    sampling_dtype : np.dtype
        Data type of the samples. Usually either 'np.float*' or 'np.complex*'

    use_full_fisher: boolean
        Determines if the proper Fisher information metric should be used as
        `metric`. If False, the same approximation as in `get_transformation`
        is used. Default is True.
    """

    def __init__(self, domain, residual_key, inverse_covariance_key,
                 sampling_dtype, use_full_fisher=True):
        self._kr = str(residual_key)
        self._ki = str(inverse_covariance_key)
        dom = DomainTuple.make(domain)
        self._domain = MultiDomain.make({self._kr: dom, self._ki: dom})
        self._dt = {self._kr: sampling_dtype, self._ki: np.float64}
        _check_sampling_dtype(self._domain, self._dt)
        self._cplx = iscomplextype(sampling_dtype)
        self._use_full_fisher = use_full_fisher
        super(VariableCovarianceGaussianEnergy, self).__init__(
                Operator.identity_operator(dom).ducktape(self._kr),
                lambda x: makeOp(x[self._ki].sqrt()))

    def apply(self, x):
        self._check_input(x)
        r, i = x[self._kr], x[self._ki]
        if self._cplx:
            res = 0.5*r.vdot(r*i.real).real - i.log().sum()
        else:
            res = 0.5*(r.vdot(r*i) - i.log().sum())
        if not x.want_metric:
            return res
        if not self._use_full_fisher:
            return res.add_metric(self.get_metric_at(x.val))
        fct = 1. if self._cplx else 0.5
        met = {self._kr: i.val, self._ki: fct*i.val**(-2)}
        met = MultiField.from_dict(met, domain=self._domain)
        met = makeOp(met, sampling_dtype=self._dt)
        return res.add_metric(met)

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        from .simplify_for_const import ConstantLikelihoodEnergyOperator
        myassert(len(c_inp.keys()) == 1)
        key = c_inp.keys()[0]
        myassert(key in self._domain.keys())
        cst = c_inp[key]
        if key == self._kr:
            res = _SpecialGammaEnergy(cst).ducktape(self._ki)
        else:
            icov = makeOp(cst, sampling_dtype=self._dt[self._kr])
            res = GaussianEnergy(data=None, inverse_covariance=icov).ducktape(self._kr)
            trlog = cst.log().sum().val_rw()
            if not self._cplx:
                trlog /= 2
            res = res + ConstantLikelihoodEnergyOperator(-trlog)
        res = res + ConstantLikelihoodEnergyOperator(0.)
        myassert(res.target is self.target)
        return None, res

    def get_transformation(self):
        """
        Note
        ----
        For `VariableCovarianceGaussianEnergy`, a global transformation to
        Euclidean space does not exist. A local approximation invoking the
        residual is used instead.
        """
        r = FieldAdapter(self._domain[self._kr], self._kr)
        ivar = FieldAdapter(self._domain[self._kr], self._ki).real
        sc = 1. if self._cplx else 0.5
        f = r.adjoint @ (ivar.sqrt()*r) + ivar.adjoint @ (sc*ivar.log())
        return self._dt, f


class _SpecialGammaEnergy(LikelihoodEnergyOperator):
    def __init__(self, residual):
        from .simplify_for_const import ConstantOperator

        self._domain = DomainTuple.make(residual.domain)
        self._resi = residual
        self._cplx = iscomplextype(self._resi.dtype)
        self._dt = self._resi.dtype
        super(_SpecialGammaEnergy, self).__init__(
                ConstantOperator(self._resi, domain=self._domain),
                lambda x: self.get_metric_at(x).get_sqrt())

    def apply(self, x):
        self._check_input(x)
        r = self._resi
        if self._cplx:
            res = 0.5*(r*x.real).vdot(r).real - x.log().sum()
        else:
            res = 0.5*((r*x).vdot(r) - x.log().sum())
        if not x.want_metric:
            return res
        return res.add_metric(self.get_metric_at(x.val))

    def get_transformation(self):
        sc = 1. if self._cplx else np.sqrt(0.5)
        return self._dt, Operator.identity_operator(self._domain).log().scale(sc)


class GaussianEnergy(LikelihoodEnergyOperator):
    """Computes a negative-log Gaussian.

    Represents up to constants in :math:`d`:

    .. math ::
        E(f) = - \\log G(f-d, D) = 0.5 (f-d)^\\dagger D^{-1} (f-d),

    an information energy for a Gaussian distribution with data d and
    covariance D.

    Parameters
    ----------
    data : :class:`nifty8.field.Field` or None
        Observed data of the Gaussian likelihood. If `inverse_covariance` is
        `None`, the `dtype` of `data` is used for sampling. Default is
        0.
    inverse_covariance : LinearOperator
        Inverse covariance of the Gaussian. Default is the identity operator.
    domain : Domain, DomainTuple, tuple of Domain or MultiDomain
        Operator domain. By default it is inferred from `data` or
        `covariance` if specified
    sampling_dtype : dtype or dict of dtype
        Type used for sampling from the inverse covariance if
        `inverse_covariance` and `data` is `None`. Otherwise, this parameter
        does not have an effect. Default: None.

    Note
    ----
    At least one of the arguments has to be provided.
    """

    def __init__(self, data=None, inverse_covariance=None, domain=None, sampling_dtype=None):
        from ..sugar import full

        if inverse_covariance is not None and not isinstance(inverse_covariance, LinearOperator):
            raise TypeError

        self._domain = self._parseDomain(data, inverse_covariance, domain)

        if not isinstance(data, (Field, MultiField)) and data is not None:
            raise TypeError

        self._icov = inverse_covariance
        if inverse_covariance is None:
            self._op = Squared2NormOperator(self._domain).scale(0.5)
            dt = sampling_dtype if data is None else data.dtype
            self._icov = ScalingOperator(self._domain, 1., dt)
        else:
            self._op = QuadraticFormOperator(inverse_covariance)
            self._icov = inverse_covariance

        self._data = data
        if data is None:
            res = Operator.identity_operator(self._domain)
        else:
            res = Adder(data, neg=True)
        super(GaussianEnergy, self).__init__(res, lambda x: self.get_metric_at(x).get_sqrt())

        icovdtype = self._icov.sampling_dtype
        if icovdtype is not None and data is not None and icovdtype != data.dtype:
            for i0, i1 in [(icovdtype, data.dtype), (data.dtype, icovdtype)]:
                if isinstance(i0, dict) and not isinstance(i1, dict):
                    fst = list(i0.values())[0]
                    if all(elem == fst for elem in i0.values()) and i1 == fst:
                        return
            s = "Sampling dtype of inverse covariance does not match dtype of data.\n"
            s += f"icov.sampling_dtype: {icovdtype}\n"
            s += f"data.dtype: {data.dtype}"
            raise RuntimeError(s)


    @staticmethod
    def _checkEquivalence(olddom, newdom):
        newdom = makeDomain(newdom)
        if olddom is None:
            return newdom
        utilities.check_object_identity(olddom, newdom)
        return newdom

    def _parseDomain(self, data, inverse_covariance, domain):
        dom = None
        if inverse_covariance is not None:
            dom = self._checkEquivalence(dom, inverse_covariance.domain)
        if data is not None:
            dom = self._checkEquivalence(dom, data.domain)
        if domain is not None:
            dom = self._checkEquivalence(dom, domain)
        if dom is None:
            raise ValueError("no domain given")
        return dom

    def apply(self, x):
        self._check_input(x)
        residual = x if self._data is None else x - self._data
        res = self._op(residual).real
        if x.want_metric:
            return res.add_metric(self.get_metric_at(x.val))
        return res

    def get_transformation(self):
        return self._icov.sampling_dtype, self._icov.get_sqrt()

    def __repr__(self):
        dom = '()' if isinstance(self.domain, DomainTuple) else self.domain.keys()
        return f'GaussianEnergy {dom}'


class PoissonianEnergy(LikelihoodEnergyOperator):
    """Computes likelihood Hamiltonians of expected count field constrained by
    Poissonian count data.

    Represents up to an f-independent term :math:`log(d!)`:

    .. math ::
        E(f) = -\\log \\text{Poisson}(d|f) = \\sum f - d^\\dagger \\log(f),

    where f is a :class:`Field` in data space with the expectation values for
    the counts.

    Parameters
    ----------
    d : :class:`nifty8.field.Field`
        Data field with counts. Needs to have integer dtype and all field
        values need to be non-negative.
    """

    def __init__(self, d):
        if not isinstance(d, Field) or not np.issubdtype(d.dtype, np.integer):
            te = "data is of invalid data-type; counts need to be integers"
            raise TypeError(te)
        if np.any(d.val < 0):
            ve = "count data is negative and thus can not be Poissonian"
            raise ValueError(ve)
        self._d = d
        self._domain = DomainTuple.make(d.domain)
        super(PoissonianEnergy, self).__init__(Adder(d, neg=True),
                                               lambda x: self.get_metric_at(x).get_sqrt())

    def apply(self, x):
        self._check_input(x)
        res = x.sum() - x.log().vdot(self._d)
        if not x.want_metric:
            return res
        return res.add_metric(self.get_metric_at(x.val))

    def get_transformation(self):
        return np.float64, 2.*Operator.identity_operator(self._domain).sqrt()


class InverseGammaEnergy(LikelihoodEnergyOperator):
    """Computes the negative log-likelihood of the inverse gamma distribution.

    It negative log-pdf(x) is given by

    .. math ::

        \\sum_i (\\alpha_i+1)*\\ln(x_i) + \\beta_i/x_i

    This is the likelihood for the variance :math:`x=S_k` given data
    :math:`\\beta = 0.5 |s_k|^2` where the :class:`nifty8.field.Field`
    :math:`s` is known to have the covariance :math:`S_k`.

    Parameters
    ----------
    beta : :class:`nifty8.field.Field`
        beta parameter of the inverse gamma distribution
    alpha : Scalar, :class:`nifty8.field.Field`, optional
        alpha parameter of the inverse gamma distribution
    """

    def __init__(self, beta, alpha=-0.5):
        from .simplify_for_const import ConstantOperator

        if not isinstance(beta, Field):
            raise TypeError
        self._domain = DomainTuple.make(beta.domain)
        self._beta = beta
        if np.isscalar(alpha):
            alpha = Field(beta.domain, np.full(beta.shape, alpha))
        elif not isinstance(alpha, Field):
            raise TypeError
        self._alphap1 = alpha+1
        if not self._beta.dtype == np.float64:
            # FIXME Add proper complex support for this energy
            raise TypeError
        self._sampling_dtype = _field_to_dtype(self._beta)

        super(InverseGammaEnergy, self).__init__(
                2*ConstantOperator(self._beta, domain=self._domain),
                lambda x: makeOp(x.reciprocal().sqrt()))

    def apply(self, x):
        self._check_input(x)
        res = x.log().vdot(self._alphap1) + x.reciprocal().vdot(self._beta)
        if not x.want_metric:
            return res
        return res.add_metric(self.get_metric_at(x.val))

    def get_transformation(self):
        fact = self._alphap1.sqrt()
        res = makeOp(fact) @ Operator.identity_operator(self._domain).log()
        return self._sampling_dtype, res


class StudentTEnergy(LikelihoodEnergyOperator):
    """Computes likelihood energy corresponding to Student's t-distribution.

    .. math ::
        E_\\theta(f) = -\\log \\text{StudentT}_\\theta(f)
                     = \\frac{\\theta + 1}{2} \\log(1 + \\frac{f^2}{\\theta}),

    where f is a field defined on `domain`. Assumes that the data is `float64`
    for sampling.

    Parameters
    ----------
    domain : `Domain` or `DomainTuple`
        Domain of the operator
    theta : Scalar or :class:`nifty8.field.Field`
        Degree of freedom parameter for the student t distribution
    """

    def __init__(self, domain, theta):
        self._domain = DomainTuple.make(domain)
        self._theta = theta
        inp = Operator.identity_operator(self._domain)
        super(StudentTEnergy, self).__init__(inp, lambda x: self.get_metric_at(x).get_sqrt())

    def apply(self, x):
        self._check_input(x)
        res = (((self._theta+1)/2)*(x**2/self._theta).log1p()).sum()
        if not x.want_metric:
            return res
        return res.add_metric(self.get_metric_at(x.val))

    def get_transformation(self):
        if isinstance(self._theta, Field) or isinstance(self._theta, MultiField):
            th = self._theta
        else:
            from ..sugar import full
            th = full(self._domain, self._theta)
        return np.float64, makeOp(((th+1)/(th+3)).sqrt())


class BernoulliEnergy(LikelihoodEnergyOperator):
    """Computes likelihood energy of expected event frequency constrained by
    event data.

    .. math ::
        E(f) = -\\log \\text{Bernoulli}(d|f)
             = -d^\\dagger \\log f  - (1-d)^\\dagger \\log(1-f),

    where f is a field defined on `d.domain` with the expected
    frequencies of events.

    Parameters
    ----------
    d : :class:`nifty8.field.Field`
        Data field with events (1) or non-events (0).
    """

    def __init__(self, d):
        if not isinstance(d, Field) or not np.issubdtype(d.dtype, np.integer):
            raise TypeError
        if np.any(np.logical_and(d.val != 0, d.val != 1)):
            raise ValueError
        self._d = d
        self._domain = DomainTuple.make(d.domain)
        super(BernoulliEnergy, self).__init__(Adder(d, neg=True),
                                              lambda x: self.get_metric_at(x).get_sqrt())

    def apply(self, x):
        self._check_input(x)
        res = -x.log().vdot(self._d) + (1.-x).log().vdot(self._d-1.)
        if not x.want_metric:
            return res
        return res.add_metric(self.get_metric_at(x.val))

    def get_transformation(self):
        from ..sugar import full
        res = Adder(full(self._domain, 1.)) @ ScalingOperator(self._domain, -1)
        res = res * Operator.identity_operator(self._domain).reciprocal()
        return np.float64, -2.*res.sqrt().arctan()


class StandardHamiltonian(EnergyOperator):
    """Computes an information Hamiltonian in its standard form, i.e. with the
    prior being a real-valued Gaussian with unit covariance.

    Let the likelihood energy be :math:`E_{lh}`. Then this operator computes:

    .. math ::
         H(f) = 0.5 f^\\dagger f + E_{lh}(f):

    Other field priors can be represented via transformations of a white
    Gaussian field into a field with the desired prior probability structure.

    By implementing prior information this way, the field prior is represented
    by a generative model, from which NIFTy can draw samples and infer a field
    using the Maximum a Posteriori (MAP) or the Variational Bayes (VB) method.

    The metric of this operator can be used as covariance for drawing Gaussian
    samples.

    Parameters
    ----------
    lh : EnergyOperator
        The likelihood energy.
    ic_samp : IterationController
        Tells an internal :class:`SamplingEnabler` which convergence criterion
        to use to draw Gaussian samples.

    See also
    --------
    `Encoding prior knowledge in the structure of the likelihood`,
    Jakob Knollmüller, Torsten A. Ensslin,
    `<https://arxiv.org/abs/1812.04403>`_
    """

    def __init__(self, lh, ic_samp=None):
        self._lh = lh
        self._prior = GaussianEnergy(data=None, domain=lh.domain, sampling_dtype=float)
        self._ic_samp = ic_samp
        self._domain = lh.domain

    def apply(self, x):
        self._check_input(x)
        lhx, prx = self._lh(x), self._prior(x)
        if not x.want_metric or self._ic_samp is None:
            return lhx + prx
        met = SamplingEnabler(lhx.metric, prx.metric, self._ic_samp)
        return (lhx+prx).add_metric(met)

    @property
    def prior_energy(self):
        return self._prior

    @property
    def likelihood_energy(self):
        return self._lh

    @property
    def iteration_controller(self):
        return self._ic_samp

    def __repr__(self):
        subs = 'Likelihood energy:\n{}'.format(utilities.indent(self._lh.__repr__()))
        subs += '\nPrior energy:\n{}'.format(self._prior)
        return 'StandardHamiltonian:\n' + utilities.indent(subs)

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        out, lh1 = self._lh.simplify_for_constant_input(c_inp)
        return out, StandardHamiltonian(lh1, self._ic_samp)


class AveragedEnergy(EnergyOperator):
    """Averages an energy over samples.

    Parameters
    ----------
    h: Hamiltonian
       The energy to be averaged.
    res_samples : iterable of :class:`nifty8.field.Field`
       Set of residual sample points to be added to mean field for
       approximate estimation of the KL.

    Notes
    -----
    - Having symmetrized residual samples, with both :math:`v_i` and
      :math:`-v_i` being present, ensures that the distribution mean is
      exactly represented.

    - :class:`AveragedEnergy(h)` approximates
      :math:`\\left< H(f) \\right>_{G(f-m,D)}` if the residuals :math:`f-m`
      are drawn from a Gaussian distribution with covariance :math:`D`.
    """

    def __init__(self, h, res_samples):
        self._h = h
        self._domain = h.domain
        self._res_samples = tuple(res_samples)

    def apply(self, x):
        self._check_input(x)
        mymap = map(lambda v: self._h(x+v), self._res_samples)
        return utilities.my_sum(mymap)/len(self._res_samples)

    def get_transformation(self):
        dtp, trafo = self._h.get_transformation()
        mymap = map(lambda v: trafo@Adder(v), self._res_samples)
        return dtp, utilities.my_sum(mymap)/np.sqrt(len(self._res_samples))
