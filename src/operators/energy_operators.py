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

from .. import utilities
from ..domain_tuple import DomainTuple
from ..field import Field
from ..linearization import Linearization
from ..multi_domain import MultiDomain
from ..multi_field import MultiField
from ..sugar import makeDomain, makeOp
from ..utilities import myassert
from .adder import Adder
from .linear_operator import LinearOperator
from .operator import Operator
from .sampling_enabler import SamplingDtypeSetter, SamplingEnabler
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


def _iscomplex(dtype):
    return np.issubdtype(dtype, np.complexfloating)


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

    def get_metric_at(self, x):
        """Compute the Fisher information metric for a `LikelihoodEnergyOperator`
        at `x` using the Jacobian of the coordinate transformation given by
        :func:`~nifty7.operators.operator.Operator.get_transformation`. """
        dtp, f = self.get_transformation()
        ch = None
        if dtp is not None:
            ch = SamplingDtypeSetter(ScalingOperator(f.target, 1.), dtp)
        bun = f(Linearization.make_var(x)).jac
        return SandwichOperator.make(bun, ch)


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
        self._cplx = _iscomplex(sampling_dtype)
        self._use_full_fisher = use_full_fisher

    def apply(self, x):
        self._check_input(x)
        r, i = x[self._kr], x[self._ki]
        if self._cplx:
            res = 0.5*r.vdot(r*i.real).real - i.log().sum()
        else:
            res = 0.5*(r.vdot(r*i) - i.log().sum())
        if not x.want_metric:
            return res
        if self._use_full_fisher:
            met = 1. if self._cplx else 0.5
            met = MultiField.from_dict({self._kr: i.val, self._ki: met*i.val**(-2)},
                                        domain=self._domain)
            met = SamplingDtypeSetter(makeOp(met), self._dt)
        else:
            met = self.get_metric_at(x.val)
        return res.add_metric(met)

    def _simplify_for_constant_input_nontrivial(self, c_inp):
        from .simplify_for_const import ConstantEnergyOperator
        myassert(len(c_inp.keys()) == 1)
        key = c_inp.keys()[0]
        myassert(key in self._domain.keys())
        cst = c_inp[key]
        if key == self._kr:
            res = _SpecialGammaEnergy(cst).ducktape(self._ki)
        else:
            dt = self._dt[self._kr]
            res = GaussianEnergy(inverse_covariance=makeOp(cst),
                                 sampling_dtype=dt).ducktape(self._kr)
            trlog = cst.log().sum().val_rw()
            if not self._cplx:
                trlog /= 2
            res = res + ConstantEnergyOperator(-trlog)
        res = res + ConstantEnergyOperator(0.)
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
        self._domain = DomainTuple.make(residual.domain)
        self._resi = residual
        self._cplx = _iscomplex(self._resi.dtype)
        self._dt = self._resi.dtype

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
        return self._dt, sc*ScalingOperator(self._domain, 1.).log()


class GaussianEnergy(LikelihoodEnergyOperator):
    """Computes a negative-log Gaussian.

    Represents up to constants in :math:`m`:

    .. math ::
        E(f) = - \\log G(f-m, D) = 0.5 (f-m)^\\dagger D^{-1} (f-m),

    an information energy for a Gaussian distribution with mean m and
    covariance D.

    Parameters
    ----------
    mean : Field
        Mean of the Gaussian. Default is 0.
    inverse_covariance : LinearOperator
        Inverse covariance of the Gaussian. Default is the identity operator.
    domain : Domain, DomainTuple, tuple of Domain or MultiDomain
        Operator domain. By default it is inferred from `mean` or
        `covariance` if specified
    sampling_dtype : type
        Here one can specify whether the distribution is a complex Gaussian or
        not. Note that for a complex Gaussian the inverse_covariance is
        .. math ::
        (<ff^dagger>)^{-1}_P(f)/2,
        where the additional factor of 2 is necessary because the
        domain of s has double as many dimensions as in the real case.

    Note
    ----
    At least one of the arguments has to be provided.
    """

    def __init__(self, mean=None, inverse_covariance=None, domain=None, sampling_dtype=None):
        if mean is not None and not isinstance(mean, (Field, MultiField)):
            raise TypeError
        if inverse_covariance is not None and not isinstance(inverse_covariance, LinearOperator):
            raise TypeError

        self._domain = None
        if mean is not None:
            self._checkEquivalence(mean.domain)
        if inverse_covariance is not None:
            self._checkEquivalence(inverse_covariance.domain)
        if domain is not None:
            self._checkEquivalence(domain)
        if self._domain is None:
            raise ValueError("no domain given")
        self._mean = mean

        # Infer sampling dtype
        if self._mean is None:
            _check_sampling_dtype(self._domain, sampling_dtype)
        else:
            if sampling_dtype is None:
                sampling_dtype = _field_to_dtype(self._mean)
            else:
                if sampling_dtype != _field_to_dtype(self._mean):
                    raise ValueError("Sampling dtype and mean not compatible")

        self._icov = inverse_covariance
        if inverse_covariance is None:
            self._op = Squared2NormOperator(self._domain).scale(0.5)
            self._met = ScalingOperator(self._domain, 1)
        else:
            self._op = QuadraticFormOperator(inverse_covariance)
            self._met = inverse_covariance
        if sampling_dtype is not None:
            self._met = SamplingDtypeSetter(self._met, sampling_dtype)

    def _checkEquivalence(self, newdom):
        newdom = makeDomain(newdom)
        if self._domain is None:
            self._domain = newdom
        else:
            if self._domain != newdom:
                raise ValueError("domain mismatch")

    def apply(self, x):
        self._check_input(x)
        residual = x if self._mean is None else x - self._mean
        res = self._op(residual).real
        if x.want_metric:
            return res.add_metric(self.get_metric_at(x.val))
        return res

    def get_transformation(self):
        icov, dtp = self._met, None
        if isinstance(icov, SamplingDtypeSetter):
            dtp = icov._dtype
            icov = icov._op
        return dtp, icov.get_sqrt()

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
    d : Field
        Data field with counts. Needs to have integer dtype and all field
        values need to be non-negative.
    """

    def __init__(self, d):
        if not isinstance(d, Field) or not np.issubdtype(d.dtype, np.integer):
            raise TypeError
        if np.any(d.val < 0):
            raise ValueError
        self._d = d
        self._domain = DomainTuple.make(d.domain)

    def apply(self, x):
        self._check_input(x)
        res = x.sum() - x.log().vdot(self._d)
        if not x.want_metric:
            return res
        return res.add_metric(self.get_metric_at(x.val))

    def get_transformation(self):
        return np.float64, 2.*ScalingOperator(self._domain, 1.).sqrt()


class InverseGammaEnergy(LikelihoodEnergyOperator):
    """Computes the negative log-likelihood of the inverse gamma distribution.

    It negative log-pdf(x) is given by

    .. math ::

        \\sum_i (\\alpha_i+1)*\\ln(x_i) + \\beta_i/x_i

    This is the likelihood for the variance :math:`x=S_k` given data
    :math:`\\beta = 0.5 |s_k|^2` where the Field :math:`s` is known to have
    the covariance :math:`S_k`.

    Parameters
    ----------
    beta : Field
        beta parameter of the inverse gamma distribution
    alpha : Scalar, Field, optional
        alpha parameter of the inverse gamma distribution
    """

    def __init__(self, beta, alpha=-0.5):
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

    def apply(self, x):
        self._check_input(x)
        res = x.log().vdot(self._alphap1) + x.reciprocal().vdot(self._beta)
        if not x.want_metric:
            return res
        return res.add_metric(self.get_metric_at(x.val))

    def get_transformation(self):
        fact = self._alphap1.sqrt()
        res = makeOp(fact) @ ScalingOperator(self._domain, 1.).log()
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
    theta : Scalar or Field
        Degree of freedom parameter for the student t distribution
    """

    def __init__(self, domain, theta):
        self._domain = DomainTuple.make(domain)
        self._theta = theta

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
            from ..extra import full
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
    d : Field
        Data field with events (1) or non-events (0).
    """

    def __init__(self, d):
        if not isinstance(d, Field) or not np.issubdtype(d.dtype, np.integer):
            raise TypeError
        if np.any(np.logical_and(d.val != 0, d.val != 1)):
            raise ValueError
        self._d = d
        self._domain = DomainTuple.make(d.domain)

    def apply(self, x):
        self._check_input(x)
        res = -x.log().vdot(self._d) + (1.-x).log().vdot(self._d-1.)
        if not x.want_metric:
            return res
        return res.add_metric(self.get_metric_at(x.val))

    def get_transformation(self):
        from ..extra import full
        res = Adder(full(self._domain, 1.)) @ ScalingOperator(self._domain, -1)
        res = res * ScalingOperator(self._domain, 1).reciprocal()
        return np.float64, -2.*res.sqrt().arctan()


class StandardHamiltonian(EnergyOperator):
    """Computes an information Hamiltonian in its standard form, i.e. with the
    prior being a Gaussian with unit covariance.

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
    prior_dtype : numpy.dtype or dict of numpy.dtype, optional
        Data type of prior used for sampling.

    See also
    --------
    `Encoding prior knowledge in the structure of the likelihood`,
    Jakob Knollmüller, Torsten A. Ensslin,
    `<https://arxiv.org/abs/1812.04403>`_
    """

    def __init__(self, lh, ic_samp=None, prior_dtype=np.float64):
        self._lh = lh
        self._prior = GaussianEnergy(domain=lh.domain, sampling_dtype=prior_dtype)
        self._ic_samp = ic_samp
        self._domain = lh.domain

    def apply(self, x):
        self._check_input(x)
        if not x.want_metric or self._ic_samp is None:
            return (self._lh + self._prior)(x)
        lhx, prx = self._lh(x), self._prior(x)
        met = SamplingEnabler(lhx.metric, prx.metric, self._ic_samp)
        return (lhx+prx).add_metric(met)

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
    res_samples : iterable of Fields
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
