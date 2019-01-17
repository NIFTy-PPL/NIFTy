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

import numpy as np

from .. import utilities
from ..domain_tuple import DomainTuple
from ..field import Field
from ..linearization import Linearization
from ..sugar import makeDomain, makeOp
from .linear_operator import LinearOperator
from .operator import Operator
from .sampling_enabler import SamplingEnabler
from .sandwich_operator import SandwichOperator
from .simple_linear_operators import VdotOperator


class EnergyOperator(Operator):
    """Operator which has a scalar domain as target domain.

    It is intended as an objective function for field inference.

    Examples
    --------
     - Information Hamiltonian, i.e. negative-log-probabilities.
     - Gibbs free energy, i.e. an averaged Hamiltonian, aka Kullbach-Leibler
       divergence.
    """
    _target = DomainTuple.scalar_domain()


class SquaredNormOperator(EnergyOperator):
    """Computes the L2-norm of the output of an operator.

    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        Domain of the operator in which the L2-norm shall be computed.
    """

    def __init__(self, domain):
        self._domain = domain

    def apply(self, x):
        self._check_input(x)
        if isinstance(x, Linearization):
            val = Field.scalar(x.val.vdot(x.val))
            jac = VdotOperator(2*x.val)(x.jac)
            return x.new(val, jac)
        return Field.scalar(x.vdot(x))


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
        if isinstance(x, Linearization):
            t1 = self._op(x.val)
            jac = VdotOperator(t1)(x.jac)
            val = Field.scalar(0.5*x.val.vdot(t1))
            return x.new(val, jac)
        return Field.scalar(0.5*x.vdot(self._op(x)))


class GaussianEnergy(EnergyOperator):
    """Class for energies of fields with Gaussian probability distribution.

    Represents up to constants in :math:`m`:

    .. math ::
        E(f) = - \\log G(f-m, D) = 0.5 (f-m)^\\dagger D^{-1} (f-m),

    an information energy for a Gaussian distribution with mean m and
    covariance D.

    Parameters
    ----------
    mean : Field
        Mean of the Gaussian. Default is 0.
    covariance : LinearOperator
        Covariance of the Gaussian. Default is the identity operator.
    domain : Domain, DomainTuple, tuple of Domain or MultiDomain
        Operator domain. By default it is inferred from `mean` or
        `covariance` if specified

    Note
    ----
    At least one of the arguments has to be provided.
    """

    def __init__(self, mean=None, covariance=None, domain=None):
        if mean is not None and not isinstance(mean, Field):
            raise TypeError
        if covariance is not None and not isinstance(covariance,
                                                     LinearOperator):
            raise TypeError

        self._domain = None
        if mean is not None:
            self._checkEquivalence(mean.domain)
        if covariance is not None:
            self._checkEquivalence(covariance.domain)
        if domain is not None:
            self._checkEquivalence(domain)
        if self._domain is None:
            raise ValueError("no domain given")
        self._mean = mean
        if covariance is None:
            self._op = SquaredNormOperator(self._domain).scale(0.5)
        else:
            self._op = QuadraticFormOperator(covariance.inverse)
        self._icov = None if covariance is None else covariance.inverse

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
        if not isinstance(x, Linearization) or not x.want_metric:
            return res
        metric = SandwichOperator.make(x.jac, self._icov)
        return res.add_metric(metric)


class PoissonianEnergy(EnergyOperator):
    """Class for likelihood Hamiltonians of expected count field constrained
    by Poissonian count data.

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
        if np.any(d.local_data < 0):
            raise ValueError
        self._d = d
        self._domain = DomainTuple.make(d.domain)

    def apply(self, x):
        self._check_input(x)
        res = x.sum() - x.log().vdot(self._d)
        if not isinstance(x, Linearization):
            return Field.scalar(res)
        if not x.want_metric:
            return res
        metric = SandwichOperator.make(x.jac, makeOp(1./x.val))
        return res.add_metric(metric)


class InverseGammaLikelihood(EnergyOperator):
    """
    FIXME
    """

    def __init__(self, d):
        if not isinstance(d, Field):
            raise TypeError
        self._d = d
        self._domain = DomainTuple.make(d.domain)

    def apply(self, x):
        self._check_input(x)
        res = 0.5*(x.log().sum() + (1./x).vdot(self._d))
        if not isinstance(x, Linearization):
            return Field.scalar(res)
        if not x.want_metric:
            return res
        metric = SandwichOperator.make(x.jac, makeOp(0.5/(x.val**2)))
        return res.add_metric(metric)


class BernoulliEnergy(EnergyOperator):
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
        print(d.dtype)
        if not isinstance(d, Field) or not np.issubdtype(d.dtype, np.integer):
            raise TypeError
        if not np.all(np.logical_or(d.local_data == 0, d.local_data == 1)):
            raise ValueError
        self._d = d
        self._domain = DomainTuple.make(d.domain)

    def apply(self, x):
        self._check_input(x)
        v = -(x.log().vdot(self._d) + (1. - x).log().vdot(1. - self._d))
        if not isinstance(x, Linearization):
            return Field.scalar(v)
        if not x.want_metric:
            return v
        met = makeOp(1./(x.val*(1. - x.val)))
        met = SandwichOperator.make(x.jac, met)
        return v.add_metric(met)


class Hamiltonian(EnergyOperator):
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


    See also
    --------
    `Encoding prior knowledge in the structure of the likelihood`,
    Jakob Knollmüller, Torsten A. Ensslin,
    `<https://arxiv.org/abs/1812.04403>`_
    """

    def __init__(self, lh, ic_samp=None):
        self._lh = lh
        self._prior = GaussianEnergy(domain=lh.domain)
        self._ic_samp = ic_samp
        self._domain = lh.domain

    def apply(self, x):
        self._check_input(x)
        if (self._ic_samp is None or not isinstance(x, Linearization)
                or not x.want_metric):
            return self._lh(x) + self._prior(x)
        else:
            lhx, prx = self._lh(x), self._prior(x)
            mtr = SamplingEnabler(lhx.metric, prx.metric.inverse,
                                  self._ic_samp, prx.metric.inverse)
            return (lhx + prx).add_metric(mtr)

    def __repr__(self):
        subs = 'Likelihood:\n{}'.format(utilities.indent(self._lh.__repr__()))
        subs += '\nPrior: Quadratic{}'.format(self._lh.domain.keys())
        return 'Hamiltonian:\n' + utilities.indent(subs)


class AveragedEnergy(EnergyOperator):
    """Computes Kullback-Leibler (KL) divergence or Gibbs free energies.

    A sample-averaged energy, e.g. an Hamiltonian, approximates the relevant
    part of a KL to be used in Variational Bayes inference if the samples are
    drawn from the approximating Gaussian:

    .. math ::
        \\text{KL}(m) = \\frac1{\\#\\{v_i\\}} \\sum_{v_i} H(m+v_i),

    where :math:`v_i` are the residual samples and :math:`m` is the mean field
    around which the samples are drawn.

    Parameters
    ----------
    h: Hamiltonian
       The energy to be averaged.
    res_samples : iterable of Fields
       Set of residual sample points to be added to mean field for approximate
       estimation of the KL.

    Note
    ----
    Having symmetrized residual samples, with both v_i and -v_i being present
    ensures that the distribution mean is exactly represented. This reduces
    sampling noise and helps the numerics of the KL minimization process in the
    variational Bayes inference.

    See also
    --------
    Let :math:`Q(f) = G(f-m,D)` be the Gaussian distribution
    which is used to approximate the accurate posterior :math:`P(f|d)` with
    information Hamiltonian
    :math:`H(d,f) = -\\log P(d,f) = -\\log P(f|d) + \\text{const}`. In
    Variational Bayes one needs to optimize the KL divergence between those
    two distributions for m. It is:

    :math:`KL(Q,P) = \\int Df Q(f) \\log Q(f)/P(f)\\\\
    = \\left< \\log Q(f) \\right>_Q(f) - \\left< \\log P(f) \\right>_Q(f)\\\\
    = \\text{const} + \\left< H(f) \\right>_G(f-m,D)`

    in essence the information Hamiltonian averaged over a Gaussian
    distribution centered on the mean m.

    :class:`AveragedEnergy(h)` approximates
    :math:`\\left< H(f) \\right>_{G(f-m,D)}` if the residuals
    :math:`f-m` are drawn from a Gaussian distribution with covariance
    :math:`D`.
    """

    def __init__(self, h, res_samples):
        self._h = h
        self._domain = h.domain
        self._res_samples = tuple(res_samples)

    def apply(self, x):
        self._check_input(x)
        mymap = map(lambda v: self._h(x + v), self._res_samples)
        return utilities.my_sum(mymap)*(1./len(self._res_samples))
