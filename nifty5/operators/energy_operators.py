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
from ..domain_tuple import DomainTuple
from ..field import Field
from ..linearization import Linearization
from ..sugar import makeOp, makeDomain
from .operator import Operator
from .sampling_enabler import SamplingEnabler
from .sandwich_operator import SandwichOperator
from .simple_linear_operators import VdotOperator


class EnergyOperator(Operator):
    """Abstract class from which
    other specific EnergyOperator subclasses are derived.

    An EnergyOperator has a scalar domain as target domain.

    It is intended as an objective function for field inference.

    Typical usage in IFT:

     - as an information Hamiltonian (i.e. a negative log probability)
     - or as a Gibbs free energy (i.e. an averaged Hamiltonian),
       aka Kullbach-Leibler divergence.
    """
    _target = DomainTuple.scalar_domain()


class SquaredNormOperator(EnergyOperator):
    """ Class for squared field norm energy.

    Usage
    -----
    ``E = SquaredNormOperator()`` represents a field energy E that is the
    L2 norm of a field f:

    :math:`E(f) = f^\\dagger f`
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
    """Class for quadratic field energies.

    Parameters
    ----------
    op : EndomorphicOperator
         kernel of quadratic form

    Notes
    -----
    ``E = QuadraticFormOperator(op)`` represents a field energy that is a
    quadratic form in a field f with kernel op:

    :math:`E(f) = 0.5 f^\\dagger op f`
    """
    def __init__(self, op):
        from .endomorphic_operator import EndomorphicOperator
        if not isinstance(op, EndomorphicOperator):
            raise TypeError("op must be an EndomorphicOperator")
        self._op = op
        self._domain = op.domain

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

    Attributes
    ----------
    mean : Field
        mean of the Gaussian, (default 0)
    covariance : LinearOperator
        covariance of the Gaussian (default = identity operator)
    domain : Domainoid
        operator domain, inferred from mean or covariance if specified

    Notes
    -----
    - At least one of the arguments has to be provided.
    - ``E = GaussianEnergy(mean=m, covariance=D)`` represents (up to constants)

        :math:`E(f) = - \\log G(f-m, D) = 0.5 (f-m)^\\dagger D^{-1} (f-m)`,

        an information energy for a Gaussian distribution with mean m and
        covariance D.
    """

    def __init__(self, mean=None, covariance=None, domain=None):
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
        residual = x if self._mean is None else x-self._mean
        res = self._op(residual).real
        if not isinstance(x, Linearization) or not x.want_metric:
            return res
        metric = SandwichOperator.make(x.jac, self._icov)
        return res.add_metric(metric)


class PoissonianEnergy(EnergyOperator):
    """Class for likelihood-energies of expected count field constrained by
    Poissonian count data.

    Parameters
    ----------
    d : Field
        data field with counts

    Notes
    -----
    ``E = PoissonianEnergy(d)`` represents (up to an f-independent term
    log(d!))

    :math:`E(f) = -\\log \\text{Poisson}(d|f) = \\sum f - d^\\dagger \\log(f)`,

    where f is a Field in data space with the expectation values for
    the counts.
    """
    def __init__(self, d):
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
    """Special class for inverse Gamma distributed covariances.

    RL FIXME: To be documented.
    """
    def __init__(self, d):
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
    """Class for likelihood-energies of expected event frequency constrained by
    event data.

    Parameters
    ----------
    d : Field
        data field with events (=1) or non-events (=0)

    Notes
    -----
    ``E = BernoulliEnergy(d)`` represents

    :math:`E(f) = -\\log \\text{Bernoulli}(d|f) =
    -d^\\dagger \\log f  - (1-d)^\\dagger \\log(1-f)`,

    where f is a field in data space (d.domain) with the expected
    frequencies of events.
    """
    def __init__(self, d):
        self._d = d
        self._domain = DomainTuple.make(d.domain)

    def apply(self, x):
        self._check_input(x)
        v = x.log().vdot(-self._d) - (1.-x).log().vdot(1.-self._d)
        if not isinstance(x, Linearization):
            return Field.scalar(v)
        if not x.want_metric:
            return v
        met = makeOp(1./(x.val*(1.-x.val)))
        met = SandwichOperator.make(x.jac, met)
        return v.add_metric(met)


class Hamiltonian(EnergyOperator):
    """Class for information Hamiltonians.

    Parameters
    ----------
    lh : EnergyOperator
        a likelihood energy
    ic_samp : IterationController
        is passed to SamplingEnabler to draw Gaussian distributed samples
        with covariance = metric of Hamiltonian
        (= Hessian without terms that generate negative eigenvalues)

    Notes
    -----
    ``H = Hamiltonian(E_lh)`` represents

    :math:`H(f) = 0.5 f^\\dagger f + E_{lh}(f)`

    an information Hamiltonian for a field f with a white Gaussian prior
    (unit covariance) and the likelihood energy :math:`E_{lh}`.

    Other field priors can be represented via transformations of a white
    Gaussian field into a field with the desired prior probability structure.

    By implementing prior information this way, the field prior is represented
    by a generative model, from which NIFTy can draw samples and infer a field
    using the Maximum a Posteriori (MAP) or the Variational Bayes (VB) method.

    For more details see:
    "Encoding prior knowledge in the structure of the likelihood"
    Jakob Knollmüller, Torsten A. Ensslin, submitted, arXiv:1812.04403
    `<https://arxiv.org/abs/1812.04403>`_
    """
    def __init__(self, lh, ic_samp=None):
        self._lh = lh
        self._prior = GaussianEnergy(domain=lh.domain)
        self._ic_samp = ic_samp
        self._domain = lh.domain

    def apply(self, x):
        self._check_input(x)
        if (self._ic_samp is None or not isinstance(x, Linearization) or
                not x.want_metric):
            return self._lh(x)+self._prior(x)
        else:
            lhx, prx = self._lh(x), self._prior(x)
            mtr = SamplingEnabler(lhx.metric, prx.metric.inverse,
                                  self._ic_samp, prx.metric.inverse)
            return (lhx+prx).add_metric(mtr)

    def __repr__(self):
        subs = 'Likelihood:\n{}'.format(utilities.indent(self._lh.__repr__()))
        subs += '\nPrior: Quadratic{}'.format(self._lh.domain.keys())
        return 'Hamiltonian:\n' + utilities.indent(subs)


class SampledKullbachLeiblerDivergence(EnergyOperator):
    """Class for Kullbach Leibler (KL) Divergence or Gibbs free energies

    Precisely a sample averaged Hamiltonian (or other energy) that represents
    approximatively the relevant part of a KL to be used in Variational Bayes
    inference if the samples are drawn from the approximating Gaussian.

    Let :math:`Q(f) = G(f-m,D)` Gaussian used to approximate
    :math:`P(f|d)`, the correct posterior with information Hamiltonian
    :math:`H(d,f) = -\\log P(d,f) = -\\log P(f|d) + \\text{const.}`

    The KL divergence between those should then be optimized for m. It is

    :math:`KL(Q,P) = \\int Df Q(f) \\log Q(f)/P(f)\\\\
    = \\left< \\log Q(f) \\right>_Q(f) - \\left< \\log P(f) \\right>_Q(f)\\\\
    = \\text{const} + \\left< H(f) \\right>_G(f-m,D)`

    in essence the information Hamiltonian averaged over a Gaussian
    distribution centered on the mean m.

    SampledKullbachLeiblerDivergence(H) approximates
    :math:`\\left< H(f) \\right>_{G(f-m,D)}` if the residuals
    :math:`f-m` are drawn from covariance :math:`D`.

    Parameters
    ----------
    h: Hamiltonian
       the Hamiltonian/energy to be averaged
    res_samples : iterable of Fields
       set of residual sample points to be added to mean field
       for approximate estimation of the KL

    Notes
    -----
    ``KL = SampledKullbachLeiblerDivergence(H, samples)`` represents

    :math:`\\text{KL}(m) = \\sum_i H(m+v_i) / N`,

    where :math:`v_i` are the residual samples, :math:`N` is their number,
    and :math:`m` is the mean field around which the samples are drawn.

    Having symmetrized residual samples, with both v_i and -v_i being present
    ensures that the distribution mean is exactly represented. This reduces
    sampling noise and helps the numerics of the KL minimization process in the
    variational Bayes inference.
    """
    def __init__(self, h, res_samples):
        self._h = h
        self._domain = h.domain
        self._res_samples = tuple(res_samples)

    def apply(self, x):
        self._check_input(x)
        mymap = map(lambda v: self._h(x+v), self._res_samples)
        return utilities.my_sum(mymap) * (1./len(self._res_samples))
