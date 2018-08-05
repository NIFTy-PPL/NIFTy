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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import absolute_import, division, print_function

from ..compat import *
from ..domain_tuple import DomainTuple
from .operator import Operator
from .sandwich_operator import SandwichOperator
from .sampling_enabler import SamplingEnabler
from ..sugar import makeOp
from ..linearization import Linearization
from .. import utilities


class EnergyOperator(Operator):
    _target = DomainTuple.scalar_domain()

    @property
    def target(self):
        return EnergyOperator._target


class SquaredNormOperator(EnergyOperator):
    def __init__(self, domain):
        super(SquaredNormOperator, self).__init__()
        self._domain = domain

    @property
    def domain(self):
        return self._domain

    def apply(self, x):
        return Field(self._target, x.vdot(x))


class QuadraticFormOperator(EnergyOperator):
    def __init__(self, op):
        from .endomorphic_operator import EndomorphicOperator
        super(QuadraticFormOperator, self).__init__()
        if not isinstance(op, EndomorphicOperator):
            raise TypeError("op must be an EndomorphicOperator")
        self._op = op

    @property
    def domain(self):
        return self._op.domain

    def apply(self, x):
        if isinstance(x, Linearization):
            jac = self._op(x)
            val = Field(self._target, 0.5 * x.vdot(jac))
            return Linearization(val, jac)
        return Field(self._target, 0.5 * x.vdot(self._op(x)))


class GaussianEnergy(EnergyOperator):
    def __init__(self, mean=None, covariance=None, domain=None):
        super(GaussianEnergy, self).__init__()
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
        self._icov = None if covariance is None else covariance.inverse

    def _checkEquivalence(self, newdom):
        if self._domain is None:
            self._domain = newdom
        else:
            if self._domain is not newdom:
                raise ValueError("domain mismatch")

    @property
    def domain(self):
        return self._domain

    def apply(self, x):
        residual = x if self._mean is None else x-self._mean
        icovres = residual if self._icov is None else self._icov(residual)
        res = .5*residual.vdot(icovres)
        if not isinstance(x, Linearization):
            return res
        metric = SandwichOperator.make(x.jac, self._icov)
        return res.add_metric(metric)


class PoissonianEnergy(EnergyOperator):
    def __init__(self, op, d):
        self._op, self._d = op, d

    @property
    def domain(self):
        return self._op.domain

    def apply(self, x):
        x = self._op(x)
        res = x.sum() - x.log().vdot(self._d)
        if not isinstance(x, Linearization):
            return res
        metric = SandwichOperator.make(x.jac, makeOp(1./x.val))
        return res.add_metric(metric)


class BernoulliEnergy(EnergyOperator):
    def __init__(self, p, d):
        self._p = p
        self._d = d

    @property
    def domain(self):
        return self._p.domain

    def apply(self, x):
        x = self._p(x)
        v = x.log().vdot(-self._d) - (1.-x).log().vdot(1.-self._d)
        if not isinstance(x, Linearization):
            return v
        met = makeOp(1./(x.val*(1.-x.val)))
        met = SandwichOperator.make(x.jac, met)
        return v.add_metric(met)


class Hamiltonian(EnergyOperator):
    def __init__(self, lh, ic_samp=None):
        super(Hamiltonian, self).__init__()
        self._lh = lh
        self._prior = GaussianEnergy(domain=lh.domain)
        self._ic_samp = ic_samp

    @property
    def domain(self):
        return self._lh.domain

    def apply(self, x):
        if self._ic_samp is None or not isinstance(x, Linearization):
            return self._lh(x) + self._prior(x)
        else:
            lhx = self._lh(x)
            prx = self._prior(x)
            mtr = SamplingEnabler(lhx.metric, prx.metric.inverse,
                                  self._ic_samp, prx.metric.inverse)
            return (lhx+prx).add_metric(mtr)


class SampledKullbachLeiblerDivergence(EnergyOperator):
    def __init__(self, h, res_samples):
        """
        # MR FIXME: does h have to be a Hamiltonian? Couldn't it be any energy?
        h: Hamiltonian
        N: Number of samples to be used
        """
        super(SampledKullbachLeiblerDivergence, self).__init__()
        self._h = h
        self._res_samples = tuple(res_samples)

    @property
    def domain(self):
        return self._h.domain

    def apply(self, x):
        return (utilities.my_sum(map(lambda v: self._h(x+v), self._res_samples)) *
                (1./len(self._res_samples)))
