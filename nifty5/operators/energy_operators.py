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

from .. import utilities
from ..compat import *
from ..domain_tuple import DomainTuple
from ..field import Field
from ..linearization import Linearization
from ..sugar import makeOp
from .operator import Operator
from .sampling_enabler import SamplingEnabler
from .sandwich_operator import SandwichOperator
from .simple_linear_operators import VdotOperator


class EnergyOperator(Operator):
    _target = DomainTuple.scalar_domain()


class SquaredNormOperator(EnergyOperator):
    def __init__(self, domain):
        self._domain = domain

    def apply(self, x):
        if isinstance(x, Linearization):
            val = Field.scalar(x.val.vdot(x.val))
            jac = VdotOperator(2*x.val)(x.jac)
            return Linearization(val, jac)
        return Field.scalar(x.vdot(x))


class QuadraticFormOperator(EnergyOperator):
    def __init__(self, op):
        from .endomorphic_operator import EndomorphicOperator
        if not isinstance(op, EndomorphicOperator):
            raise TypeError("op must be an EndomorphicOperator")
        self._op = op
        self._domain = op.domain

    def apply(self, x):
        if isinstance(x, Linearization):
            t1 = self._op(x.val)
            jac = VdotOperator(t1)(x.jac)
            val = Field.scalar(0.5*x.val.vdot(t1))
            return Linearization(val, jac)
        return Field.scalar(0.5*x.vdot(self._op(x)))


class GaussianEnergy(EnergyOperator):
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
        if self._domain is None:
            self._domain = newdom
        else:
            if self._domain != newdom:
                raise ValueError("domain mismatch")

    def apply(self, x):
        residual = x if self._mean is None else x-self._mean
        res = self._op(residual).real
        if not isinstance(x, Linearization):
            return res
        metric = SandwichOperator.make(x.jac, self._icov)
        return res.add_metric(metric)


class PoissonianEnergy(EnergyOperator):
    def __init__(self, op, d):
        self._op, self._d = op, d
        self._domain = d.domain

    def apply(self, x):
        x = self._op(x)
        res = x.sum() - x.log().vdot(self._d)
        if not isinstance(x, Linearization):
            return Field.scalar(res)
        metric = SandwichOperator.make(x.jac, makeOp(1./x.val))
        return res.add_metric(metric)

class InverseGammaLikelihood(EnergyOperator):
    def __init__(self, op, d):
        self._op, self._d = op, d
        self._domain = d.domain

    def apply(self, x):
        x = self._op(x)
        res = 0.5*(x.log().sum() + (0.5/x).vdot(self._d))
        if not isinstance(x, Linearization):
            return Field.scalar(res)
        metric = SandwichOperator.make(x.jac, makeOp(0.5/(x.val**2)))
        return res.add_metric(metric)



class BernoulliEnergy(EnergyOperator):
    def __init__(self, p, d):
        self._p = p
        self._d = d
        self._domain = d.domain

    def apply(self, x):
        x = self._p(x)
        v = x.log().vdot(-self._d) - (1.-x).log().vdot(1.-self._d)
        if not isinstance(x, Linearization):
            return Field.scalar(v)
        met = makeOp(1./(x.val*(1.-x.val)))
        met = SandwichOperator.make(x.jac, met)
        return v.add_metric(met)


class Hamiltonian(EnergyOperator):
    def __init__(self, lh, ic_samp=None):
        self._lh = lh
        self._prior = GaussianEnergy(domain=lh.domain)
        self._ic_samp = ic_samp
        self._domain = lh.domain

    def apply(self, x):
        if self._ic_samp is None or not isinstance(x, Linearization):
            return self._lh(x)+self._prior(x)
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
        self._h = h
        self._domain = h.domain
        self._res_samples = tuple(res_samples)

    def apply(self, x):
        mymap = map(lambda v: self._h(x+v), self._res_samples)
        return utilities.my_sum(mymap) * (1./len(self._res_samples))
