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
# Copyright(C) 2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

#import numpy as np
#import pytest
from numpy.testing import assert_allclose#, assert_equal, assert_raises

import nifty6 as ift
#from .common import setup_function, teardown_function


def test_leibnitz_simple():
    dom = ift.RGSpace(10)
    a = ift.from_random(dom)
    b = ift.from_random(dom)
    c = ift.from_random(dom)

    x = ift.Taylor.make_var(a, 4)
    assert x[0].vec is a
    y = ift.Taylor.make_var(b, 4)
    assert y[0].vec is b
    ops = x.new_from_prod(y)
    assert_allclose(ops[0].vec.val, (a*b).val)

    assert_allclose(ops[2].getVec((c,c)).val,(2.*c*c).val)
    assert_allclose(ops[1].getVec((c,)).val,(b*c+a*c).val)

    assert_allclose(ops[1].getLinop()._ldiag,(a+b).val)

    assert isinstance(ops[3].getLinop((c,c)), ift.NullOperator)


def test_tensors():
    dom = ift.RGSpace(5)
    a = ift.from_random(dom)
    x = ift.Taylor.make_var(a, 4)
    op = ift.ScalingOperator(dom, 4.)
    y = op(x)
    assert_allclose(y.val.val, op(a).val)

    ht = ift.HarmonicTransformOperator(dom.get_default_codomain())
    op = ht.ducktape("hi")
    op = op.exp()
    a = ift.from_random(op.domain)
    x = ift.Taylor.make_var(a, 5)
    y = op(x)

    bs = tuple(ift.from_random(op.domain) for _ in range(6))
    for i in range(6):
        res1 = y[i].getVec(bs[:i])
        rr = y[i].getVec(bs[:i][::-1])
        assert_allclose(res1.val,rr.val)
        if i > 0:
            rr = y[i].getLinop(bs[:i-1])
            rr = rr(bs[i-1])
            assert_allclose(rr.val,res1.val)
        tm = 1.
        for j in range(i):
            tm = tm*ht(bs[j]['hi'])
        res2 = ht(a['hi']).exp() * tm
        assert_allclose(res1.val,res2.val)

    op2 = ift.ScalingOperator(dom, 1.).ducktape('ho')
    r = (op+op2).reciprocal()
    r2 = op2.exp().reciprocal()
    a = ift.from_random(r.domain)
    z = r(ift.Taylor.make_var(a, 4))
    a2 = ift.from_random(r2.domain)
    z2 = r2(ift.Taylor.make_var(a2, 4))

    bs = tuple(ift.from_random(r.domain) for _ in range(5))
    bs2 = tuple(ift.from_random(r2.domain) for _ in range(5))
    for i in range(5):
        res = z[i].getVec(bs[:i])
        rr = z[i].getVec(bs[:i][::-1])
        assert_allclose(res.val,rr.val)
        if i>0:
            res2 = z[i].getLinop(bs[:i-1])(bs[i-1])
            assert_allclose(res.val,res2.val)
        tm = 1.
        for bb in bs2[:i]:
            tm = tm*bb
        re = (1.-2.*(i%2))*(-a2).exp()*tm
        assert_allclose(re['ho'].val,z2[i].getVec(bs2[:i]).val)


def test_comp():
    dom = ift.RGSpace(10)
    ht = ift.ScalingOperator(dom,3.)
    op = ht.ducktape("hi")
    op = op.exp()
    
    a = ift.from_random(op.domain)
    x = ift.Taylor.make_var(a,5)
    y = op(x)

    b1 = ift.from_random(op.domain)
    b2 = ift.from_random(op.domain)
    bs = (b1,b2)
    
    r1 = y[2].getVec(bs[:2])
    r2 = y[2].getLinop(bs[:1])(bs[1])
    assert_allclose(r1.val,r2.val)

    r1 = y[1].getVec((b1,))
    r2 = y[1].getLinop()(b1)
    assert_allclose(r1.val,r2.val)


def test_cf():
    sp = ift.RGSpace(128)
    cf = ift.CorrelatedFieldMaker.make(0.,1.,1.,'pre')
    fluctuations_dict = {
        # Amplitude of field fluctuations
        'fluctuations_mean':   2.0,  # 1.0
        'fluctuations_stddev': 1.0,  # 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope_mean': -2.0,  # -3.0
        'loglogavgslope_stddev': 0.5,  #  0.5

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility_mean':   2.5,  # 1.0
        'flexibility_stddev': 1.0,  # 0.5

        # How ragged the integrated Wiener process component is
        'asperity_mean':   0.5,  # 0.1
        'asperity_stddev': 0.5  # 0.5
    }
    cf.add_fluctuations(sp, **fluctuations_dict)
    correlated_field = cf.finalize(prior_info=0)

    x = ift.from_random(correlated_field.domain)
    t = ift.Taylor.make_var(x, 2)
    y = correlated_field(t)

    assert_allclose(y.val.val,correlated_field(x).val)
    bs = (ift.from_random(y.domain), ift.from_random(y.domain))
    dd = y[2].getLinop(bs[:1])
    assert_allclose(y[2].getVec(bs).val, dd(bs[1]).val)

    rl = correlated_field(ift.Linearization.make_var(x))
    assert_allclose(y.jac(bs[0]).val,rl.jac(bs[0]).val)
    c = ift.from_random(y.jac.target)
    r1 = y.jac.adjoint(c)
    r2 = rl.jac.adjoint(c)
    for k in r1.domain.keys():
        assert_allclose(r1[k].val,r2[k].val)

    
test_comp()
test_leibnitz_simple()
test_tensors()
test_cf()