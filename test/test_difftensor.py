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


def test_simple():
    a = ift.Field.full(ift.RGSpace(10),3.)
    c = ift.Field.full(ift.RGSpace(10),2.)
    b = ift.Field.full(ift.RGSpace(11),3.)
    t3 = ift.DiffTensor.makeDiagonal(a, 3)
    t2 = t3.contract((a,))
    ts = t2 + t2
    print(t2)
    print(t2.linop(a).val)
    t1 = t2.contract((a,))
    print(t1._arg, t1.rank)
    print(t1.vec)
    x = ift.diff_tensor.Taylor.make_var(a,3)
    bla = ift.diff_tensor.GenLeibnizTensor(x,x,1)
    print(type(bla.getVec((c,))))
    #print(type(bla.getLinop((a,a))))
    foo = ift.diff_tensor.ComposedTensor(x,x,2)
    print(type(foo.getVec((a,a))))
    print(type(foo.getLinop((a,))))

    print(bla.getVec((c,)).val)

def test_leibnitz_simple():
    dom = ift.RGSpace(10)
    a = ift.from_random(dom)
    b = ift.from_random(dom)
    c = ift.from_random(dom)

    x = ift.diff_tensor.Taylor.make_var(a, 4)
    assert x[0].vec is a
    y = ift.diff_tensor.Taylor.make_var(b, 4)
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
    x = ift.diff_tensor.Taylor.make_var(a, 4)
    op = ift.ScalingOperator(dom, 4.)
    y = op(x)
    assert_allclose(y.val.val, op(a).val)

    ht = ift.ScalingOperator(dom,3.)#ift.HarmonicTransformOperator(dom.get_default_codomain())
    op = ht.ducktape("hi")
    op = op.exp()
    a = ift.from_random(op.domain)
    x = ift.diff_tensor.Taylor.make_var(a, 5)
    y = op(x)

    bs = tuple(ift.from_random(op.domain) for _ in range(6))
    for i in range(6):
        print(i)
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
    #r = (op+op2).reciprocal()
    r = op2.exp().reciprocal()
    a = ift.from_random(r.domain)
    x = ift.diff_tensor.Taylor.make_var(a, 5)
    z = r(x)

    bs = tuple(ift.from_random(r.domain) for _ in range(6))
    for i in range(6):
        print(i)
        res = z[i].getVec(bs[:i])
        rr = z[i].getVec(bs[:i][::-1])
        assert_allclose(res.val,rr.val)
        if i>0:
            res2 = z[i].getLinop(bs[:i-1])(bs[i-1])
            #assert_allclose(res.val,res2.val)
        tm = 1.
        for bb in bs[:i]:
            tm = tm*bb
        re = (1.-2.*(i%2))*(-a).exp()*tm
        print(res.val)
        print(re['ho'].val)
        #assert_allclose(re['ho'].val,res.val)

def test_comp():
    dom = ift.RGSpace(10)
    ht = ift.ScalingOperator(dom,3.)
    op = ht.ducktape("hi")
    op = op.exp()
    
    a = ift.from_random(op.domain)
    x = ift.diff_tensor.Taylor.make_var(a,5)
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

test_comp()
test_leibnitz_simple()
test_tensors()

