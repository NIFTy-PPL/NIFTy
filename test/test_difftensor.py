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

def test_composed_simple():
    dom = ift.RGSpace(10)
    a = ift.from_random(dom)
    b = ift.from_random(dom)
    c = ift.from_random(dom)

    x = ift.diff_tensor.Taylor.make_var(a, 4)
    y = ift.diff_tensor.Taylor.make_var(b, 4)
    z = x.new_from_prod(y)
    p = ift.diff_tensor.Taylor.make_var(ift.full(z.domain,5.),4)

    

    ops = x.new(y)
    print(ops[4].getVec((c,c,c,c)).val)

    assert ops[0].vec is b
    print(ops[1].linop)
    print(ops[1].getVec((c,)).val)

    
    
test_leibnitz_simple()
test_composed_simple()