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

from numpy.testing import assert_allclose, assert_raises

import nifty7 as ift
from nifty7.operators.simplify_for_const import ConstantOperator


def test_simplification():
    dom = {"a": ift.RGSpace(10)}
    f1 = ift.full(dom, 2.)
    op = ift.FFTOperator(f1.domain["a"]).ducktape("a")
    _, op2 = op.simplify_for_constant_input(f1)
    ift.myassert(isinstance(op2, ConstantOperator))
    assert_allclose(op(f1).val, op2.force(f1).val)

    dom = {"a": ift.RGSpace(10), "b": ift.RGSpace(5)}
    f1 = ift.full(dom, 2.)
    pdom = {"a": ift.RGSpace(10)}
    f2 = ift.full(pdom, 2.)
    o1 = ift.FFTOperator(f1.domain["a"])
    o2 = ift.FFTOperator(f1.domain["b"])
    op = (o1.ducktape("a").ducktape_left("a") +
          o2.ducktape("b").ducktape_left("b"))
    _, op2 = op.simplify_for_constant_input(f2)
    assert_allclose(op(f1)["a"].val, op2.force(f1)["a"].val)
    assert_allclose(op(f1)["b"].val, op2.force(f1)["b"].val)
    # FIXME Add test for ChainOperator._simplify_for_constant_input_nontrivial()


def test_modify_sample_domain():
    func = ift.minimization.kl_energies._modify_sample_domain
    dom0 = ift.RGSpace(1)
    dom1 = ift.RGSpace(2)
    field = ift.full(dom0, 1.)
    ift.extra.assert_equal(func(field, dom0), field)

    mdom0 = ift.makeDomain({'a': dom0, 'b': dom1})
    mdom1 = ift.makeDomain({'a': dom0})
    mfield0 = ift.full(mdom0, 1.)
    mfield1 = ift.full(mdom1, 1.)
    mfield01 = ift.MultiField.from_dict({'a': ift.full(dom0, 1.),
                                         'b': ift.full(dom1, 0.)})

    ift.extra.assert_equal(func(mfield0, mdom0), mfield0)
    ift.extra.assert_equal(func(mfield0, mdom1), mfield1)
    ift.extra.assert_equal(func(mfield1, mdom0), mfield01)
    ift.extra.assert_equal(func(mfield1, mdom1), mfield1)

    with assert_raises(TypeError):
        func(mfield0, dom0)
    with assert_raises(TypeError):
        func(field, dom1)
