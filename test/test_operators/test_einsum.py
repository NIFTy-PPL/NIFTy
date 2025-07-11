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
# Copyright(C) 2013-2020 Max-Planck-Society
# Authors: Gordian Edenhofer, Philipp Frank
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
from nifty8.extra import check_linear_operator, check_operator
from numpy.testing import assert_allclose

from ..common import list2fixture, setup_function, teardown_function

spaces = (ift.UnstructuredDomain(4),
          ift.RGSpace((3, 2)),
          ift.LMSpace(5),
          ift.GLSpace(4))

space1 = list2fixture(spaces)
space2 = list2fixture(spaces)
dtype = list2fixture([np.float64, np.complex128])


def test_linear_einsum_outer(space1, space2, dtype, n_invocations=10):
    mf_dom = ift.MultiDomain.make({
        "dom01": space1,
        "dom02": ift.DomainTuple.make((space1, space2))})
    mf = ift.from_random(mf_dom, "normal", dtype=dtype)
    ss = "i,ij,j->ij"
    key_order = ("dom01", "dom02")
    le = ift.LinearEinsum(space2, mf, ss, key_order=key_order)
    ift.myassert(check_linear_operator(le, domain_dtype=dtype, target_dtype=dtype) is None)

    le_ift = ift.DiagonalOperator(mf["dom01"], domain=mf_dom["dom02"], spaces=0) @ ift.DiagonalOperator(mf["dom02"])
    le_ift = le_ift @ ift.OuterProduct(ift.DomainTuple.make(mf_dom["dom02"][1]),
                                       ift.full(mf_dom["dom01"], 1.))

    for _ in range(n_invocations):
        r = ift.from_random(le.domain, "normal", dtype=dtype)
        assert_allclose(le(r).val, le_ift(r).val)
        r_adj = ift.from_random(le.target, "normal", dtype=dtype)
        assert_allclose(le.adjoint(r_adj).val, le_ift.adjoint(r_adj).val)


def test_linear_einsum_contraction(space1, space2, dtype, n_invocations=10):
    mf_dom = ift.MultiDomain.make({
        "dom01": space1,
        "dom02": ift.DomainTuple.make((space1, space2))})
    mf = ift.from_random(mf_dom, "normal", dtype=dtype)
    ss = "i,ij,j->i"
    key_order = ("dom01", "dom02")
    le = ift.LinearEinsum(space2, mf, ss, key_order=key_order)
    ift.myassert(check_linear_operator(le, domain_dtype=dtype, target_dtype=dtype) is None)

    le_ift = ift.ContractionOperator(mf_dom["dom02"], 1)
    le_ift = le_ift @ ift.DiagonalOperator(mf["dom01"], domain=mf_dom["dom02"], spaces=0)
    le_ift = le_ift @ ift.DiagonalOperator(mf["dom02"])
    le_ift = le_ift @ ift.OuterProduct(ift.DomainTuple.make(mf_dom["dom02"][1]),
                                       ift.full(mf_dom["dom01"], 1.),)

    for _ in range(n_invocations):
        r = ift.from_random(le.domain, "normal", dtype=dtype)
        assert_allclose(le(r).val, le_ift(r).val)
        r_adj = ift.from_random(le.target, "normal", dtype=dtype)
        assert_allclose(le.adjoint(r_adj).val, le_ift.adjoint(r_adj).val)


class _SwitchSpacesOperator(ift.LinearOperator):
    """Operator to permutate the domain entries of fields.

    Exchanges the entries `space1` and `space2` of the input's domain.
    """
    def __init__(self, domain, space1, space2=0):
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._domain = ift.DomainTuple.make(domain)

        n_spaces = len(self._domain)
        if space1 >= n_spaces or space1 < 0 \
           or space2 >= n_spaces or space2 < 0:
            raise ValueError("invalid space value")

        tgt = list(self._domain)
        tgt[space2] = self._domain[space1]
        tgt[space1] = self._domain[space2]
        self._target = ift.DomainTuple.make(tgt)

        dom_axes = self._domain.axes
        tgt_axes = self._target.axes

        self._axes_dom = dom_axes[space1] + dom_axes[space2]
        self._axes_tgt = tgt_axes[space2] + tgt_axes[space1]

    def apply(self, x, mode):
        self._check_input(x, mode)
        args = self._axes_dom, self._axes_tgt
        if mode == self.ADJOINT_TIMES:
            args = args[::-1]
        return ift.Field(self._tgt(mode), np.moveaxis(x.val, *args))


def test_linear_einsum_transpose(space1, space2, dtype, n_invocations=10):
    dom = ift.DomainTuple.make((space1, space2))
    mf = ift.MultiField.from_dict({})
    ss = "ij->ji"
    le = ift.LinearEinsum(dom, mf, ss)
    ift.myassert(check_linear_operator(le, domain_dtype=dtype, target_dtype=dtype) is None)

    # SwitchSpacesOperator is equivalent to LinearEinsum with "ij->ji"
    le_ift = _SwitchSpacesOperator(dom, 1)

    for _ in range(n_invocations):
        r = ift.from_random(le.domain, "normal", dtype=dtype)
        assert_allclose(le(r).val, le_ift(r).val)
        r_adj = ift.from_random(le.target, "normal", dtype=dtype)
        assert_allclose(le.adjoint(r_adj).val, le_ift.adjoint(r_adj).val)


def test_multi_linear_einsum_outer(space1, space2, dtype):
    ntries = 10
    n_invocations = 5
    mf_dom = ift.MultiDomain.make({
        "dom01": space1,
        "dom02": ift.DomainTuple.make((space1, space2)),
        "dom03": space2})
    ss = "i,ij,j->ij"
    key_order = ("dom01", "dom02", "dom03")
    mle = ift.MultiLinearEinsum(mf_dom, ss, key_order=key_order)
    check_operator(mle, ift.from_random(mle.domain, "normal", dtype=dtype), ntries=ntries)

    outer_i = ift.OuterProduct(
        ift.DomainTuple.make(mf_dom["dom02"][0]), ift.full(mf_dom["dom03"], 1.)
    )
    outer_j = ift.OuterProduct(
        ift.DomainTuple.make(mf_dom["dom02"][1]), ift.full(mf_dom["dom01"], 1.)
    )
    # SwitchSpacesOperator is equivalent to LinearEinsum with "ij->ji"
    mle_ift = _SwitchSpacesOperator(outer_i.target, 1) @ outer_i @ \
        ift.FieldAdapter(mf_dom["dom01"], "dom01") * \
        ift.FieldAdapter(mf_dom["dom02"], "dom02") * \
        (outer_j @ ift.FieldAdapter(mf_dom["dom03"], "dom03"))

    for _ in range(n_invocations):
        rl = ift.Linearization.make_var(ift.from_random(mle.domain, "normal", dtype=dtype))
        mle_rl, mle_ift_rl = mle(rl), mle_ift(rl)
        assert_allclose(mle_rl.val.val, mle_ift_rl.val.val)
        assert_allclose(mle_rl.jac(rl.val).val, mle_ift_rl.jac(rl.val).val)

        rj_adj = ift.from_random(mle_rl.jac.target, "normal", dtype=dtype)
        mle_j_val = mle_rl.jac.adjoint(rj_adj).val
        mle_ift_j_val = mle_ift_rl.jac.adjoint(rj_adj).val
        for k in mle_ift.domain.keys():
            assert_allclose(mle_j_val[k], mle_ift_j_val[k])
