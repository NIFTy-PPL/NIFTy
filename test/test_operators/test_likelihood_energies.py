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
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce
from operator import add

import nifty8 as ift
import numpy as np
import pytest

from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize

doms = [ift.HPSpace(4), {"a": ift.UnstructuredDomain(2), "b": ift.HPSpace(4)}]

@pmp("with_names", [0, 1, 2])
@pmp("dom", doms)
def test_likelihood_single(with_names, dom):
    dom = ift.makeDomain(dom)
    data = ift.from_random(dom)
    icov = ift.from_random(dom).exp()
    lh = ift.GaussianEnergy(data=data, inverse_covariance=ift.makeOp(icov))
    if with_names == 1:
        lh.name = f"myname"
        assert lh.name == f"myname"
    if isinstance(dom, ift.DomainTuple):
        lh = lh.ducktape("b")
    if with_names == 2:
        lh.name = f"myname"
    if with_names in [1, 2]:
        assert lh.name == f"myname"
    print(lh)
    samples = ift.SampleList([ift.from_random(lh.domain) for _ in range(2)])
    print(ift.extra.minisanity(lh, samples))


@pmp("with_names", [0, 1, 2, 3])
def test_likelihood_sum(with_names):
    lhs = []
    for ii, dom in enumerate(doms):
        dom = ift.makeDomain(dom)
        data = ift.from_random(dom)
        icov = ift.from_random(dom).exp()
        lh = ift.GaussianEnergy(data=data, inverse_covariance=ift.makeOp(icov))
        if with_names == 1:
            lh.name = f"myname{ii}"
        if isinstance(dom, ift.DomainTuple):
            lh = lh.ducktape("b")
        if with_names == 2:
            lh.name = f"myname{ii}"
        if with_names == 3:
            lh.name = f"myname"
        lhs.append(lh)
    if with_names == 3:
        with pytest.raises(ValueError):
            lh = reduce(add, lhs)
        return
    lh = reduce(add, lhs)
    print(lh)
    samples = ift.SampleList([ift.from_random(lh.domain) for _ in range(2)])
    print(ift.extra.minisanity(lh, samples))
