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
# Authors: Gordian Edenhofer
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty8 as ift
import numpy as np
import pytest
from nifty8.extra import check_linear_operator
from numpy.testing import assert_allclose, assert_array_equal

from ..common import list2fixture, setup_function, teardown_function

pmp = pytest.mark.parametrize

# The test cases do not work on a multi-dimensional RGSpace yet
spaces = (
    ift.UnstructuredDomain(4),
    ift.LMSpace(5),
    ift.GLSpace(4),
)
space1 = list2fixture(spaces)
space2 = list2fixture(spaces)
dtype = list2fixture([np.float64, np.complex128])


def test_split_operator_first_axes_without_intersections(
    space1, space2, n_splits=3
):
    rng = ift.random.current_rng()

    dom = ift.DomainTuple.make((space1, space2))
    orig_idx = np.arange(space1.shape[0])
    rng.shuffle(orig_idx)
    split_idx = np.array_split(orig_idx, n_splits)
    split = ift.SplitOperator(
        dom, {f"{i:06d}": (si, )
              for i, si in enumerate(split_idx)}
    )
    assert check_linear_operator(split) is None

    r = ift.from_random(dom, "normal")
    split_r = split(r)
    # This relies on the keys of the target domain either being in the order of
    # insertion or being alphabetically sorted
    for idx, v in zip(split_idx, split_r.val.values()):
        assert_array_equal(r.val[idx], v)
    # Here, the adjoint must be the inverse as the field is split fully among
    # the generated indices and without intersections.
    assert_array_equal(split.adjoint(split_r).val, r.val)


def test_split_operator_first_axes_with_intersections(
    space1, space2, n_splits=3
):
    rng = ift.random.current_rng()

    dom = ift.DomainTuple.make((space1, space2))
    orig_idx = np.arange(space1.shape[0])
    split_idx = [
        rng.choice(orig_idx, rng.integers(1, space1.shape[0]), replace=False)
        for _ in range(n_splits)
    ]
    split = ift.SplitOperator(
        dom, {f"{i:06d}": (si, )
              for i, si in enumerate(split_idx)}
    )
    print(split_idx)
    assert check_linear_operator(split) is None

    r = ift.from_random(dom, "normal")
    split_r = split(r)
    # This relies on the keys of the target domain either being in the order of
    # insertion or being alphabetically sorted
    for idx, v in zip(split_idx, split_r.val.values()):
        assert_array_equal(r.val[idx], v)

    r_diy = np.copy(r.val)
    unique_freq = np.unique(np.concatenate(split_idx), return_counts=True)
    # Null values that were not selected
    r_diy[list(set(unique_freq[0]) ^ set(range(space1.shape[0])))] = 0.
    for idx, freq in zip(*unique_freq):
        r_diy[idx] *= freq
    assert_allclose(split.adjoint(split_r).val, r_diy)
