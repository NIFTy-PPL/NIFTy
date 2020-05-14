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

import pytest
from numpy.testing import assert_allclose, assert_array_equal
from nifty6.extra import consistency_check

import numpy as np
import nifty6 as ift
from ..common import setup_function, teardown_function

pmp = pytest.mark.parametrize


@pmp("n_unstructured", (3, 9))
@pmp("nside", (4, 8))
def test_split_operator_first_axes_without_intersections(
    n_unstructured, nside, n_splits=3
):
    setup_function()
    rng = ift.random.current_rng()

    pos_space = ift.HPSpace(nside)
    dom = ift.DomainTuple.make(
        (ift.UnstructuredDomain(n_unstructured), pos_space)
    )
    orig_idx = np.arange(n_unstructured)
    rng.shuffle(orig_idx)
    split_idx = np.split(orig_idx, n_splits)
    split = ift.SplitOperator(
        dom, {f"{i:06d}": (si, )
              for i, si in enumerate(split_idx)}
    )
    assert consistency_check(split) is None

    r = ift.from_random("normal", dom)
    split_r = split(r)
    # This relies on the keys of the target domain either being in the order of
    # insertion or being alphabetically sorted
    for idx, v in zip(split_idx, split_r.val.values()):
        assert_array_equal(r.val[idx], v)
    # Here, the adjoint must be the inverse as the field is split fully among
    # the generated indices and without intersections.
    assert_array_equal(split.adjoint(split_r).val, r.val)

    teardown_function()


@pmp("n_unstructured", (3, 9))
@pmp("nside", (4, 8))
def test_split_operator_first_axes_with_intersections(
    n_unstructured, nside, n_splits=3
):
    setup_function()
    rng = ift.random.current_rng()

    pos_space = ift.HPSpace(nside)
    dom = ift.DomainTuple.make(
        (ift.UnstructuredDomain(n_unstructured), pos_space)
    )
    orig_idx = np.arange(n_unstructured)
    split_idx = [
        rng.choice(orig_idx, rng.integers(1, n_unstructured), replace=False)
        for _ in range(n_splits)
    ]
    split = ift.SplitOperator(
        dom, {f"{i:06d}": (si, )
              for i, si in enumerate(split_idx)}
    )
    print(split_idx)
    assert consistency_check(split) is None

    r = ift.from_random("normal", dom)
    split_r = split(r)
    # This relies on the keys of the target domain either being in the order of
    # insertion or being alphabetically sorted
    for idx, v in zip(split_idx, split_r.val.values()):
        assert_array_equal(r.val[idx], v)

    r_diy = np.copy(r.val)
    unique_freq = np.unique(np.concatenate(split_idx), return_counts=True)
    # Null values that were not selected
    r_diy[list(set(unique_freq[0]) ^ set(range(n_unstructured)))] = 0.
    for idx, freq in zip(*unique_freq):
        r_diy[idx] *= freq
    assert_allclose(split.adjoint(split_r).val, r_diy)

    teardown_function()
