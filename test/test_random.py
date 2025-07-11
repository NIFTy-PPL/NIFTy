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

import nifty8 as ift
import numpy as np


class RandomStateIsSane:
    def __init__(self):
        self._n_states = None

    def __enter__(self):
        self._n_states = len(ift.random._rng)
        assert self._n_states == len(ift.random._sseq)

    def __exit__(self, exc_type, exc_value, tb):
        assert self._n_states == len(ift.random._rng)
        assert self._n_states == len(ift.random._sseq)
        return exc_type is None


def test_rand1():
    with RandomStateIsSane():
        with ift.random.Context(31):
            a = ift.random.current_rng().integers(0,1000000000)
        with ift.random.Context(31):
            b = ift.random.current_rng().integers(0,1000000000)
    np.testing.assert_equal(a,b)


def test_rand2():
    with RandomStateIsSane():
        sseq = ift.random.spawn_sseq(10)
        with ift.random.Context(sseq[2]):
            a = ift.random.current_rng().integers(0,1000000000)
        with ift.random.Context(sseq[2]):
            b = ift.random.current_rng().integers(0,1000000000)
    np.testing.assert_equal(a,b)


def test_rand3():
    with RandomStateIsSane():
        with ift.random.Context(31):
            sseq = ift.random.spawn_sseq(10)
            with ift.random.Context(sseq[2]):
                a = ift.random.current_rng().integers(0,1000000000)
        with ift.random.Context(31):
            sseq = ift.random.spawn_sseq(1)
            sseq = ift.random.spawn_sseq(1)
            sseq = ift.random.spawn_sseq(1)
            with ift.random.Context(sseq[0]):
                b = ift.random.current_rng().integers(0,1000000000)
    np.testing.assert_equal(a,b)


def test_rand4():
    with ift.random.Context(31):
        a = ift.random.current_rng().integers(0,1000000000)
        with ift.random.Context(31):
            b = ift.random.current_rng().integers(0,1000000000)
    np.testing.assert_equal(a,b)


def test_rand5():
    with RandomStateIsSane():
        ift.random.push_sseq_from_seed(31)
        a = ift.random.current_rng().integers(0,1000000000)
        ift.random.push_sseq_from_seed(31)
        b = ift.random.current_rng().integers(0,1000000000)
        c = ift.random.current_rng().integers(0,1000000000)
        ift.random.pop_sseq()
        d = ift.random.current_rng().integers(0,1000000000)
        ift.random.pop_sseq()
    np.testing.assert_equal(a,b)
    np.testing.assert_equal(c,d)


def test_rand5b():
    with RandomStateIsSane():
        with ift.random.Context(31):
            a = ift.random.current_rng().integers(0,1000000000)
            with ift.random.Context(31):
                b = ift.random.current_rng().integers(0,1000000000)
                c = ift.random.current_rng().integers(0,1000000000)
            d = ift.random.current_rng().integers(0,1000000000)
    np.testing.assert_equal(a,b)
    np.testing.assert_equal(c,d)


def test_rand6():
    with RandomStateIsSane():
        ift.random.push_sseq_from_seed(31)
        state = ift.random.getState()
        a = ift.random.current_rng().integers(0,1000000000)
        ift.random.setState(state)
        b = ift.random.current_rng().integers(0,1000000000)
        np.testing.assert_equal(a,b)
        ift.random.pop_sseq()


def test_rand6b():
    with RandomStateIsSane():
        with ift.random.Context(31):
            state = ift.random.getState()
            a = ift.random.current_rng().integers(0,1000000000)
            ift.random.setState(state)
            b = ift.random.current_rng().integers(0,1000000000)
            np.testing.assert_equal(a,b)
