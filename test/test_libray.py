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

import numpy as np
import pytest
from numpy.testing import assert_

import nifty5 as ift


@pytest.mark.parametrize('power_space', [
    ift.PowerSpace(ift.RGSpace((4, 4), harmonic=True)),
    ift.PowerSpace(ift.LMSpace(5)),
])

@pytest.mark.parametrize('seed', [13, 2])

@pytest.mark.parametrize('space', [None, 0, 1, 2])

def test_SLAmplitude(power_space, seed, space):
    np.random.seed(seed)
    domains = [ift.UnstructuredDomain([2,2]), ift.RGSpace(2)]
    if space is None:
        ps = power_space
        dct = { 'target': ps, 'n_pix': 32,
               'a': 1.5,  'k0': 5,  'sm': -2,  'sv': 0.2, 'im':  0, 'iv': .02 }
    else:
        domains.insert(space, power_space)
        ps = ift.makeDomain(domains)
        dct = { 'target': ps, 'n_pix': 32,
            'a': np.ones([2,2,2]),  'k0': 5,  
            'sm': [[[-2, -3],[-1, -1]], [[-2, -3], [-2, -1]]], 'sv': 0.2, 'im':  0, 'iv': .02 }
    A = ift.SLAmplitude(**dct, space = space)
    x = ift.from_random('normal', A.domain)
    ift.extra.check_jacobian_consistency(A, x)
    #assert_(A(x) is not None)
