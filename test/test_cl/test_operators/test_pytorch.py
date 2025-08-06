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
# Copyright(C) 2025 LambdaFields GmbH
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import nifty.cl as ift
import numpy as np
import pytest
import torch

from ..common import list2fixture, setup_function, teardown_function

# Input: f(x), f(x, y)
# Output: Scalar, Field, MultiField

# TODO: dtypes: single, double, complex single, complex double

def test_torch_operator():
    def func(x):
        return torch.exp(x**2)[:5]

    domain = ift.UnstructuredDomain(10)
    target = ift.UnstructuredDomain(5)
    op = ift.TorchOperator(domain, target, func)
    ift.extra.check_operator(op, ift.from_random(op.domain))
