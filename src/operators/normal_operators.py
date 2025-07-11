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
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.unstructured_domain import UnstructuredDomain
from ..operators.adder import Adder
from ..operators.diagonal_operator import DiagonalOperator
from ..operators.simple_linear_operators import ducktape
from ..sugar import makeField
from ..utilities import lognormal_moments, value_reshaper


def NormalTransform(mean, sigma, key, N_copies=0):
    """Opchain that transforms standard normally distributed values to
    normally distributed values with given mean an standard deviation.

    Parameters
    ----------
    mean : float
        Mean of the field
    sigma : float
        Standard deviation of the field
    key : string
        Name of the operators domain (Multidomain)
    N_copies : integer
        If == 0, target will be a scalar field.
        If >= 1, target will be an
        :class:`~nifty8.domains.unstructured_domain.UnstructuredDomain`.
    """
    if N_copies == 0:
        domain = DomainTuple.scalar_domain()
        mean, sigma = np.asarray(mean, dtype=float), np.asarray(sigma, dtype=float)
        mean_adder = Adder(makeField(domain, mean))
        return mean_adder @ (sigma * ducktape(domain, None, key))

    domain = UnstructuredDomain(N_copies)
    mean, sigma = (value_reshaper(param, N_copies) for param in (mean, sigma))
    mean_adder = Adder(makeField(domain, mean))
    sigma_op = DiagonalOperator(makeField(domain, sigma))
    return mean_adder @ sigma_op @ ducktape(domain, None, key)


def LognormalTransform(mean, sigma, key, N_copies):
    """Opchain that transforms standard normally distributed values to
    log-normally distributed values with given mean an standard deviation.

    Parameters
    ----------
    mean : float
        Mean of the field
    sigma : float
        Standard deviation of the field
    key : string
        Name of the domain
    N_copies : integer
        If == 0, target will be a scalar field.
        If >= 1, target will be an
        :class:`~nifty8.domains.unstructured_domain.UnstructuredDomain`.
    """
    logmean, logsigma = lognormal_moments(mean, sigma, N_copies)
    return NormalTransform(logmean, logsigma, key, N_copies).ptw("exp")
