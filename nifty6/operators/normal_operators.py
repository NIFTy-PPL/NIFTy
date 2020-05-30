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
from ..operators.operator import Operator
from ..operators.adder import Adder
from ..operators.simple_linear_operators import ducktape
from ..operators.diagonal_operator import DiagonalOperator
from ..sugar import makeField


def _reshaper(x, N):
    x = np.asfarray(x)
    if x.shape in [(), (1, )]:
        return np.full(N, x) if N != 0 else x.reshape(())
    elif x.shape == (N, ):
        return x
    else:
        raise TypeError("Shape of parameters cannot be interpreted")


def NormalTransform(mean, sigma, key, N=0):
    """Opchain that transforms standard normally distributed values to
    normally distributed values with given mean an standard deviation.

    Parameters:
    -----------
    mean : float
        Mean of the field
    sigma : float
        Standard deviation of the field
    key : string
        Name of the operators domain (Multidomain)
    N_copies : integer
        If == 0, target will be a scalar field.
        If >= 1, target will be an
        :class:`~nifty6.unstructured_domain.UnstructuredDomain`.
    """
    if N == 0:
        domain = DomainTuple.scalar_domain()
        mean, sigma = np.asfarray(mean), np.asfarray(sigma)
        mean_adder = Adder(makeField(domain, mean))
        return mean_adder @ (sigma * ducktape(domain, None, key))

    domain = UnstructuredDomain(N)
    mean, sigma = (_reshaper(param, N) for param in (mean, sigma))
    mean_adder = Adder(makeField(domain, mean))
    sigma_op = DiagonalOperator(makeField(domain, sigma))
    return mean_adder @ sigma_op @ ducktape(domain, None, key)


def _lognormal_moments(mean, sig, N=0):
    if N == 0:
        mean, sig = np.asfarray(mean), np.asfarray(sig)
    else:
        mean, sig = (_reshaper(param, N) for param in (mean, sig))
    if not np.all(mean > 0):
        raise ValueError("mean must be greater 0; got {!r}".format(mean))
    if not np.all(sig > 0):
        raise ValueError("sig must be greater 0; got {!r}".format(sig))

    logsig = np.sqrt(np.log1p((sig / mean)**2))
    logmean = np.log(mean) - logsig**2 / 2
    return logmean, logsig


class LognormalTransform(Operator):
    """Opchain that transforms standard normally distributed values to
    log-normally distributed values with given mean an standard deviation.

    Parameters:
    -----------
    mean : float
        Mean of the field
    sigma : float
        Standard deviation of the field
    key : string
        Name of the domain
    N_copies : integer
        If == 0, target will be a scalar field.
        If >= 1, target will be an
        :class:`~nifty6.unstructured_domain.UnstructuredDomain`.
    """
    def __init__(self, mean, sigma, key, N_copies):
        key = str(key)
        logmean, logsigma = _lognormal_moments(mean, sigma, N_copies)
        self._mean = mean
        self._sigma = sigma
        op = NormalTransform(logmean, logsigma, key, N_copies).ptw("exp")
        self._domain, self._target = op.domain, op.target
        self.apply = op.apply
        self._repr_str = f"LognormalTransform: " + op.__repr__()

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._sigma

    def __repr__(self):
        return self._repr_str
