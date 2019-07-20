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

import numpy as np
import scipy.sparse.linalg as ssl

from .domain_tuple import DomainTuple
from .domains.unstructured_domain import UnstructuredDomain
from .multi_domain import MultiDomain
from .operators.linear_operator import LinearOperator
from .operators.sandwich_operator import SandwichOperator
from .sugar import from_global_data, makeDomain


class _DomRemover(LinearOperator):
    """Operator which transforms between a structured MultiDomain
    and an unstructured domain.

    Parameters
    ----------
    domain: MultiDomain
        the full input domain of the operator.

    Notes
    -----
    The operator converts the full domain of its input domain to an
    UnstructuredDomain
    """

    def __init__(self, domain):
        self._domain = makeDomain(domain)
        if isinstance(self._domain, MultiDomain):
            self._size_array = np.array([0] +
                                        [d.size for d in domain.values()])
        else:
            self._size_array = np.array([0, domain.size])
        np.cumsum(self._size_array, out=self._size_array)
        target = UnstructuredDomain(self._size_array[-1])
        self._target = makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        x = x.to_global_data()
        if isinstance(self._domain, DomainTuple):
            res = x.ravel() if mode == self.TIMES else x.reshape(
                self._domain.shape)
        else:
            res = np.empty(self.target.shape) if mode == self.TIMES else {}
            for ii, (kk, dd) in enumerate(self.domain.items()):
                i0, i1 = self._size_array[ii:ii + 2]
                if mode == self.TIMES:
                    res[i0:i1] = x[kk].ravel()
                else:
                    res[kk] = x[i0:i1].reshape(dd.shape)
        return from_global_data(self._tgt(mode), res)


def _op2lambda(op):
    remover = _DomRemover(op.domain).adjoint
    op = SandwichOperator.make(remover, op)
    return lambda x: op(from_global_data(op.domain, x)).to_global_data()


def operator_spectrum(op, n, hermitian, tol=1e-5):
    if not isinstance(op, LinearOperator):
        raise TypeError('Operator needs to be linear.')
    if op.domain is not op.target:
        raise TypeError('Operator needs to be endomorphism.')
    size = op.domain.size
    M = ssl.LinearOperator(shape=2*(size,), matvec=_op2lambda(op))
    f = ssl.eigsh if hermitian else ssl.eigs
    eigs = f(M, k=n, tol=tol, return_eigenvectors=False)
    return np.flip(np.sort(eigs), axis=0)
