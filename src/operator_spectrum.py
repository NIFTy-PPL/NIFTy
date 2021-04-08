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

import numpy as np
import scipy.sparse.linalg as ssl

from .domain_tuple import DomainTuple
from .domains.unstructured_domain import UnstructuredDomain
from .field import Field
from .multi_domain import MultiDomain
from .multi_field import MultiField
from .operators.linear_operator import LinearOperator
from .operators.sandwich_operator import SandwichOperator
from .sugar import makeDomain, makeField


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
        self._check_input(x, mode)
        self._check_float_dtype(x)
        x = x.val
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
        return makeField(self._tgt(mode), res)

    @staticmethod
    def _check_float_dtype(fld):
        if isinstance(fld, MultiField):
            dts = [ff.dtype for ff in fld.values()]
        elif isinstance(fld, Field):
            dts = [fld.dtype]
        else:
            raise TypeError
        for dt in dts:
            if not np.issubdtype(dt, np.float64):
                raise TypeError('Operator supports only floating point dtypes')


def operator_spectrum(A, k, hermitian, which='LM', tol=0, return_eigenvectors=False):
    '''
    Find k eigenvalues and eigenvectors of the endomorphism A.

    Parameters
    ----------
    A : LinearOperator
        Operator of which eigenvalues shall be computed.
    k : int
        The number of eigenvalues and eigenvectors desired. `k` must be
        smaller than N-1. It is not possible to compute all eigenvectors of a
        matrix.
    hermitian: bool
        Specifies whether A is hermitian or not.
    which : str, ['LM' | 'SM' | 'LR' | 'SR' | 'LI' | 'SI'], optional
        Which `k` eigenvectors and eigenvalues to find:

            'LM' : largest magnitude

            'SM' : smallest magnitude

            'LR' : largest real part

            'SR' : smallest real part

            'LI' : largest imaginary part

            'SI' : smallest imaginary part

    tol : float, optional
        Relative accuracy for eigenvalues (stopping criterion)
        The default value of 0 implies machine precision.

    return_eigenvectors: bool, optional
        Return eigenvectors (True) in addition to eigenvalues


    Returns
    -------
    w : ndarray
        Array of k eigenvalues.

    v : ndarray
        An array representing the k eigenvectors. The column v[:, i] is the
        eigenvector corresponding to the eigenvalue w[i].

    Raises
    ------
    ArpackNoConvergence
        When the requested convergence is not obtained.
        The currently converged eigenvalues and eigenvectors can be found
        as ``eigenvalues`` and ``eigenvectors`` attributes of the exception
        object.
    '''
    if not isinstance(A, LinearOperator):
        raise TypeError('Operator needs to be linear.')
    if A.domain is not A.target:
        raise TypeError('Operator needs to be endomorphism.')
    size = A.domain.size
    Ar = SandwichOperator.make(_DomRemover(A.domain).adjoint, A)
    M = ssl.LinearOperator(
        shape=2*(size,),
        matvec=lambda x: Ar(makeField(Ar.domain, x)).val)
    f = ssl.eigsh if hermitian else ssl.eigs
    eigs = f(M, k=k, tol=tol, return_eigenvectors=return_eigenvectors, which=which)
    if return_eigenvectors:
        eigval, eigvec = eigs
        inds = np.argsort(eigval)
        return np.flip(eigval[inds]), np.flip(eigvec[:,inds],axis = 1)
    else:
        return np.flip(np.sort(eigs))
