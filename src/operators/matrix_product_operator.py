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
import scipy.sparse

from .. import utilities
from ..domain_tuple import DomainTuple
from ..field import Field
from .endomorphic_operator import EndomorphicOperator


class MatrixProductOperator(EndomorphicOperator):
    """Endomorphic matrix multiplication with input field.

    This operator supports scipy.sparse matrices and numpy arrays
    as the matrix to be applied.

    For numpy array matrices, can apply the matrix over a subspace
    of the input.

    If the input arrays have more than one dimension, for
    scipy.sparse matrices the `flatten` keyword argument must be
    set to true. This means that the input field will be flattened
    before applying the matrix and reshaped to its original shape
    afterwards.

    Matrices are tested regarding their compatibility with the
    called for application method.

    Flattening and subspace application are mutually exclusive.

    Parameters
    ----------
    domain: Domain or DomainTuple
        Domain of the operator.
        If :class:`DomainTuple` it is assumed to have only one entry.
    matrix: scipy.sparse.spmatrix or numpy.ndarray
        Quadratic matrix of shape `(domain.shape, domain.shape)`
        (if `not flatten`) that supports `matrix.transpose()`.
        If it is not a numpy array, needs to be applicable to the val
        array of input fields by `matrix.dot()`.
    spaces: int or tuple of int, optional
        The subdomain(s) of "domain" which the operator acts on.
        If None, it acts on all elements.
        Only possible for numpy array matrices.
        If `len(domain) > 1` and `flatten=False`, this parameter is
        mandatory.
    flatten: boolean, optional
        Whether the input value array should be flattened before
        applying the matrix and reshaped to its original shape
        afterwards.
        Needed for scipy.sparse matrices if `len(domain) > 1`.
    """
    def __init__(self, domain, matrix, spaces=None, flatten=False):
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._domain = DomainTuple.make(domain)

        mat_dim = len(matrix.shape)

        if mat_dim % 2 != 0 or \
           matrix.shape != (matrix.shape[:mat_dim//2] + matrix.shape[:mat_dim//2]):
            raise ValueError("Matrix must be quadratic.")
        appl_dim = mat_dim // 2  # matrix application space dimension

        # take shortcut for trivial case
        if spaces is not None:
            if len(self._domain.shape) == 1 and spaces == (0, ):
                spaces = None

        if spaces is None:
            self._spaces = None
            self._active_axes = utilities.my_sum(self._domain.axes)
            appl_space_shape = self._domain.shape
            if flatten:
                appl_space_shape = (utilities.my_product(appl_space_shape), )
        else:
            if flatten:
                raise ValueError(
                    "Cannot flatten input AND apply to a subspace")
            if not isinstance(matrix, np.ndarray):
                raise ValueError(
                    "Application to subspaces only supported for numpy array matrices."
                )
            self._spaces = utilities.parse_spaces(spaces, len(self._domain))
            appl_space_shape = []
            active_axes = []
            for space_idx in spaces:
                appl_space_shape += self._domain[space_idx].shape
                active_axes += self._domain.axes[space_idx]
            appl_space_shape = tuple(appl_space_shape)
            self._active_axes = tuple(active_axes)

            self._mat_last_n = tuple([-appl_dim + i for i in range(appl_dim)])
            self._mat_first_n = np.arange(appl_dim)

        # Test if the matrix and the array it will be applied to fit
        if matrix.shape[:appl_dim] != appl_space_shape:
            raise ValueError(
                "Matrix and domain shapes are incompatible under the requested "
                + "application scheme.\n" +
                f"Matrix appl shape: {matrix.shape[:appl_dim]}, " +
                f"appl_space_shape: {appl_space_shape}.")

        self._mat = matrix
        self._mat_tr = matrix.transpose().conjugate()
        self._flatten = flatten

    def apply(self, x, mode):
        self._check_input(x, mode)
        times = (mode == self.TIMES)
        m = self._mat if times else self._mat_tr

        if self._spaces is None:
            if not self._flatten:
                res = m.dot(x.val)
            else:
                res = m.dot(x.val.flatten()).reshape(self._domain.shape)
            return Field(self._domain, res)

        mat_axes = self._mat_last_n if times else np.flip(self._mat_last_n)
        move_axes = self._mat_first_n if times else np.flip(self._mat_first_n)
        res = np.tensordot(m, x.val, axes=(mat_axes, self._active_axes))
        res = np.moveaxis(res, move_axes, self._active_axes)
        return Field(self._domain, res)
