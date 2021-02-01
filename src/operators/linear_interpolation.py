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

from functools import reduce
from operator import add

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import aslinearoperator

from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..sugar import makeDomain
from .linear_operator import LinearOperator


class LinearInterpolator(LinearOperator):
    """Multilinear interpolation for points in an RGSpace

    Parameters
    ----------
    domain : RGSpace
    sampling_points : numpy.ndarray
        Positions at which to interpolate, shape (dim, ndata),

    Notes
    -----
    Positions that are not within the RGSpace are wrapped according to
    periodic boundary conditions. This reflects the general property of
    RGSpaces to be tori topologically.
    """
    def __init__(self, domain, sampling_points):
        self._domain = makeDomain(domain)
        for dom in self.domain:
            if not isinstance(dom, RGSpace):
                raise TypeError
        dims = [len(dom.shape) for dom in self.domain]

        # FIXME This needs to be removed as soon as the bug below is fixed.
        if dims.count(dims[0]) != len(dims):
            raise TypeError("This is a bug. Please extend"
                            "LinearInterpolator's functionality!")

        shp = sampling_points.shape
        if not (isinstance(sampling_points, np.ndarray) and len(shp) == 2):
            raise TypeError
        n_dim, n_points = shp
        if n_dim != reduce(add, dims):
            raise TypeError
        self._target = makeDomain(UnstructuredDomain(n_points))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._build_mat(sampling_points, n_points)

    def _build_mat(self, sampling_points, N_points):
        ndim = sampling_points.shape[0]
        mg = np.mgrid[(slice(0, 2),)*ndim]
        mg = np.array(list(map(np.ravel, mg)))
        dist = [list(dom.distances) for dom in self.domain]
        # FIXME This breaks as soon as not all domains have the same number of
        # dimensions.
        dist = np.array(dist).reshape(-1, 1)
        pos = sampling_points/dist
        excess = pos - np.floor(pos)
        pos = np.floor(pos).astype(np.int64)
        max_index = np.array(self.domain.shape).reshape(-1, 1)
        data = np.zeros((len(mg[0]), N_points))
        ii = np.zeros((len(mg[0]), N_points), dtype=np.int64)
        jj = np.zeros((len(mg[0]), N_points), dtype=np.int64)
        for i in range(len(mg[0])):
            factor = np.prod(
                np.abs(1 - mg[:, i].reshape(-1, 1) - excess), axis=0)
            data[i, :] = factor
            fromi = (pos + mg[:, i].reshape(-1, 1)) % max_index
            ii[i, :] = np.arange(N_points)
            jj[i, :] = np.ravel_multi_index(fromi, self.domain.shape)
        self._mat = coo_matrix((data.reshape(-1),
                                (ii.reshape(-1), jj.reshape(-1))),
                               (N_points, np.prod(self.domain.shape)))
        self._mat = aslinearoperator(self._mat)

    def apply(self, x, mode):
        self._check_input(x, mode)
        x_val = x.val
        if mode == self.TIMES:
            res = self._mat.matvec(x_val.reshape(-1))
        else:
            res = self._mat.rmatvec(x_val).reshape(self.domain.shape)
        return Field(self._tgt(mode), res)
