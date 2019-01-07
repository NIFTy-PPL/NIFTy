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
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import aslinearoperator

from ..compat import *
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..sugar import makeDomain
from .linear_operator import LinearOperator


class LinearInterpolator(LinearOperator):
    def __init__(self, domain, positions):
        """
        Multilinear interpolation for points in an RGSpace

        :param domain:
            RGSpace
        :param positions:
            positions at which to interpolate
            Field with UnstructuredDomain, shape (dim, ndata)
            positions that are not within the RGSpace are wrapped
            according to periodic boundary conditions
        """
        self._domain = makeDomain(domain)
        N_points = positions.shape[1]
        self._target = makeDomain(UnstructuredDomain(N_points))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._build_mat(positions, N_points)

    def _build_mat(self, positions, N_points):
        ndim = positions.shape[0]
        mg = np.mgrid[(slice(0, 2),)*ndim]
        mg = np.array(list(map(np.ravel, mg)))
        dist = []
        for dom in self.domain:
            if not isinstance(dom, RGSpace):
                raise TypeError
            dist.append(list(dom.distances))
        dist = np.array(dist).reshape(-1, 1)
        pos = positions/dist
        excess = pos-pos.astype(np.int64)
        pos = pos.astype(np.int64)
        max_index = np.array(self.domain.shape).reshape(-1, 1)
        data = np.zeros((len(mg[0]), N_points))
        ii = np.zeros((len(mg[0]), N_points), dtype=np.int64)
        jj = np.zeros((len(mg[0]), N_points), dtype=np.int64)
        for i in range(len(mg[0])):
            factor = np.prod(np.abs(1-mg[:, i].reshape(-1, 1)-excess),
                             axis=0)
            data[i, :] = factor
            fromi = (pos+mg[:, i].reshape(-1, 1)) % max_index
            ii[i, :] = np.arange(N_points)
            jj[i, :] = np.ravel_multi_index(fromi, self.domain.shape)
        self._mat = coo_matrix((data.reshape(-1),
                               (ii.reshape(-1), jj.reshape(-1))),
                               (N_points, np.prod(self.domain.shape)))
        self._mat = aslinearoperator(self._mat)

    def apply(self, x, mode):
        self._check_input(x, mode)
        x_val = x.to_global_data()
        if mode == self.TIMES:
            res = self._mat.matvec(x_val.reshape(-1))
            return Field.from_global_data(self.target, res)
        res = self._mat.rmatvec(x_val).reshape(self.domain.shape)
        return Field.from_global_data(self.domain, res)


# import numpy as np
# from ..domains.rg_space import RGSpace
# import itertools
#
#
# class LinearInterpolator(LinearOperator):
#     def __init__(self, domain, positions):
#         """
#         :param domain:
#             RGSpace
#         :param target:
#             UnstructuredDomain, shape (ndata,)
#         :param positions:
#             positions at which to interpolate
#             Field with UnstructuredDomain, shape (dim, ndata)
#         """
#         if not isinstance(domain, RGSpace):
#             raise TypeError("RGSpace needed")
#         if np.any(domain.shape < 2):
#             raise ValueError("RGSpace shape too small")
#         if positions.ndim != 2:
#             raise ValueError("positions must be a 2D array")
#         ndim = len(domain.shape)
#         if positions.shape[0] != ndim:
#             raise ValueError("shape mismatch")
#         self._domain = makeDomain(domain)
#         N_points = positions.shape[1]
#         dist = np.array(domain.distances).reshape((ndim, -1))
#         self._pos = positions/dist
#         shp = np.array(domain.shape, dtype=int).reshape((ndim, -1))
#         self._idx = np.maximum(0, np.minimum(shp-2, self._pos.astype(int)))
#         self._pos -= self._idx
#         tmp = tuple([0, 1] for i in range(ndim))
#         self._corners = np.array(list(itertools.product(*tmp)))
#         self._target = makeDomain(UnstructuredDomain(N_points))
#         self._capability = self.TIMES | self.ADJOINT_TIMES
#
#     def apply(self, x, mode):
#         self._check_input(x, mode)
#         x = x.to_global_data()
#         ndim = len(self._domain.shape)
#
#         res = np.zeros(self._tgt(mode).shape, dtype=x.dtype)
#         for corner in self._corners:
#             corner = corner.reshape(ndim, -1)
#             idx = self._idx+corner
#             idx2 = tuple(idx[t, :] for t in range(idx.shape[0]))
#             wgt = np.prod(self._pos*corner+(1-self._pos)*(1-corner), axis=0)
#             if mode == self.TIMES:
#                 res += wgt*x[idx2]
#             else:
#                 np.add.at(res, idx2, wgt*x)
#         return Field.from_global_data(self._tgt(mode), res)
