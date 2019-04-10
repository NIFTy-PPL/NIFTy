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
# Copyright(C) 2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..fft import hartley
from ..operators.linear_operator import LinearOperator
from ..sugar import from_global_data, makeDomain


class GridderMaker(object):
    def __init__(self, domain, eps=1e-15):
        domain = makeDomain(domain)
        if (len(domain) != 1 or not isinstance(domain[0], RGSpace) or
                not len(domain.shape) == 2):
            raise ValueError("need domain with exactly one 2D RGSpace")
        nu, nv = domain.shape
        if nu % 2 != 0 or nv % 2 != 0:
            raise ValueError("dimensions must be even")
        rat = 3 if eps < 1e-11 else 2
        nu2, nv2 = rat*nu, rat*nv

        nspread = int(-np.log(eps)/(np.pi*(rat-1)/(rat-.5)) + .5) + 1
        nu2 = max([nu2, 2*nspread])
        nv2 = max([nv2, 2*nspread])
        r2lamb = rat*rat*nspread/(rat*(rat-.5))

        oversampled_domain = RGSpace(
            [nu2, nv2], distances=[1, 1], harmonic=False)

        self._nspread = nspread
        self._r2lamb = r2lamb
        self._rest = _RestOperator(domain, oversampled_domain, r2lamb)

    def getReordering(self, uv):
        from nifty_gridder import peanoindex
        nu2, nv2 = self._rest._domain.shape
        return peanoindex(uv, nu2, nv2)

    def getGridder(self, uv):
        return RadioGridder(self._rest.domain, self._nspread, self._r2lamb, uv)

    def getRest(self):
        return self._rest

    def getFull(self, uv):
        return self.getRest() @ self.getGridder(uv)


class _RestOperator(LinearOperator):
    def __init__(self, domain, oversampled_domain, r2lamb):
        self._domain = makeDomain(oversampled_domain)
        self._target = domain
        nu, nv = domain.shape
        nu2, nv2 = oversampled_domain.shape

        # compute deconvolution operator
        rng = np.arange(nu)
        k = np.minimum(rng, nu-rng)
        c = np.pi*r2lamb/nu2**2
        self._deconv_u = np.roll(np.exp(c*k**2), -nu//2).reshape((-1, 1))
        rng = np.arange(nv)
        k = np.minimum(rng, nv-rng)
        c = np.pi*r2lamb/nv2**2
        self._deconv_v = np.roll(
            np.exp(c*k**2)/r2lamb, -nv//2).reshape((1, -1))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        nu, nv = self._target.shape
        res = x.to_global_data()
        if mode == self.TIMES:
            res = hartley(res)
            res = np.roll(res, (nu//2, nv//2), axis=(0, 1))
            res = res[:nu, :nv]
            res *= self._deconv_u
            res *= self._deconv_v
        else:
            res = res*self._deconv_u
            res *= self._deconv_v
            nu2, nv2 = self._domain.shape
            res = np.pad(res, ((0, nu2-nu), (0, nv2-nv)), mode='constant',
                         constant_values=0)
            res = np.roll(res, (-nu//2, -nv//2), axis=(0, 1))
            res = hartley(res)
        return from_global_data(self._tgt(mode), res)


class RadioGridder(LinearOperator):
    def __init__(self, target, nspread, r2lamb, uv):
        self._domain = DomainTuple.make(
            UnstructuredDomain((uv.shape[0],)))
        self._target = DomainTuple.make(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._nspread, self._r2lamb = int(nspread), float(r2lamb)
        self._uv = uv  # FIXME: should we write-protect this?

    def apply(self, x, mode):
        from nifty_gridder import to_grid, from_grid
        self._check_input(x, mode)
        nu2, nv2 = self._target.shape
        x = x.to_global_data()
        if mode == self.TIMES:
            res = to_grid(self._uv, x, nu2, nv2, self._nspread, self._r2lamb)
            res += np.conj(np.roll(res[::-1, ::-1], (1, 1), axis=(0, 1)))
            res = 0.5*(res.real+res.imag)
        else:
            mirr = np.roll(x[::-1, ::-1], (1, 1), axis=(0, 1))
            x = 0.5*(x+mirr + 1j*(x-mirr))
            res = from_grid(self._uv, x, nu2, nv2, self._nspread, self._r2lamb)
        return from_global_data(self._tgt(mode), res)
