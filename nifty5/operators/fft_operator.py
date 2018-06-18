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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import numpy as np
from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from .linear_operator import LinearOperator
from .. import dobj
from .. import utilities
from ..field import Field


class FFTOperator(LinearOperator):
    """Transforms between a pair of position and harmonic RGSpaces.

    Parameters
    ----------
    domain: Domain, tuple of Domain or DomainTuple
        The domain of the data that is input by "times" and output by
        "adjoint_times".
    target: Domain, optional
        The target (sub-)domain of the transform operation.
        If omitted, a domain will be chosen automatically.
    space: int, optional
        The index of the subdomain on which the operator should act
        If None, it is set to 0 if `domain` contains exactly one space.
        `domain[space]` must be an RGSpace.
    """

    def __init__(self, domain, target=None, space=None):
        super(FFTOperator, self).__init__()

        # Initialize domain and target
        self._domain = DomainTuple.make(domain)
        self._space = utilities.infer_space(self._domain, space)

        adom = self._domain[self._space]
        if not isinstance(adom, RGSpace):
            raise TypeError("FFTOperator only works on RGSpaces")
        if target is None:
            target = adom.get_default_codomain()

        self._target = [dom for dom in self._domain]
        self._target[self._space] = target
        self._target = DomainTuple.make(self._target)
        adom.check_codomain(target)
        target.check_codomain(adom)

        utilities.fft_prep()

    def apply(self, x, mode):
        self._check_input(x, mode)
        if np.issubdtype(x.dtype, np.complexfloating):
            return (self._apply_cartesian(x.real, mode) +
                    1j*self._apply_cartesian(x.imag, mode))
        else:
            return self._apply_cartesian(x, mode)

    def _apply_cartesian(self, x, mode):
        axes = x.domain.axes[self._space]
        tdom = self._target if x.domain == self._domain else self._domain
        oldax = dobj.distaxis(x.val)
        if oldax not in axes:  # straightforward, no redistribution needed
            ldat = x.local_data
            ldat = utilities.hartley(ldat, axes=axes)
            tmp = dobj.from_local_data(x.val.shape, ldat, distaxis=oldax)
        elif len(axes) < len(x.shape) or len(axes) == 1:
            # we can use one Hartley pass in between the redistributions
            tmp = dobj.redistribute(x.val, nodist=axes)
            newax = dobj.distaxis(tmp)
            ldat = dobj.local_data(tmp)
            ldat = utilities.hartley(ldat, axes=axes)
            tmp = dobj.from_local_data(tmp.shape, ldat, distaxis=newax)
            tmp = dobj.redistribute(tmp, dist=oldax)
        else:  # two separate, full FFTs needed
            # ideal strategy for the moment would be:
            # - do real-to-complex FFT on all local axes
            # - fill up array
            # - redistribute array
            # - do complex-to-complex FFT on remaining axis
            # - add re+im
            # - redistribute back
            rem_axes = tuple(i for i in axes if i != oldax)
            tmp = x.val
            ldat = dobj.local_data(tmp)
            ldat = utilities.my_fftn_r2c(ldat, axes=rem_axes)
            if oldax != 0:
                raise ValueError("bad distribution")
            ldat2 = ldat.reshape((ldat.shape[0],
                                  np.prod(ldat.shape[1:])))
            shp2d = (x.val.shape[0], np.prod(x.val.shape[1:]))
            tmp = dobj.from_local_data(shp2d, ldat2, distaxis=0)
            tmp = dobj.transpose(tmp)
            ldat2 = dobj.local_data(tmp)
            ldat2 = utilities.my_fftn(ldat2, axes=(1,))
            ldat2 = ldat2.real+ldat2.imag
            tmp = dobj.from_local_data(tmp.shape, ldat2, distaxis=0)
            tmp = dobj.transpose(tmp)
            ldat2 = dobj.local_data(tmp).reshape(ldat.shape)
            tmp = dobj.from_local_data(x.val.shape, ldat2, distaxis=0)
        Tval = Field(tdom, tmp)
        if mode & (LinearOperator.TIMES | LinearOperator.ADJOINT_TIMES):
            fct = self._domain[self._space].scalar_dvol
        else:
            fct = self._target[self._space].scalar_dvol
        if fct != 1:
            Tval *= fct

        return Tval

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        return self._all_ops
