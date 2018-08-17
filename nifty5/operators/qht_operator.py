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

from __future__ import absolute_import, division, print_function

from .. import dobj
from ..compat import *
from ..domain_tuple import DomainTuple
from ..field import Field
from ..utilities import hartley, infer_space
from .linear_operator import LinearOperator


class QHTOperator(LinearOperator):
    """
    Does a Hartley transform on LogRGSpace

    This operator takes a field on a LogRGSpace and transforms it
    according to the Hartley transform. The zero modes are not transformed
    because they are infinitely far away.

    Parameters
    ----------
    target : domain, tuple of domains or DomainTuple
        The full output domain
    space : int
        The index of the domain on which the operator acts.
        target[space] must be a nonharmonic LogRGSpace.
    """

    def __init__(self, target, space=0):
        self._target = DomainTuple.make(target)
        self._space = infer_space(self._target, space)

        from ..domains.log_rg_space import LogRGSpace
        if not isinstance(self._target[self._space], LogRGSpace):
            raise ValueError("target[space] has to be a LogRGSpace!")

        if self._target[self._space].harmonic:
            raise TypeError("target[space] must be a nonharmonic space")

        self._domain = [dom for dom in self._target]
        self._domain[self._space] = \
            self._target[self._space].get_default_codomain()
        self._domain = DomainTuple.make(self._domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        dom = self._domain[self._space]
        v = x.val * dom.scalar_dvol
        n = self._domain.axes[self._space]
        rng = n if mode == self.TIMES else reversed(n)
        for i in rng:
            sl = (slice(None),)*i + (slice(1, None),)
            v, tmp = dobj.ensure_not_distributed(v, (i,))
            tmp[sl] = hartley(tmp[sl], axes=(i,))
        return Field(self._tgt(mode), dobj.ensure_default_distributed(v))
