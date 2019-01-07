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

from .. import dobj, utilities
from ..domain_tuple import DomainTuple
from ..domains.log_rg_space import LogRGSpace
from ..field import Field
from .endomorphic_operator import EndomorphicOperator


class SymmetrizingOperator(EndomorphicOperator):
    def __init__(self, domain, space=0):
        self._domain = DomainTuple.make(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._space = utilities.infer_space(self._domain, space)
        dom = self._domain[self._space]
        if not (isinstance(dom, LogRGSpace) and not dom.harmonic):
            raise TypeError("nonharmonic LogRGSpace needed")

    def apply(self, x, mode):
        self._check_input(x, mode)
        v = x.val.copy()
        for i in self._domain.axes[self._space]:
            lead = (slice(None),)*i
            v, loc = dobj.ensure_not_distributed(v, (i,))
            loc[lead+(slice(1, None),)] -= loc[lead+(slice(None, 0, -1),)]
            loc /= 2
        return Field(self.target, dobj.ensure_default_distributed(v))
