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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

import numpy as np
from .dof_projection_operator import DOFProjectionOperator
from .. import Field, DomainTuple, dobj
from ..spaces import PowerSpace


class PowerProjectionOperator(DOFProjectionOperator):
    def __init__(self, domain, power_space=None, space=None):
        # Initialize domain and target
        self._domain = DomainTuple.make(domain)
        if space is None and len(self._domain) == 1:
            space = 0
        space = int(space)
        if space < 0 or space >= len(self.domain):
            raise ValueError("space index out of range")
        hspace = self._domain[space]
        if not hspace.harmonic:
            raise ValueError("Operator acts on harmonic spaces only")
        if power_space is None:
            power_space = PowerSpace(hspace)
        else:
            if not isinstance(power_space, PowerSpace):
                raise TypeError("power_space argument must be a PowerSpace")
            if power_space.harmonic_partner != hspace:
                raise ValueError("power_space does not match its partner")

        self._init2(power_space.pindex, space, power_space)
