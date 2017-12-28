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

from .dof_projection_operator import DOFProjectionOperator
from .. import DomainTuple
from ..utilities import infer_space
from ..spaces import PowerSpace


class PowerProjectionOperator(DOFProjectionOperator):
    def __init__(self, domain, power_space=None, space=None):
        # Initialize domain and target
        self._domain = DomainTuple.make(domain)
        self._space = infer_space(self._domain, space)
        hspace = self._domain[self._space]
        if not hspace.harmonic:
            raise ValueError("Operator acts on harmonic spaces only")
        if power_space is None:
            power_space = PowerSpace(hspace)
        else:
            if not isinstance(power_space, PowerSpace):
                raise TypeError("power_space argument must be a PowerSpace")
            if power_space.harmonic_partner != hspace:
                raise ValueError("power_space does not match its partner")

        self._init2(power_space.pindex, self._space, power_space)
