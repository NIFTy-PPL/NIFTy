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

from ..compat import *
from ..domain_tuple import DomainTuple
from ..domains.power_space import PowerSpace
from ..utilities import infer_space
from .dof_distributor import DOFDistributor


class PowerDistributor(DOFDistributor):
    """Operator which transforms between a PowerSpace and a harmonic domain.

    Parameters
    ----------
    target: Domain, tuple of Domain, or DomainTuple
        the total *target* domain of the operator.
    power_space: PowerSpace, optional
        the input sub-domain on which the operator acts.
        If not supplied, a matching PowerSpace with natural binbounds will be
        used.
    space: int, optional:
       The index of the sub-domain on which the operator acts.
       Can be omitted if `target` only has one sub-domain.
    """

    def __init__(self, target, power_space=None, space=None):
        # Initialize domain and target
        self._target = DomainTuple.make(target)
        self._space = infer_space(self._target, space)
        hspace = self._target[self._space]
        if not hspace.harmonic:
            raise ValueError("Operator requires harmonic target space")
        if power_space is None:
            power_space = PowerSpace(hspace)
        else:
            if not isinstance(power_space, PowerSpace):
                raise TypeError("power_space argument must be a PowerSpace")
            if power_space.harmonic_partner != hspace:
                raise ValueError("power_space does not match its partner")

        self._init2(power_space.pindex, self._space, power_space)
