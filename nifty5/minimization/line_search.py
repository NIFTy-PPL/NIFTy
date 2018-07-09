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
import abc
from ..utilities import NiftyMetaBase


class LineSearch(NiftyMetaBase()):
    """Class for determining the optimal step size along some descent
       direction.

    Parameters
    ----------
    preferred_initial_step_size : float, optional
        Newton-based methods should intialize this to 1.
    """

    def __init__(self, preferred_initial_step_size=None):
        self.preferred_initial_step_size = preferred_initial_step_size

    @abc.abstractmethod
    def perform_line_search(self, energy, pk, f_k_minus_1=None):
        """Find step size and advance.

        Determines a good step size and advances the current estimate
        by this step size in the search direction.

        Parameters
        ----------
        energy : Energy
            Energy object from which we will calculate the energy and the
            gradient at a specific point.
        pk : Field
            Vector pointing into the search direction.
        f_k_minus_1 : float, optional
            Value of the fuction (which is being minimized) at the k-1
            iteration of the line search procedure. (Default: None)

        Returns
        -------
        Energy
            The new Energy object on the new position.
        bool
            whether the line search was considered successful or not
        """
        raise NotImplementedError
