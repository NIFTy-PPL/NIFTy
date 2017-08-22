from __future__ import division
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

from __future__ import division
from .descent_minimizer import DescentMinimizer


class SteepestDescent(DescentMinimizer):
    def get_descent_direction(self, energy):
        """ Implementation of the steepest descent minimization scheme.

        Also known as 'gradient descent'. This algorithm simply follows the
        functionals gradient for minization.

        Parameters
        ----------
        energy : Energy
            An instance of the Energy class which shall be minized. The
            position of `energy` is used as the starting point of minization.

        Returns
        -------
        descent_direction : Field
            Returns the descent direction.

        """

        return -energy.gradient
