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

from .descent_minimizer import DescentMinimizer
from .line_search_strong_wolfe import LineSearchStrongWolfe


class RelaxedNewton(DescentMinimizer):
    def __init__(self, controller, line_searcher=LineSearchStrongWolfe()):
        super(RelaxedNewton, self).__init__(controller=controller,
                                            line_searcher=line_searcher)
        # FIXME: this does not look idiomatic
        self.line_searcher.preferred_initial_step_size = 1.

    def get_descent_direction(self, energy):
        """ Calculates the descent direction according to a Newton scheme.

        The descent direction is determined by weighting the gradient at the
        current parameter position with the inverse local curvature, provided
        by the Energy object.


        Parameters
        ----------
        energy : Energy
            An instance of the Energy class which shall be minized. The
            position of `energy` is used as the starting point of minization.

        Returns
        -------
        descent_direction : Field
           Returns the descent direction with proposed step length. In a
           quadratic potential this corresponds to the optimal step.
        """
        return -energy.curvature.inverse_times(energy.gradient)
