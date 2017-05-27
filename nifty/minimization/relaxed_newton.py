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
from .line_searching import LineSearchStrongWolfe


class RelaxedNewton(DescentMinimizer):

    def __init__(self, line_searcher=LineSearchStrongWolfe(), callback=None,
                 convergence_tolerance=1E-4, convergence_level=3,
                 iteration_limit=None):
        super(RelaxedNewton, self).__init__(
                                line_searcher=line_searcher,
                                callback=callback,
                                convergence_tolerance=convergence_tolerance,
                                convergence_level=convergence_level,
                                iteration_limit=iteration_limit)

        self.line_searcher.prefered_initial_step_size = 1.

    def get_descend_direction(self, energy):
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
        descend_direction : Field
           Returns the descent direction with proposed step length. In a
           quadratic potential this corresponds to the optimal step.

        """
        gradient = energy.gradient
        curvature = energy.curvature
        descend_direction = curvature.inverse_times(gradient)
        return descend_direction * -1
