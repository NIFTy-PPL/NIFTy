# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

from .quasi_newton_minimizer import QuasiNewtonMinimizer
from .line_searching import LineSearchStrongWolfe


class RelaxedNewton(QuasiNewtonMinimizer):
    """ A implementation of the relaxed Newton minimization scheme.
    The relaxed Newton minimization exploits gradient and curvature information to
    propose a step. A linesearch optimizes along this direction.

    Parameter
    ---------
    line_searcher : LineSearch,
        An implementation of a line-search algorithm.
    callback :
    convergence_tolerance : float,
        Specifies the required accuracy for convergence. (default : 10e-4)
    convergence_level : integer
        Specifies the demanded level of convergence. (default : 3)
    iteration_limit : integer
        Limiting the maximum number of steps. (default : None)

    """
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

    def _get_descend_direction(self, energy):
        """ Calculates the descent direction according to a Newton scheme.
        The descent direction is determined by weighting the gradient at the
        current parameter position with the inverse local curvature, provided by the
        Energy object.

        Parameters
        ----------
        energy : Energy
            The energy object providing implementations of the to be minimized function,
            its gradient and curvature.

        Returns
        -------
        out : Field
           Returns the descent direction with proposed step length. In a quadratic
            potential this corresponds to the optimal step.

        """
        gradient = energy.gradient
        curvature = energy.curvature
        descend_direction = curvature.inverse_times(gradient)
        return descend_direction * -1

