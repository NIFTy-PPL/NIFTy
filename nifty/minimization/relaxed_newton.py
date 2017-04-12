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
        gradient = energy.gradient
        curvature = energy.curvature
        descend_direction = curvature.inverse_times(gradient)
        return descend_direction * -1
        #norm = descend_direction.norm()
#        if norm != 1:
#            return descend_direction / -norm
#        else:
#            return descend_direction * -1
