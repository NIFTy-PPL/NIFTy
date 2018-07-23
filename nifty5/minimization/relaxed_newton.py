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
from .descent_minimizer import DescentMinimizer
from .line_search_strong_wolfe import LineSearchStrongWolfe


class RelaxedNewton(DescentMinimizer):
    """ Calculates the descent direction according to a Newton scheme.

    The descent direction is determined by weighting the gradient at the
    current parameter position with the inverse local metric.
    """

    def __init__(self, controller, line_searcher=None):
        if line_searcher is None:
            line_searcher = LineSearchStrongWolfe(
                preferred_initial_step_size=1.)
        super(RelaxedNewton, self).__init__(controller=controller,
                                            line_searcher=line_searcher)

    def get_descent_direction(self, energy):
        return -energy.metric.inverse_times(energy.gradient)
