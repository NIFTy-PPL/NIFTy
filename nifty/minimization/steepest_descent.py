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

from .descent_minimizer import DescentMinimizer


class SteepestDescent(DescentMinimizer):
    def _get_descend_direction(self, energy):
        descend_direction = energy.gradient
        norm = descend_direction.norm()
        if norm != 1:
            return descend_direction / -norm
        else:
            return descend_direction * -1
