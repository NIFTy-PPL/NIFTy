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

from .energy import Energy


class LineEnergy(Energy):
    """A Energy object restricting an underlying Energy along some descent direction.
    Given some Energy and descent direction, its position is parametrized by a scalar
    step size along the descent direction.

    Parameters
    ----------
    position : float
        The step length parameter along the given line direction.

    Attributes
    ----------
    position :  float
        The step length along the given line direction.
    value : float
        The evaluation of the energy functional at given position.
    gradient : float
        The gradient along the line direction projected on the current line position.
    curvature : callable
        A positive semi-definite operator or function describing the curvature of the potential
        at given position.
    line_direction : field
        Direction along which the movement is restricted. Does not have to be normalized.
    energy : Energy
        The underlying Energy at the resulting position along the line according to the step length.


    """
    def __init__(self, position, energy, line_direction, zero_point=None):
        super(LineEnergy, self).__init__(position=position)
        self.line_direction = line_direction

        if zero_point is None:
            zero_point = energy.position
        self._zero_point = zero_point

        position_on_line = self._zero_point + self.position*line_direction
        self.energy = energy.at(position=position_on_line)

    def at(self, position):
        return self.__class__(position,
                              self.energy,
                              self.line_direction,
                              zero_point=self._zero_point)

    @property
    def value(self):
        return self.energy.value

    @property
    def gradient(self):
        return self.energy.gradient.dot(self.line_direction)

    @property
    def curvature(self):
        return self.energy.curvature
