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
    """A Energy object restricting an underlying Energy along only some line direction.
    Given some Energy and line direction, its position is parametrized by a scalar
    step size along the descent direction.

    Parameters
    ----------
    position : float
        The step length parameter along the given line direction.
    energy : Energy
        The Energy object which will be restricted along the given line direction
    line_direction : Field, float
        Line direction restricting the Energy.
    zero_point :  Field, float
        Fixing the zero point of the line restriction. Used to memorize this position in new
        initializations (default : None)

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
    line_direction : Field
        Direction along which the movement is restricted. Does not have to be normalized.
    energy : Energy
        The underlying Energy at the resulting position along the line according to the step length.

    Raises
    ------
    NotImplementedError
        Raised if
            * value, gradient or curvature of the attribute energy is not implemented.

    Notes
    -----
    The LineEnergy is used in minimization schemes in order to determine the step size along
    some descent direction using a line search. It describes an underlying Energy which is restricted
    along one direction, only requiring the step size parameter to determine a new position.


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
        """ Initializes and returns new LineEnergy object at new position, memorizing the zero point.

        Parameters
        ----------
        position : float
            Parameter for the new position.

        Returns
        -------
        out : LineEnergy
            LineEnergy object at new position with same zero point.

        """
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
