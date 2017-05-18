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
    """ Evaluates an underlying Energy along a certain line direction.

    Given an Energy class and a line direction, its position is parametrized by
    a scalar step size along the descent direction relative to a zero point.

    Parameters
    ----------
    position : float
        The step length parameter along the given line direction.
    energy : Energy
        The Energy object which will be evaluated along the given direction.
    line_direction : Field
        Direction used for line evaluation.
    zero_point :  Field *optional*
        Fixing the zero point of the line restriction. Used to memorize this
        position in new initializations. By the default the current position
        of the supplied `energy` instance is used (default : None).

    Attributes
    ----------
    position : float
        The position along the given line direction relative to the zero point.
    value : float
        The value of the energy functional at given `position`.
    gradient : float
        The gradient of the underlying energy instance along the line direction
        projected on the line direction.
    curvature : callable
        A positive semi-definite operator or function describing the curvature
        of the potential at given `position`.
    line_direction : Field
        Direction along which the movement is restricted. Does not have to be
        normalized.
    energy : Energy
        The underlying Energy at the `position` along the line direction.

    Raises
    ------
    NotImplementedError
        Raised if
            * value, gradient or curvature of the attribute energy are not
              implemented.

    Notes
    -----
    The LineEnergy is used in minimization schemes in order perform line
    searches. It describes an underlying Energy which is restricted along one
    direction, only requiring the step size parameter to determine a new
    position.

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
        """ Returns LineEnergy at new position, memorizing the zero point.

        Parameters
        ----------
        position : float
            Parameter for the new position on the line direction.

        Returns
        -------
        out : LineEnergy
            LineEnergy object at new position with same zero point as `self`.

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
