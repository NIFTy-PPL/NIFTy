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

from __future__ import print_function

from keepers import Loggable
from ..nifty_meta import NiftyMeta
from future.utils import with_metaclass


class LineEnergy((with_metaclass(NiftyMeta,
                                 type('NewBase', (Loggable, object), {})))):
    """ Evaluates an underlying Energy along a certain line direction.

    Given an Energy class and a line direction, its position is parametrized by
    a scalar step size along the descent direction relative to a zero point.

    Parameters
    ----------
    line_position : float
        Defines the full spatial position of this energy via
        self.energy.position = zero_point + line_position*line_direction
    energy : Energy
        The Energy object which will be evaluated along the given direction.
    line_direction : Field
        Direction used for line evaluation. Does not have to be normalized.
    offset :  float *optional*
        Indirectly defines the zero point of the line via the equation
        energy.position = zero_point + offset*line_direction
        (default : 0.).

    Attributes
    ----------
    line_position : float
        The position along the given line direction relative to the zero point.
    value : float
        The value of the energy functional at the given position
    directional_derivative : float
        The directional derivative at the given position
    line_direction : Field
        Direction along which the movement is restricted. Does not have to be
        normalized.
    energy : Energy
        The underlying Energy at the given position

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

    def __init__(self, line_position, energy, line_direction, offset=0.):
        super(LineEnergy, self).__init__()
        self._line_position = float(line_position)
        self._line_direction = line_direction

        if self._line_position == float(offset):
            self.energy = energy
        else:
            pos = energy.position \
                + (self._line_position-float(offset))*self._line_direction
            self.energy = energy.at(position=pos)

    def at(self, line_position):
        """ Returns LineEnergy at new position, memorizing the zero point.

        Parameters
        ----------
        line_position : float
            Parameter for the new position on the line direction.

        Returns
        -------
            LineEnergy object at new position with same zero point as `self`.

        """

        return self.__class__(line_position,
                              self.energy,
                              self.line_direction,
                              offset=self.line_position)

    @property
    def value(self):
        return self.energy.value

    @property
    def line_position(self):
        return self._line_position

    @property
    def line_direction(self):
        return self._line_direction

    @property
    def directional_derivative(self):
        res = self.energy.gradient.vdot(self.line_direction)
        if abs(res.imag) / max(abs(res.real), 1.) > 1e-12:
            self.logger.warn("directional derivative has non-negligible "
                             "imaginary part:", res)
        return res.real
