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

from keepers import Loggable


class Energy(Loggable, object):
    """ The Energy object provides the structure required for minimization schemes.

    It is the abstract implementation of a scalar function with its gradient and curvature at some position.

    Parameters
    ----------
    position : Field
        The parameter of the scalar function and its first and second derivative.

    Attributes
    ----------
    position : Field
        The Field location in parameter space where value, gradient and curvature is evaluated.
    value : float
        The evaluation of the energy functional at given position.
    gradient : Field
        The gradient at given position in parameter direction.
    curvature : InvertibleOperator
        An implicit operator encoding the curvature at given position.

    Raises
    ------
    NotImplementedError
        Raised if
            * value, gradient or curvature is called
    AttributeError
        Raised if
            * copying of the position fails

    Notes
    -----
    The Energy object gives the blueprint how to formulate the model in order to apply
    various inference schemes. The functions value, gradient and curvature have to be
    implemented according to the concrete inference problem.

    Memorizing the evaluations of some quantities minimizes the computational effort
    for multiple calls.


    """
    def __init__(self, position):
        self._cache = {}
        try:
            position = position.copy()
        except AttributeError:
            pass
        self.position = position

    def at(self, position):
        """ Initializes and returns new Energy object at new position.

        Parameters
        ----------
        position : Field
            Parameter for the new Energy object.

        Returns
        -------
        out : Energy
            Energy object at new position.

        """
        return self.__class__(position)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position = position

    @property
    def value(self):
        raise NotImplementedError

    @property
    def gradient(self):
        raise NotImplementedError

    @property
    def curvature(self):
        raise NotImplementedError
