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
# Copyright(C) 2013-2023 Max-Planck-Society
# Authors: Philipp Arras, Vincent Eberle
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .structured_domain import StructuredDomain

class IRGSpace(StructuredDomain):
    """Represents non-equidistantly binned and non-periodic one-dimensional spaces.

    Parameters
    ----------
    coordinates : np.ndarray
        Must be sorted and strictly ascending.
    """

    _needed_for_hash = ["_coordinates"]

    def __init__(self, coordinates):
        bb = np.array(coordinates)
        if bb.ndim != 1:
            raise TypeError
        if np.any(np.diff(bb) <= 0.0):
            raise ValueError("Coordinates must be sorted and strictly ascending")
        self._coordinates = tuple(bb)

    def __repr__(self):
        return f"IRGSpace(coordinates={self._coordinates})"

    @property
    def harmonic(self):
        """bool : Always False for this class."""
        return False

    @property
    def shape(self):
        return (len(self._coordinates),)

    @property
    def size(self):
        return self.shape[0]

    @property
    def scalar_dvol(self):
        return None

    @property
    def dvol(self):
        """Assume that the coordinates are the center of symmetric pixels."""
        return np.diff(self.binbounds())

    def binbounds(self):
        if len(self._coordinates) == 1:
            return np.array([-np.inf, np.inf])
        c = np.array(self._coordinates)
        bounds = np.empty(self.size + 1)
        bounds[1:-1] = c[:-1] + 0.5*np.diff(c)
        bounds[0] = c[0] - 0.5*(c[1] - c[0])
        bounds[-1] = c[-1] + 0.5*(c[-1] - c[-2])
        return bounds

    @property
    def distances(self):
        return np.diff(self._coordinates)

    @property
    def regular(self):
        if self.size == 1:
            return True
        return np.all(self.distances[0] == self.distances)

    @property
    def coordinates(self):
        return self._coordinates
