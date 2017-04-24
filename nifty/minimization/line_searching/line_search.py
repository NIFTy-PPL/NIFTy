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

import abc

from keepers import Loggable

from nifty import LineEnergy


class LineSearch(Loggable, object):
    """
    Class for finding a step size.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):

        """
        Parameters
        ----------

        f : callable f(x, *args)
            Objective function.

        fprime : callable f'(x, *args)
            Objective functions gradient.

        f_args : tuple (optional)
            Additional arguments passed to objective function and its
            derivation.
        """

        self.line_energy = None
        self.f_k_minus_1 = None
        self.prefered_initial_step_size = None

    def _set_line_energy(self, energy, pk, f_k_minus_1=None):
        """
        Set the coordinates for a new line search.

        Parameters
        ----------
        xk : ndarray, d2o, field
            Starting point.

        pk : ndarray, d2o, field
            Unit vector in search direction.

        f_k : float (optional)
            Function value f(x_k).

        fprime_k : ndarray, d2o, field (optional)
            Function value fprime(xk).

        """
        self.line_energy = LineEnergy(position=0.,
                                      energy=energy,
                                      line_direction=pk)
        if f_k_minus_1 is not None:
            f_k_minus_1 = f_k_minus_1.copy()
        self.f_k_minus_1 = f_k_minus_1

    @abc.abstractmethod
    def perform_line_search(self, energy, pk, f_k_minus_1=None):
        raise NotImplementedError
