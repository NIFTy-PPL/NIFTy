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
    """Class for determining the optimal step size along some descent direction.
    
    Initialize the line search procedure which can be used by a specific line
    search method. Its finds the step size in a specific direction in the
    minimization process.
    
    Attributes
    ----------
    line_energy : LineEnergy Object
        LineEnergy object from which we can extract energy at a specific point.
    f_k_minus_1 : Field
        Value of the field at the k-1 iteration of the line search procedure.
    prefered_initial_step_size : float
        Initial guess for the step length.
    
    """
    
    __metaclass__ = abc.ABCMeta

    def __init__(self):

        

        self.line_energy = None
        self.f_k_minus_1 = None
        self.prefered_initial_step_size = None

    def _set_line_energy(self, energy, pk, f_k_minus_1=None):
        """Set the coordinates for a new line search.

        Parameters
        ----------
        energy : Energy object
            Energy object from which we can calculate the energy, gradient and
            curvature at a specific point.        
        pk : Field
            Unit vector pointing into the search direction.
        f_k_minus_1 : float
            Value of the fuction (energy) which will be minimized at the k-1 
            iteration of the line search procedure. (Default: None)
            
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
