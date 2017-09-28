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

import abc
from ..nifty_meta import NiftyMeta

from keepers import Loggable
from future.utils import with_metaclass


class Minimizer(
        with_metaclass(NiftyMeta, type('NewBase', (Loggable, object), {}))):
    """ A base class used by all minimizers.
    """
    def __init__(self, controller):
        self._controller = controller
        # overwrite the logger from controller to show contex correctly
        self._controller.logger = self.logger

    @property
    def controller(self):
        return self._controller

    @abc.abstractmethod
    def __call__(self, energy):
        """ Performs the minimization of the provided Energy functional.

        Parameters
        ----------
        energy : Energy object
           Energy object which provides value, gradient and curvature at a
           specific position in parameter space.

        Returns
        -------
        energy : Energy object
            Latest `energy` of the minimization.
        status : integer
        """

        raise NotImplementedError
