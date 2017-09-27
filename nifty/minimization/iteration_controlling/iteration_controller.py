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

from builtins import range
import abc
from ..nifty_meta import NiftyMeta

from keepers import Loggable
from future.utils import with_metaclass


class IterationController(
        with_metaclass(NiftyMeta, type('NewBase', (Loggable, object), {}))):
    """The abstract base class for all iteration controllers.
    An iteration controller is an object that monitors the progress of a
    minimization iteration. At the begin of the minimization, its start()
    method is called with the energy object at the initial position.
    Afterwards, its check() method is called during every iteration step with
    the energy object describing the current position.
    Based on that information, the iteration controller has to decide whether
    iteration needs to progress further (in this case it returns CONTINUE), or
    if sufficient convergence has been reached (in this case it returns
    CONVERGED), or if some error has been detected (then it returns ERROR).

    The concrete convergence criteria can be chosen by inheriting from this
    class; the implementer has full flexibility to use whichever criteria are
    appropriate for a particular problem - as ong as they can be computed from
    the information passed to the controller during the iteration process.
    """

    CONVERGED, CONTINUE, STOPPED, ERROR = list(range(4))

    def __init__(self):
        self._iteration_count = 0
        self._convergence_count = 0

    @property
    def iteration_count(self):
        return self._iteration_count

    @property
    def convergence_count(self):
        return self._convergence_count

    @abc.abstractmethod
    def reset(self, energy):
        """
        Parameters
        ----------
        energy : Energy object
           Energy object at the start of the iteration

        """

        raise NotImplementedError

    @abc.abstractmethod
    def check(self, energy):
        """
        Parameters
        ----------
        energy : Energy object
           Energy object at the current position

        Returns
        -------
        status : integer status, can be CONVERGED, CONTINUE or ERROR
        """

        raise NotImplementedError
