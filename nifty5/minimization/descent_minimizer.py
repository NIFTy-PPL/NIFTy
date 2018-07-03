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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import division
import abc
from .minimizer import Minimizer
from .line_search_strong_wolfe import LineSearchStrongWolfe
from ..logger import logger


class DescentMinimizer(Minimizer):
    """ A base class used by gradient methods to find a local minimum.

    Descent minimization methods are used to find a local minimum of a scalar
    function by following a descent direction. This class implements the
    minimization procedure once a descent direction is known. The descent
    direction has to be implemented separately.

    Parameters
    ----------
    controller : IterationController
        Object that decides when to terminate the minimization.
    line_searcher : callable *optional*
        Function which infers the step size in the descent direction
        (default : LineSearchStrongWolfe()).
    """

    def __init__(self, controller, line_searcher=LineSearchStrongWolfe()):
        super(DescentMinimizer, self).__init__()
        self._controller = controller
        self.line_searcher = line_searcher

    def __call__(self, energy):
        """ Performs the minimization of the provided Energy functional.

        Parameters
        ----------
        energy : Energy
           Energy object which provides value, gradient and curvature at a
           specific position in parameter space.

        Returns
        -------
        Energy
            Latest `energy` of the minimization.
        int
            Can be controller.CONVERGED or controller.ERROR

        Notes
        -----
        The minimization is stopped if
            * the controller returns controller.CONVERGED or controller.ERROR,
            * a perfectly flat point is reached,
            * according to the line-search the minimum is found,
        """
        f_k_minus_1 = None
        controller = self._controller
        status = controller.start(energy)
        if status != controller.CONTINUE:
            return energy, status

        while True:
            # check if position is at a flat point
            if energy.gradient_norm == 0:
                return energy, controller.CONVERGED

            # compute a step length that reduces energy.value sufficiently
            new_energy, success = self.line_searcher.perform_line_search(
                energy=energy, pk=self.get_descent_direction(energy),
                f_k_minus_1=f_k_minus_1)
            if not success:
                self.reset()

            f_k_minus_1 = energy.value

            if new_energy.value > energy.value:
                logger.error("Error: Energy has increased")
                return energy, controller.ERROR

            if new_energy.value == energy.value:
                logger.warning(
                    "Warning: Energy has not changed. Assuming convergence...")
                return new_energy, controller.CONVERGED

            energy = new_energy
            status = self._controller.check(energy)
            if status != controller.CONTINUE:
                return energy, status

    def reset(self):
        pass

    @abc.abstractmethod
    def get_descent_direction(self, energy):
        """ Calculates the next descent direction.

        Parameters
        ----------
        energy : Energy
            An instance of the Energy class which shall be minimized. The
            position of `energy` is used as the starting point of minimization.

        Returns
        -------
        Field
           The descent direction.
        """
        raise NotImplementedError
