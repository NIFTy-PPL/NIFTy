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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..logger import logger
from ..utilities import NiftyMeta
import numpy as np


class IterationController(metaclass=NiftyMeta):
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
    appropriate for a particular problem - as long as they can be computed from
    the information passed to the controller during the iteration process.
    """

    CONVERGED, CONTINUE, ERROR = list(range(3))

    def start(self, energy):
        """Starts the iteration.

        Parameters
        ----------
        energy : Energy object
           Energy object at the start of the iteration

        Returns
        -------
        status : integer status, can be CONVERGED, CONTINUE or ERROR
        """
        raise NotImplementedError

    def check(self, energy):
        """Checks the state of the iteration. Called after every step.

        Parameters
        ----------
        energy : Energy object
           Energy object at the start of the iteration

        Returns
        -------
        status : integer status, can be CONVERGED, CONTINUE or ERROR
        """
        raise NotImplementedError


class GradientNormController(IterationController):
    """An iteration controller checking (mainly) the L2 gradient norm.

    Parameters
    ----------
    tol_abs_gradnorm : float, optional
        If the L2 norm of the energy gradient is below this value, the
        convergence counter will be increased in this iteration.
    tol_rel_gradnorm : float, optional
        If the L2 norm of the energy gradient divided by its initial L2 norm
        is below this value, the convergence counter will be increased in this
        iteration.
    convergence_level : int, default=1
        The number which the convergence counter must reach before the
        iteration is considered to be converged
    iteration_limit : int, optional
        The maximum number of iterations that will be carried out.
    name : str, optional
        if supplied, this string and some diagnostic information will be
        printed after every iteration
    p : float
        Order of norm, default is the 2-Norm (p=2)
    """

    def __init__(self, tol_abs_gradnorm=None, tol_rel_gradnorm=None,
                 convergence_level=1, iteration_limit=None, name=None, p=2):
        self._tol_abs_gradnorm = tol_abs_gradnorm
        self._tol_rel_gradnorm = tol_rel_gradnorm
        self._convergence_level = convergence_level
        self._iteration_limit = iteration_limit
        self._name = name
        self._p = p

    def start(self, energy):
        self.energyhistory = []
        self._itcount = -1
        self._ccount = 0
        if self._tol_rel_gradnorm is not None:
            self._tol_rel_gradnorm_now = self._tol_rel_gradnorm * self._norm(energy)
        return self.check(energy)

    def _norm(self, energy):
        # FIXME Only p=2 norm is cached in energy class
        if self._p == 2:
            return energy.gradient_norm
        return energy.gradient.norm(self._p)

    def check(self, energy):
        self._itcount += 1

        inclvl = False
        if self._tol_abs_gradnorm is not None:
            if self._norm(energy) <= self._tol_abs_gradnorm:
                inclvl = True
        if self._tol_rel_gradnorm is not None:
            if self._norm(energy) <= self._tol_rel_gradnorm_now:
                inclvl = True
        if inclvl:
            self._ccount += 1
        else:
            self._ccount = max(0, self._ccount-1)

        # report
        if self._name is not None:
            logger.info(
                "{}: Iteration #{} energy={:.6E} gradnorm={:.2E} clvl={}"
                .format(self._name, self._itcount, energy.value,
                        self._norm(energy), self._ccount))
        self.energyhistory.append(energy.value)

        # Are we done?
        if self._iteration_limit is not None:
            if self._itcount >= self._iteration_limit:
                logger.warning(
                    "{}Iteration limit reached. Assuming convergence"
                    .format("" if self._name is None else self._name+": "))
                return self.CONVERGED
        if self._ccount >= self._convergence_level:
            return self.CONVERGED

        return self.CONTINUE


class GradInfNormController(IterationController):
    """An iteration controller checking (mainly) the L_infinity gradient norm.

    Parameters
    ----------
    tol : float
        If the L_infinity norm of the energy gradient is below this value, the
        convergence counter will be increased in this iteration.
    convergence_level : int, default=1
        The number which the convergence counter must reach before the
        iteration is considered to be converged
    iteration_limit : int, optional
        The maximum number of iterations that will be carried out.
    name : str, optional
        if supplied, this string and some diagnostic information will be
        printed after every iteration
    """

    def __init__(self, tol, convergence_level=1, iteration_limit=None,
                 name=None):
        self._tol = tol
        self._convergence_level = convergence_level
        self._iteration_limit = iteration_limit
        self._name = name

    def start(self, energy):
        self._itcount = -1
        self._ccount = 0
        return self.check(energy)

    def check(self, energy):
        self._itcount += 1

        crit = energy.gradient.norm(np.inf) / abs(energy.value)
        if self._tol is not None and crit <= self._tol:
            self._ccount += 1
        else:
            self._ccount = max(0, self._ccount-1)

        # report
        if self._name is not None:
            logger.info(
                "{}: Iteration #{} energy={:.6E} crit={:.2E} clvl={}"
                .format(self._name, self._itcount, energy.value,
                        crit, self._ccount))

        # Are we done?
        if self._iteration_limit is not None:
            if self._itcount >= self._iteration_limit:
                logger.warning(
                    "{} Iteration limit reached. Assuming convergence"
                    .format("" if self._name is None else self._name+": "))
                return self.CONVERGED
        if self._ccount >= self._convergence_level:
            return self.CONVERGED

        return self.CONTINUE


class DeltaEnergyController(IterationController):
    """An iteration controller checking (mainly) the energy change from one
    iteration to the next.

    Parameters
    ----------
    tol_rel_deltaE : float
        If the difference between the last and current energies divided by
        the current energy is below this value, the convergence counter will
        be increased in this iteration.
    convergence_level : int, default=1
        The number which the convergence counter must reach before the
        iteration is considered to be converged
    iteration_limit : int, optional
        The maximum number of iterations that will be carried out.
    name : str, optional
        if supplied, this string and some diagnostic information will be
        printed after every iteration
    """

    def __init__(self, tol_rel_deltaE, convergence_level=1,
                 iteration_limit=None, name=None):
        self._tol_rel_deltaE = tol_rel_deltaE
        self._convergence_level = convergence_level
        self._iteration_limit = iteration_limit
        self._name = name

    def start(self, energy):
        self._itcount = -1
        self._ccount = 0
        self._Eold = 0.
        return self.check(energy)

    def check(self, energy):
        self._itcount += 1

        inclvl = False
        Eval = energy.value
        rel = abs(self._Eold-Eval)/max(abs(self._Eold), abs(Eval))
        if self._itcount > 0:
            if rel < self._tol_rel_deltaE:
                inclvl = True
        self._Eold = Eval
        if inclvl:
            self._ccount += 1
        else:
            self._ccount = max(0, self._ccount-1)

        # report
        if self._name is not None:
            logger.info(
                "{}: Iteration #{} energy={:.6E} reldiff={:.6E} clvl={}"
                .format(self._name, self._itcount, Eval, rel, self._ccount))

        # Are we done?
        if self._iteration_limit is not None:
            if self._itcount >= self._iteration_limit:
                logger.warning(
                    "{} Iteration limit reached. Assuming convergence"
                    .format("" if self._name is None else self._name+": "))
                return self.CONVERGED
        if self._ccount >= self._convergence_level:
            return self.CONVERGED

        return self.CONTINUE
