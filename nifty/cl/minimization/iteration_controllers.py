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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import functools
from time import time

import numpy as np

from ..logger import logger
from ..utilities import NiftyMeta


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

    For analyzing minimization procedures IterationControllers can log energy
    values together with the respective time stamps. In order to activate this
    feature `enable_logging()` needs to be called.
    """

    CONVERGED, CONTINUE, ERROR = list(range(3))

    def __init__(self):
        self._history = None

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

    def enable_logging(self):
        """Enables the logging functionality. If the log has been populated
        before, it stays as it is."""
        if self._history is None:
            self._history = EnergyHistory()

    def disable_logging(self):
        """Disables the logging functionality. If the log has been populated
        before, it is dropped."""
        self._history = None

    @property
    def history(self):
        return self._history


class EnergyHistory:
    def __init__(self):
        self._lst = []

    def append(self, x):
        if len(x) != 2:
            raise ValueError
        self._lst.append((float(x[0]), float(x[1])))

    def reset(self):
        self._lst = []

    def __getitem__(self, i):
        return self._lst[i]

    @property
    def time_stamps(self):
        return [x for x, _ in self._lst]

    @property
    def energy_values(self):
        return [x for _, x in self._lst]

    def __add__(self, other):
        if not isinstance(other, EnergyHistory):
            return NotImplemented
        res = EnergyHistory()
        res._lst = self._lst + other._lst
        return res

    def __iadd__(self, other):
        if not isinstance(other, EnergyHistory):
            return NotImplemented
        self._lst += other._lst
        return self

    def __len__(self):
        return len(self._lst)


def append_history(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        hist = args[0].history
        if isinstance(hist, EnergyHistory):
            hist.append((time(), args[1].value))
        return func(*args, **kwargs)
    return wrapper


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
    """

    def __init__(self, tol_abs_gradnorm=None, tol_rel_gradnorm=None,
                 convergence_level=1, iteration_limit=None, name=None):
        super(GradientNormController, self).__init__()
        self._tol_abs_gradnorm = tol_abs_gradnorm
        self._tol_rel_gradnorm = tol_rel_gradnorm
        self._convergence_level = convergence_level
        self._iteration_limit = iteration_limit
        self._name = name

    @append_history
    def start(self, energy):
        self._itcount = -1
        self._ccount = 0
        if self._tol_rel_gradnorm is not None:
            self._tol_rel_gradnorm_now = self._tol_rel_gradnorm \
                                       * energy.gradient_norm
        return self.check(energy)

    @append_history
    def check(self, energy):
        self._itcount += 1

        inclvl = False
        if self._tol_abs_gradnorm is not None:
            if energy.gradient_norm <= self._tol_abs_gradnorm:
                inclvl = True
        if self._tol_rel_gradnorm is not None:
            if energy.gradient_norm <= self._tol_rel_gradnorm_now:
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
                        energy.gradient_norm, self._ccount))

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
        super(GradInfNormController, self).__init__()
        self._tol = tol
        self._convergence_level = convergence_level
        self._iteration_limit = iteration_limit
        self._name = name

    @append_history
    def start(self, energy):
        self._itcount = -1
        self._ccount = 0
        return self.check(energy)

    @append_history
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
    """An iteration controller checking (mainly) the relative energy change
    from one iteration to the next.

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
        super(DeltaEnergyController, self).__init__()
        self._tol_rel_deltaE = tol_rel_deltaE
        self._convergence_level = convergence_level
        self._iteration_limit = iteration_limit
        self._name = name

    @append_history
    def start(self, energy):
        self._itcount = -1
        self._ccount = 0
        self._Eold = 0.
        return self.check(energy)

    @append_history
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


class AbsDeltaEnergyController(IterationController):
    """An iteration controller checking (mainly) the energy change from one
    iteration to the next.

    Parameters
    ----------
    deltaE : float
        If the difference between the last and current energies is below this
        value, the convergence counter will be increased in this iteration.
    convergence_level : int, default=1
        The number which the convergence counter must reach before the
        iteration is considered to be converged
    iteration_limit : int, optional
        The maximum number of iterations that will be carried out.
    name : str, optional
        if supplied, this string and some diagnostic information will be
        printed after every iteration
    """

    def __init__(self, deltaE, convergence_level=1, iteration_limit=None,
                 name=None):
        super(AbsDeltaEnergyController, self).__init__()
        self._deltaE = deltaE
        self._convergence_level = convergence_level
        self._iteration_limit = iteration_limit
        self._name = name

    @append_history
    def start(self, energy):
        self._itcount = -1
        self._ccount = 0
        self._Eold = 0.
        return self.check(energy)

    @append_history
    def check(self, energy):
        self._itcount += 1

        inclvl = False
        Eval = energy.value
        diff = abs(self._Eold-Eval)
        if self._itcount > 0:
            if diff < self._deltaE:
                inclvl = True
        self._Eold = Eval
        if inclvl:
            self._ccount += 1
        else:
            self._ccount = max(0, self._ccount-1)

        # report
        if self._name is not None:
            logger.info(
                "{}: Iteration #{} energy={:.6E} diff={:.6E} crit={:.1E} clvl={}"
                .format(self._name, self._itcount, Eval, diff, self._deltaE,
                        self._ccount))

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


class StochasticAbsDeltaEnergyController(IterationController):
    """Check the standard deviation over a period of iterations.

    Convergence is reported once this quantity falls below the given threshold.


    Parameters
    ----------
    deltaE : float
        If the standard deviation of the last energies is below this
        value, the convergence counter will be increased in this iteration.
    convergence_level : int, optional
        The number which the convergence counter must reach before the
        iteration is considered to be converged. Defaults to 1.
    iteration_limit : int, optional
        The maximum number of iterations that will be carried out.
    name : str, optional
        If supplied, this string and some diagnostic information will be
        printed after every iteration.
    memory_length : int, optional
        The number of last energies considered for determining convergence,
        defaults to 10.
    """

    def __init__(self, deltaE, convergence_level=1, iteration_limit=None,
                 name=None, memory_length=10):
        super(StochasticAbsDeltaEnergyController, self).__init__()
        self._deltaE = deltaE
        self._convergence_level = convergence_level
        self._iteration_limit = iteration_limit
        self._name = name
        self.memory_length = memory_length

    @append_history
    def start(self, energy):
        self._itcount = -1
        self._ccount = 0
        self._memory = []
        return self.check(energy)

    @append_history
    def check(self, energy):
        self._itcount += 1

        inclvl = False
        Eval = energy.value
        self._memory.append(Eval)
        if len(self._memory) > self.memory_length:
            self._memory = self._memory[1:]
        diff = np.std(self._memory)
        if self._itcount > 0:
            if diff < self._deltaE:
                inclvl = True
        if inclvl:
            self._ccount += 1
        else:
            self._ccount = max(0, self._ccount-1)

        # report
        if self._name is not None:
            logger.info(
                "{}: Iteration #{} energy={:.6E} diff={:.6E} crit={:.1E} clvl={}"
                .format(self._name, self._itcount, Eval, diff, self._deltaE,
                        self._ccount))

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
