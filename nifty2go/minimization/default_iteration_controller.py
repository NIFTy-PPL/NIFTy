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
from .iteration_controller import IterationController


class DefaultIterationController(IterationController):
    def __init__(self, tol_abs_gradnorm=None, tol_rel_gradnorm=None,
                 convergence_level=1, iteration_limit=None, name=None,
                 verbose=None):
        super(DefaultIterationController, self).__init__()
        self._tol_abs_gradnorm = tol_abs_gradnorm
        self._tol_rel_gradnorm = tol_rel_gradnorm
        self._convergence_level = convergence_level
        self._iteration_limit = iteration_limit
        self._name = name
        self._verbose = verbose

    def start(self, energy):
        self._itcount = -1
        self._ccount = 0
        if self._tol_rel_gradnorm is not None:
            self._tol_rel_gradnorm_now = self._tol_rel_gradnorm \
                                       * energy.gradient_norm
        return self.check(energy)

    def check(self, energy):
        self._itcount += 1

        if self._tol_abs_gradnorm is not None:
            if energy.gradient_norm <= self._tol_abs_gradnorm:
                self._ccount += 1
        if self._tol_rel_gradnorm is not None:
            if energy.gradient_norm <= self._tol_rel_gradnorm_now:
                self._ccount += 1

        # report
        if self._verbose:
            msg = ""
            if self._name is not None:
                msg += self._name+":"
            msg += " Iteration #" + str(self._itcount)
            msg += " energy=" + str(energy.value)
            msg += " gradnorm=" + str(energy.gradient_norm)
            msg += " clvl=" + str(self._ccount)
            print(msg)
            # self.logger.info(msg)

        # Are we done?
        if self._iteration_limit is not None:
            if self._itcount >= self._iteration_limit:
                return self.CONVERGED
        if self._ccount >= self._convergence_level:
            return self.CONVERGED

        return self.CONTINUE
