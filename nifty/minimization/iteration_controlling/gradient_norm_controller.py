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


class GradientNormController(IterationController):
    def __init__(self, tol_abs_gradnorm=None, tol_rel_gradnorm=None,
                 convergence_level=1, iteration_limit=None, callback=None,
                 logger=None):
        super(GradientNormController, self).__init__(callback=callback,
                                                     logger=logger)
        self._tol_abs_gradnorm = tol_abs_gradnorm
        self._tol_rel_gradnorm = tol_rel_gradnorm
        self._tol_rel_gradnorm_now = None
        self._convergence_level = convergence_level
        self._iteration_limit = iteration_limit

    def reset(self, energy):
        self._iteration_count = 0
        self._convergence_count = 0
        if self._tol_rel_gradnorm is not None:
            self._tol_rel_gradnorm_now = self._tol_rel_gradnorm \
                                       * energy.gradient_norm

    def check(self, energy):
        super_check = super(GradientNormController, self).check(energy)
        if super_check != self.CONTINUE:
            return super_check

        # check if position is at a flat point
        if energy.gradient_norm == 0:
            self._print_debug_info(energy)
            self.logger.info("Reached perfectly flat point. Stopping.")
            return self.CONVERGED

        if self._tol_abs_gradnorm is not None:
            if energy.gradient_norm <= self._tol_abs_gradnorm:
                self._convergence_count += 1
        if self._tol_rel_gradnorm is not None:
            if energy.gradient_norm <= self._tol_rel_gradnorm_now:
                self._convergence_count += 1

        if self._iteration_limit is not None:
            if self._iteration_count > self._iteration_limit:
                self._print_debug_info(energy)
                self.logger.info("Reached iteration limit. Stopping.")
                return self.STOPPED

        if self._convergence_count >= self._convergence_level:
            self._print_debug_info(energy)
            self.logger.info("Reached convergence limit. Stopping.")
            return self.CONVERGED

        self._print_debug_info(energy)
        return self.CONTINUE

    def _print_debug_info(self, energy):
        self.logger.debug(
                "Iteration %08u gradient-norm %3.1E convergence-level %i" %
                (self._iteration_count,
                 energy.gradient_norm,
                 self._convergence_count))
