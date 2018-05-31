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

from .energy import Energy
from ..utilities import memo


class EnergySum(Energy):
    def __init__(self, position, energies, minimizer_controller=None,
                 preconditioner=None, precon_idx=None):
        super(EnergySum, self).__init__(position=position)
        self._energies = [energy.at(position) for energy in energies]
        self._min_controller = minimizer_controller
        self._preconditioner = preconditioner
        self._precon_idx = precon_idx

    def at(self, position):
        return self.__class__(position, self._energies, self._min_controller,
                              self._preconditioner, self._precon_idx)

    @property
    @memo
    def value(self):
        res = self._energies[0].value
        for e in self._energies[1:]:
            res += e.value
        return res

    @property
    @memo
    def gradient(self):
        res = self._energies[0].gradient.copy()
        for e in self._energies[1:]:
            res += e.gradient
        return res.lock()

    @property
    @memo
    def curvature(self):
        res = self._energies[0].curvature
        for e in self._energies[1:]:
            res = res + e.curvature
        if self._min_controller is None:
            return res
        precon = self._preconditioner
        if precon is None and self._precon_idx is not None:
            precon = self._energies[self._precon_idx].curvature
        from ..operators.inversion_enabler import InversionEnabler
        from .conjugate_gradient import ConjugateGradient
        return InversionEnabler(
            res, ConjugateGradient(self._min_controller), precon)

