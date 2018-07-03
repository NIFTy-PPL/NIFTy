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

from builtins import *
from ..utilities import memo, my_lincomb_simple, my_lincomb
from .energy import Energy


class EnergySum(Energy):
    def __init__(self, position, energies, factors):
        super(EnergySum, self).__init__(position=position)
        self._energies = tuple(e.at(position) for e in energies)
        self._factors = tuple(factors)

    @staticmethod
    def make(energies, factors=None):
        if factors is None:
            factors = (1,)*len(energies)
        # unpack energies
        eout = []
        fout = []
        EnergySum._unpackEnergies(energies, factors, 1., eout, fout)
        for e in eout[1:]:
            if not e.position.isEquivalentTo(eout[0].position):
                raise ValueError("position mismatch")
        return EnergySum(eout[0].position, eout, fout)

    @staticmethod
    def _unpackEnergies(e_in, f_in, prefactor, e_out, f_out):
        for e, f in zip(e_in, f_in):
            if isinstance(e, EnergySum):
                EnergySum._unpackEnergies(e._energies, e._factors, prefactor*f,
                                          e_out, f_out)
            else:
                e_out.append(e)
                f_out.append(prefactor*f)

    def at(self, position):
        return self.__class__(position, self._energies, self._factors)

    @property
    @memo
    def value(self):
        return my_lincomb_simple(map(lambda v: v.value, self._energies),
                                 self._factors)

    @property
    @memo
    def gradient(self):
        return my_lincomb(map(lambda v: v.gradient, self._energies),
                          self._factors).lock()

    @property
    @memo
    def curvature(self):
        return my_lincomb(map(lambda v: v.curvature, self._energies),
                          self._factors)
