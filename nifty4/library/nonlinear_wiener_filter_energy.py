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
from numpy.testing import assert_allclose

from ..field import Field
from ..sugar import sqrt
from ..minimization.energy import Energy
from ..operators import DiagonalOperator
from ..operators.inversion_enabler import InversionEnabler
from ..symbolic import (SymbolicCABF, SymbolicAdd, SymbolicConstant, SymbolicExp, SymbolicLinear, SymbolicQuad,
                        SymbolicTanh, SymbolicVariable, Tensor)
from ..utilities import memo
from .nonlinearities import Exponential, Linear, Tanh
from .wiener_filter_curvature import WienerFilterCurvature


class NonlinearWienerFilterEnergy(Energy):
    def __init__(self, position, d, Instrument, nonlinearity, ht, power, N, S,
                 inverter=None):
        super(NonlinearWienerFilterEnergy, self).__init__(position=position)
        self.d = d.lock()
        self.Instrument = Instrument
        self.nonlinearity = nonlinearity
        self.ht = ht
        self.power = power
        m = ht(power*position)

        residual = d - Instrument(nonlinearity(m))
        self.N = N
        self.S = S
        self.inverter = inverter
        t1 = S.inverse_times(position)
        t2 = N.inverse_times(residual)
        self._value = 0.5 * (position.vdot(t1) + residual.vdot(t2)).real
        self.R = Instrument * nonlinearity.derivative(m) * ht * power
        self._gradient = (t1 - self.R.adjoint_times(t2)).lock()
        self._curvature = WienerFilterCurvature(self.R, self.N, self.S, self.inverter)

        # Nonlinear implementation
        if isinstance(nonlinearity, Exponential):
            SymbolicNonlinearity = SymbolicExp
        elif isinstance(nonlinearity, Linear):
            SymbolicNonlinearity = SymbolicLinear
        elif isinstance(nonlinearity, Tanh):
            SymbolicNonlinearity = SymbolicTanh
        else:
            raise NotImplementedError

        pos_nl = SymbolicVariable(position.domain)
        Ninv_nextgen = DiagonalOperator(sqrt(self.N.inverse(Field.full(self.N.target, 1.))))
        Ninv_nextgen_nl = SymbolicConstant(Tensor(Ninv_nextgen, 2, name='Ninv_nextgen'), (-1, -1))
        Sinv_nextgen = DiagonalOperator(sqrt(self.S.inverse(Field.full(self.S.target, 1.))))
        Sinv_nextgen_nl = SymbolicConstant(Tensor(Sinv_nextgen, 2, name='Sinv_nextgen'), (-1, -1))
        mh_nl = SymbolicCABF(SymbolicConstant(Tensor(DiagonalOperator(power), 2, name='power'), (1, -1)), pos_nl)
        m_nl = SymbolicCABF(SymbolicConstant(Tensor(ht, 2, name='HT'), (1, -1)), mh_nl)
        sky_nl = SymbolicNonlinearity(m_nl)
        d_nl = SymbolicConstant(Tensor(d, 1, name='d'), (1,))
        MinusR_nl = SymbolicConstant(Tensor((-1) * Instrument, 2, name='-R'), (1, -1))
        rec_nl = SymbolicCABF(MinusR_nl, sky_nl)
        residual_nl = SymbolicAdd(d_nl, rec_nl)
        residual_nextgen_nl = SymbolicCABF(Ninv_nextgen_nl, residual_nl)
        pos_nextgen_nl = SymbolicCABF(Sinv_nextgen_nl, pos_nl)

        likelihood_nl = SymbolicQuad(residual_nextgen_nl)
        prior_nl = SymbolicQuad(pos_nextgen_nl)
        energy_nl = SymbolicAdd(likelihood_nl, prior_nl)

        new_energy = energy_nl.eval(position)
        new_gradient = energy_nl.derivative.eval(position)
        new_curvature = energy_nl.curvature.eval(position)
        # End Nonlinear implementation

        # Compare old and new implementation
        assert_allclose(self._value, new_energy)
        assert_allclose(self._gradient.val, new_gradient.val, rtol=1e-5)
        rand_field = Field.from_random('normal', new_curvature.domain)
        assert_allclose(self._curvature(rand_field).val, new_curvature(rand_field).val)
        # End Compare old and new implementation

        self._value = new_energy
        self._gradient = new_gradient
        self._curvature = InversionEnabler(new_curvature, inverter, S.inverse)

    def at(self, position):
        return self.__class__(position, self.d, self.Instrument,
                              self.nonlinearity, self.ht, self.power, self.N,
                              self.S, self.inverter)

    @property
    def value(self):
        return self._value

    @property
    def gradient(self):
        return self._gradient

    @property
    @memo
    def curvature(self):
        return self._curvature
