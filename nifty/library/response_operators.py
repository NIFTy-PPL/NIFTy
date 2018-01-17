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

from ..field import exp


def LinearizedSignalResponse(Instrument, nonlinearity, FFT, power, m):
    position = FFT.adjoint_times(power*m)
    return (Instrument * nonlinearity.derivative(position) *
            FFT.adjoint * power)

def LinearizedPowerResponse(Instrument, nonlinearity, FFT, Projection, t, m):
    power = exp(0.5*t)
    position = FFT.adjoint_times(Projection.adjoint_times(power) * m)
    linearization = nonlinearity.derivative(position)
    return (0.5 * Instrument * linearization * FFT.adjoint * m *
            Projection.adjoint * power)
