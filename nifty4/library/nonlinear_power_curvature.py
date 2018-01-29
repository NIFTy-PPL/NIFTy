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

from ..operators.inversion_enabler import InversionEnabler
from .response_operators import LinearizedPowerResponse


def NonlinearPowerCurvature(
        tau,
        ht,
        Instrument,
        nonlinearity,
        Projection,
        N,
        T,
        xi_sample_list,
        inverter,
        munit=1.,
        sunit=1.):
    result = None
    for xi_sample in xi_sample_list:
        LinearizedResponse = LinearizedPowerResponse(
            Instrument, nonlinearity, ht, Projection, tau, xi_sample, munit, sunit)
        op = LinearizedResponse.adjoint * N.inverse * LinearizedResponse
        result = op if result is None else result + op
    result = result * (1. / len(xi_sample_list)) + T
    return InversionEnabler(result, inverter)
