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

from ..minimization.quadratic_energy import QuadraticEnergy
from .wiener_filter_curvature import WienerFilterCurvature


def WienerFilterEnergy(position, d, R, N, S, inverter):
    """The Energy for the Wiener filter.

    It covers the case of linear measurement with
    Gaussian noise and Gaussian signal prior with known covariance.

    Parameters
    ----------
    position : Field
        The current map in harmonic space.
    d :  Field
        the data
    R : LinearOperator
        The response operator, description of the measurement process. It needs
        to map from harmonic signal space to data space.
    N : EndomorphicOperator
        The noise covariance in data space.
    S : EndomorphicOperator
        The prior signal covariance in harmonic space.
    inverter : Minimizer
        the minimization strategy to use for operator inversion
    """
    op = WienerFilterCurvature(R, N, S, inverter)
    vec = R.adjoint_times(N.inverse_times(d))
    return QuadraticEnergy(position, op, vec)
