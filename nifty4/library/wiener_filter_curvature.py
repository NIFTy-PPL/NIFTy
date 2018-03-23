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

from ..operators.endomorphic_operator import EndomorphicOperator
from ..operators.inversion_enabler import InversionEnabler
import numpy as np


class WienerFilterCurvature(EndomorphicOperator):
    """The curvature of the WienerFilterEnergy.

    This operator implements the second derivative of the
    WienerFilterEnergy used in some minimization algorithms or
    for error estimates of the posterior maps. It is the
    inverse of the propagator operator.

    Parameters
    ----------
    R : LinearOperator
        The response operator of the Wiener filter measurement.
    N : EndomorphicOperator
        The noise covariance.
    S : DiagonalOperator
        The prior signal covariance
    inverter : Minimizer
        The minimizer to use during numerical inversion
    """

    def __init__(self, R, N, S, inverter):
        super(WienerFilterCurvature, self).__init__()
        self.R = R
        self.N = N
        self.S = S
        op = R.adjoint*N.inverse*R + S.inverse
        self._op = InversionEnabler(op, inverter, S.times)

    @property
    def domain(self):
        return self._op.domain

    @property
    def capability(self):
        return self._op.capability

    def apply(self, x, mode):
        return self._op.apply(x, mode)

    def draw_inverse_sample(self, dtype=np.float64):
        n = self.N.draw_sample(dtype)
        s = self.S.draw_sample(dtype)

        d = self.R(s) + n

        j = self.R.adjoint_times(self.N.inverse_times(d))
        m = self.inverse_times(j)
        return s - m
