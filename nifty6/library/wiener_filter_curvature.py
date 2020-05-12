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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from ..operators.inversion_enabler import InversionEnabler
from ..operators.sampling_enabler import SamplingEnabler
from ..operators.sandwich_operator import SandwichOperator


def WienerFilterCurvature(R, N, S, iteration_controller=None,
                          iteration_controller_sampling=None,
                          data_sampling_dtype=None,
                          prior_sampling_dtype=None):
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
    iteration_controller : IterationController
        The iteration controller to use during numerical inversion via
        ConjugateGradient.
    iteration_controller_sampling : IterationController
        The iteration controller to use for sampling.
    """
    Ninv = N.inverse
    Sinv = S.inverse
    if data_sampling_dtype is not None:
        from ..operators.energy_operators import SamplingDtypeEnabler
        Ninv = SamplingDtypeEnabler(Ninv, data_sampling_dtype)
    if prior_sampling_dtype is not None:
        from ..operators.energy_operators import SamplingDtypeEnabler
        Sinv = SamplingDtypeEnabler(Sinv, data_sampling_dtype)
    M = SandwichOperator.make(R, Ninv)
    if iteration_controller_sampling is not None:
        op = SamplingEnabler(M, Sinv, iteration_controller_sampling,
                             Sinv)
    else:
        op = M + Sinv
    op = InversionEnabler(op, iteration_controller, Sinv)
    return op
