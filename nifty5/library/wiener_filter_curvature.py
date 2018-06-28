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

from ..operators.sandwich_operator import SandwichOperator
from ..operators.inversion_enabler import InversionEnabler
from ..operators.sampling_enabler import SamplingEnabler


def WienerFilterCurvature(R, N, S, iteration_controller=None,
                          iteration_controller_sampling=None):
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
    M = SandwichOperator.make(R, N.inverse)
    if iteration_controller_sampling is not None:
        op = SamplingEnabler(M, S.inverse, iteration_controller_sampling,
                             S.inverse)
    else:
        op = M + S.inverse
    return InversionEnabler(op, iteration_controller, S.inverse)
