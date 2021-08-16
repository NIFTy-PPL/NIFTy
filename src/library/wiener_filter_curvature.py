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
# Copyright(C) 2013-2020 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..operators.inversion_enabler import InversionEnabler
from ..operators.sampling_enabler import SamplingDtypeSetter, SamplingEnabler
from ..operators.sandwich_operator import SandwichOperator


def WienerFilterCurvature(R, N, S, iteration_controller=None,
                          iteration_controller_sampling=None,
                          data_sampling_dtype=np.float64,
                          prior_sampling_dtype=np.float64):
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
    S : EndomorphicOperator
        The prior signal covariance.
    iteration_controller : IterationController
        The iteration controller to use during numerical inversion via
        ConjugateGradient.
    iteration_controller_sampling : IterationController
        The iteration controller to use for sampling.
    data_sampling_dtype : numpy.dtype or dict of numpy.dtype
        Data type used for sampling from likelihood. Conincides with the data
        type of the data used in the inference problem. Default is float64.
    prior_sampling_dtype : numpy.dtype or dict of numpy.dtype
        Data type used for sampling from likelihood. Coincides with the data
        type of the parameters of the forward model used for the inference
        problem. Default is float64.

    Note
    ----
    It must be possible to set the sampling dtype of `N` and `S` with the help
    of an `SamplingDtypeSetter`. In practice this means that
    `data_sampling_dtype` is not `None`, `N` must be a `ScalingOperator`, a
    `DiagonalOperator`, or something similar.
    """
    Ninv = N.inverse
    Sinv = S.inverse
    if data_sampling_dtype is not None:
        Ninv = SamplingDtypeSetter(Ninv, data_sampling_dtype)
    if prior_sampling_dtype is not None:
        Sinv = SamplingDtypeSetter(Sinv, prior_sampling_dtype)
    M = SandwichOperator.make(R, Ninv)
    if iteration_controller_sampling is not None:
        op = SamplingEnabler(M, Sinv, iteration_controller_sampling,
                             Sinv)
    else:
        op = M + Sinv
    return InversionEnabler(op, iteration_controller, Sinv)
