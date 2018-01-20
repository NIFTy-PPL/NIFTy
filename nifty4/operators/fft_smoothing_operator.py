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

from .scaling_operator import ScalingOperator
from .fft_operator import FFTOperator
from ..utilities import infer_space
from .diagonal_operator import DiagonalOperator
from .. import DomainTuple


def FFTSmoothingOperator(domain, sigma, space=None):
    sigma = float(sigma)
    if sigma < 0.:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0.:
        return ScalingOperator(1., domain)

    domain = DomainTuple.make(domain)
    space = infer_space(domain, space)
    FFT = FFTOperator(domain, space=space)
    codomain = FFT.domain[space].get_default_codomain()
    kernel = codomain.get_k_length_array()
    smoother = codomain.get_fft_smoothing_kernel_function(sigma)
    kernel = smoother(kernel)
    ddom = list(domain)
    ddom[space] = codomain
    diag = DiagonalOperator(kernel, ddom, space)
    return FFT.adjoint*diag*FFT
