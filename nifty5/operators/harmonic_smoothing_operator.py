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

from __future__ import absolute_import, division, print_function

from ..compat import *
from ..domain_tuple import DomainTuple
from ..utilities import infer_space
from .diagonal_operator import DiagonalOperator
from .hartley_operator import HartleyOperator
from .scaling_operator import ScalingOperator


def HarmonicSmoothingOperator(domain, sigma, space=None):
    """ This function returns an operator that carries out a smoothing with
    a Gaussian kernel of width `sigma` on the part of `domain` given by
    `space`.

    Parameters
    ----------
    domain : Domain, tuple of Domain, or DomainTuple
       The total domain of the operator's input and output fields
    sigma : float>=0
       The sigma of the Gaussian used for smoothing. It has the same units as
       the RGSpace the operator is working on.
       If `sigma==0`, an identity operator will be returned.
    space : int, optional
       The index of the sub-domain on which the smoothing is performed.
       Can be omitted if `domain` only has one sub-domain.

    Notes
    -----
       The sub-domain on which the smoothing is carried out *must* be a
       non-harmonic `RGSpace`.
    """

    sigma = float(sigma)
    if sigma < 0.:
        raise ValueError("sigma must be nonnegative")
    if sigma == 0.:
        return ScalingOperator(1., domain)

    domain = DomainTuple.make(domain)
    space = infer_space(domain, space)
    if domain[space].harmonic:
        raise TypeError("domain must not be harmonic")
    Hartley = HartleyOperator(domain, space=space)
    codomain = Hartley.domain[space].get_default_codomain()
    kernel = codomain.get_k_length_array()
    smoother = codomain.get_fft_smoothing_kernel_function(sigma)
    kernel = smoother(kernel)
    ddom = list(domain)
    ddom[space] = codomain
    diag = DiagonalOperator(kernel, ddom, space)
    return Hartley.inverse*diag*Hartley
