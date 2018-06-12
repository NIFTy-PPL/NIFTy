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
from .laplace_operator import LaplaceOperator


def SmoothnessOperator(domain, strength=1., logarithmic=True, space=None):
    """An operator measuring the smoothness on an irregular grid with respect
    to some scale.

    This operator applies the irregular LaplaceOperator and its adjoint to some
    Field over a PowerSpace which corresponds to its smoothness and weights the
    result with a scale parameter sigma. For this purpose we use free boundary
    conditions in the LaplaceOperator, having no curvature at both ends. In
    addition the first entry is ignored as well, corresponding to the overall
    mean of the map. The mean is therefore not considered in the smoothness
    prior.


    Parameters
    ----------
    domain : Domain, tuple of Domain, or DomainTuple
       The total domain of the operator's input and output fields
    strength : nonnegative float
        Specifies the strength of the SmoothnessOperator
    logarithmic : bool, optional
        Whether smoothness is calculated on a logarithmic scale or linear scale
        default : True
    space : int, optional
       The index of the sub-domain on which the operator acts.
       Can be omitted if `domain` only has one sub-domain.
    """
    if strength < 0:
        raise ValueError("ERROR: strength must be nonnegative.")
    if strength == 0.:
        return ScalingOperator(0., domain)
    laplace = LaplaceOperator(domain, logarithmic=logarithmic, space=space)
    return (strength**2)*laplace.adjoint*laplace
