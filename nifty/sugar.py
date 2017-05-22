# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

from nifty import PowerSpace,\
                  Field,\
                  DiagonalOperator

__all__ = ['create_power_operator']


def create_power_operator(domain, power_spectrum, dtype=None,
                          distribution_strategy='not'):
    """ Creates a diagonal operator with the given power spectrum.

    Constructs a diagonal operator that lives over the specified domain.

    Parameters
    ----------
    domain : DomainObject
        Domain over which the power operator shall live. 
    power_spectrum : (array-like, method)
        An array-like object, or a method that implements the square root
        of a power spectrum as a function of k.
    dtype : type *optional*
        dtype that the field holding the power spectrum shall use
        (default : None).
        if dtype == None: the dtype of `power_spectrum` will be used.
    distribution_strategy : string *optional*
        Distributed strategy to be used by the underlying d2o objects.
        (default : 'not')

    Returns
    -------
    DiagonalOperator : An operator that implements the given power spectrum.

    """

    power_domain = PowerSpace(domain,
                              distribution_strategy=distribution_strategy)
    fp = Field(power_domain, val=power_spectrum, dtype=dtype,
               distribution_strategy=distribution_strategy)
    fp **= 2
    f = fp.power_synthesize(mean=1, std=0, real_signal=False)

    return DiagonalOperator(domain, diagonal=f, bare=True)
