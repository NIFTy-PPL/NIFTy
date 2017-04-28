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
                  DiagonalOperator,\
                  FFTOperator

__all__ = ['create_power_operator']


def create_power_operator(domain, power_spectrum, dtype=None,
                          distribution_strategy='not'):
    if not domain.harmonic:
        fft = FFTOperator(domain)
        domain = fft.target[0]

    power_domain = PowerSpace(domain,
                              distribution_strategy=distribution_strategy)

    fp = Field(power_domain,
               val=power_spectrum,dtype=dtype,
               distribution_strategy=distribution_strategy)

    f = fp.power_synthesize(mean=1, std=0, real_signal=False)

    power_operator = DiagonalOperator(domain, diagonal=f)

    return power_operator
