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
                  FFTOperator,\
                  sqrt

__all__ = ['create_power_operator']


def create_power_operator(domain, power_spectrum, power_domain=None, dtype=None,
                          distribution_strategy='not'):
    if not domain.harmonic:
        fft = FFTOperator(domain)
        domain = fft.target[0]
    if isinstance(power_spectrum, Field):
        power_domain = power_spectrum.domain
    elif power_domain is None:
        power_domain = PowerSpace(domain,
                              distribution_strategy=distribution_strategy)

    fp = Field(power_domain,
               val=power_spectrum, dtype=dtype,
               distribution_strategy=distribution_strategy)

    f = fp.power_synthesize(mean=1, std=0, real_signal=False)

    power_operator = DiagonalOperator(domain, diagonal=f, bare=True)

    return power_operator


def generate_posterior_sample(mean, covariance):
    S = covariance.S
    R = covariance.R
    N = covariance.N
    power = sqrt(S.diagonal().power_analyze())
    mock_signal = power.power_synthesize(real_signal=True)


    noise = N.diagonal().val

    mock_noise = Field.from_random(random_type="normal", domain=N.domain,
                                   std = sqrt(noise), dtype = noise.dtype)
    mock_data = R.derived_times(mock_signal, mean) + mock_noise

    mock_j = R.derived_adjoint_times(N.inverse_times(mock_data), mean)
    mock_m = covariance.inverse_times(mock_j)
    sample = mock_signal - mock_m + mean
    return sample

