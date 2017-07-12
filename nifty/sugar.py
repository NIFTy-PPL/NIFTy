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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from . import PowerSpace,\
              Field,\
              DiagonalOperator,\
              sqrt

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

    if isinstance(power_spectrum, Field):
        power_domain = power_spectrum.domain
    else:
        power_domain = PowerSpace(domain,
                                  distribution_strategy=distribution_strategy)

    fp = Field(power_domain, val=power_spectrum, dtype=dtype,
               distribution_strategy=distribution_strategy)
    f = fp.power_synthesize(mean=1, std=0, real_signal=False)
    f **= 2
    return DiagonalOperator(domain, diagonal=f, bare=True)


def generate_posterior_sample(mean, covariance):
    """ Generates a posterior sample from a Gaussian distribution with given
    mean and covariance

    This method generates samples by setting up the observation and
    reconstruction of a mock signal in order to obtain residuals of the right
    correlation which are added to the given mean.

    Parameters
    ----------
    mean : Field
        the mean of the posterior Gaussian distribution
    covariance : WienerFilterCurvature
        The posterior correlation structure consisting of a
        response operator, noise covariance and prior signal covariance

    Returns
    -------
    sample : Field
        Returns the a sample from the Gaussian of given mean and covariance.

    """

    S = covariance.S
    R = covariance.R
    N = covariance.N

    power = S.diagonal().power_analyze()**.5
    mock_signal = power.power_synthesize(real_signal=True)

    noise = N.diagonal(bare=True).val

    mock_noise = Field.from_random(random_type="normal", domain=N.domain,
                                   std=sqrt(noise), dtype=noise.dtype)
    mock_data = R(mock_signal) + mock_noise

    mock_j = R.adjoint_times(N.inverse_times(mock_data))
    mock_m = covariance.inverse_times(mock_j)
    sample = mock_signal - mock_m + mean
    return sample
