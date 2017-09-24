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

import numpy as np

from . import Space,\
                  PowerSpace,\
                  Field,\
                  ComposedOperator,\
                  DiagonalOperator,\
                  FFTOperator,\
                  sqrt

__all__ = ['create_power_operator',
           'generate_posterior_sample',
           'create_composed_fft_operator']


def create_power_operator(domain, power_spectrum, dtype=None):
    """ Creates a diagonal operator with the given power spectrum.

    Constructs a diagonal operator that lives over the specified domain.

    Parameters
    ----------
    domain : DomainObject
        Domain over which the power operator shall live.
    power_spectrum : callable
        A method that implements the square root of a power spectrum as a
        function of k.
    dtype : type *optional*
        dtype that the field holding the power spectrum shall use
        (default : None).
        if dtype == None: the dtype of `power_spectrum` will be used.

    Returns
    -------
    DiagonalOperator : An operator that implements the given power spectrum.

    """

    if not callable(power_spectrum):
        raise TypeError("power_spectrum must be callable")
    power_domain = PowerSpace(domain)

    fp = Field(power_domain, val=power_spectrum(power_domain.k_lengths),
               dtype=dtype)
    f = fp.power_synthesize_special()

    if not issubclass(fp.dtype.type, np.complexfloating):
        f = f.real

    f **= 2
    return DiagonalOperator(domain, diagonal=Field(domain,f).weight(1))


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

    power = sqrt(S.diagonal().power_analyze())
    mock_signal = power.power_synthesize(real_signal=True)

    noise = N.diagonal().weight(-1)

    mock_noise = Field.from_random(random_type="normal", domain=N.domain,
                                   dtype=noise.dtype.type)
    mock_noise *= sqrt(noise)

    mock_data = R(mock_signal) + mock_noise

    mock_j = R.adjoint_times(N.inverse_times(mock_data))
    mock_m = covariance.inverse_times(mock_j)
    sample = mock_signal - mock_m + mean
    return sample


def create_composed_fft_operator(domain, codomain=None, all_to='other'):
    fft_op_list = []
    space_index_list = []

    if codomain is None:
        codomain = [None]*len(domain)
    for i, space in enumerate(domain):
        cospace = codomain[i]
        if not isinstance(space, Space):
            continue
        if (all_to == 'other' or
                (all_to == 'position' and space.harmonic) or
                (all_to == 'harmonic' and not space.harmonic)):
            fft_op_list += [FFTOperator(domain=space, target=cospace)]
            space_index_list += [i]
    result = ComposedOperator(fft_op_list, default_spaces=space_index_list)
    return result
