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
from . import Space, PowerSpace, Field, ComposedOperator, DiagonalOperator,\
              PowerProjectionOperator, FFTOperator, sqrt, DomainTuple, dobj,\
              utilities

__all__ = ['PS_field',
           'power_analyze',
           'power_synthesize',
           'power_synthesize_special',
           'create_power_field',
           'create_power_operator',
           'generate_posterior_sample',
           'create_composed_fft_operator']


def PS_field(pspace, func, dtype=None):
    if not isinstance(pspace, PowerSpace):
        raise TypeError
    data = dobj.from_global_data(func(pspace.k_lengths))
    return Field(pspace, val=data, dtype=dtype)


def _single_power_analyze(field, idx, binbounds):
    from .operators.power_projection_operator import PowerProjectionOperator
    power_domain = PowerSpace(field.domain[idx], binbounds)
    ppo = PowerProjectionOperator(field.domain, power_domain, idx)
    return ppo(field.weight(-1))


def power_analyze(field, spaces=None, binbounds=None,
                  keep_phase_information=False):
    """ Computes the square root power spectrum for a subspace of `field`.

    Creates a PowerSpace for the space addressed by `spaces` with the given
    binning and computes the power spectrum as a Field over this
    PowerSpace. This can only be done if the subspace to  be analyzed is a
    harmonic space. The resulting field has the same units as the initial
    field, corresponding to the square root of the power spectrum.

    Parameters
    ----------
    field : Field
        The field to be analyzed
    spaces : int *optional*
        The subspace for which the powerspectrum shall be computed.
        (default : None).
    binbounds : array-like *optional*
        Inner bounds of the bins (default : None).
        if binbounds==None : bins are inferred.
    keep_phase_information : boolean, *optional*
        If False, return a real-valued result containing the power spectrum
        of the input Field.
        If True, return a complex-valued result whose real component
        contains the power spectrum computed from the real part of the
        input Field, and whose imaginary component contains the power
        spectrum computed from the imaginary part of the input Field.
        The absolute value of this result should be identical to the output
        of power_analyze with keep_phase_information=False.
        (default : False).

    Raise
    -----
    TypeError
        Raised if any of the input field's domains is not harmonic

    Returns
    -------
    out : Field
        The output object. Its domain is a PowerSpace and it contains
        the power spectrum of 'field'.
    """

    # check if all spaces in `field.domain` are either harmonic or
    # power_space instances
    for sp in field.domain:
        if not sp.harmonic and not isinstance(sp, PowerSpace):
            dobj.mprint("WARNING: Field has a space in `domain` which is "
                        "neither harmonic nor a PowerSpace.")

    # check if the `spaces` input is valid
    if spaces is None:
        spaces = range(len(field.domain))
    else:
        spaces = utilities.cast_iseq_to_tuple(spaces)

    if len(spaces) == 0:
        raise ValueError("No space for analysis specified.")

    if keep_phase_information:
        parts = [field.real*field.real, field.imag*field.imag]
    else:
        parts = [field.real*field.real + field.imag*field.imag]

    parts = [part.weight(1, spaces) for part in parts]
    for space_index in spaces:
        parts = [_single_power_analyze(field=part,
                                       idx=space_index,
                                       binbounds=binbounds)
                 for part in parts]

    return parts[0] + 1j*parts[1] if keep_phase_information else parts[0]


def _compute_spec(field, spaces):
    from .operators.power_projection_operator import PowerProjectionOperator
    if spaces is None:
        spaces = range(len(field.domain))
    else:
        spaces = utilities.cast_iseq_to_tuple(spaces)

    # create the result domain
    result_domain = list(field.domain)

    spec = sqrt(field)
    for i in spaces:
        result_domain[i] = field.domain[i].harmonic_partner
        ppo = PowerProjectionOperator(result_domain, field.domain[i], i)
        spec = ppo.adjoint_times(spec)

    return spec


def power_synthesize(field, spaces=None, real_power=True, real_signal=True):
    """ Yields a sampled field with `field`**2 as its power spectrum.

    This method draws a Gaussian random field in the harmonic partner
    domain of this field's domains, using this field as power spectrum.

    Parameters
    ----------
    field : Field
        The input field containing the square root of the power spectrum
    spaces : {tuple, int, None} *optional*
        Specifies the subspace containing all the PowerSpaces which
        should be converted (default : None).
        if spaces==None : Tries to convert the whole domain.
    real_power : boolean *optional*
        Determines whether the power spectrum is treated as intrinsically
        real or complex (default : True).
    real_signal : boolean *optional*
        True will result in a purely real signal-space field
        (default : True).

    Returns
    -------
    out : Field
        The output object. A random field created with the power spectrum
        stored in the `spaces` in `field`.

    Notes
    -----
    For this the spaces specified by `spaces` must be a PowerSpace.
    This expects this field to be the square root of a power spectrum, i.e.
    to have the unit of the field to be sampled.

    Raises
    ------
    ValueError : If domain specified by `spaces` is not a PowerSpace.

    """

    spec = _compute_spec(field, spaces)

    # create random samples: one or two, depending on whether the
    # power spectrum is real or complex
    result = [field.from_random('normal', mean=0., std=1.,
                                domain=spec.domain,
                                dtype=np.float if real_signal
                                else np.complex)
              for x in range(1 if real_power else 2)]

    # MR: dummy call - will be removed soon
    if real_signal:
        field.from_random('normal', mean=0., std=1.,
                          domain=spec.domain, dtype=np.float)

    # apply the rescaler to the random fields
    result[0] *= spec.real
    if not real_power:
        result[1] *= spec.imag

    return result[0] if real_power else result[0] + 1j*result[1]


def power_synthesize_special(field, spaces=None):
    spec = _compute_spec(field, spaces)

    # MR: dummy call - will be removed soon
    field.from_random('normal', mean=0., std=1.,
                      domain=spec.domain, dtype=np.complex)

    return spec.real


def create_power_field(domain, power_spectrum, dtype=None):
    if not callable(power_spectrum):  # we have a Field living on a PowerSpace
        if not isinstance(power_spectrum, Field):
            raise TypeError("Field object expected")
        if len(power_spectrum.domain) != 1:
            raise ValueError("exactly one domain required")
        if not isinstance(power_spectrum.domain[0], PowerSpace):
            raise TypeError("PowerSpace required")
        power_domain = power_spectrum.domain[0]
        fp = Field(power_domain, val=power_spectrum.val, dtype=dtype)
    else:
        power_domain = PowerSpace(domain)
        fp = PS_field(power_domain, power_spectrum, dtype)
    P = PowerProjectionOperator(domain, power_domain)
    f = P.adjoint_times(fp)

    if not issubclass(fp.dtype.type, np.complexfloating):
        f = f.real

    return f


def create_power_operator(domain, power_spectrum, space=None, dtype=None):
    """ Creates a diagonal operator with the given power spectrum.

    Constructs a diagonal operator that lives over the specified domain.

    Parameters
    ----------
    domain : DomainObject
        Domain over which the power operator shall live.
    power_spectrum : callable of Field
        An object that implements the power spectrum as a function of k.
    space : int
            the domain index on which the power operator will work
    dtype : type *optional*
        dtype that the field holding the power spectrum shall use
        (default : None).
        if dtype == None: the dtype of `power_spectrum` will be used.

    Returns
    -------
    DiagonalOperator : An operator that implements the given power spectrum.

    """
    domain = DomainTuple.make(domain)
    if space is None:
        if len(domain) != 1:
            raise ValueError("space keyword must be set")
        else:
            space = 0
    space = int(space)
    return DiagonalOperator(
        create_power_field(domain[space],
                           power_spectrum, dtype).weight(1),
        domain=domain,
        spaces=space)


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

    S = covariance.op.S
    R = covariance.op.R
    N = covariance.op.N

    power = sqrt(power_analyze(S.diagonal()))
    mock_signal = power_synthesize(power, real_signal=True)

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

    if codomain is None:
        codomain = [None]*len(domain)
    interdomain = list(domain.domains)
    for i, space in enumerate(domain):
        if not isinstance(space, Space):
            continue
        if (all_to == 'other' or
                (all_to == 'position' and space.harmonic) or
                (all_to == 'harmonic' and not space.harmonic)):
            if codomain[i] is None:
                interdomain[i] = domain[i].get_default_codomain()
            else:
                interdomain[i] = codomain[i]
            fft_op_list += [FFTOperator(domain=domain, target=interdomain,
                                        space=i)]
        domain = interdomain
    return ComposedOperator(fft_op_list)
