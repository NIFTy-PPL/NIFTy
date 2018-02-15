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

import numpy as np
from .domains.structured_domain import StructuredDomain
from .domains.power_space import PowerSpace
from .field import Field, sqrt
from .operators.diagonal_operator import DiagonalOperator
from .operators.power_distributor import PowerDistributor
from .operators.harmonic_transform_operator import HarmonicTransformOperator
from .domain_tuple import DomainTuple
from . import dobj, utilities

__all__ = ['PS_field',
           'power_analyze',
           'power_synthesize',
           'power_synthesize_nonrandom',
           'create_power_field',
           'create_power_operator',
           'create_composed_ht_operator',
           'create_harmonic_smoothing_operator']


def PS_field(pspace, func, dtype=None):
    if not isinstance(pspace, PowerSpace):
        raise TypeError
    data = dobj.from_global_data(func(pspace.k_lengths))
    return Field(pspace, val=data, dtype=dtype)


def _single_power_analyze(field, idx, binbounds):
    power_domain = PowerSpace(field.domain[idx], binbounds)
    pd = PowerDistributor(field.domain, power_domain, idx)
    return pd.adjoint_times(field.weight(1)).weight(-1)  # divides by bin size


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
    spaces : None or int or tuple of int , optional
        The set of subdomains for which the powerspectrum shall be computed.
        If None, all subdomains will be converted.
        (default : None).
    binbounds : None or array-like, optional
        Inner bounds of the bins (default : None).
        if binbounds is None : bins are inferred.
    keep_phase_information : bool, optional
        If False, return a real-valued result containing the power spectrum
        of the input Field.
        If True, return a complex-valued result whose real component
        contains the power spectrum computed from the real part of the
        input Field, and whose imaginary component contains the power
        spectrum computed from the imaginary part of the input Field.
        The absolute value of this result should be identical to the output
        of power_analyze with keep_phase_information=False.
        (default : False).

    Returns
    -------
    Field
        The output object. Its domain is a PowerSpace and it contains
        the power spectrum of 'field'.
    """

    for sp in field.domain:
        if not sp.harmonic and not isinstance(sp, PowerSpace):
            dobj.mprint("WARNING: Field has a space in `domain` which is "
                        "neither harmonic nor a PowerSpace.")

    spaces = utilities.parse_spaces(spaces, len(field.domain))

    if len(spaces) == 0:
        raise ValueError("No space for analysis specified.")

    field_real = not np.issubdtype(field.dtype, np.complexfloating)
    if (not field_real) and keep_phase_information:
        raise ValueError("cannot keep phase from real-valued input Field")

    if keep_phase_information:
        parts = [field.real*field.real, field.imag*field.imag]
    else:
        if field_real:
            parts = [field**2]
        else:
            parts = [field.real*field.real + field.imag*field.imag]

    for space_index in spaces:
        parts = [_single_power_analyze(field=part,
                                       idx=space_index,
                                       binbounds=binbounds)
                 for part in parts]

    return parts[0] + 1j*parts[1] if keep_phase_information else parts[0]


def power_synthesize_nonrandom(field, spaces=None):
    spaces = utilities.parse_spaces(spaces, len(field.domain))

    result_domain = list(field.domain)
    spec = sqrt(field)
    for i in spaces:
        result_domain[i] = field.domain[i].harmonic_partner
        pd = PowerDistributor(result_domain, field.domain[i], i)
        spec = pd(spec)

    return spec


def power_synthesize(field, spaces=None, real_power=True, real_signal=True):
    """Returns a sampled field with `field`**2 as its power spectrum.

    This method draws a Gaussian random field in the harmonic partner
    domain of this field's domains, using this field as power spectrum.

    Parameters
    ----------
    field : Field
        The input field containing the square root of the power spectrum
    spaces : None, int, or tuple of int, optional
        Specifies the subdomains containing all the PowerSpaces which
        should be converted (default : None).
        if spaces is None : Tries to convert the whole domain.
    real_power : bool, optional
        Determines whether the power spectrum is treated as intrinsically
        real or complex (default : True).
    real_signal : bool, optional
        True will result in a purely real signal-space field
        (default : True).

    Returns
    -------
    Field
        The output object. A random field created with the power spectrum
        stored in the `spaces` in `field`.

    Notes
    -----
    For this the spaces specified by `spaces` must be a PowerSpace.
    This expects this field to be the square root of a power spectrum, i.e.
    to have the unit of the field to be sampled.

    Raises
    ------
    ValueError : If a domain specified by `spaces` is not a PowerSpace.
    """

    spec = power_synthesize_nonrandom(field, spaces)
    self_real = not np.issubdtype(spec.dtype, np.complexfloating)
    if (not real_power) and self_real:
        raise ValueError("can't draw complex realizations from real spectrum")

    # create random samples: one or two, depending on whether the
    # power spectrum is real or complex
    result = [field.from_random('normal', mean=0., std=1.,
                                domain=spec.domain,
                                dtype=np.float64 if real_signal
                                else np.complex128)
              for x in range(1 if real_power else 2)]

    result[0] *= spec if self_real else spec.real
    if not real_power:
        result[1] *= spec.imag

    return result[0] if real_power else result[0] + 1j*result[1]


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

    return PowerDistributor(domain, power_domain)(fp)


def create_power_operator(domain, power_spectrum, space=None, dtype=None):
    """ Creates a diagonal operator with the given power spectrum.

    Constructs a diagonal operator that lives over the specified domain.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        Domain over which the power operator shall live.
    power_spectrum : callable or Field
        An object that contains the power spectrum as a function of k.
    space : int
        the domain index on which the power operator will work
    dtype : None or type, optional
        dtype that the field holding the power spectrum shall use
        (default : None).
        if dtype is None: the dtype of `power_spectrum` will be used.

    Returns
    -------
    DiagonalOperator
        An operator that implements the given power spectrum.
    """
    domain = DomainTuple.make(domain)
    if space is None:
        if len(domain) != 1:
            raise ValueError("space keyword must be set")
        else:
            space = 0
    space = int(space)
    return DiagonalOperator(
        create_power_field(domain[space], power_spectrum, dtype),
        domain=domain,
        spaces=space)


def create_composed_ht_operator(domain, codomain=None):
    if codomain is None:
        codomain = [None]*len(domain)
    res = None
    for i, space in enumerate(domain):
        if isinstance(space, StructuredDomain) and space.harmonic:
            tdom = domain if res is None else res.target
            op = HarmonicTransformOperator(tdom, codomain[i], i)
            res = op if res is None else op*res
    if res is None:
        raise ValueError("empty operator")
    return res


def create_harmonic_smoothing_operator(domain, space, sigma):
    kfunc = domain[space].get_fft_smoothing_kernel_function(sigma)
    return DiagonalOperator(kfunc(domain[space].get_k_length_array()), domain,
                            space)
