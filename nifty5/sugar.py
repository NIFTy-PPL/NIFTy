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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import sys

import numpy as np

from . import dobj, utilities
from .domain_tuple import DomainTuple
from .domains.power_space import PowerSpace
from .field import Field
from .logger import logger
from .multi_domain import MultiDomain
from .multi_field import MultiField
from .operators.block_diagonal_operator import BlockDiagonalOperator
from .operators.diagonal_operator import DiagonalOperator
from .operators.distributors import PowerDistributor

__all__ = ['PS_field', 'power_analyze', 'create_power_operator',
           'create_harmonic_smoothing_operator', 'from_random',
           'full', 'from_global_data', 'from_local_data',
           'makeDomain', 'sqrt', 'exp', 'log', 'tanh', 'sigmoid',
           'sin', 'cos', 'tan', 'sinh', 'cosh',
           'absolute', 'one_over', 'clip', 'sinc',
           'conjugate', 'get_signal_variance', 'makeOp', 'domain_union',
           'get_default_codomain']


def PS_field(pspace, func):
    """Convenience function sampling a power spectrum

    Parameters
    ----------
    pspace : PowerSpace
        space at whose `k_lengths` the power spectrum function is evaluated
    func : function taking and returning a numpy.ndarray(float)
        the power spectrum function
    Returns
    -------
    Field : a field living on (pspace,) containing the computed function values
    """
    if not isinstance(pspace, PowerSpace):
        raise TypeError
    data = dobj.from_global_data(func(pspace.k_lengths))
    return Field(DomainTuple.make(pspace), data)


def get_signal_variance(spec, space):
    """
    Computes how much a field with a given power spectrum will vary in space

    This is a small helper function that computes how the expected variance
    of a harmonically transformed sample of this power spectrum.

    Parameters
    ---------
    spec: method
        a method that takes one k-value and returns the power spectrum at that
        location
    space: PowerSpace or any harmonic Domain
        If this function is given a harmonic domain, it creates the naturally
        binned PowerSpace to that domain.
        The field, for which the signal variance is then computed, is assumed
        to have this PowerSpace as naturally binned PowerSpace
    """
    if space.harmonic:
        space = PowerSpace(space)
    if not isinstance(space, PowerSpace):
        raise ValueError(
            "space must be either a harmonic space or Power space.")
    field = PS_field(space, spec)
    dist = PowerDistributor(space.harmonic_partner, space)
    k_field = dist(field)
    return k_field.weight(2).sum()


def _single_power_analyze(field, idx, binbounds):
    power_domain = PowerSpace(field.domain[idx], binbounds)
    pd = PowerDistributor(field.domain, power_domain, idx)
    return pd.adjoint_times(field.weight(1)).weight(-1)  # divides by bin size


# MR FIXME: this function is not well suited for analyzing more than one
# subdomain at once, because it allows only one set of binbounds.
def power_analyze(field, spaces=None, binbounds=None,
                  keep_phase_information=False):
    """ Computes the power spectrum for a subspace of `field`.

    Creates a PowerSpace for the space addressed by `spaces` with the given
    binning and computes the power spectrum as a Field over this
    PowerSpace. This can only be done if the subspace to  be analyzed is a
    harmonic space. The resulting field has the same units as the square of the
    initial field.

    Parameters
    ----------
    field : Field
        The field to be analyzed
    spaces : None or int or tuple of int, optional
        The indices of subdomains for which the power spectrum shall be
        computed.
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
        the power spectrum of `field`.
    """

    for sp in field.domain:
        if not sp.harmonic and not isinstance(sp, PowerSpace):
            logger.warning("WARNING: Field has a space in `domain` which is "
                           "neither harmonic nor a PowerSpace.")

    spaces = utilities.parse_spaces(spaces, len(field.domain))

    if len(spaces) == 0:
        raise ValueError("No space for analysis specified.")

    field_real = not utilities.iscomplextype(field.dtype)
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
        parts = [_single_power_analyze(part, space_index, binbounds)
                 for part in parts]

    return parts[0] + 1j*parts[1] if keep_phase_information else parts[0]


def _create_power_field(domain, power_spectrum):
    if not callable(power_spectrum):  # we have a Field living on a PowerSpace
        if not isinstance(power_spectrum, Field):
            raise TypeError("Field object expected")
        if len(power_spectrum.domain) != 1:
            raise ValueError("exactly one domain required")
        if not isinstance(power_spectrum.domain[0], PowerSpace):
            raise TypeError("PowerSpace required")
        power_domain = power_spectrum.domain[0]
        fp = power_spectrum
    else:
        power_domain = PowerSpace(domain)
        fp = PS_field(power_domain, power_spectrum)

    return PowerDistributor(domain, power_domain)(fp)


def create_power_operator(domain, power_spectrum, space=None):
    """ Creates a diagonal operator with the given power spectrum.

    Constructs a diagonal operator that is defined on the specified domain.

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        Domain on which the power operator shall be defined.
    power_spectrum : callable or Field
        An object that contains the power spectrum as a function of k.
    space : int
        the domain index on which the power operator will work

    Returns
    -------
    DiagonalOperator
        An operator that implements the given power spectrum.
    """
    domain = DomainTuple.make(domain)
    space = utilities.infer_space(domain, space)
    field = _create_power_field(domain[space], power_spectrum)
    return DiagonalOperator(field, domain, space)


def create_harmonic_smoothing_operator(domain, space, sigma):
    kfunc = domain[space].get_fft_smoothing_kernel_function(sigma)
    return DiagonalOperator(kfunc(domain[space].get_k_length_array()), domain,
                            space)


def full(domain, val):
    if isinstance(domain, (dict, MultiDomain)):
        return MultiField.full(domain, val)
    return Field.full(domain, val)


def from_random(random_type, domain, dtype=np.float64, **kwargs):
    if isinstance(domain, (dict, MultiDomain)):
        return MultiField.from_random(random_type, domain, dtype, **kwargs)
    return Field.from_random(random_type, domain, dtype, **kwargs)


def from_global_data(domain, arr, sum_up=False):
    if isinstance(domain, (dict, MultiDomain)):
        return MultiField.from_global_data(domain, arr, sum_up)
    return Field.from_global_data(domain, arr, sum_up)


def from_local_data(domain, arr):
    if isinstance(domain, (dict, MultiDomain)):
        return MultiField.from_local_data(domain, arr)
    return Field.from_local_data(domain, arr)


def makeDomain(domain):
    if isinstance(domain, (MultiDomain, dict)):
        return MultiDomain.make(domain)
    return DomainTuple.make(domain)


def makeOp(input):
    if input is None:
        return None
    if isinstance(input, Field):
        return DiagonalOperator(input)
    if isinstance(input, MultiField):
        return BlockDiagonalOperator(
            input.domain, tuple(makeOp(val) for val in input.values()))
    raise NotImplementedError


def domain_union(domains):
    if isinstance(domains[0], DomainTuple):
        if any(dom != domains[0] for dom in domains[1:]):
            raise ValueError("domain mismatch")
        return domains[0]
    return MultiDomain.union(domains)


# Arithmetic functions working on Fields


_current_module = sys.modules[__name__]

for f in ["sqrt", "exp", "log", "tanh", "sigmoid",
          "conjugate", 'sin', 'cos', 'tan', 'sinh', 'cosh',
          'absolute', 'one_over', 'sinc']:
    def func(f):
        def func2(x):
            from .linearization import Linearization
            from .operators.operator import Operator
            if isinstance(x, (Field, MultiField, Linearization, Operator)):
                return getattr(x, f)()
            else:
                return getattr(np, f)(x)
        return func2
    setattr(_current_module, f, func(f))


def clip(a, a_min=None, a_max=None):
    return a.clip(a_min, a_max)


def get_default_codomain(domainoid, space=None):
    """For `RGSpace`, returns the harmonic partner domain.
    For `DomainTuple`, returns a copy of the object in which the domain
    indexed by `space` is substituted by its harmonic partner domain.
    In this case, if `space` is None, it is set to 0 if the `DomainTuple`
    contains exactly one domain.

    Parameters
    ----------
    domain: `RGSpace` or `DomainTuple`
        Domain for which to constuct the default harmonic partner
    space: int
        Optional index of the subdomain to be replaced by its default
        codomain. `domain[space]` must be of class `RGSpace`.
    """
    from .domains.rg_space import RGSpace
    if isinstance(domainoid, RGSpace):
        return domainoid.get_default_codomain()
    if not isinstance(domainoid, DomainTuple):
        raise TypeError(
            'Works only on RGSpaces and DomainTuples containing those')
    space = utilities.infer_space(domainoid, space)
    if not isinstance(domainoid[space], RGSpace):
        raise TypeError("can only codomain RGSpaces")
    ret = [dom for dom in domainoid]
    ret[space] = domainoid[space].get_default_codomain()
    return DomainTuple.make(ret)
