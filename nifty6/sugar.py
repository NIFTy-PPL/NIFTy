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
from time import time

import numpy as np

from .logger import logger
from . import utilities
from .domain_tuple import DomainTuple
from .domains.power_space import PowerSpace
from .field import Field
from .multi_domain import MultiDomain
from .multi_field import MultiField
from .operators.block_diagonal_operator import BlockDiagonalOperator
from .operators.diagonal_operator import DiagonalOperator
from .operators.distributors import PowerDistributor
from .operators.operator import Operator
from .operators.scaling_operator import ScalingOperator
from .plot import Plot
from . import pointwise


__all__ = ['PS_field', 'power_analyze', 'create_power_operator',
           'create_harmonic_smoothing_operator', 'from_random',
           'full', 'makeField',
           'makeDomain', 'get_signal_variance', 'makeOp', 'domain_union',
           'get_default_codomain', 'single_plot', 'exec_time',
           'calculate_position'] + list(pointwise.ptw_dict.keys())


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
    Field
        A field defined on (pspace,) containing the computed function values
    """
    if not isinstance(pspace, PowerSpace):
        raise TypeError
    data = func(pspace.k_lengths)
    return Field(DomainTuple.make(pspace), data)


def get_signal_variance(spec, space):
    """
    Computes how much a field with a given power spectrum will vary in space

    This is a small helper function that computes the expected variance
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
    return k_field.weight(2).s_sum()


def _single_power_analyze(field, idx, binbounds):
    power_domain = PowerSpace(field.domain[idx], binbounds)
    pd = PowerDistributor(field.domain, power_domain, idx)
    return pd.adjoint_times(field.weight(1)).weight(-1)  # divides by bin size


# MR FIXME: this function is not well suited for analyzing more than one
# subdomain at once, because it allows only one set of binbounds.
def power_analyze(field, spaces=None, binbounds=None,
                  keep_phase_information=False):
    """Computes the power spectrum for a subspace of `field`.

    Creates a PowerSpace for the space addressed by `spaces` with the given
    binning and computes the power spectrum as a :class:`Field` over this
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
        of `field`.
        If True, return a complex-valued result whose real component
        contains the power spectrum computed from the real part of `field`,
        and whose imaginary component contains the power
        spectrum computed from the imaginary part of `field`.
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
    if not callable(power_spectrum):  # we have a Field defined on a PowerSpace
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
    """Creates a diagonal operator with the given power spectrum.

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
    """Creates an operator which smoothes a subspace of a harmonic domain.

    Parameters
    ----------
    domain: DomainTuple
        The total domain and target of the operator
    space : int
        the index of the subspace on which the operator acts.
        This must be a harmonic space
    sigma : float
        The sigma of the Gaussian smoothing kernel

    Returns
    -------
    DiagonalOperator
        The requested smoothing operator
    """
    kfunc = domain[space].get_fft_smoothing_kernel_function(sigma)
    return DiagonalOperator(kfunc(domain[space].get_k_length_array()), domain,
                            space)


def full(domain, val):
    """Convenience function creating Fields/MultiFields with uniform values.

    Parameters
    ----------
    domain : Domainoid
        the intended domain of the output field
    val : scalar value
        the uniform value to be placed into all entries of the result

    Returns
    -------
    Field or MultiField
        The newly created uniform field
    """
    if isinstance(domain, (dict, MultiDomain)):
        return MultiField.full(domain, val)
    return Field.full(domain, val)


def from_random(random_type, domain, dtype=np.float64, **kwargs):
    """Convenience function creating Fields/MultiFields with random values.

    Parameters
    ----------
    random_type : 'pm1', 'normal', or 'uniform'
            The random distribution to use.
    domain : Domainoid
        the intended domain of the output field
    dtype : type
        data type of the output field (e.g. numpy.float64)
    **kwargs : additional parameters for the random distribution
        ('mean' and 'std' for 'normal', 'low' and 'high' for 'uniform')

    Returns
    -------
    Field or MultiField
        The newly created random field

    Notes
    -----
    The ordering of the keys within a multi-domain WILL change the resulting
    :class:`MultiField`. Note, since keys in a :class:`MultiDomain` are sorted,
    renaming keys may also impacts the ordering of the generated random numbers
    within the returned :class:`MultiField`.
    """
    if isinstance(domain, (dict, MultiDomain)):
        return MultiField.from_random(random_type, domain, dtype, **kwargs)
    return Field.from_random(random_type, domain, dtype, **kwargs)


def makeField(domain, arr):
    """Convenience function creating Fields/MultiFields from Numpy arrays or
    dicts of Numpy arrays.

    Parameters
    ----------
    domain : Domainoid
        the intended domain of the output field
    arr : Numpy array if `domain` corresponds to a `DomainTuple`,
          dictionary of Numpy arrays if `domain` corresponds to a `MultiDomain`

    Returns
    -------
    Field or MultiField
        The newly created random field
    """
    if isinstance(domain, (dict, MultiDomain)):
        return MultiField.from_raw(domain, arr)
    return Field.from_raw(domain, arr)


def makeDomain(domain):
    """Convenience function creating DomainTuples/MultiDomains Domainoids.

    Parameters
    ----------
    domain : Domainoid (can be DomainTuple, MultiDomain, dict, Domain or list of Domains)
        the description of the requested (multi-)domain

    Returns
    -------
    DomainTuple or MultiDomain
        The newly created domain object
    """
    if isinstance(domain, (MultiDomain, dict)):
        return MultiDomain.make(domain)
    return DomainTuple.make(domain)


def makeOp(input, dom=None):
    """Converts a Field or MultiField to a diagonal operator.

    Parameters
    ----------
    input : None, Field or MultiField
        - if None, None is returned.
        - if Field on scalar-domain, a ScalingOperator with the coefficient
            given by the Field is returned.
        - if Field, a DiagonalOperator with the coefficients given by this
            Field is returned.
        - if MultiField, a BlockDiagonalOperator with entries given by this
            MultiField is returned.

    dom : DomainTuple or MultiDomain
        if `input` is a scalar, this is used as the operator's domain

    Notes
    -----
    No volume factors are applied.
    """
    if input is None:
        return None
    if np.isscalar(input):
        if not isinstance(dom, (DomainTuple, MultiDomain)):
            raise TypeError("need proper `dom` argument")
        return ScalingOperator(dom, input)
    if dom is not None:
        if not dom == input.domain:
            raise ValueError("domain mismatch")
    if input.domain is DomainTuple.scalar_domain():
        return ScalingOperator(input.domain, input.val[()])
    if isinstance(input, Field):
        return DiagonalOperator(input)
    if isinstance(input, MultiField):
        return BlockDiagonalOperator(
            input.domain, {key: makeOp(val) for key, val in input.items()})
    raise NotImplementedError


def domain_union(domains):
    """Computes the union of multiple DomainTuples/MultiDomains.

    Parameters
    ----------
    domains : list of DomainTuple or MultiDomain
        - if DomainTuple, all entries must be equal
        - if MultiDomain, there must not be any conflicting components
    """
    if isinstance(domains[0], DomainTuple):
        if any(dom != domains[0] for dom in domains[1:]):
            raise ValueError("domain mismatch")
        return domains[0]
    return MultiDomain.union(domains)


# Pointwise functions

_current_module = sys.modules[__name__]

for f in pointwise.ptw_dict.keys():
    def func(f):
        def func2(x, *args, **kwargs):
           return x.ptw(f, *args, **kwargs)
        return func2
    setattr(_current_module, f, func(f))


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
    from .domains.hp_space import HPSpace
    from .domains.gl_space import GLSpace
    from .domains.lm_space import LMSpace
    if isinstance(domainoid, RGSpace):
        return domainoid.get_default_codomain()
    if not isinstance(domainoid, DomainTuple):
        raise TypeError(
            'Works only on RGSpaces and DomainTuples containing those')
    space = utilities.infer_space(domainoid, space)
    if not isinstance(domainoid[space], (RGSpace, HPSpace, GLSpace, LMSpace)):
        raise TypeError("can only codomain structrued spaces")
    ret = [dom for dom in domainoid]
    ret[space] = domainoid[space].get_default_codomain()
    return DomainTuple.make(ret)


def single_plot(field, **kwargs):
    """Creates a single plot using `Plot`.
    Keyword arguments are passed to both `Plot.add` and `Plot.output`.
    """
    p = Plot()
    p.add(field, **kwargs)
    if 'title' in kwargs:
        del(kwargs['title'])
    p.output(**kwargs)


def exec_time(obj, want_metric=True):
    """Times the execution time of an operator or an energy."""
    from .linearization import Linearization
    from .minimization.energy import Energy
    from .operators.energy_operators import EnergyOperator
    if isinstance(obj, Energy):
        t0 = time()
        obj.at(0.99*obj.position)
        logger.info('Energy.at(): {}'.format(time() - t0))

        t0 = time()
        obj.value
        logger.info('Energy.value: {}'.format(time() - t0))
        t0 = time()
        obj.gradient
        logger.info('Energy.gradient: {}'.format(time() - t0))
        t0 = time()
        obj.metric
        logger.info('Energy.metric: {}'.format(time() - t0))

        t0 = time()
        obj.apply_metric(obj.position)
        logger.info('Energy.apply_metric: {}'.format(time() - t0))

        t0 = time()
        obj.metric(obj.position)
        logger.info('Energy.metric(position): {}'.format(time() - t0))
    elif isinstance(obj, Operator):
        want_metric = bool(want_metric)
        pos = from_random('normal', obj.domain)
        t0 = time()
        obj(pos)
        logger.info('Operator call with field: {}'.format(time() - t0))

        lin = Linearization.make_var(pos, want_metric=want_metric)
        t0 = time()
        res = obj(lin)
        logger.info('Operator call with linearization: {}'.format(time() - t0))

        if isinstance(obj, EnergyOperator):
            t0 = time()
            res.gradient
            logger.info('Gradient evaluation: {}'.format(time() - t0))

            if want_metric:
                t0 = time()
                res.metric(pos)
                logger.info('Metric apply: {}'.format(time() - t0))
    else:
        raise TypeError


def calculate_position(operator, output):
    """Finds approximate preimage of an operator for a given output."""
    from .minimization.descent_minimizers import NewtonCG
    from .minimization.iteration_controllers import GradientNormController
    from .minimization.metric_gaussian_kl import MetricGaussianKL
    from .operators.scaling_operator import ScalingOperator
    from .operators.energy_operators import GaussianEnergy, StandardHamiltonian
    if not isinstance(operator, Operator):
        raise TypeError
    if output.domain != operator.target:
        raise TypeError
    if isinstance(output, MultiField):
        cov = 1e-3*max([vv.max() for vv in output.val.values()])**2
    else:
        cov = 1e-3*output.val.max()**2
    invcov = ScalingOperator(output.domain, cov).inverse
    d = output + invcov.draw_sample_with_dtype(dtype=output.dtype, from_inverse=True)
    lh = GaussianEnergy(d, invcov) @ operator
    H = StandardHamiltonian(
        lh, ic_samp=GradientNormController(iteration_limit=200))
    pos = 0.1*from_random('normal', operator.domain)
    minimizer = NewtonCG(GradientNormController(iteration_limit=10))
    for ii in range(3):
        kl = MetricGaussianKL(pos, H, 3, mirror_samples=True)
        kl, _ = minimizer(kl)
        pos = kl.position
    return pos
