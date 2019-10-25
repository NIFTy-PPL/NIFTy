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

from functools import reduce
from operator import mul

from ..domain_tuple import DomainTuple
from ..field import Field
from ..operators.adder import Adder
from ..operators.contraction_operator import ContractionOperator
from ..operators.distributors import PowerDistributor
from ..operators.harmonic_operators import HarmonicTransformOperator
from ..operators.operator import Operator
from ..operators.simple_linear_operators import VdotOperator, ducktape
from ..sugar import full, makeOp


def CorrelatedField(target, amplitude_operator, name='xi', codomain=None):
    """Constructs an operator which turns a white Gaussian excitation field
    into a correlated field.

    This function returns an operator which implements:

        ht @ (vol * A * xi),

    where `ht` is a harmonic transform operator, `A` is the square root of the
    prior covariance and `xi` is the excitation field.

    Parameters
    ----------
    target : Domain, DomainTuple or tuple of Domain
        Target of the operator. Must contain exactly one space.
    amplitude_operator: Operator
    name : string
        :class:`MultiField` key for the xi-field.
    codomain : Domain
        The codomain for target[0]. If not supplied, it is inferred.

    Returns
    -------
    Operator
        Correlated field

    Notes
    -----
    In NIFTy, non-harmonic RGSpaces are by definition periodic. Therefore
    the operator constructed by this method will output a correlated field
    with *periodic* boundary conditions. If a non-periodic field is needed,
    one needs to combine this operator with a :class:`FieldZeroPadder`.
    """
    tgt = DomainTuple.make(target)
    if len(tgt) > 1:
        raise ValueError
    if codomain is None:
        codomain = tgt[0].get_default_codomain()
    h_space = codomain
    ht = HarmonicTransformOperator(h_space, target=tgt[0])
    if isinstance(amplitude_operator, Operator):
        p_space = amplitude_operator.target[0]
    elif isinstance(amplitude_operator, Field):
        p_space = amplitude_operator.domain[0]
    else:
        raise TypeError
    power_distributor = PowerDistributor(h_space, p_space)
    A = power_distributor(amplitude_operator)
    vol = h_space.scalar_dvol**-0.5
    xi = ducktape(h_space, None, name)
    # When doubling the resolution of `tgt` the value of the highest k-mode
    # will scale with a square root. `vol` cancels this effect such that the
    # same power spectrum can be used for the spaces with the same volume,
    # different resolutions and the same object in them.
    if isinstance(amplitude_operator, Field):
        return ht(makeOp(A) @ xi).scale(vol)
    return ht(A*xi).scale(vol)


def MfCorrelatedField(target, amplitudes, name='xi'):
    """Constructs an operator which turns white Gaussian excitation fields
    into a correlated field defined on a DomainTuple with two entries and two
    separate correlation structures.

    This operator may be used as a model for multi-frequency reconstructions
    with a correlation structure in both spatial and energy direction.

    Parameters
    ----------
    target : Domain, DomainTuple or tuple of Domain
        Target of the operator. Must contain exactly two spaces.
    amplitudes: iterable of Operator
        List of two amplitude operators.
    name : string
        :class:`MultiField` key for xi-field.

    Returns
    -------
    Operator
        Correlated field

    Notes
    -----
    In NIFTy, non-harmonic RGSpaces are by definition periodic. Therefore
    the operator constructed by this method will output a correlated field
    with *periodic* boundary conditions. If a non-periodic field is needed,
    one needs to combine this operator with a :class:`FieldZeroPadder` or even
    two (one for the energy and one for the spatial subdomain)
    """
    tgt = DomainTuple.make(target)
    if len(tgt) != 2:
        raise ValueError
    if len(amplitudes) != 2:
        raise ValueError

    hsp = DomainTuple.make([tt.get_default_codomain() for tt in tgt])
    ht1 = HarmonicTransformOperator(hsp, target=tgt[0], space=0)
    ht2 = HarmonicTransformOperator(ht1.target, target=tgt[1], space=1)
    ht = ht2 @ ht1

    psp = [aa.target[0] for aa in amplitudes]
    pd0 = PowerDistributor(hsp, psp[0], 0)
    pd1 = PowerDistributor(pd0.domain, psp[1], 1)
    pd = pd0 @ pd1

    dd0 = ContractionOperator(pd.domain, 1).adjoint
    dd1 = ContractionOperator(pd.domain, 0).adjoint
    d = [dd0, dd1]

    a = [dd @ amplitudes[ii] for ii, dd in enumerate(d)]
    a = reduce(mul, a)
    A = pd @ a
    # For `vol` see comment in `CorrelatedField`
    vol = reduce(mul, [sp.scalar_dvol**-0.5 for sp in hsp])
    return ht(vol*A*ducktape(hsp, None, name))


def CorrelatedFieldNormAmplitude(target,
                                 amplitudes,
                                 stdmean,
                                 stdstd,
                                 names=['xi', 'std']):
    """Constructs an operator which turns white Gaussian excitation fields
    into a correlated field defined on a DomainTuple with n entries and n
    separate correlation structures.

    This operator may be used as a model for multi-frequency reconstructions
    with a correlation structure in both spatial and energy direction.

    Parameters
    ----------
    target : Domain, DomainTuple or tuple of Domain
        Target of the operator. Must contain exactly n spaces.
    amplitudes: Opertor, iterable of Operator
        Amplitude operator if n = 1 or list of n amplitude operators.
    stdmean : float
        Prior mean of the overall standart deviation.
    stdstd : float
        Prior standart deviation of the overall standart deviation.
    names : iterable of string
        :class:`MultiField` keys for xi-field and std-field.

    Returns
    -------
    Operator
        Correlated field

    Notes
    -----
    In NIFTy, non-harmonic RGSpaces are by definition periodic. Therefore
    the operator constructed by this method will output a correlated field
    with *periodic* boundary conditions. If a non-periodic field is needed,
    one needs to combine this operator with a :class:`FieldZeroPadder` or even
    two (one for the energy and one for the spatial subdomain)
    """

    amps = [
        amplitudes,
    ] if isinstance(amplitudes, (Operator, Field)) else amplitudes

    cls = Operator if isinstance(amps[0], Operator) else Field
    assert all([isinstance(aa, cls) for aa in amps])

    tgt = DomainTuple.make(target)
    if len(tgt) != len(amps):
        raise ValueError
    stdmean, stdstd = float(stdmean), float(stdstd)
    if stdstd <= 0:
        raise ValueError

    if cls == Field:
        psp = [aa.domain[0] for aa in amps]
    else:
        psp = [aa.target[0] for aa in amps]
    hsp = DomainTuple.make([tt.get_default_codomain() for tt in tgt])

    ht = HarmonicTransformOperator(hsp, target=tgt[0], space=0)
    pd = PowerDistributor(hsp, psp[0], 0)

    for i in range(1, len(amps)):
        ht = HarmonicTransformOperator(ht.target, target=tgt[i], space=i) @ ht
        pd = pd @ PowerDistributor(pd.domain, psp[i], space=i)

    spaces = tuple(range(len(amps)))

    a = ContractionOperator(pd.domain, spaces[1:]).adjoint(amps[0])
    for i in range(1, len(amps)):
        a = a*(ContractionOperator(
            pd.domain, spaces[:i] + spaces[(i + 1):]).adjoint(amps[i]))

    expander = VdotOperator(full(pd.domain, 1.)).adjoint

    Std = stdstd*ducktape(expander.domain, None, names[1])
    Std = expander @ (Adder(full(expander.domain, stdmean)) @ Std).exp()

    if cls == Field:
        A = pd @ (makeOp(a) @ Std)
    else:
        A = pd @ (Std*a)
    # For `vol` see comment in `CorrelatedField`
    vol = reduce(mul, [sp.scalar_dvol**-0.5 for sp in hsp])
    return ht(vol*A*ducktape(hsp, None, names[0]))
