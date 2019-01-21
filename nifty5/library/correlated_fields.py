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

from ..domain_tuple import DomainTuple
from ..operators.contraction_operator import ContractionOperator
from ..operators.distributors import PowerDistributor
from ..operators.harmonic_operators import HarmonicTransformOperator
from ..operators.simple_linear_operators import ducktape


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
    Correlated field : Operator
    """
    tgt = DomainTuple.make(target)
    if len(tgt) > 1:
        raise ValueError
    if codomain is None:
        codomain = tgt[0].get_default_codomain()
    h_space = codomain
    ht = HarmonicTransformOperator(h_space, target=tgt[0])
    p_space = amplitude_operator.target[0]
    power_distributor = PowerDistributor(h_space, p_space)
    A = power_distributor(amplitude_operator)
    vol = h_space.scalar_dvol**-0.5
    return ht(vol*A*ducktape(h_space, None, name))


def MfCorrelatedField(target, amplitudes, name='xi'):
    """Constructs an operator which turns white Gaussian excitation fields
    into a correlated field defined on a DomainTuple with two entries and two
    separate correlation structures.

    This operator may be used as a model for multi-frequency reconstructions
    with a correlation structure in both spatial and energy direction.

    Parameters
    ----------
    target : Domain, DomainTuple or tuple of Domain
        Target of the operator. Must contain exactly one space.
    amplitudes: iterable of Operator
        List of two amplitude operators.
    name : string
        :class:`MultiField` key for xi-field.

    Returns
    -------
    Correlated field : Operator
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
    a = reduce(lambda x, y: x*y, a)
    A = pd @ a
    vol = reduce(lambda x, y: x*y, [sp.scalar_dvol**-0.5 for sp in hsp])
    return ht(vol*A*ducktape(hsp, None, name))
