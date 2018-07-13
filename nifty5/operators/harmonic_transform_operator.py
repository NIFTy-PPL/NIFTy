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

from __future__ import absolute_import, division, print_function

from .. import utilities
from ..compat import *
from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from .hartley_operator import HartleyOperator
from .linear_operator import LinearOperator
from .sht_operator import SHTOperator


class HarmonicTransformOperator(LinearOperator):
    """Transforms between a harmonic domain and a position domain counterpart.

    Built-in domain pairs are
      - a harmonic and a non-harmonic RGSpace (with matching distances)
      - an LMSpace and a HPSpace
      - an LMSpace and a GLSpace

    The supported operations are times() and adjoint_times().

    Parameters
    ----------
    domain : Domain, tuple of Domain or DomainTuple
        The domain of the data that is input by "times" and output by
        "adjoint_times".
    target : Domain, optional
        The target domain of the transform operation.
        If omitted, a domain will be chosen automatically.
        Whenever the input domain of the transform is an RGSpace, the codomain
        (and its parameters) are uniquely determined.
        For LMSpace, a GLSpace of sufficient resolution is chosen.
    space : int, optional
        The index of the domain on which the operator should act
        If None, it is set to 0 if domain contains exactly one subdomain.
        domain[space] must be a harmonic domain.
    """

    def __init__(self, domain, target=None, space=None):
        super(HarmonicTransformOperator, self).__init__()

        domain = DomainTuple.make(domain)
        space = utilities.infer_space(domain, space)

        hspc = domain[space]
        if not hspc.harmonic:
            raise TypeError(
                "HarmonicTransformOperator only works on a harmonic space")
        if isinstance(hspc, RGSpace):
            self._op = HartleyOperator(domain, target, space)
        else:
            self._op = SHTOperator(domain, target, space)

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._op.apply(x, mode)

    @property
    def domain(self):
        return self._op.domain

    @property
    def target(self):
        return self._op.target

    @property
    def capability(self):
        return self.TIMES | self.ADJOINT_TIMES
