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
from .. import DomainTuple
from ..spaces import RGSpace
from ..utilities import infer_space
from .linear_operator import LinearOperator
from .fft_operator_support import RGRGTransformation, SphericalTransformation


class FFTOperator(LinearOperator):
    """Transforms between a pair of harmonic and position domains.

    Built-in domain pairs are
      - harmonic RGSpace / nonharmonic RGSpace (with matching distances)
      - LMSpace / HPSpace
      - LMSpace / GLSpace
    The times() operation always transforms from the harmonic to the
    position domain.

    Parameters
    ----------
    domain: Space or single-element tuple of Spaces
        The domain of the data that is input by "times" and output by
        "adjoint_times".
    space: the index of the space on which the operator should act
        If None, it is set to 0 if domain contains exactly one space
    target: Space or single-element tuple of Spaces (optional)
        The domain of the data that is output by "times" and input by
        "adjoint_times".
        If omitted, a co-domain will be chosen automatically.
        Whenever "domain" is a harmonic RGSpace, the codomain
        (and its parameters) are uniquely determined.
        For LMSpace, a sensible (but not unique)
        co-domain is chosen that should work satisfactorily in most situations,
        but for full control, the user should explicitly specify a codomain.
    """

    def __init__(self, domain, target=None, space=None):
        super(FFTOperator, self).__init__()

        # Initialize domain and target
        self._domain = DomainTuple.make(domain)
        self._space = infer_space(self._domain, space)
        if not self._domain[self._space].harmonic:
            raise TypeError("H2POperator must work on a harmonic domain")

        adom = self.domain[self._space]
        if target is None:
            target = adom.get_default_codomain()

        self._target = [dom for dom in self.domain]
        self._target[self._space] = target
        self._target = DomainTuple.make(self._target)
        adom.check_codomain(target)
        target.check_codomain(adom)

        hdom, pdom = (self._domain, self._target)
        if isinstance(pdom[self._space], RGSpace):
            self._trafo = RGRGTransformation(hdom, pdom, self._space)
        else:
            self._trafo = SphericalTransformation(hdom, pdom, self._space)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if np.issubdtype(x.dtype, np.complexfloating):
            res = (self._trafo.apply(x.real, mode) +
                   1j * self._trafo.apply(x.imag, mode))
        else:
            res = self._trafo.apply(x, mode)
        return res

    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def capability(self):
        res = self.TIMES | self.ADJOINT_TIMES
        if self._trafo.unitary:
            res |= self.INVERSE_TIMES | self.ADJOINT_INVERSE_TIMES
        return res
