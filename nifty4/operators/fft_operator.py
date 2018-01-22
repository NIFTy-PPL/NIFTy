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
from ..domain_tuple import DomainTuple
from ..spaces.rg_space import RGSpace
from ..utilities import infer_space
from .linear_operator import LinearOperator
from .fft_operator_support import RGRGTransformation, SphericalTransformation


class FFTOperator(LinearOperator):
    """Transforms between a pair of position and harmonic domains.

    Built-in domain pairs are

      - a harmonic and a non-harmonic RGSpace (with matching distances)
      - a HPSpace and a LMSpace
      - a GLSpace and a LMSpace

    Within a domain pair, both orderings are possible.

    The operator provides a "times" and an "adjoint_times" operation.
    For a pair of RGSpaces, the "adjoint_times" operation is equivalent to
    "inverse_times"; for the sphere-related domains this is not the case, since
    the operator matrix is not square.

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
        Whenever "domain" is an RGSpace, the codomain (and its parameters) are
        uniquely determined.
        For GLSpace, HPSpace, and LMSpace, a sensible (but not unique)
        co-domain is chosen that should work satisfactorily in most situations,
        but for full control, the user should explicitly specify a codomain.
    """

    def __init__(self, domain, target=None, space=None):
        super(FFTOperator, self).__init__()

        # Initialize domain and target
        self._domain = DomainTuple.make(domain)
        self._space = infer_space(self._domain, space)

        adom = self.domain[self._space]
        if target is None:
            target = adom.get_default_codomain()

        self._target = [dom for dom in self.domain]
        self._target[self._space] = target
        self._target = DomainTuple.make(self._target)
        adom.check_codomain(target)
        target.check_codomain(adom)

        if self._target[self._space].harmonic:
            pdom, hdom = (self._domain, self._target)
        else:
            pdom, hdom = (self._target, self._domain)
        if isinstance(pdom[self._space], RGSpace):
            self._trafo = RGRGTransformation(pdom, hdom, self._space)
        else:
            self._trafo = SphericalTransformation(pdom, hdom, self._space)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if np.issubdtype(x.dtype, np.complexfloating):
            res = (self._trafo.transform(x.real) +
                   1j * self._trafo.transform(x.imag))
        else:
            res = self._trafo.transform(x)
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
