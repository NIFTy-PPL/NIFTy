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

from __future__ import division
import numpy as np
from ..field import Field, sqrt
from ..domain_tuple import DomainTuple
from .endomorphic_operator import EndomorphicOperator
from .. import utilities
from .. import dobj


class DiagonalOperator(EndomorphicOperator):
    """ NIFTy class for diagonal operators.

    The NIFTy DiagonalOperator class is a subclass derived from the
    EndomorphicOperator. It multiplies an input field pixel-wise with its
    diagonal.

    Parameters
    ----------
    diagonal : Field
        The diagonal entries of the operator
        (already containing volume factors).
    domain : tuple of DomainObjects
        The domain on which the Operator's input Field lives.
        If None, use the domain of "diagonal".
    spaces : tuple of int
        The elements of "domain" on which the operator acts.
        If None, it acts on all elements.

    NOTE: the fields given to __init__ and returned from .diagonal are
    considered to be non-bare, i.e. during operator application, no additional
    volume factors are applied!
    """

    def __init__(self, diagonal, domain=None, spaces=None):
        super(DiagonalOperator, self).__init__()

        if not isinstance(diagonal, Field):
            raise TypeError("Field object required")
        if domain is None:
            self._domain = diagonal.domain
        else:
            self._domain = DomainTuple.make(domain)
        if spaces is None:
            self._spaces = None
            if diagonal.domain != self._domain:
                raise ValueError("domain mismatch")
        else:
            self._spaces = utilities.parse_spaces(spaces, len(self._domain))
            if len(self._spaces) != len(diagonal.domain):
                raise ValueError("spaces and domain must have the same length")
            # if nspc==len(self.diagonal.domain),
            # we could do some optimization
            for i, j in enumerate(self._spaces):
                if diagonal.domain[i] != self._domain[j]:
                    raise ValueError("domain mismatch")
            if self._spaces == tuple(range(len(self._domain))):
                self._spaces = None  # shortcut

        self._diagonal = diagonal.locked_copy()

        if self._spaces is not None:
            active_axes = []
            for space_index in self._spaces:
                active_axes += self._domain.axes[space_index]

            if self._spaces[0] == 0:
                self._ldiag = dobj.local_data(self._diagonal.val)
            else:
                self._ldiag = dobj.to_global_data(self._diagonal.val)
            locshape = dobj.local_shape(self._domain.shape, 0)
            self._reshaper = [shp if i in active_axes else 1
                              for i, shp in enumerate(locshape)]
            self._ldiag = self._ldiag.reshape(self._reshaper)

        else:
            self._ldiag = dobj.local_data(self._diagonal.val)

    def apply(self, x, mode):
        self._check_input(x, mode)

        if mode == self.TIMES:
            return Field(x.domain, val=x.val*self._ldiag)
        elif mode == self.ADJOINT_TIMES:
            if np.issubdtype(self._ldiag.dtype, np.floating):
                return Field(x.domain, val=x.val*self._ldiag)
            else:
                return Field(x.domain, val=x.val*self._ldiag.conj())
        elif mode == self.INVERSE_TIMES:
            return Field(x.domain, val=x.val/self._ldiag)
        else:
            if np.issubdtype(self._ldiag.dtype, np.floating):
                return Field(x.domain, val=x.val/self._ldiag)
            else:
                return Field(x.domain, val=x.val/self._ldiag.conj())

    @property
    def diagonal(self):
        """ Returns the diagonal of the Operator."""
        return self._diagonal

    @property
    def domain(self):
        return self._domain

    @property
    def capability(self):
        return self._all_ops

    @property
    def inverse(self):
        return DiagonalOperator(1./self._diagonal, self._domain, self._spaces)

    @property
    def adjoint(self):
        return DiagonalOperator(self._diagonal.conjugate(), self._domain,
                                self._spaces)

    def draw_sample(self):
        if self._spaces is not None:
            raise ValueError("Cannot draw (yet) from this operator")

        res = Field.from_random(random_type="normal",
                                domain=self._domain,
                                dtype=self._diagonal.dtype)
        res *= sqrt(self._diagonal)
        return res
