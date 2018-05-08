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
from ..field import Field
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
        The diagonal entries of the operator.
    domain : Domain, tuple of Domain or DomainTuple, optional
        The domain on which the Operator's input Field lives.
        If None, use the domain of "diagonal".
    spaces : int or tuple of int, optional
        The elements of "domain" on which the operator acts.
        If None, it acts on all elements.

    Notes
    -----
    Formally, this operator always supports all operation modes (times,
    adjoint_times, inverse_times and inverse_adjoint_times), even if there
    are diagonal elements with value 0 or infinity. It is the user's
    responsibility to apply the operator only in appropriate ways (e.g. call
    inverse_times only if there are no zeros on the diagonal).

    This shortcoming will hopefully be fixed in the future.
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
            for i, j in enumerate(self._spaces):
                if diagonal.domain[i] != self._domain[j]:
                    raise ValueError("domain mismatch")
            if self._spaces == tuple(range(len(self._domain))):
                self._spaces = None  # shortcut

        if self._spaces is not None:
            active_axes = []
            for space_index in self._spaces:
                active_axes += self._domain.axes[space_index]

            if self._spaces[0] == 0:
                self._ldiag = diagonal.local_data
            else:
                self._ldiag = diagonal.to_global_data()
            locshape = dobj.local_shape(self._domain.shape, 0)
            self._reshaper = [shp if i in active_axes else 1
                              for i, shp in enumerate(locshape)]
            self._ldiag = self._ldiag.reshape(self._reshaper)
        else:
            self._ldiag = diagonal.local_data
        self._update_diagmin()

    def __str__(self):
        return 'Diag({})'.format(self(Field.ones(self.domain)).val)

    def _update_diagmin(self):
        self._ldiag.flags.writeable = False
        if not np.issubdtype(self._ldiag.dtype, np.complexfloating):
            lmin = self._ldiag.min() if self._ldiag.size > 0 else 1.
            self._diagmin = dobj.np_allreduce_min(np.array(lmin))[()]

    def _from_ldiag(self, spc, ldiag):
        res = DiagonalOperator.__new__(DiagonalOperator)
        res._domain = self._domain
        if self._spaces is None or spc is None:
            res._spaces = None
        else:
            res._spaces = tuple(set(self._spaces) | set(spc))
        res._ldiag = ldiag
        res._update_diagmin()
        return res

    def _scale(self, fct):
        if not np.isscalar(fct):
            raise TypeError("scalar value required")
        return self._from_ldiag((), self._ldiag*fct)

    def _add(self, sum):
        if not np.isscalar(sum):
            raise TypeError("scalar value required")
        return self._from_ldiag((), self._ldiag+sum)

    def _combine_prod(self, op):
        if not isinstance(op, DiagonalOperator):
            raise TypeError("DiagonalOperator required")
        return self._from_ldiag(op._spaces, self._ldiag*op._ldiag)

    def _combine_sum(self, op, selfneg, opneg):
        if not isinstance(op, DiagonalOperator):
            raise TypeError("DiagonalOperator required")
        tdiag = (self._ldiag * (-1 if selfneg else 1) +
                 op._ldiag * (-1 if opneg else 1))
        return self._from_ldiag(op._spaces, tdiag)

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
    def domain(self):
        return self._domain

    @property
    def capability(self):
        return self._all_ops

    def _flip_modes(self, trafo):
        ADJ = self.ADJOINT_BIT
        INV = self.INVERSE_BIT

        if trafo == 0:
            return self
        if trafo == ADJ and np.issubdtype(self._ldiag.dtype, np.floating):
            return self
        if trafo == ADJ:
            return self._from_ldiag((), self._ldiag.conjugate())
        elif trafo == INV:
            return self._from_ldiag((), 1./self._ldiag)
        elif trafo == ADJ | INV:
            return self._from_ldiag((), 1./self._ldiag.conjugate())
        raise ValueError("invalid operator transformation")

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        if np.issubdtype(self._ldiag.dtype, np.complexfloating):
            raise ValueError("operator not positive definite")
        if self._diagmin < 0.:
            raise ValueError("operator not positive definite")
        if self._diagmin == 0. and from_inverse:
            raise ValueError("operator not positive definite")
        res = Field.from_random(random_type="normal", domain=self._domain,
                                dtype=dtype)
        if from_inverse:
            res.local_data[()] /= np.sqrt(self._ldiag)
        else:
            res.local_data[()] *= np.sqrt(self._ldiag)
        return res
