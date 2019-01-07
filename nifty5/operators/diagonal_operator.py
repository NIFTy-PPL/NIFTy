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

import numpy as np

from .. import dobj, utilities
from ..domain_tuple import DomainTuple
from ..field import Field
from .endomorphic_operator import EndomorphicOperator


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
        self._fill_rest()

    def _fill_rest(self):
        self._ldiag.flags.writeable = False
        self._complex = utilities.iscomplextype(self._ldiag.dtype)
        self._capability = self._all_ops
        if not self._complex:
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
        res._fill_rest()
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
        # shortcut for most common cases
        if mode == 1 or (not self._complex and mode == 2):
            return Field.from_local_data(x.domain, x.local_data*self._ldiag)

        xdiag = self._ldiag
        if self._complex and (mode & 10):  # adjoint or inverse adjoint
            xdiag = xdiag.conj()

        if mode & 3:
            return Field.from_local_data(x.domain, x.local_data*xdiag)
        return Field.from_local_data(x.domain, x.local_data/xdiag)

    def _flip_modes(self, trafo):
        if trafo == self.ADJOINT_BIT and not self._complex:  # shortcut
            return self
        xdiag = self._ldiag
        if self._complex and (trafo & self.ADJOINT_BIT):
            xdiag = xdiag.conj()
        if trafo & self.INVERSE_BIT:
            xdiag = 1./xdiag
        return self._from_ldiag((), xdiag)

    def process_sample(self, samp, from_inverse):
        if (self._complex or (self._diagmin < 0.) or
                (self._diagmin == 0. and from_inverse)):
                    raise ValueError("operator not positive definite")
        if from_inverse:
            res = samp.local_data/np.sqrt(self._ldiag)
        else:
            res = samp.local_data*np.sqrt(self._ldiag)
        return Field.from_local_data(self._domain, res)

    def draw_sample(self, from_inverse=False, dtype=np.float64):
        res = Field.from_random(random_type="normal", domain=self._domain,
                                dtype=dtype)
        return self.process_sample(res, from_inverse)

    def __repr__(self):
        return "DiagonalOperator"
