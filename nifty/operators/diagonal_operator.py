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

from __future__ import division
import numpy as np
from ..field import Field
from ..domain_tuple import DomainTuple
from .endomorphic_operator import EndomorphicOperator
from ..utilities import cast_iseq_to_tuple
from .. import dobj


class DiagonalOperator(EndomorphicOperator):
    """ NIFTY class for diagonal operators.

    The NIFTY DiagonalOperator class is a subclass derived from the
    EndomorphicOperator. It multiplies an input field pixel-wise with its
    diagonal.

    Parameters
    ----------
    diagonal : Field
        The diagonal entries of the operator
        (already containing volume factors).
    domain : tuple of DomainObjects, i.e. Spaces and FieldTypes
        The domain on which the Operator's input Field lives.
        If None, use the domain of "diagonal".
    spaces : tuple of int
        The elements of "domain" on which the operator acts.
        If None, it acts on all elements.

    Attributes
    ----------
    domain : tuple of DomainObjects, i.e. Spaces and FieldTypes
        The domain on which the Operator's input Field lives.
    target : tuple of DomainObjects, i.e. Spaces and FieldTypes
        The domain in which the outcome of the operator lives. As the Operator
        is endomorphic this is the same as its domain.
    unitary : boolean
        Indicates whether the Operator is unitary or not.
    self_adjoint : boolean
        Indicates whether the operator is self-adjoint or not.

    NOTE: the fields given to __init__ and returned from .diagonal() are
    considered to be non-bare, i.e. during operator application, no additional
    volume factors are applied!

    See Also
    --------
    EndomorphicOperator
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
            self._spaces = cast_iseq_to_tuple(spaces)
            nspc = len(self._spaces)
            if nspc != len(diagonal.domain.domains):
                raise ValueError("spaces and domain must have the same length")
            if nspc > len(self._domain.domains):
                raise ValueError("too many spaces")
            if nspc > len(set(self._spaces)):
                raise ValueError("non-unique space indices")
            # if nspc==len(self.diagonal.domain),
            # we could do some optimization
            for i, j in enumerate(self._spaces):
                if diagonal.domain[i] != self._domain[j]:
                    raise ValueError("domain mismatch")
            if self._spaces == tuple(range(len(self._domain.domains))):
                self._spaces = None  # shortcut

        self._diagonal = diagonal.copy()

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

        self._self_adjoint = None
        self._unitary = None

    def _times(self, x):
        return Field(x.domain, val=x.val*self._ldiag)

    def _adjoint_times(self, x):
        return Field(x.domain, val=x.val*self._ldiag.conj())

    def _inverse_times(self, x):
        return Field(x.domain, val=x.val/self._ldiag)

    def _adjoint_inverse_times(self, x):
        return Field(x.domain, val=x.val/self._ldiag.conj())

    def diagonal(self):
        """ Returns the diagonal of the Operator."""
        return self._diagonal.copy()

    @property
    def domain(self):
        return self._domain

    @property
    def self_adjoint(self):
        if self._self_adjoint is None:
            if not issubclass(self._diagonal.dtype.type, np.complexfloating):
                self._self_adjoint = True
            else:
                self._self_adjoint = (self._diagonal.val.imag == 0).all()
        return self._self_adjoint

    @property
    def unitary(self):
        if self._unitary is None:
            self._unitary = (abs(self._diagonal.val) == 1.).all()
        return self._unitary
