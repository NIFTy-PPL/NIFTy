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
from builtins import range
import numpy as np
from ..field import Field
from ..domain_tuple import DomainTuple
from .endomorphic_operator import EndomorphicOperator
from ..nifty_utilities import cast_iseq_to_tuple

class DiagonalOperator(EndomorphicOperator):
    """ NIFTY class for diagonal operators.

    The NIFTY DiagonalOperator class is a subclass derived from the
    EndomorphicOperator. It multiplies an input field pixel-wise with its
    diagonal.


    Parameters
    ----------
    diagonal : Field
        The diagonal entries of the operator.
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
        Indicates whether the operator is self_adjoint or not.

    See Also
    --------
    EndomorphicOperator

    """

    # ---Overwritten properties and methods---

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
            # if nspc==len(self.diagonal.domain.domains, we could do some optimization
            for i, j  in enumerate(self._spaces):
                if diagonal.domain[i] != self._domain[j]:
                    raise ValueError("domain mismatch")

        self._diagonal = diagonal.weight(1)
        self._self_adjoint = None
        self._unitary = None

    def _times(self, x):
        return self._times_helper(x, lambda z: z.__mul__)

    def _adjoint_times(self, x):
        return self._times_helper(x, lambda z: z.conjugate().__mul__)

    def _inverse_times(self, x):
        return self._times_helper(x, lambda z: z.__rtruediv__)

    def _adjoint_inverse_times(self, x):
        return self._times_helper(x, lambda z: z.conjugate().__rtruediv__)

    def diagonal(self):
        """ Returns the diagonal of the Operator.

        Returns
        -------
        out : Field
            The diagonal of the Operator.

        """
        return self._diagonal.weight(-1)

    # ---Mandatory properties and methods---

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

    # ---Added properties and methods---

    def _times_helper(self, x, operation):
        if self._spaces is None:
            return operation(self._diagonal)(x)

        active_axes = []
        for space_index in self._spaces:
            active_axes += x.domain.axes[space_index]

        reshaper = [x.shape[i] if i in active_axes else 1
                    for i in range(len(x.shape))]
        reshaped_local_diagonal = np.reshape(self._diagonal.val, reshaper)

        # here the actual multiplication takes place
        return Field(x.domain, val=operation(reshaped_local_diagonal)(x.val))
