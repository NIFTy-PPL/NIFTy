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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import partial
from operator import mul

import numpy as np

from .. import utilities
from ..domain_tuple import DomainTuple
from ..field import Field
from .endomorphic_operator import EndomorphicOperator


class DiagonalOperator(EndomorphicOperator):
    """Represents a :class:`LinearOperator` which is diagonal.

    The NIFTy DiagonalOperator class is a subclass derived from the
    :class:`EndomorphicOperator`. It multiplies an input field pixel-wise with
    its diagonal.

    Parameters
    ----------
    diagonal : :class:`nifty8.field.Field`
        The diagonal entries of the operator.
    domain : Domain, tuple of Domain or DomainTuple, optional
        The domain on which the Operator's input Field is defined.
        If None, use the domain of "diagonal".
    spaces : int or tuple of int, optional
        The elements of "domain" on which the operator acts.
        If None, it acts on all elements.
    sampling_dtype :
        If this operator represents the covariance of a Gaussian probabilty
        distribution, `sampling_dtype` specifies if it is real or complex
        Gaussian. If `sampling_dtype` is `None`, the operator cannot be used as
        a covariance, i.e. no samples can be drawn. Default: None.

    Notes
    -----
    Formally, this operator always supports all operation modes (times,
    adjoint_times, inverse_times and inverse_adjoint_times), even if there
    are diagonal elements with value 0 or infinity. It is the user's
    responsibility to apply the operator only in appropriate ways (e.g. call
    inverse_times only if there are no zeros on the diagonal).

    This shortcoming will hopefully be fixed in the future.
    """

    def __init__(self, diagonal, domain=None, spaces=None, sampling_dtype=None):
        if not isinstance(diagonal, Field):
            raise TypeError("Field object required")
        utilities.check_dtype_or_none(sampling_dtype)
        self._dtype = sampling_dtype
        if domain is None:
            self._domain = diagonal.domain
        else:
            self._domain = DomainTuple.make(domain)
        if spaces is None:
            self._spaces = None
            utilities.check_object_identity(diagonal.domain, self._domain)
        else:
            self._spaces = utilities.parse_spaces(spaces, len(self._domain))
            if len(self._spaces) != len(diagonal.domain):
                raise ValueError("spaces and domain must have the same length")
            for i, j in enumerate(self._spaces):
                if diagonal.domain[i] != self._domain[j]:
                    raise ValueError("Mismatch:\n{diagonal.domain[i]}\n{self._domain[j]}")
            if self._spaces == tuple(range(len(self._domain))):
                self._spaces = None  # shortcut

        if self._spaces is not None:
            active_axes = []
            for space_index in self._spaces:
                active_axes += self._domain.axes[space_index]

            self._ldiag = diagonal.val
            self._reshaper = [shp if i in active_axes else 1
                              for i, shp in enumerate(self._domain.shape)]
            self._ldiag = self._ldiag.reshape(self._reshaper)
        else:
            self._ldiag = diagonal.val
        self._fill_rest()

        self._jax_expr = partial(mul, self._ldiag)

    def _fill_rest(self):
        self._ldiag.flags.writeable = False
        self._complex = utilities.iscomplextype(self._ldiag.dtype)
        self._capability = self._all_ops
        if not self._complex:
            self._diagmin = self._ldiag.min()

    def _from_ldiag(self, spc, ldiag, sampling_dtype):
        res = DiagonalOperator.__new__(DiagonalOperator)
        res._dtype = sampling_dtype
        res._domain = self._domain
        if self._spaces is None or spc is None:
            res._spaces = None
        else:
            res._spaces = tuple(set(self._spaces) | set(spc))
        res._ldiag = np.array(ldiag)
        res._fill_rest()

        res._jax_expr = partial(mul, res._ldiag)

        return res

    def _scale(self, fct):
        if not np.isscalar(fct):
            raise TypeError("scalar value required")
        return self._from_ldiag((), self._ldiag*fct, self._dtype)

    def _add(self, sum_):
        if not np.isscalar(sum_):
            raise TypeError("scalar value required")
        return self._from_ldiag((), self._ldiag+sum_, self._dtype)

    def _combine_prod(self, op):
        if not isinstance(op, DiagonalOperator):
            raise TypeError("DiagonalOperator required")
        dtype = self._dtype if self._dtype == op._dtype else None
        return self._from_ldiag(op._spaces, self._ldiag*op._ldiag, dtype)

    def _combine_sum(self, op, selfneg, opneg):
        if not isinstance(op, DiagonalOperator):
            raise TypeError("DiagonalOperator required")
        tdiag = (self._ldiag * (-1 if selfneg else 1) +
                 op._ldiag * (-1 if opneg else 1))
        dtype = self._dtype if self._dtype == op._dtype else None
        return self._from_ldiag(op._spaces, tdiag, dtype)

    def apply(self, x, mode):
        self._check_input(x, mode)
        # shortcut for most common cases
        if mode == 1 or (not self._complex and mode == 2):
            return Field(x.domain, x.val*self._ldiag)

        xdiag = self._ldiag
        if self._complex and (mode & 10):  # adjoint or inverse adjoint
            xdiag = xdiag.conj()

        if mode & 3:
            return Field(x.domain, x.val*xdiag)
        return Field(x.domain, x.val/xdiag)

    def _flip_modes(self, trafo):
        if trafo == self.ADJOINT_BIT and not self._complex:  # shortcut
            return self
        xdiag = self._ldiag
        if self._complex and (trafo & self.ADJOINT_BIT):
            xdiag = xdiag.conj()
        if trafo & self.INVERSE_BIT:
            # dividing by zero is OK here, we can deal with infinities
            with np.errstate(divide='ignore'):
                xdiag = 1./xdiag
        return self._from_ldiag((), xdiag, self._dtype)

    def process_sample(self, samp, from_inverse):
        if (self._complex or (self._diagmin < 0.) or
                (self._diagmin == 0. and from_inverse)):
            raise ValueError("operator not positive definite")
        if from_inverse:
            res = samp.val/np.sqrt(self._ldiag)
        else:
            res = samp.val*np.sqrt(self._ldiag)
        return Field(self._domain, res)

    def draw_sample(self, from_inverse=False):
        if self._dtype is None:
            s = "Need to specify dtype to be able to sample from this operator:\n"
            s += self.__repr__()
            raise RuntimeError(s)
        res = Field.from_random(domain=self._domain, random_type="normal",
                                dtype=self._dtype)
        return self.process_sample(res, from_inverse)

    def get_sqrt(self):
        if np.iscomplexobj(self._ldiag) or (self._ldiag < 0).any():
            raise ValueError("get_sqrt() works only for positive definite operators.")
        return self._from_ldiag((), np.sqrt(self._ldiag), self._dtype)

    def __repr__(self):
        from ..multi_domain import MultiDomain
        s = "DiagonalOperator (domain/target "
        if isinstance(self.domain, MultiDomain):
            s += f"keys: {self._domain.keys()}"
        else:
            s += f"shape: {self._domain.shape}"
        if self._dtype is not None:
            s += f", sampling dtype {self._dtype}"
        s += ")"
        return s
