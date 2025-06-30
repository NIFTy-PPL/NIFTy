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
# Copyright(C) 2013-2025 Max-Planck-Society
# Copyright(C) 2025 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from .. import utilities
from ..any_array import AnyArray
from ..domain_tuple import DomainTuple
from ..field import Field
from .endomorphic_operator import EndomorphicOperator

# TODO: Eventually enforce somewhat modern ducc version (>=0.37.0) as nifty
# dependency and remove the try statement below (also at other places)
try:
    from ducc0.misc.experimental import div_conj, mul_conj
    def mul_conj2(a, b):
        assert a.device_id == b.device_id
        if a.device_id == -1:
            return AnyArray(mul_conj(a.val, b.val))
        return a*b.conj()

    def div_conj2(a, b):
        assert a.device_id == b.device_id
        if a.device_id == -1:
            return AnyArray(div_conj(a.val, b.val))
        return a/b.conj()

except ImportError:
    def mul_conj2(a, b):
        return a*b.conj()

    def div_conj2(a, b):
        return a/b.conj()


class DiagonalOperator(EndomorphicOperator):
    """Represents a :class:`LinearOperator` which is diagonal.

    The NIFTy DiagonalOperator class is a subclass derived from the
    :class:`EndomorphicOperator`. It multiplies an input field pixel-wise with
    its diagonal.

    Parameters
    ----------
    diagonal : :class:`nifty.cl.field.Field`
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

    def __init__(self, diagonal, domain=None, spaces=None, sampling_dtype=None,
                 _trafo=0):
# MR: _trafo is more or less deliberately undocumented, since it is not supposed
# to be necessary for "end users". It describes the type of transform for which
# the diagonal can be used without modification
# (0:TIMES, 1:ADJOINT, 2:INVERSE, 3:ADJOINT_INVERSE)
        if not isinstance(diagonal, Field):
            raise TypeError("Field object required")
        utilities.check_dtype_or_none(sampling_dtype)
        self._dtype = sampling_dtype
        self._trafo = _trafo
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
                    raise ValueError(f"Mismatch between:\n{diagonal.domain[i]}\n"
                                     f"and:\n{self._domain[j]}")
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
        assert isinstance(self._ldiag, AnyArray)
        self._fill_rest()

    def _fill_rest(self):
        self._ldiag.lock()
        self._complex = utilities.iscomplextype(self._ldiag.dtype)
        self._capability = self._all_ops
        if not self._complex:
            self._diagmin_cache = None

    @property
    def _diagmin(self):
        if self._complex:
            raise RuntimeError("complex DiagonalOperator does not have _diagmin")
        if self._diagmin_cache is None:
            self._diagmin_cache = self._ldiag.min()
        return self._diagmin_cache

    def _from_ldiag(self, spc, ldiag, sampling_dtype, trafo):
        res = DiagonalOperator.__new__(DiagonalOperator)
        res._dtype = sampling_dtype
        res._trafo = trafo
        res._domain = self._domain
        if self._spaces is None or spc is None:
            res._spaces = None
        else:
            res._spaces = tuple(set(self._spaces) | set(spc))
        res._ldiag = AnyArray(ldiag)
        res._fill_rest()
        return res

    def _get_actual_diag(self):
        if self._trafo == 0:
            return self._ldiag
        if self._trafo == 1:
            return np.conj(self._ldiag) if self._complex else self._ldiag
        if self._trafo == 2:
            return 1./self._ldiag
        if self._trafo == 3:
            return np.conj(1./self._ldiag) if self._complex else 1./self._ldiag

    def _scale(self, fct):
        if not np.isscalar(fct):
            raise TypeError("scalar value required")
        return self._from_ldiag((), self._get_actual_diag()*fct, self._dtype, 0)

    def _add(self, sum_):
        if not np.isscalar(sum_):
            raise TypeError("scalar value required")
        return self._from_ldiag((), self._get_actual_diag()+sum_, self._dtype, 0)

    def _combine_prod(self, op):
        if not isinstance(op, DiagonalOperator):
            raise TypeError("DiagonalOperator required")
        dtype = self._dtype if self._dtype == op._dtype else None
        return self._from_ldiag(op._spaces, self._get_actual_diag()*op._get_actual_diag(),
                                dtype, 0)

    def _combine_sum(self, op, selfneg, opneg):
        if not isinstance(op, DiagonalOperator):
            raise TypeError("DiagonalOperator required")
        tdiag = (self._get_actual_diag() * (-1 if selfneg else 1) +
                 op._get_actual_diag() * (-1 if opneg else 1))
        dtype = self._dtype if self._dtype == op._dtype else None
        return self._from_ldiag(op._spaces, tdiag, dtype, 0)

    def _device_preparation(self, x, mode):
        self._ldiag = self._ldiag.at(x.device_id)

    def apply(self, x, mode):
        self._check_input(x, mode)
        # To save both time and memory, we remap the `mode` (via `self._trafo`)
        # and do not compute and store a new `self._ldiag`s for adjoint, inverse
        # or adjoint-inverse DiagonalOperators.
        self._device_preparation(x, mode)
        trafo = self._ilog[mode] ^ self._trafo

        if trafo == 0:  # straight application
            return Field(x.domain, x.val*self._ldiag)

        if trafo == 1:  # adjoint
            return Field(x.domain, mul_conj2(x.val, self._ldiag)
                                   if self._complex else x.val*self._ldiag)

        if trafo == 2:  # inverse
            return Field(x.domain, x.val/self._ldiag)

        # adjoint inverse
        return Field(x.domain, div_conj2(x.val, self._ldiag)
            if self._complex else x.val/self._ldiag)

    def _flip_modes(self, trafo):
        return self._from_ldiag((), self._ldiag, self._dtype, self._trafo ^ trafo)

    def process_sample(self, samp, from_inverse):
        from_inverse2 = from_inverse ^ (self._trafo >= 2)
        # `from_inverse2` captures if the inverse of `self._ldiag` needs to be
        # taken or not (can happen for nontrivial `self._trafo`).
        if (self._complex or (self._diagmin < 0.) or
                (self._diagmin == 0. and from_inverse2)):
            raise ValueError("operator not positive definite")
        if from_inverse2:
            res = samp.val/np.sqrt(self._ldiag)
        else:
            res = samp.val*np.sqrt(self._ldiag)
        return Field(self._domain, res)

    def draw_sample(self, from_inverse=False, device_id=-1):
        if self._dtype is None:
            s = "Need to specify dtype to be able to sample from this operator:\n"
            s += self.__repr__()
            raise RuntimeError(s)
        res = Field.from_random(domain=self._domain, random_type="normal",
                                dtype=self._dtype, device_id=device_id)
        return self.process_sample(res, from_inverse)

    def get_sqrt(self):
        if self._complex or self._diagmin < 0.:
            raise ValueError("get_sqrt() works only for positive definite operators.")
        return self._from_ldiag((), np.sqrt(self._ldiag), self._dtype, self._trafo)

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
