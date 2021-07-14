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
# Authors: Gordian Edenhofer
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..multi_domain import MultiDomain
from ..multi_field import MultiField
from .linear_operator import LinearOperator


class SliceOperator(LinearOperator):
    """Geometry preserving mask operator

    Takes a field, slices it into the desired shape and returns the values of
    the field in the sliced domain all while preserving the original distances.

    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        The operator's input domain.
    new_shape : tuple of tuples or integers, or None
        The shape of the target domain with None indicating to copy the shape
        of the original domain for this axis. For example ((10, 5), 100) for a
        DomainTuple with two entires, the first having shape (10, 5) and the
        second having shape 100
    center : bool, optional
        Whether to center the slice that is selected in the input field.
    preserve_dist: bool, optional
        Whether to preserve the distance of the input field.
    """
    def __init__(self, domain, new_shape, center=False, preserve_dist=True):
        self._domain = DomainTuple.make(domain)
        if len(new_shape) != len(self._domain):
            ve = (
                f"shape ({new_shape}) is incompatible with the shape of the"
                f" domain ({self._domain.shape})"
            )
            raise ValueError(ve)
        for i, shape in enumerate(new_shape):
            if len(np.atleast_1d(shape)) != len(self._domain[i].shape):
                ve = (
                    f"shape of subspace ({i}) is incompatible with the domain"
                )
                raise ValueError(ve)

        tgt = []
        slc_by_ax = []
        for i, d in enumerate(self._domain):
            if new_shape[i] is None or np.all(
                np.array(self._domain.shape[i]) == np.array(new_shape[i])
            ):
                tgt += [d]
            elif np.all(np.array(new_shape[i]) <= np.array(d.shape)):
                dom_kw = dict()
                if isinstance(d, RGSpace):
                    if preserve_dist:
                        dom_kw["distances"] = d.distances
                    dom_kw["harmonic"] = d.harmonic
                elif not isinstance(d, UnstructuredDomain):
                    # Some domains like HPSpace or LMSPace can not be sliced
                    ve = f"{d.__class__.__name__} can not be sliced"
                    raise ValueError(ve)
                tgt += [d.__class__(new_shape[i], **dom_kw)]
            else:
                ve = (
                    f"domain axes ({d}) is smaller than the target shape"
                    f"{new_shape[i]}"
                )
                raise ValueError(ve)

            if center:
                for j, n_pix in enumerate(np.atleast_1d(new_shape[i])):
                    slc_start = np.floor((d.shape[j] - n_pix) / 2.).astype(int)
                    slc_end = slc_start + n_pix
                    slc_by_ax += [slice(slc_start, slc_end)]
            else:
                for n_pix in np.atleast_1d(new_shape[i]):
                    slc_start = 0
                    slc_end = n_pix
                    slc_by_ax += [slice(slc_start, slc_end)]

        self._slc_by_ax = tuple(slc_by_ax)
        self._target = DomainTuple.make(tgt)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = x[self._slc_by_ax]
            return Field.from_raw(self.target, res)
        res = np.zeros(self.domain.shape, x.dtype)
        res[self._slc_by_ax] = x
        return Field.from_raw(self.domain, res)

    def __str__(self):
        ss = (
            f"{self.__class__.__name__}"
            f"({self.domain.shape} -> {self.target.shape})"
        )
        return ss


class SplitOperator(LinearOperator):
    """Split a single field into a multi-field

    Takes a field, selects the desired entries for each multi-field key and
    puts the result into a multi-field. Along sliced axis, the domain will
    be replaced by an UnstructuredDomain as no distance measures are preserved.

    Note, slices may intersect, i.e. slices may reference the same input
    multiple times if the `intersecting_slices` option is set. However, a
    single field in the output may not contain the same part of the input more
    than once.

    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        The operator's input domain.
    slices_by_key : dict{key: tuple of integers or None}
        The key-value pairs of which the values indicate the parts to be
        selected. The result will be a multi-field with the given keys as
        entries and the selected slices of the domain as values. `None`
        indicates to select the whole input along this axis.
    intersecting_slices : bool, optional
        Tells the operator whether slices may contain intersections. If true,
        the adjoint is constructed a little less efficiently.  Set this
        parameter to `False` to gain a little more efficiency.
    """
    def __init__(self, domain, slices_by_key, intersecting_slices=True):
        self._domain = DomainTuple.make(domain)
        self._intersec_slc = intersecting_slices

        tgt = dict()
        self._k_slc = dict()
        for k, slc in slices_by_key.items():
            if len(slc) > len(self._domain):
                ve = f"slice at key {k!r} has more dimensions than the input"
                raise ValueError(ve)
            k_tgt = []
            k_slc_by_ax = []
            for i, d in enumerate(self._domain):
                if i >= len(slc) or slc[i] is None or (
                    isinstance(slc[i], slice) and slc[i] == slice(None)
                ):
                    k_tgt += [d]
                    k_slc_by_ax += [slice(None)]
                elif isinstance(slc[i], slice):
                    start = slc[i].start if slc[i].start is not None else 0
                    stop = slc[i].stop if slc[i].stop is not None else d.size
                    step = slc[i].step if slc[i].step is not None else 1
                    frac = np.floor((stop - start) / np.abs(step))
                    k_tgt += [UnstructuredDomain(frac.astype(int))]
                    k_slc_by_ax += [slc[i]]
                elif isinstance(slc[i],
                                np.ndarray) and slc[i].dtype is np.dtype(bool):
                    if slc[i].size != d.size:
                        raise ValueError(
                            "shape mismatch between desired slice {slc[i]}"
                            "and the shape of the domain {d.size}"
                        )
                    k_tgt += [UnstructuredDomain(slc[i].sum())]
                    k_slc_by_ax += [slc[i]]
                elif isinstance(slc[i], (tuple, list, np.ndarray)):
                    k_tgt += [UnstructuredDomain(len(slc[i]))]
                    k_slc_by_ax += [slc[i]]
                elif isinstance(slc[i], int):
                    k_slc_by_ax += [slc[i]]
                else:
                    ve = f"invalid type for specifying a slice; got {slc[i]}"
                    raise ValueError(ve)
            tgt[k] = DomainTuple.make(k_tgt)
            self._k_slc[k] = tuple(k_slc_by_ax)

        self._target = MultiDomain.make(tgt)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = dict()
            for k, slc in self._k_slc.items():
                res[k] = x[slc]
            return MultiField.from_raw(self.target, res)

        # Note, not-selected parts must be zero. Hence, using the quicker
        # `np.empty` method is unfortunately not possible
        res = np.zeros(self.domain.shape, tuple(x.values())[0].dtype)
        if self._intersec_slc:
            for k, slc in self._k_slc.items():
                # Mind the `+` here for coping with intersections
                res[slc] += x[k]
            return Field.from_raw(self.domain, res)
        for k, slc in self._k_slc.items():
            res[slc] = x[k]
        return Field.from_raw(self.domain, res)

    def __str__(self):
        return f"{self.__class__.__name__} {self._target.keys()!r} <-"
