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
# Authors: Gordian Edenhofer, Philipp Frank
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import string

import numpy as np

from ..domain_tuple import DomainTuple
from ..field import Field
from ..linearization import Linearization
from ..multi_domain import MultiDomain
from ..multi_field import MultiField
from ..utilities import myassert
from .linear_operator import LinearOperator
from .operator import Operator


class MultiLinearEinsum(Operator):
    """Multi-linear Einsum operator with corresponding derivates

    Parameters
    ----------
    domain : MultiDomain or dict{name: DomainTuple}
        The operator's input domain.
    subscripts : str
        The subscripts which is passed to einsum.
    key_order: tuple of str, optional
        The order of the keys in the multi-field. If not specified, defaults to
        the order of the keys in the multi-field.
    static_mf: MultiField or dict{name: Field}, optional
        A dictionary like type from which Fields are to be taken if the key from
        `key_order` is not part of the `domain`. Fields in this object are
        supposed to be static as they will not appear as FieldAdapter in the
        Linearization.
    optimize: bool, String or List, optional
        Parameter passed on to einsum_path.

    Notes
    -----
    The indices of the tensors run along spaces, not the axes of the spaces.
    For example, a matrix requires a product of two spaces, not one space with
    the combined shapes.

    By convention :class:`MultiLinearEinsum` only performs operations with
    lower indices. Therefore no complex conjugation is performed on complex
    inputs. To achieve operations with upper/lower indices use
    :class:`PartialConjugate` before applying this operator.
    """
    def __init__(self, domain, subscripts,
                 key_order=None, static_mf=None, optimize='optimal'):
        self._domain = MultiDomain.make(domain)
        if key_order is None:
            self._key_order = tuple(self._domain.keys())
        else:
            self._key_order = key_order
        if static_mf is not None and key_order is None:
            ve = "`key_order` mus be specified if additional fields are munged"
            raise ValueError(ve)
        self._stat_mf = static_mf
        iss, oss, *rest = subscripts.split("->")
        iss_spl = iss.split(",")
        len_consist = len(self._key_order) == len(iss_spl)
        sscr_consist = all(o in iss for o in oss)
        if rest or not sscr_consist or "," in oss or not len_consist:
            raise ValueError(f"invalid subscripts specified; got {subscripts}")
        ve = f"invalid order of keys {self._key_order} for subscripts {subscripts}"
        shapes, numpy_subscripts, subscriptmap = {}, '', {}
        alphabet = list(string.ascii_lowercase)[::-1]
        for k, ss in zip(self._key_order, iss_spl):
            dom = self._domain[k] if k in self._domain.keys(
            ) else self._stat_mf[k].domain
            if len(dom) != len(ss):
                raise ValueError(ve)
            for i, a in enumerate(list(ss)):
                if a not in subscriptmap.keys():
                    subscriptmap[a] = [alphabet.pop() for _ in
                                       range(len(dom[i].shape))]
                numpy_subscripts += ''.join(subscriptmap[a])
            numpy_subscripts += ','
            shapes[k] = dom.shape
        numpy_subscripts = numpy_subscripts[:-1] + '->'
        dom_sscr = dict(zip(self._key_order, iss_spl))
        tgt = []
        for o in oss:
            k_hit = tuple(k for k, sscr in dom_sscr.items() if o in sscr)[0]
            dom_k_idx = dom_sscr[k_hit].index(o)
            if k_hit in self._domain.keys():
                tgt += [self._domain[k_hit][dom_k_idx]]
            else:
                if k_hit not in self._stat_mf.keys():
                    ve = f"{k_hit} is not in domain nor in static_mf"
                    raise ValueError(ve)
                tgt += [self._stat_mf[k_hit].domain[dom_k_idx]]
            numpy_subscripts += ''.join(subscriptmap[o])
        self._target = DomainTuple.make(tgt)

        numpy_iss, numpy_oss, *_ = numpy_subscripts.split("->")
        numpy_iss_spl = numpy_iss.split(",")

        self._sscr_endswith = dict()
        self._linpaths = dict()
        for k, (i, ss) in zip(self._key_order, enumerate(numpy_iss_spl)):
            left_ss_spl = (*numpy_iss_spl[:i], *numpy_iss_spl[i + 1:], ss)
            linpath = '->'.join((','.join(left_ss_spl), numpy_oss))

            plc = tuple(np.broadcast_to(np.nan, shapes[q]) for q in shapes if q != k)
            plc += (np.broadcast_to(np.nan, shapes[k]),)
            self._sscr_endswith[k] = linpath
            self._linpaths[k] = np.einsum_path(linpath, *plc, optimize=optimize)[0]
        if isinstance(optimize, list):
            path = optimize
        else:
            plc = (np.broadcast_to(np.nan, shapes[k]) for k in shapes)
            path = np.einsum_path(numpy_subscripts, *plc, optimize=optimize)[0]
        self._sscr = numpy_subscripts
        self._ein_kw = {"optimize": path}

    def apply(self, x):
        self._check_input(x)
        if isinstance(x, Linearization):
            val = x.val.val
        else:
            val = x.val
        v = (
            val[k] if k in val else self._stat_mf[k].val
            for k in self._key_order
        )
        res = np.einsum(self._sscr, *v, **self._ein_kw)

        if isinstance(x, Linearization):
            jac = None
            for wrt in self.domain.keys():
                plc = {
                    k: x.val[k] if k in x.val else self._stat_mf[k]
                    for k in self._key_order if k != wrt
                }
                mf_wo_k = MultiField.from_dict(plc)
                ss = self._sscr_endswith[wrt]
                # Use the fact that the insertion order in a dictionary is the
                # ordering of keys as to pass on `key_order`
                jac_k = LinearEinsum(
                    self.domain[wrt],
                    mf_wo_k,
                    ss,
                    key_order=tuple(plc.keys()),
                    optimize=self._linpaths[wrt],
                    _target=self._target,
                    _calling_as_lin=True
                ).ducktape(wrt)
                jac = jac + jac_k if jac is not None else jac_k
            return x.new(Field.from_raw(self.target, res), jac)
        return Field.from_raw(self.target, res)


class LinearEinsum(LinearOperator):
    """Linear Einsum operator with exactly one freely varying field

    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        The operator's input domain.
    mf : :class:`nifty8.multi_field.MultiField`
        The first part of the left-hand side of the einsum.
    subscripts : str
        The subscripts which is passed to einsum. Everything before the very
        last scripts before the '->' is treated as part of the fixed mulfi-
        field while the last scripts are taken to correspond to the freely
        varying field.
    key_order: tuple of str, optional
        The order of the keys in the multi-field. If not specified, defaults to
        the order of the keys in the multi-field.
    optimize: bool, String or List, optional
        Parameter passed on to einsum_path.

    Notes
    -----
    The indices of the tensors run along spaces, not the axes of the spaces.
    For example, a matrix requires a product of two spaces, not one space with
    the combined shapes.

    By convention :class:`LinearEinsum` only performs operations with
    lower indices. Therefore no complex conjugation is performed on complex
    inputs or mf. To achieve operations with upper/lower indices use
    :class:`PartialConjugate` before applying this operator.
    """
    def __init__(self, domain, mf, subscripts, key_order=None, optimize='optimal',
                 _target=None, _calling_as_lin=False):
        self._domain = DomainTuple.make(domain)
        if _calling_as_lin:
            self._init_wo_preproc(mf, subscripts, key_order, optimize, _target)
        else:
            self._mf = mf
            if key_order is None:
                _key_order = tuple(self._mf.domain.keys())
            else:
                _key_order = key_order
            self._ein_kw = {"optimize": optimize}
            iss, oss, *rest = subscripts.split("->")
            iss_spl = iss.split(",")
            sscr_consist = all(o in iss for o in oss)
            len_consist = len(_key_order) == len(iss_spl[:-1])
            if rest or not sscr_consist or "," in oss or not len_consist:
                raise ValueError(f"invalid subscripts specified; got {subscripts}")
            ve = f"invalid order of keys {_key_order} for subscripts {subscripts}"
            shapes, numpy_subscripts, subscriptmap = (), '', {}
            alphabet = list(string.ascii_lowercase)
            for k, ss in zip(_key_order, iss_spl[:-1]):
                dom = self._mf[k].domain
                if len(dom) != len(ss):
                    raise ValueError(ve)
                for i, a in enumerate(list(ss)):
                    if a not in subscriptmap.keys():
                        subscriptmap[a] = [alphabet.pop() for _ in
                                           range(len(dom[i].shape))]
                    numpy_subscripts += ''.join(subscriptmap[a])
                numpy_subscripts += ','
                shapes += (dom.shape,)
            if len(self._domain) != len(iss_spl[-1]):
                raise ValueError(ve)
            for i, a in enumerate(list(iss_spl[-1])):
                if a not in subscriptmap.keys():
                    subscriptmap[a] = [alphabet.pop() for _ in
                                       range(len(self._domain[i].shape))]
                numpy_subscripts += ''.join(subscriptmap[a])
            shapes += (self._domain.shape,)
            numpy_subscripts += '->'

            dom_sscr = dict(zip(_key_order, iss_spl[:-1]))
            dom_sscr[id(self)] = iss_spl[-1]
            tgt = []
            for o in oss:
                k_hit = tuple(k for k, sscr in dom_sscr.items() if o in sscr)[0]
                dom_k_idx = dom_sscr[k_hit].index(o)
                if k_hit in _key_order:
                    tgt += [self._mf.domain[k_hit][dom_k_idx]]
                else:
                    myassert(k_hit == id(self))
                    tgt += [self._domain[dom_k_idx]]
                numpy_subscripts += "".join(subscriptmap[o])
            _target = DomainTuple.make(tgt)
            self._sscr = numpy_subscripts

            if isinstance(optimize, list):
                path = optimize
            else:
                plc = (np.broadcast_to(np.nan, shp) for shp in shapes)
                path = np.einsum_path(numpy_subscripts, *plc, optimize=optimize)[0]
            self._init_wo_preproc(mf, numpy_subscripts, _key_order, path, _target)

    def _init_wo_preproc(self, mf, subscripts, keyorder, optimize, target):
        self._ein_kw = {"optimize": optimize}
        self._mf = mf
        self._sscr = subscripts
        self._key_order = keyorder
        self._target = target
        iss, oss, *_ = subscripts.split("->")
        iss_spl = iss.split(",")

        if len(iss_spl) == 1:
            self._adj_sscr = "->".join((oss, iss))
        else:
            adj_iss = ",".join((",".join(iss_spl[:-1]), oss))
            self._adj_sscr = "->".join((adj_iss, iss_spl[-1]))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            dom, ss, mf = self.target, self._sscr, self._mf
        else:
            dom, ss, mf = self.domain, self._adj_sscr, self._mf.conjugate()
        res = np.einsum(
            ss, *(mf[k].val for k in self._key_order), x.val,
            **self._ein_kw
        )
        return Field.from_raw(dom, res)
