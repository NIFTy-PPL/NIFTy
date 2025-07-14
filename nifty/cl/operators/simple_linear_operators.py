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
# Copyright(C) 2013-2020 Max-Planck-Society
# Copyright(C) 2025 Philipp Arras
# Copyright(C) 2025 LambdaFields GmbH
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import numpy as np

from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..multi_domain import MultiDomain
from ..multi_field import MultiField
from ..utilities import check_object_identity
from .endomorphic_operator import EndomorphicOperator
from .linear_operator import LinearOperator


class VdotOperator(LinearOperator):
    """Operator computing the scalar product of its input with a given Field.

    Parameters
    ----------
    field : :class:`nifty.cl.field.Field` or :class:`nifty.cl.multi_field.MultiField`
        The field used to build the scalar product with the operator input
    """
    def __init__(self, field):
        self._field = field
        self._domain = field.domain
        self._target = DomainTuple.scalar_domain()
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def _device_preparation(self, x):
        self._field = self._field.at(x.device_id)

    def apply(self, x, mode):
        self._check_mode(mode)
        if mode == self.TIMES:
            # device_preparation only here since scalars are always on host
            self._device_preparation(x)
            return self._field.vdot(x)
        return self._field*x.asnumpy()[()]


class ConjugationOperator(EndomorphicOperator):
    """Operator computing the complex conjugate of its input.

    Parameters
    ----------
    domain: Domain, tuple of domains or DomainTuple
        domain of the input field

    """
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._capability = self._all_ops

    def apply(self, x, mode):
        self._check_input(x, mode)
        return x.conjugate()


class WeightApplier(EndomorphicOperator):
    """Operator multiplying its input by a given power of dvol.

    Parameters
    ----------
    domain: Domain, tuple of domains or DomainTuple
        domain of the input field
    spaces: list or tuple of int
        indices of subdomains for which the weights shall be applied
    power: int
        the power of to be used for the volume factors

    """
    def __init__(self, domain, spaces, power):
        from .. import utilities
        self._domain = DomainTuple.make(domain)
        if spaces is None:
            self._spaces = None
        else:
            self._spaces = utilities.parse_spaces(spaces, len(self._domain))
        self._power = int(power)
        self._capability = self._all_ops

    def apply(self, x, mode):
        self._check_input(x, mode)
        power = self._power if (mode & 3) else -self._power
        return x.weight(power, spaces=self._spaces)


class Realizer(EndomorphicOperator):
    """Operator returning the real component of its input.

    Parameters
    ----------
    domain: Domain, tuple of domains, DomainTuple or MultiDomain
        domain of the input field

    """
    def __init__(self, domain):
        from ..sugar import makeDomain

        self._domain = makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return x.real


class Imaginizer(EndomorphicOperator):
    """Operator returning the imaginary component of its input.

    Parameters
    ----------
    domain: Domain, tuple of domains, DomainTuple or MultiDomain
        domain of the input field

    """
    def __init__(self, domain):
        from ..sugar import makeDomain

        self._domain = makeDomain(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            if not np.issubdtype(x.dtype, np.complexfloating):
                raise ValueError
            return x.imag
        if x.dtype not in (np.float64, np.float32):
            raise ValueError
        return 1j*x


class FieldAdapter(LinearOperator):
    """Operator for conversion between Fields and MultiFields.

    Parameters
    ----------
    tgt : Domain, tuple of domains, DomainTuple, dict or MultiDomain:
        If this is a Domain, tuple of Domain or DomainTuple, this will be the
        operator's target, and its domain will be a MultiDomain consisting of
        its domain with the supplied `name`
        If this is a dict or MultiDomain, everything except for `name` will
        be stripped out of it, and the result will be the operator's target.
        Its domain will then be the DomainTuple corresponding to the single
        entry in the operator's domain.

    name : String
        The relevant key of the MultiDomain.
    """

    def __init__(self, target, name):
        from ..sugar import makeDomain
        tmp = makeDomain(target)
        if isinstance(tmp, DomainTuple):
            self._target = tmp
            self._domain = MultiDomain.make({name: tmp})
        else:
            self._domain = tmp[name]
            self._target = MultiDomain.make({name: tmp[name]})
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if isinstance(x, MultiField):
            return x.values()[0]
        else:
            return MultiField(self._tgt(mode), (x,))

    def __repr__(self):
        dom = self.domain.keys() if isinstance(self.domain, MultiDomain) else '()'
        tgt = self.target.keys() if isinstance(self.target, MultiDomain) else '()'
        return f'{tgt} <- {dom}'


class _SlowFieldAdapter(LinearOperator):
    """Operator for conversion between Fields and MultiFields.
    The operator is built so that the MultiDomain is always the target.
    Its domain is `tgt[name]`

    Parameters
    ----------
    dom : dict or MultiDomain:
        the operator's dom

    name : String
        The relevant key of the MultiDomain.
    """

    def __init__(self, domain, name):
        from ..sugar import makeDomain
        tmp = makeDomain(domain)
        if not isinstance(tmp, MultiDomain):
            raise TypeError("MultiDomain expected")
        self._name = str(name)
        self._domain = tmp
        self._target = tmp[name]
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if isinstance(x, MultiField):
            return x[self._name]
        return MultiField.from_dict({self._name: x}, domain=self._tgt(mode))

    def __repr__(self):
        return '_SlowFieldAdapter'


def ducktape(left, right, name):
    """Convenience function creating an operator that converts between a
    DomainTuple and a MultiDomain.

    Parameters
    ----------
    left : None, Operator, or Domainoid
        Something describing the new operator's target domain.
        If `left` is an `Operator`, its domain is used as `left`.

    right : None, Operator, or Domainoid
        Something describing the new operator's input domain.
        If `right` is an `Operator`, its target is used as `right`.

    name : string
        The component of the `MultiDomain` that will be extracted/inserted

    Notes
    -----
    - one of the involved domains must be a `DomainTuple`, the other a
      `MultiDomain`.
    - `left` and `right` must not be both `None`, but one of them can (and
      probably should) be `None`. In this case, the missing information is
      inferred.

    Returns
    -------
    FieldAdapter or _SlowFieldAdapter
        an adapter operator converting between the two (possibly
        partially inferred) domains.
    """
    from ..sugar import makeDomain
    from .operator import Operator
    if isinstance(right, Operator):
        right = right.target
    elif right is not None:
        right = makeDomain(right)
    if isinstance(left, Operator):
        left = left.domain
    elif left is not None:
        left = makeDomain(left)
    if left is None:  # need to infer left from right
        if isinstance(right, MultiDomain):
            left = right[name]
        else:
            left = MultiDomain.make({name: right})
    elif right is None:  # need to infer right from left
        if isinstance(left, MultiDomain):
            right = left[name]
        else:
            right = MultiDomain.make({name: left})
    lmulti = isinstance(left, MultiDomain)
    rmulti = isinstance(right, MultiDomain)
    if lmulti + rmulti != 1:
        raise ValueError("need exactly one MultiDomain")
    if lmulti:
        if len(left) == 1:
            return FieldAdapter(left, name)
        else:
            return _SlowFieldAdapter(left, name).adjoint
    if rmulti:
        if len(right) == 1:
            return FieldAdapter(left, name)
        else:
            return _SlowFieldAdapter(right, name)
    raise ValueError("must not arrive here")


def Variable(target, key):
    """Operator that picks an entry from a MultiDomain and makes it available on
    a DomainTuple.

    The domain of the operator is a MultiDomain with one entry: {key: target}.

    This operator is often used at the very beginning of a complicated operator
    chain.

    Parameters
    ----------
    target: Domain, tuple of Domain, or DomainTuple
        The target of the operator.
    key: str
        The key of the domain of the operator.
    """
    from .scaling_operator import ScalingOperator
    return ScalingOperator(target, 1.).ducktape(key)


class GeometryRemover(LinearOperator):
    """Operator which transforms between a structured and an unstructured
    domain.

    Parameters
    ----------
    domain: Domain, tuple of Domain, or DomainTuple:
        the full input domain of the operator.
    space: int, optional
        The index of the subdomain on which the operator should act.
        If None, it acts on all spaces.

    Notes
    -----
    The operator will convert every sub-domain of its input domain to an
    UnstructuredDomain with the same shape. No weighting by volume factors
    is carried out.
    """

    def __init__(self, domain, space=None):
        self._domain = DomainTuple.make(domain)
        if space is not None:
            tgt = [dom for dom in self._domain]
            tgt[space] = UnstructuredDomain(self._domain[space].shape)
        else:
            tgt = [UnstructuredDomain(dom.shape) for dom in self._domain]
        self._target = DomainTuple.make(tgt)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return x.cast_domain(self._tgt(mode))


class NullOperator(LinearOperator):
    """Operator corresponding to a matrix of all zeros.

    Parameters
    ----------
    domain : DomainTuple or MultiDomain
        input domain
    target : DomainTuple or MultiDomain
        output domain
    default_domain_device_id : int or dict of int
        Fallback device_id for output of adjoint if it cannot be determined
        otherwise.
    default_target_device_id : int or dict of int
        Fallback device_id for output if it cannot be determined otherwise.
    """

    def __init__(self, domain, target,
                 default_domain_device_id=-1, default_target_device_id=-1):
        from ..sugar import makeDomain
        self._domain = makeDomain(domain)
        self._target = makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._domain_did = default_domain_device_id
        self._target_did = default_target_device_id
        self._check_device_id(self._domain, self._domain_did)
        self._check_device_id(self._target, self._target_did)

    @staticmethod
    def _check_device_id(domain, device_id):
        if isinstance(domain, DomainTuple):
            assert isinstance(device_id, int) and device_id >= -1
        else:
            assert isinstance(device_id, (dict, int))
            if isinstance(device_id, dict):
                assert all(isinstance(vv, int) for vv in device_id.values())
                assert set(device_id.keys()) == set(domain.keys())

    @staticmethod
    def _nullfield(dom, device_id):
        if isinstance(dom, DomainTuple):
            return Field.full(dom, 0., device_id=device_id)
        else:
            return MultiField.full(dom, 0., device_id=device_id)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if self.domain is self.target or \
           (isinstance(self.domain, DomainTuple) and isinstance(self.target, DomainTuple)):
            # Domain and target are the same or Domain and target are both DomainTuple
            device_id = x.device_id
        elif isinstance(x.domain, DomainTuple):
            # Input is Field and trivially on a single device
            device_id = x.device_id
        elif len(set(x.device_id.values())) == 1:
            # Input is MultiField and on single device
            device_id = list(set(x.device_id.values()))[0]
        else:
            # Fallback
            if mode in [self.TIMES, self.ADJOINT_INVERSE_TIMES]:
                device_id = self._target_did
            else:
                device_id = self._domain_did
        return self._nullfield(self._tgt(mode), device_id)

    def __repr__(self):
        dom = self.domain.keys() if isinstance(self.domain, MultiDomain) else '()'
        tgt = self.target.keys() if isinstance(self.target, MultiDomain) else '()'
        return f'{tgt} <- NullOperator <- {dom}'


class PartialExtractor(LinearOperator):
    def __init__(self, domain, target):
        if not isinstance(domain, MultiDomain):
            raise TypeError("MultiDomain expected")
        if not isinstance(target, MultiDomain):
            raise TypeError("MultiDomain expected")
        self._domain = domain
        self._target = target
        for key in self._target.keys():
            check_object_identity(self._domain[key], self._target[key])
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._compldomain = MultiDomain.make({kk: self._domain[kk]
                                              for kk in self._domain.keys()
                                              if kk not in self._target.keys()})

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            return x.extract(self._target)
        res0 = MultiField.from_dict({key: x[key] for key in x.domain.keys()})
        res1 = MultiField.full(self._compldomain, 0.)
        return res0.unite(res1)

    def __repr__(self):
        return f'{self.target.keys()} <- {self.domain.keys()}'


class PrependKey(LinearOperator):
    """Prepend a string to all keys of a MultiDomain.

    Parameters
    ----------
    domain : MultiDomain
    pre : str
    """
    def __init__(self, domain, pre):
        if not isinstance(domain, MultiDomain):
            raise ValueError
        from ..sugar import makeDomain
        self._domain = makeDomain(domain)
        self._pre = str(pre)
        target = {self._pre+k: domain[k] for k in domain.keys()}
        self._target = makeDomain(MultiDomain.make(target))
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            res = {self._pre+k:x[k] for k in self._domain.keys()}
        else:
            res = {k:x[self._pre+k] for k in self._domain.keys()}
        return MultiField.from_dict(res, domain=self._tgt(mode))


class DomainChangerAndReshaper(LinearOperator):
    """Convert nifty domains into each other and reshape field.

    This is only possible if `domain` and `target` have the same number of pixels.

    Parameters
    ----------
    domain : DomainTuple
        Domain of the operator
    target : DomainTuple
        Target of the operator
    """
    def __init__(self, domain, target):
        from ..sugar import makeDomain

        self._domain = makeDomain(domain)
        self._target = makeDomain(target)
        if self._domain.size != self._target.size:
            s = ["Domain and target do not have the same number of pixels",
                f"Domain: {self._domain.shape}",
                f"Target: {self._target.shape}"]
            raise ValueError("\n".join(s))
        self._capability = self.TIMES | self.ADJOINT_TIMES
        if isinstance(self._domain, MultiDomain) or isinstance(self._target, MultiDomain):
            raise NotImplementedError("MultiDomains are not supported yet")

    def apply(self, x, mode):
        from ..sugar import makeField

        self._check_input(x, mode)
        tgt = self._tgt(mode)
        return makeField(tgt, x.val.reshape(tgt.shape))

    def __repr__(self):
        return f"Reshape {self._shapes(self.target)} <- {self._shapes(self.domain)}"

    @staticmethod
    def _shapes(dom):
        return " ".join([str(dd.shape) for dd in dom])


class ExtractAtIndices(LinearOperator):
    """Extract Field values at given indices along a subspace of a DomainTuple,
    and puts them in a field with an unstructured domain for the given subspace.
    Note: This operator supports having the same index several times.
    If this is not the case also the numerically faster `GeometryRemover`
    and `MaskOperator` might be useful.

    Parameters
    ----------
    domain : DomainTuple
        Domain of the operator
    indices: tuple
        Tuple of indices to extract from the field along a given space.
        The length of the tuple needs to equal the number of axes of the space.
        Each entry of the tuple should be a tuple of indices to take along the
        respective axis of the space.
        Example for a 2d space: indices=((0,1,1,0), (3,4,1,5)) will extract
        the pixels (0,3), (1,4), (1,1) and (0,5).
    space: int
        The index of the space in the domain along which to extract values
    """
    def __init__(self, domain, indices, space=0):
        from ..sugar import makeDomain

        self._domain = makeDomain(domain)
        if not isinstance(indices, tuple):
            raise TypeError("indices need to be a tuple")
        if not len(self._domain[space].shape) == len(indices):
            raise ValueError("Shape of indices don't match dimension of space")
        target = [
                UnstructuredDomain(len(indices[0])) if i == space else sp
                for i, sp in enumerate(self._domain)]
        self._target = makeDomain(target)

        inds = [slice(None)] * len(domain.shape)
        dims_of_sps = [len(sp.shape) for sp in self._domain]
        start_ax_sp = int(np.sum(dims_of_sps[:space]))
        stop_ax_sp = start_ax_sp + dims_of_sps[space]
        for i, ax in enumerate(range(start_ax_sp, stop_ax_sp)):
            inds[ax] = indices[i]
        self._inds = tuple(inds)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode == self.TIMES:
            res = x[self._inds]
        else:
            res = np.zeros_like(x, shape=self._domain.shape, dtype=x.dtype)
            if x.device_id == -1:
                np.add.at(res, self._inds, x)
            else:
                # FIXME: Understand how np.add.at works for GPUs, then enable device copy test
                device_id = x.device_id
                res, x = res.at(-1), x.at(-1)
                np.add.at(res, self._inds, x)
                res = res.at(x.device_id).at(device_id)
        return Field.from_raw(self._tgt(mode), res)


class SqueezeOperator(LinearOperator):
    def __init__(self, domain, aggressive=False):
        """Removes trivial axes from a DomainTuple.

        This operator performs the equivalent of `np.squeeze` along selected
        dimensions (by default only truly trivial spaces, optionally also
        "aggressively" compressing RGSpaces and UnstructuredDomains).

        Parameters
        ----------
        domain : DomainTuple or compatible
            The input domain from which trivial axes may be removed.

        aggressive : bool, optional
            If True, also remove all singleton axes from certain domain types
            (`UnstructuredDomain`, `RGSpace`), effectively changing the
            dimensionality of the respective subspaces. If False (default), only
            entire domains of shape `(1,)` are removed.
        """
        self._domain = DomainTuple.make(domain)
        self._capability = self._all_ops

        ta, tgt, ax = [], [], 0
        for d in self._domain:
            if d.shape == (1,):  # Trivially removable
                ta.append(ax)
            elif aggressive and isinstance(d, (UnstructuredDomain, RGSpace)):  # Agressively removable
                shp, dst = [], []
                for ii, ss in enumerate(d.shape):
                    if ss == 1:
                        ta.append(ax+ii)
                    else:
                        shp.append(ss)
                        if isinstance(d, RGSpace):
                            dst.append(d.distances[ii])
                if isinstance(d, RGSpace):
                    tgt.append(RGSpace(shp, dst, d.harmonic))
                else:
                    tgt.append(UnstructuredDomain(shp))
            else:  # Not removeable
                tgt.append(d)
            ax += len(d.shape)

        self._target = DomainTuple.make(tgt)
        self._trivial_axes = tuple(ta)
        self._fwd_indexer = tuple(0 if i in ta else slice(None)
                                  for i in range(len(self.domain.shape)))
        if len(self._trivial_axes) == 0:
            raise RuntimeError("Nothing found to be squeezed")

    def apply(self, x, mode):
        self._check_input(x, mode)
        x = x.val
        if mode & (self.TIMES | self.ADJOINT_INVERSE_TIMES):
            x = x[self._fwd_indexer]
        else:
            for ax in self._trivial_axes:
                x = np.expand_dims(x, axis=ax)
        return Field(self._tgt(mode), x)
