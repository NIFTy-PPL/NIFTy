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

from ..domain_tuple import DomainTuple
from ..multi_domain import MultiDomain
from ..domains.unstructured_domain import UnstructuredDomain
from ..field import Field
from ..multi_field import MultiField
from .linear_operator import LinearOperator
from .endomorphic_operator import EndomorphicOperator
from .. import utilities
import numpy as np


class VdotOperator(LinearOperator):
    """Operator computing the scalar product of its input with a given Field.

    Parameters
    ----------
    field : Field or MultiField
        The field used to build the scalar product with the operator input
    """
    def __init__(self, field):
        self._field = field
        self._domain = field.domain
        self._target = DomainTuple.scalar_domain()
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_mode(mode)
        if mode == self.TIMES:
            return self._field.vdot(x)
        return self._field*x.val[()]


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
    domain: Domain, tuple of domains or DomainTuple
        domain of the input field

    """
    def __init__(self, domain):
        self._domain = DomainTuple.make(domain)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return x.real


class FieldAdapter(LinearOperator):
    """Operator for conversion between Fields and MultiFields.

    Parameters
    ----------
    tgt : Domain, tuple of Domain, DomainTuple, dict or MultiDomain:
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

    def __init__(self, tgt, name):
        from ..sugar import makeDomain
        tmp = makeDomain(tgt)
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
        s = 'FieldAdapter'
        dom = isinstance(self._domain, MultiDomain)
        tgt = isinstance(self._target, MultiDomain)
        if dom and tgt:
            s += ' {} <- {}'.format(self._target.keys(), self._domain.keys())
        elif dom:
            s += ' <- {}'.format(self._domain.keys())
        elif tgt:
            s += ' {} <-'.format(self._target.keys())
        return s


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

    def __init__(self, dom, name):
        from ..sugar import makeDomain
        tmp = makeDomain(dom)
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
    """

    def __init__(self, domain, target):
        from ..sugar import makeDomain
        self._domain = makeDomain(domain)
        self._target = makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    @staticmethod
    def _nullfield(dom):
        if isinstance(dom, DomainTuple):
            return Field(dom, 0)
        else:
            return MultiField.full(dom, 0)

    def apply(self, x, mode):
        self._check_input(x, mode)
        return self._nullfield(self._tgt(mode))


class PartialExtractor(LinearOperator):
    def __init__(self, domain, target):
        if not isinstance(domain, MultiDomain):
            raise TypeError("MultiDomain expected")
        if not isinstance(target, MultiDomain):
            raise TypeError("MultiDomain expected")
        self._domain = domain
        self._target = target
        for key in self._target.keys():
            if self._domain[key] is not self._target[key]:
                raise ValueError("domain mismatch")
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


class MatrixProductOperator(EndomorphicOperator):
    """Endomorphic matrix multiplication with input field.

    This operator supports scipy.sparse matrices and numpy arrays
    as the matrix to be applied.

    For numpy array matrices, can apply the matrix over a subspace
    of the input.

    If the input arrays have more than one dimension, for
    scipy.sparse matrices the `flatten` keyword argument must be
    set to true. This means that the input field will be flattened
    before applying the matrix and reshaped to its original shape
    afterwards.

    Matrices are tested regarding their compatibility with the
    called for application method.

    Flattening and subspace application are mutually exclusive.

    Parameters
    ----------
    domain: :class:`Domain` or :class:`DomainTuple`
        Domain of the operator.
        If :class:`DomainTuple` it is assumed to have only one entry.
    matrix: scipy.sparse matrix or numpy array
        Quadratic matrix of shape `(domain.shape, domain.shape)`
        (if `not flatten`) that supports `matrix.transpose()`.
        If it is not a numpy array, needs to be applicable to the val
        array of input fields by `matrix.dot()`.
    spaces: int or tuple of int, optional
        The subdomain(s) of "domain" which the operator acts on.
        If None, it acts on all elements.
        Only possible for numpy array matrices.
        If `len(domain) > 1` and `flatten=False`, this parameter is
        mandatory.
    flatten: boolean, optional
        Whether the input value array should be flattened before
        applying the matrix and reshaped to its original shape
        afterwards.
        Needed for scipy.sparse matrices if `len(domain) > 1`.
    """
    def __init__(self, domain, matrix, spaces=None, flatten=False):
        self._capability = self.TIMES | self.ADJOINT_TIMES
        self._domain = DomainTuple.make(domain)

        mat_dim = len(matrix.shape)

        if mat_dim % 2 != 0 or \
           matrix.shape != (matrix.shape[:mat_dim//2] + matrix.shape[:mat_dim//2]):
            raise ValueError("Matrix must be quadratic.")
        appl_dim = mat_dim // 2  # matrix application space dimension

        # take shortcut for trivial case
        if spaces is not None:
            if len(self._domain.shape) == 1 and spaces == (0, ):
                spaces = None

        if spaces is None:
            self._spaces = None
            self._active_axes = utilities.my_sum(self._domain.axes)
            appl_space_shape = self._domain.shape
            if flatten:
                appl_space_shape = (utilities.my_product(appl_space_shape), )
        else:
            if flatten:
                raise ValueError(
                    "Cannot flatten input AND apply to a subspace")
            if not isinstance(matrix, np.ndarray):
                raise ValueError(
                    "Application to subspaces only supported for numpy array matrices."
                )
            self._spaces = utilities.parse_spaces(spaces, len(self._domain))
            appl_space_shape = []
            active_axes = []
            for space_idx in spaces:
                appl_space_shape += self._domain[space_idx].shape
                active_axes += self._domain.axes[space_idx]
            appl_space_shape = tuple(appl_space_shape)
            self._active_axes = tuple(active_axes)

            self._mat_last_n = tuple([-appl_dim + i for i in range(appl_dim)])
            self._mat_first_n = np.arange(appl_dim)

        # Test if the matrix and the array it will be applied to fit
        if matrix.shape[:appl_dim] != appl_space_shape:
            raise ValueError(
                "Matrix and domain shapes are incompatible under the requested "
                + "application scheme.\n" +
                f"Matrix appl shape: {matrix.shape[:appl_dim]}, " +
                f"appl_space_shape: {appl_space_shape}.")

        self._mat = matrix
        self._mat_tr = matrix.transpose().conjugate()
        self._flatten = flatten

    def apply(self, x, mode):
        self._check_input(x, mode)
        times = (mode == self.TIMES)
        m = self._mat if times else self._mat_tr

        if self._spaces is None:
            if not self._flatten:
                res = m.dot(x.val)
            else:
                res = m.dot(x.val.flatten()).reshape(self._domain.shape)
            return Field(self._domain, res)

        mat_axes = self._mat_last_n if times else np.flip(self._mat_last_n)
        move_axes = self._mat_first_n if times else np.flip(self._mat_first_n)
        res = np.tensordot(m, x.val, axes=(mat_axes, self._active_axes))
        res = np.moveaxis(res, move_axes, self._active_axes)
        return Field(self._domain, res)
