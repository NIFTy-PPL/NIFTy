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
# Copyright(C) 2013-2022 Max-Planck-Society
# Copyright(C) 2025 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import collections
import contextlib
import pickle
from functools import reduce
from itertools import product

import numpy as np

__all__ = ["get_slice_list", "safe_cast", "parse_spaces", "infer_space",
           "memo", "NiftyMeta", "my_sum", "my_lincomb_simple",
           "my_lincomb", "indent", "my_product", "frozendict",
           "special_add_at", "iscomplextype", "issingleprec",
           "value_reshaper", "lognormal_moments", "check_object_identity",
           "check_MPI_equality", "check_MPI_synced_random_state"]


def my_sum(iterable):
    return reduce(lambda x, y: x+y, iterable)


def my_lincomb_simple(terms, factors):
    terms2 = map(lambda v: v[0]*v[1], zip(terms, factors))
    return my_sum(terms2)


def my_lincomb(terms, factors):
    terms2 = map(lambda v: v[0] if v[1] == 1. else v[0]*v[1],
                 zip(terms, factors))
    return my_sum(terms2)


def my_product(iterable):
    return reduce(lambda x, y: x*y, iterable)


def get_slice_list(shape, axes):
    """
    Helper function which generates slice list(s) to traverse over all
    combinations of axes, other than the selected axes.

    Parameters
    ----------
    shape: tuple
        Shape of the data array to traverse over.
    axes: tuple
        Axes which should not be iterated over.

    Yields
    ------
    list
        The next list of indices and/or slice objects for each dimension.

    Raises
    ------
    ValueError
        If shape is empty.
        If axes(axis) does not match shape.
    """
    if shape is None:
        raise ValueError("shape cannot be None.")

    if axes:
        if not all(axis < len(shape) for axis in axes):
            raise ValueError("axes(axis) does not match shape.")
        axes_select = [0 if x in axes else 1 for x in range(len(shape))]
        axes_iterables = \
            [list(range(y)) for x, y in enumerate(shape) if x not in axes]
        for index in product(*axes_iterables):
            it_iter = iter(index)
            slice_list = tuple(
                next(it_iter)
                if axis else slice(None, None) for axis in axes_select
            )
            yield slice_list
    else:
        yield [slice(None, None)]


def safe_cast(tfunc, val):
    tmp = tfunc(val)
    if val != tmp:
        raise ValueError("value changed during cast")
    return tmp


def parse_spaces(spaces, nspc):
    nspc = safe_cast(int, nspc)
    if spaces is None:
        return tuple(range(nspc))
    elif np.isscalar(spaces):
        spaces = (safe_cast(int, spaces),)
    else:
        spaces = tuple(safe_cast(int, item) for item in spaces)
    if len(spaces) == 0:
        return spaces
    tmp = tuple(set(spaces))
    if tmp[0] < 0 or tmp[-1] >= nspc:
        raise ValueError("space index out of range")
    if len(tmp) != len(spaces):
        raise ValueError("multiply defined space indices")
    return spaces


def infer_space(domain, space):
    if space is None:
        if len(domain) != 1:
            raise ValueError("'space' index must be given for objects based on"
                             " DomainTuples containing more than one domain")
        space = 0
    space = int(space)
    if space < 0 or space >= len(domain):
        raise ValueError("space index out of range")
    return space


def memo(f):
    name = f.__name__

    def wrapped_f(self):
        if not hasattr(self, "_cache"):
            self._cache = {}
        try:
            return self._cache[name]
        except KeyError:
            self._cache[name] = f(self)
            return self._cache[name]
    return wrapped_f


class _DocStringInheritor(type):
    """
    A variation on
    https://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
    by Paul McGuire
    """
    def __new__(meta, name, bases, clsdict):
        if not('__doc__' in clsdict and clsdict['__doc__']):
            for mro_cls in (mro_cls for base in bases
                            for mro_cls in base.mro()):
                doc = mro_cls.__doc__
                if doc:
                    clsdict['__doc__'] = doc
                    break
        for attr, attribute in clsdict.items():
            if not attribute.__doc__:
                for mro_cls in (mro_cls for base in bases
                                for mro_cls in base.mro()
                                if hasattr(mro_cls, attr)):
                    doc = getattr(getattr(mro_cls, attr), '__doc__')
                    if doc:
                        if isinstance(attribute, property):
                            clsdict[attr] = property(attribute.fget,
                                                     attribute.fset,
                                                     attribute.fdel,
                                                     doc)
                        else:
                            attribute.__doc__ = doc
                        break
        return super(_DocStringInheritor, meta).__new__(meta, name,
                                                        bases, clsdict)


class NiftyMeta(_DocStringInheritor):
    pass


class frozendict(collections.abc.Mapping):
    """
    An immutable wrapper around dictionaries that implements the complete
    :py:class:`collections.Mapping` interface. It can be used as a drop-in
    replacement for dictionaries where immutability is desired.
    """

    dict_cls = dict

    def __init__(self, *args, **kwargs):
        self._dict = self.dict_cls(*args, **kwargs)
        self._hash = None

    def __getitem__(self, key):
        return self._dict[key]

    def __contains__(self, key):
        return key in self._dict

    def copy(self, **add_or_replace):
        return self.__class__(self, **add_or_replace)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __repr__(self):
        return '<{} {}>'.format(self.__class__.__name__, self._dict)

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(frozenset(self._dict.items()))
        return self._hash


def _special_add_at(a, axis, index, b):
    if type(a) is not type(b):
        raise TypeError(f"type mismatch, got {type(a)} and {type(b)}")
    if a.dtype != b.dtype:
        raise TypeError("data type mismatch")
    if index.device_id != a.device_id:
        raise RuntimeError("`index` needs to be on same device as `a`")
    if b.device_id != a.device_id:
        raise RuntimeError("`b` needs to be on same device as `a`")
    sz1 = int(np.prod(a.shape[:axis]))
    sz3 = int(np.prod(a.shape[axis+1:]))
    a2 = a.reshape([sz1, -1, sz3])
    b2 = b.reshape([sz1, -1, sz3])
    if iscomplextype(a.dtype):
        dt2 = a.real.dtype
        a2 = a2.view(dt2)
        b2 = b2.view(dt2)
        sz3 *= 2
    for i1 in range(sz1):
        for i3 in range(sz3):
            a2[i1, :, i3] += np.bincount(index, b2[i1, :, i3],
                                         minlength=a2.shape[1])
    if iscomplextype(a.dtype):
        a2 = a2.view(a.dtype)
    return a2.reshape(a.shape)



def iscomplextype(dtype):
    if isinstance(dtype, dict):
        return _getunique(iscomplextype, dtype.values())
    return np.issubdtype(dtype, np.complexfloating)


try:
    from ducc0.misc import special_add_at
except ImportError:
    special_add_at = _special_add_at


def issingleprec(dtype):
    if isinstance(dtype, dict):
        return _getunique(issingleprec, dtype.values())
    return dtype.type in (np.float32, np.complex64)


def _getunique(f, iterable):
    lst = list(f(vv) for vv in iterable)
    if len(set(lst)) == 1:
        return lst[0]
    raise RuntimeError("Value is not unique", lst)


def indent(inp):
    return "\n".join((("  "+s).rstrip() for s in inp.splitlines()))


def shareRange(nwork, nshares, myshare):
    """Divides a number of work items as fairly as possible into a given number
    of shares.

    Parameters
    ----------
    nwork: int
        number of work items
    nshares: int
        number of shares among which the work should be distributed
    myshare: int
        the share for which the range of work items is requested


    Returns
    -------
    lo, hi: int
        index range of work items for this share
    """

    nbase = nwork//nshares
    additional = nwork % nshares
    lo = myshare*nbase + min(myshare, additional)
    hi = lo + nbase + int(myshare < additional)
    return lo, hi


def get_MPI_params_from_comm(comm):
    if comm is None:
        return 1, 0, True
    size = comm.Get_size()
    rank = comm.Get_rank()
    return size, rank, rank == 0


def get_MPI_params():
    """Returns basic information about the MPI setup of the running script.

    To enable transferring large objects (>2 GiB) via MPI, wrap the returned `comm`
    object with `mpi4py.utils.pkl5.Intracomm` before passing it to nifty
    functions.

    Returns
    -------
    comm: MPI communicator or None
        if MPI is detected _and_ more than one task is active, returns
        the world communicator, else returns None
    size: int
        the number of tasks running in total
    rank: int
        the rank of this task
    master: bool
        True if rank == 0, else False
    """

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        if size == 1:
            return None, 1, 0, True
        rank = comm.Get_rank()
        return comm, size, rank, rank == 0
    except ImportError:
        return None, 1, 0, True


def allreduce_sum(obj, comm):
    """ This is a deterministic implementation of MPI allreduce

    Numeric addition is not associative due to rounding errors.
    Therefore we provide our own implementation that is consistent
    no matter if MPI is used and how many tasks there are.

    At the beginning, a list `who` is constructed, that states which obj can
    be found on which MPI task.
    Then elements are added pairwise, with increasing pair distance.
    In the first round, the distance between pair members is 1:
      v[0] := v[0] + v[1]
      v[2] := v[2] + v[3]
      v[4] := v[4] + v[5]
    Entries whose summation partner lies beyond the end of the array
    stay unchanged.
    When both summation partners are not located on the same MPI task,
    the second summand is sent to the task holding the first summand and
    the operation is carried out there.
    For the next round, the distance is doubled:
      v[0] := v[0] + v[2]
      v[4] := v[4] + v[6]
      v[8] := v[8] + v[10]
    This is repeated until the distance exceeds the length of the array.
    At this point v[0] contains the sum of all entries, which is then
    broadcast to all tasks.
    """
    vals = list(obj)
    if comm is None:
        nobj = len(vals)
        who = np.zeros(nobj, dtype=np.int32)
        rank = 0
        dtype = type(vals[0])
    else:
        rank = comm.Get_rank()
        nobj_list = comm.allgather(len(vals))
        all_hi = list(np.cumsum(nobj_list))
        all_lo = [0] + all_hi[:-1]
        nobj = all_hi[-1]
        rank_lo_hi = [(l, h) for l, h in zip(all_lo, all_hi)]
        lo, hi = rank_lo_hi[rank]
        vals = [None]*lo + vals + [None]*(nobj-hi)
        who = [t for t, (l, h) in enumerate(rank_lo_hi) for cnt in range(h-l)]
        # Determine dtype of elements
        dtype = comm.allreduce([type(x) for x in vals if x is not None])
        dtype = list(set(dtype))
        assert len(dtype) == 1
        dtype = dtype[0]

    step = 1
    while step < nobj:
        for j in range(0, nobj, 2*step):
            if j+step < nobj:  # summation partner found
                if rank == who[j]:
                    if who[j] == who[j+step]:  # no communication required
                        vals[j] = vals[j] + vals[j+step]
                        vals[j+step] = None
                    else:
                        vals[j] = vals[j] + _recv(comm, source=who[j+step], dtype=dtype)
                elif rank == who[j+step]:
                    _send(comm, vals[j+step], dest=who[j], dtype=dtype)
                    vals[j+step] = None
        step *= 2
    if comm is None:
        return vals[0]
    return _bcast(comm, vals[0], root=who[0])


# TODO: Later make sure that all this also works directly with GPU arrays https://mpi4py.readthedocs.io/en/stable/tutorial.html#gpu-aware-mpi-python-gpu-arrays
def _send(comm, obj, dest, dtype):
    from .field import Field
    from .multi_field import MultiField

    assert isinstance(obj, dtype)
    if dtype is np.ndarray:
        shp_orig = obj.shape
        obj = np.ascontiguousarray(obj).reshape(shp_orig)
        comm.send((obj.shape, obj.dtype), dest=dest)
        comm.Send(obj, dest=dest)
        return
    elif dtype is Field:
        comm.send((obj.domain, type(obj.val)), dest=dest)
        _send(comm, obj.val, dest, type(obj.val))
        return
    elif dtype is MultiField:
        dct = obj.to_dict()
        keys = tuple(dct.keys())
        _send(comm, keys, dest, tuple)
        for kk, vv in dct.items():
            _send(comm, vv, dest, Field)
        return
    return comm.send(obj, dest=dest)

def _recv(comm, source, dtype):
    from .field import Field
    from .multi_field import MultiField

    if dtype is np.ndarray:
        shape, dtype = comm.recv(source=source)
        buf = np.empty(shape, dtype)  # assume C-style contiguous arrays
        comm.Recv(buf, source)
        return buf
    elif dtype is Field:
        dom, dtype = comm.recv(source=source)
        return Field(dom, _recv(comm, source, dtype))
    elif dtype is MultiField:
        dct = {}
        keys = _recv(comm, source, tuple)
        for kk in keys:
            dct[kk] = _recv(comm, source, Field)
        return MultiField.from_dict(dct)
    return comm.recv(source=source)

def _bcast(comm, obj, root):
    from .field import Field
    from .multi_field import MultiField

    master = comm.Get_rank() == root
    dtype = comm.bcast(type(obj), root=root)
    if dtype is np.ndarray:
        shape, dtype = (obj.shape, obj.dtype) if master else (None, None)
        shape, dtype = comm.bcast((shape, dtype), root=root)
        data = obj if master else np.empty(shape, dtype)
        comm.Bcast(data, root=root)
        return data
    elif dtype is Field:
        dom, val, dtype = (obj.domain, obj.val, obj.dtype) if master else (None, None, None)
        dom, dtype = comm.bcast((dom, dtype), root=root)
        return Field(dom, _bcast(comm, val, root))
    elif dtype is MultiField:
        keys = tuple(obj.keys()) if master else None
        keys = comm.bcast(keys, root=root)
        dct = {}
        for kk in keys:
            dct[kk] = _bcast(comm, obj[kk] if master else None, root)
        return MultiField.from_dict(dct)
    return comm.bcast(obj, root=root)


def value_reshaper(x, N):
    """Produce arrays of shape `(N,)`.
    If `x` is a scalar or array of length one, fill the target array with it.
    If `x` is an array, check if it has the right shape."""
    x = np.asarray(x, dtype=float)
    if x.shape in [(), (1, )]:
        return np.full(N, x) if N != 0 else x.reshape(())
    elif x.shape == (N, ):
        return x
    raise TypeError("x and N are incompatible")


def lognormal_moments(mean, sigma, N=0):
    """Calculates the parameters for a normal distribution `n(x)`
    such that `exp(n)(x)` has the mean and standard deviation given.

    Used in :func:`~nifty.cl.normal_operators.LognormalTransform`."""
    mean, sigma = (value_reshaper(param, N) for param in (mean, sigma))
    if not np.all(mean > 0):
        raise ValueError("mean must be greater 0; got {!r}".format(mean))
    if not np.all(sigma > 0):
        raise ValueError("sig must be greater 0; got {!r}".format(sigma))

    logsigma = np.sqrt(np.log1p((sigma / mean)**2))
    logmean = np.log(mean) - logsigma**2 / 2
    return logmean, logsigma


def myassert(val):
    """Safe alternative to python's assert statement which is active even if
    `__debug__` is False."""
    if not val:
        raise AssertionError


def check_object_identity(obj0, obj1):
    """Check if two objects are the same and throw ValueError if not."""
    if obj0 is not obj1:
        raise ValueError(f"Mismatch:\n{obj0}\n{obj1}")


def check_MPI_equality(obj, comm, hash=False):
    """Check that object is the same on all MPI tasks associated to a given
    communicator.

    Raises a RuntimeError if it differs.

    Parameters
    ----------
    obj :
        Any Python object that implements __eq__.
    comm : MPI communicator or None
        If comm is None, no check will be performed
    """
    if comm is None:
        return
    if not _MPI_unique(obj, comm, hash=hash):
        raise RuntimeError("MPI tasks are not in sync")


def _MPI_unique(obj, comm, hash=False):
    from hashlib import blake2b

    obj = pickle.dumps(obj)
    obj = blake2b(obj).hexdigest() if hash else obj
    return len(set(comm.allgather(obj))) == 1


def check_MPI_synced_random_state(comm):
    """Check that random state is the same on all MPI tasks associated to a
    given communicator.

    Raises a RuntimeError if it differs.

    Parameters
    ----------
    comm : MPI communicator or None
        If comm is None, no check will be performed
    """
    from .random import getState
    if comm is None:
        return
    check_MPI_equality(getState(), comm)


@contextlib.contextmanager
def ensure_all_tasks_succeed(comm):
    if comm is not None:
        comm.Barrier()
    all_good, exception_message = True, ""
    try:
        yield
    except Exception as e:
        all_good, exception_message = False, str(e)
    finally:
        all_good = [all_good] if comm is None else comm.allgather(all_good)
        if not all(all_good):
            raise RuntimeError(exception_message)


def check_dtype_or_none(obj, domain=None):
    """Check that dtype is compatible with a given domain.

    If domain is None or a DomainTuple, the obj is checked to be either a
    floating dtype or None. If domain is a MultiDomain, obj can still be a
    single quantity and is treated like in the previous case. Or it can be a
    dictionary; then all its entries need to satisfy the previous condition to
    pass the test.

    Raises a TypeError if any incompatibility is detected.

    Parameters
    ----------
    obj : object
        The object to be checked.
    domain : DomainTuple or MultiDomain
        If it is a MultiDomain,
    """
    from .multi_domain import MultiDomain
    from .sugar import makeDomain
    if domain is not None:
        domain = makeDomain(domain)
        if isinstance(domain, MultiDomain) and isinstance(obj, dict):
            for kk in domain.keys():
                check_dtype_or_none(obj[kk])
            return
    check = obj in [np.float32, np.float64, float,
                    np.complex64, np.complex128, complex,
                    None]
    if not check:
        s = "Need to pass floating dtype (e.g. np.float64, complex) "
        s += f"or `None` to this function.\nHave recieved:\n{obj}"
        raise TypeError(s)


class Nop:
    def nop(*args, **kw):
        return Nop()
    def __getattr__(self, _):
        return self.nop


def strtobool(val):
    val = val.lower().strip()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError(f"invalid truth value {val!r}")


try:
    import cupy
    _device_available = cupy.cuda.runtime.getDeviceCount() > 0
except ImportError:
    _device_available = False
def device_available():
    return _device_available


def assert_device_available():
    if not device_available():
        raise RuntimeError("Cupy not installed or cuda device not available")
