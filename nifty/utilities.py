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

from builtins import next, range
import numpy as np
from itertools import product
import abc


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
    -------
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
        axes_select = [0 if x in axes else 1 for x, y in enumerate(shape)]
        axes_iterables = \
            [list(range(y)) for x, y in enumerate(shape) if x not in axes]
        for index in product(*axes_iterables):
            it_iter = iter(index)
            slice_list = [
                next(it_iter)
                if axis else slice(None, None) for axis in axes_select
                ]
            yield slice_list
    else:
        yield [slice(None, None)]


def cast_iseq_to_tuple(seq):
    if seq is None:
        return None
    if np.isscalar(seq):
        return (int(seq),)
    return tuple(int(item) for item in seq)


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
    http://groups.google.com/group/comp.lang.python/msg/26f7b4fcb4d66c95
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
        for attr, attribute in list(clsdict.items()):
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


class NiftyMeta(_DocStringInheritor, abc.ABCMeta):
    pass


def _fill_upper_half(tmp, res, axes):
    lastaxis = axes[-1]
    nlast = res.shape[lastaxis]
    ntmplast = tmp.shape[lastaxis]
    nrem = nlast - ntmplast
    slice1 = [slice(None)]*lastaxis + [slice(ntmplast, None)]
    slice2 = [slice(None)]*lastaxis + [slice(nrem, 0, -1)]
    for i in axes[:-1]:
        slice1[i] = slice(1, None)
        slice2[i] = slice(None, 0, -1)
    np.subtract(tmp[slice2].real, tmp[slice2].imag, out=res[slice1])
    for i, ax in enumerate(axes[:-1]):
        dim1 = [slice(None)]*ax + [slice(0, 1)]
        axes2 = axes[:i] + axes[i+1:]
        _fill_upper_half(tmp[dim1], res[dim1], axes2)


def _fill_array(tmp, res, axes):
    if axes is None:
        axes = tuple(range(tmp.ndim))
    lastaxis = axes[-1]
    ntmplast = tmp.shape[lastaxis]
    slice1 = [slice(None)]*lastaxis + [slice(0, ntmplast)]
    np.add(tmp.real, tmp.imag, out=res[slice1])
    _fill_upper_half(tmp, res, axes)
    return res


def hartley(a, axes=None):
    # Check if the axes provided are valid given the shape
    if axes is not None and \
            not all(axis < len(a.shape) for axis in axes):
        raise ValueError("Provided axes do not match array shape")
    if issubclass(a.dtype.type, np.complexfloating):
        raise TypeError("Hartley tansform requires real-valued arrays.")

    from pyfftw.interfaces.numpy_fft import rfftn
    tmp = rfftn(a, axes=axes)
    return _fill_array(tmp, np.empty_like(a), axes)


def general_axpy(a, x, y, out):
    if x.domain != y.domain or x.domain != out.domain:
        raise ValueError("Incompatible domains")

    if out is x:
        if a != 1.:
            out *= a
        out += y
    elif out is y:
        if a != 1.:
            out += a*x
        else:
            out += x
    else:
        out.copy_content_from(y)
        if a != 1.:
            out += a*x
        else:
            out += x
    return out
