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
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from builtins import next, range
import numpy as np
from itertools import product
import abc
from future.utils import with_metaclass


__all__ = ["get_slice_list", "safe_cast", "parse_spaces", "infer_space",
           "memo", "NiftyMetaBase", "hartley", "my_fftn_r2c"]

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


def safe_cast(tfunc, val):
    tmp = tfunc(val)
    if val != tmp:
        raise ValueError("value changed during cast")
    return tmp


def parse_spaces(spaces, maxidx):
    maxidx = safe_cast(int, maxidx)
    if spaces is None:
        return tuple(range(maxidx))
    elif np.isscalar(spaces):
        spaces = (safe_cast(int, spaces),)
    else:
        spaces = tuple(safe_cast(int, item) for item in spaces)
    tmp = tuple(set(spaces))
    if tmp[0] < 0 or tmp[-1] >= maxidx:
        raise ValueError("space index out of range")
    if len(tmp) != len(spaces):
        raise ValueError("multiply defined space indices")
    return spaces


def infer_space(domain, space):
    if space is None:
        if len(domain) != 1:
            raise ValueError("need a Field with exactly one domain")
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


def NiftyMetaBase():
    return with_metaclass(NiftyMeta, type('NewBase', (object,), {}))


def hartley(a, axes=None):
    # Check if the axes provided are valid given the shape
    if axes is not None and \
            not all(axis < len(a.shape) for axis in axes):
        raise ValueError("Provided axes do not match array shape")
    if np.issubdtype(a.dtype, np.complexfloating):
        raise TypeError("Hartley transform requires real-valued arrays.")

    from pyfftw.interfaces.numpy_fft import rfftn
    tmp = rfftn(a, axes=axes)

    def _fill_array(tmp, res, axes):
        if axes is None:
            axes = tuple(range(tmp.ndim))
        lastaxis = axes[-1]
        ntmplast = tmp.shape[lastaxis]
        slice1 = [slice(None)]*lastaxis + [slice(0, ntmplast)]
        np.add(tmp.real, tmp.imag, out=res[slice1])

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

        _fill_upper_half(tmp, res, axes)
        return res

    return _fill_array(tmp, np.empty_like(a), axes)


# Do a real-to-complex forward FFT and return the _full_ output array
def my_fftn_r2c(a, axes=None):
    # Check if the axes provided are valid given the shape
    if axes is not None and \
            not all(axis < len(a.shape) for axis in axes):
        raise ValueError("Provided axes do not match array shape")
    if np.issubdtype(a.dtype, np.complexfloating):
        raise TypeError("Transform requires real-valued input arrays.")

    from pyfftw.interfaces.numpy_fft import rfftn
    tmp = rfftn(a, axes=axes)

    def _fill_complex_array(tmp, res, axes):
        if axes is None:
            axes = tuple(range(tmp.ndim))
        lastaxis = axes[-1]
        ntmplast = tmp.shape[lastaxis]
        slice1 = [slice(None)]*lastaxis + [slice(0, ntmplast)]
        res[slice1] = tmp

        def _fill_upper_half_complex(tmp, res, axes):
            lastaxis = axes[-1]
            nlast = res.shape[lastaxis]
            ntmplast = tmp.shape[lastaxis]
            nrem = nlast - ntmplast
            slice1 = [slice(None)]*lastaxis + [slice(ntmplast, None)]
            slice2 = [slice(None)]*lastaxis + [slice(nrem, 0, -1)]
            for i in axes[:-1]:
                slice1[i] = slice(1, None)
                slice2[i] = slice(None, 0, -1)
            # np.conjugate(tmp[slice2], out=res[slice1])
            res[slice1] = np.conjugate(tmp[slice2])
            for i, ax in enumerate(axes[:-1]):
                dim1 = [slice(None)]*ax + [slice(0, 1)]
                axes2 = axes[:i] + axes[i+1:]
                _fill_upper_half_complex(tmp[dim1], res[dim1], axes2)

        _fill_upper_half_complex(tmp, res, axes)
        return res

    return _fill_complex_array(tmp, np.empty_like(a, dtype=tmp.dtype), axes)
