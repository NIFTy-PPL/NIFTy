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
# Copyright(C) 2024-2025 Philipp Arras
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import contextlib
from functools import reduce
from itertools import chain
from operator import mul
from warnings import warn

import numpy as np

from ..config import _config
from .ducc_dispatch import vdot as cpu_vdot
from .logger import logger
from .utilities import assert_device_available, device_available, myassert

ALLOWED_WRAPPEES = [np.ndarray]
if device_available():
    import cupy
    ALLOWED_WRAPPEES += [cupy.ndarray]
ALLOWED_WRAPPEES = tuple(ALLOWED_WRAPPEES)


def _make_scalar_if_scalar(a):
    if a.shape == tuple():
        if device_available() and isinstance(a, cupy.ndarray):
            return cupy.asnumpy(a)[()]
        else:
            return a[()]
    return a


@contextlib.contextmanager
def assert_no_device_copies():
    from ..config import update
    old = _config["fail_on_device_copy"]
    update("fail_on_device_copy", True)
    try:
        yield
    finally:
        update("fail_on_device_copy", old)


class AnyArray(np.lib.mixins.NDArrayOperatorsMixin):
    """Wrapper for numpy and cupy arrays for flexible computations on CPU and GPUs

    AnyArray supports both CPU (`numpy.ndarray`) and GPU (`cupy.ndarray`)
    arrays. It provides automatic device-aware operations and conversions. It
    implements numpy-like operations such as element-wise operations, vector dot
    products, and norms. Additionally it can also be directly plugged into numpy
    functions (like `np.fft.fftn` or `np.transpose`) - the work will be
    dispatched correctly to numpy and cupy.

    `ALLOWED_WRAPPEES` are `numpy.ndarray` and, if cupy is install installed and
    GPUs are available, also `cupy.ndarray`.

    Parameters
    ----------
    arr: ALLOWED_WRAPPEES or AnyArray
        The input array that will be wrapped into an `AnyArray` instance. It can
        be either a `numpy.ndarray` (for CPU-based computations) or a
        `cupy.ndarray` (for GPU-based computations, if `cupy` is available). If
        `arr` is already an `AnyArray`, it is returned as is without
        modification. Scalars are automatically converted into `numpy.ndarray`
        instances. The stored array retains its original memory location, and no
        implicit device transfers are performed during initialization.

    """

    # ---Constructors---
    def __new__(cls, obj=None, *args, **kwargs):
        if isinstance(obj, AnyArray):
            return obj
        return super().__new__(cls)

    def __init__(self, arr):
        if isinstance(arr, AnyArray):
            return
        if np.isscalar(arr):
            arr = np.array(arr)
        if not isinstance(arr, ALLOWED_WRAPPEES):
            raise TypeError(f"arr must be in {ALLOWED_WRAPPEES}, got {arr}")
        if isinstance(arr, np.ndarray):
            self._device_id = -1
        elif isinstance(arr, cupy.ndarray):
            self._device_id = arr.device.id
        self._val = arr
        self._shape = arr.shape
        self._strides = arr.strides
        if self._val.shape == tuple():
            assert not isinstance(self._val[()], AnyArray)
        self._writeable = True
        if (
            _config["fail_on_nontrivial_anyarray_creation_on_host"]
            and (self._device_id == -1)
            and (self.size > 1)
        ):
            raise RuntimeError(
                "forbidden AnyArray creation on host: "
                f"shape {self._shape}, device_id {self._device_id}"
            )

    @staticmethod
    def full(shape, val, device_id=-1):
        """Create an AnyArray filled with a specified scalar value.

        This method generates an `AnyArray` of the given shape, filled with the
        specified scalar value. The resulting array can be placed on a specific
        device (CPU or GPU) by specifying `device_id`.

        Parameters
        ----------
        shape : tuple of int
            The shape of the output array.
        val : scalar (int, float, complex)
            The value to fill the array with. Must be a real or complex scalar.
        device_id : int, optional (default: -1)
            The target device for the array:
            - `-1` for CPU (`numpy.ndarray`).
            - `>=0` for GPU (`cupy.ndarray` with the specified device ID).

        Returns
        -------
        AnyArray
            An `AnyArray` instance of the specified shape, filled with `val`,
            located on the specified device.

        Examples
        --------
        Create a 3x3 array filled with 5 on the CPU:

        >>> AnyArray.full((3, 3), 5)

        Create a 2x2 complex array filled with `2 + 3j` on GPU (device 0):

        >>> AnyArray.full((2, 2), 2 + 3j, device_id=0)

        """
        if not np.isscalar(val):
            raise TypeError("val must be a scalar")
        if not (np.isreal(val) or np.iscomplex(val)):
            raise TypeError("need arithmetic scalar")
        xp = np if device_id == -1 else cupy
        return AnyArray(np.broadcast_to(xp.array(val), shape))

    # ---Views, copies, rights, etc.---
    def lock(self):
        """Make the AnyArray instance read-only.

        This method sets the underlying array to be non-writable, preventing any
        modifications to its contents. For `numpy.ndarray`, it updates the
        `writeable` flag to `False`.

        Once locked, the array cannot be modified unless a writable copy is
        created.

        Note
        ----
        For `cupy.ndarray`, write protection is not yet implemented due to
        limitations in `cupy` (see https://github.com/cupy/cupy/issues/2616).
        However, we still try to track the read-only status via an internal
        attribute (`AnyArray._writeable`).

        Examples
        --------
        >>> arr = AnyArray(np.array([1, 2, 3]))
        >>> arr.lock()
        >>> arr[0] = 10  # Raises ValueError

        """
        if isinstance(self, np.ndarray):
            self._val.flags.writeable = False
        # TODO: Set writable to False for cupy arrays as well, as soon as
        # https://github.com/cupy/cupy/issues/2616 is resolved
        self._writeable = False

    @property
    def readonly(self):
        """Indicates whether the AnyArray instance is read-only.

        Note
        ----
        For `cupy.ndarray`, write protection is not fully supported yet.

        Returns
        -------
        bool
            True if the array is read-only, False otherwise.

        Examples
        --------
        >>> arr = AnyArray(np.array([1, 2, 3]))
        >>> arr.readonly
        False
        >>> arr.lock()
        >>> arr.readonly
        True

        """
        return not self._writeable

    def at(self, device_id, *, check_fail=True):
        """Returns a copy of the AnyArray on the specified device.

        This method ensures that the AnyArray instance is located on the
        requested device (CPU or GPU). If the array is already on the target
        device, it is returned as-is. Otherwise, a copy is made and transferred
        to the target device.

        If the source array is read-only, the copied array on the new device
        will also be locked as read-only.

        Parameters
        ----------
        device_id : int
            The target device ID:
            - `-1` for CPU (`numpy.ndarray`)
            - `>= 0` for a specific GPU (`cupy.ndarray`)
        check_fail : bool, optional
            If True, additional debugging checks are performed to catch
            unexpected device transfers (default: True).

        Returns
        -------
        AnyArray
            A new AnyArray instance located on the specified device.

        Raises
        ------
        RuntimeError
            If `_config["fail_on_device_copy"]` is enabled and an unexpected device
            transfer occurs.

        Examples
        --------
        >>> arr = AnyArray(np.array([1, 2, 3]))
        >>> arr.device_id
        -1  # Currently on CPU

        >>> gpu_arr = arr.at(0)  # Move to GPU (if available)
        >>> gpu_arr.device_id
        0  # Now on GPU

        >>> cpu_arr = gpu_arr.at(-1)  # Move back to CPU
        >>> cpu_arr.device_id
        -1  # Now on CPU again

        """
        myassert(isinstance(device_id, int))
        origin, target = self._device_id, device_id
        if origin != target:
            s = f"AnyArray copy {self._device_id} -> {device_id}: shape {self._shape}"
            if self.shape != tuple():
                logger.debug(s)
            if _config["break_on_device_copy"] and check_fail:
                breakpoint()
            if _config["fail_on_device_copy"] and check_fail and self.shape != tuple():
                raise RuntimeError("Unexpected Device Copy:" + s)
        if origin == target:  # no transfer
            return self
        elif target == -1:  # gpu->cpu
            res = AnyArray(cupy.asnumpy(self._val))
        elif target > -1:  # cpu/gpu->gpu
            assert_device_available()
            with cupy.cuda.Device(device_id):
                arr = cupy.asarray(self._val)
            myassert(arr.device.id == device_id)
            res = AnyArray(arr)
        else:
            raise ValueError(f"device_id needs to be >= -1, got {device_id}")
        if self.readonly:
            res.lock()
        return res

    def asnumpy(self):
        """Returns a NumPy representation of the AnyArray.

        This method ensures that the array is located on the CPU
        (`numpy.ndarray`). If the `AnyArray` is already on the CPU, it is
        returned as-is. Otherwise, it is transferred from GPU to CPU.

        If the AnyArray is read-only, the returned NumPy array will also be
        flagged as read-only.

        Returns
        -------
        numpy.ndarray
            The content of the AnyArray as NumPy array.

        """
        res = self.at(-1).val
        if self.readonly:
            res.flags.writeable = False
        return res

    def view(self, *args, **kwargs):
        """Returns a new view of the array with the specified shape or type.

        This method creates a view of the underlying data without making a copy.
        It behaves similarly to `numpy.ndarray.view()`, returning an AnyArray
        instance wrapping the new view.

        Views share memory with the original array; modifying one modifies the
        other.

        Parameters
        ----------
        See numpy or cupy documentation.

        Returns
        -------
        AnyArray
            A new AnyArray instance representing the view.

        Examples
        --------
        >>> arr = AnyArray(np.array([[1, 2], [3, 4]]))
        >>> view = arr.view()
        >>> view is arr
        False  # Different object, but shares memory

        """
        return AnyArray(self._val.view(*args, **kwargs))

    def copy(self):
        """Returns a deep copy of the AnyArray.

        This method creates an independent copy of the array, ensuring that
        modifications to the copy do not affect the original.

        The copy is always writeable, even if the original AnyArray is
        read-only.

        Returns
        -------
        AnyArray
            A new AnyArray instance containing a copy of the original data.

        Examples
        --------
        >>> arr = AnyArray(np.array([1, 2, 3]))
        >>> arr_copy = arr.copy()
        >>> (arr_copy.val is arr.val)
        False  # The data is copied, not shared

        """
        return AnyArray(self._val.copy())

    # ---Properties, etc.---
    @property
    def val(self):
        """Returns the underlying array.

        This property provides access to the raw `numpy.ndarray` or
        `cupy.ndarray` wrapped by `AnyArray`.

        Returns
        -------
        numpy.ndarray or cupy.ndarray
            The wrapped array.

        """
        return self._val

    @property
    def device_id(self):
        """Returns the device ID where the array is stored.

        - `-1` represents a CPU (NumPy) array.
        - A non-negative integer represents a GPU (CuPy) array on the
          corresponding GPU device.

        Returns
        -------
        int
            The device ID of the array.

        """
        return self._device_id

    @property
    def shape(self):
        return self._val.shape

    @property
    def ndim(self):
        return len(self._val.shape)

    @property
    def size(self):
        if self._val.shape == tuple():
            return 1
        return reduce(mul, self._val.shape)

    @property
    def nbytes(self):
        return self._val.nbytes

    @property
    def strides(self):
        return self._val.strides

    @property
    def dtype(self):
        return self._val.dtype

    def astype(self, *args, **kwargs):
        return AnyArray(self._val.astype(*args, **kwargs))

    # ---Helpers---
    def _preprocess_index(self, index):
        if isinstance(index, tuple):
            out = []
            for ind in index:
                if isinstance(index, ALLOWED_WRAPPEES):
                    raise TypeError("Need to wrap numpy and cupy arrays into AnyArrays")
                if isinstance(ind, AnyArray):
                    if self.device_id != ind.device_id:
                        raise RuntimeError("device id mismatch")
                    ind = ind._val
                out.append(ind)
            index = tuple(out)
        elif isinstance(index, AnyArray):
            if self.device_id != index.device_id:
                raise RuntimeError("device id mismatch")
            index = index._val
        elif isinstance(index, ALLOWED_WRAPPEES):
            raise TypeError("Need to wrap numpy and cupy arrays into AnyArrays")
        return index

    def __getitem__(self, index):
        index = self._preprocess_index(index)
        return AnyArray(self._val[index])

    def __setitem__(self, index, value):
        if self.readonly:
            raise ValueError("assignment destination is read-only")
        if isinstance(value, AnyArray):
            value = value._val
        index = self._preprocess_index(index)
        if not isinstance(value, (int, float, complex)) and type(value) is not type(self._val):
            raise TypeError(f"Expected: {type(self._val)}, got: {type(value)}")
        self._val[index] = value

    # ---Point-wise nonlinearities---
    @staticmethod
    def _prep_args(args, kwargs, device_id=-1):
        for arg in args + tuple(kwargs.values()):
            if not (arg is None or np.isscalar(arg) or arg.jac is None):
                raise TypeError("bad argument")
        argstmp = tuple(arg if arg is None or np.isscalar(arg)
                        else arg._val.at(device_id)._val for arg in args)
        kwargstmp = {key: val if val is None or np.isscalar(val)
                     else val._val.at(device_id)._val
                     for key, val in kwargs.items()}
        return argstmp, kwargstmp

    def ptw(self, op, *args, **kwargs):
        """Performs an element-wise operation on the array.

        This method applies a specified pointwise operation to the array, using
        the corresponding function from :mod:`nifty.cl.pointwise`.

        Parameters
        ----------
        op : str
            The name of the pointwise operation (e.g., `exp`).
        *args : tuple
            Additional positional arguments for the operation.
        **kwargs : dict
            Additional keyword arguments for the operation.

        Returns
        -------
        AnyArray
            A new `AnyArray` instance with the result of the operation.

        """
        from .pointwise import ptw_dict
        argstmp, kwargstmp = self._prep_args(args, kwargs, self._device_id)
        return AnyArray(ptw_dict[op][0](self._val, *argstmp, **kwargstmp))

    def ptw_with_deriv(self, op, *args, **kwargs):
        """Performs element-wise operation and derivative simultaneously.

        This method applies a specified pointwise operation to the array and
        also computes its derivative, using the corresponding function from
        :mod:`nifty.cl.pointwise`.

        Parameters
        ----------
        op : str
            The name of the pointwise operation (e.g., `exp`).
        *args : tuple
            Additional positional arguments for the operation.
        **kwargs : dict
            Additional keyword arguments for the operation.

        Returns
        -------
        tuple of AnyArray
            A tuple containing:
            - The result of the operation (`AnyArray`).
            - The derivative of the operation (`AnyArray`).

        Examples
        --------
        >>> arr = AnyArray(np.array([4, 9, 16]))
        >>> res, deriv = arr.ptw_with_deriv("sqrt")
        >>> res.val
        array([2., 3., 4.])
        >>> deriv.val
        array([0.25, 0.16666667, 0.125])

        """
        from .pointwise import ptw_dict
        argstmp, kwargstmp = self._prep_args(args, kwargs, self._device_id)
        return tuple(map(AnyArray, ptw_dict[op][1](self._val, *argstmp, **kwargstmp)))

    # ---Numeric properties---
    @property
    def real(self):
        return AnyArray(self._val.real)

    @property
    def imag(self):
        return AnyArray(self._val.imag)

    # ---Operations---
    def norm(self, ord=2):
        res = np.linalg.norm(self._val.reshape(-1), ord=ord)
        return _make_scalar_if_scalar(res)

    def vdot(self, x):
        x = x.at(self.device_id)
        if self._device_id == -1:
            return cpu_vdot(self._val, x._val)
        return _make_scalar_if_scalar(cupy.vdot(self._val, x._val))

    @property
    def T(self):
        return np.transpose(self)

    @property
    def H(self):
        return np.transpose(self.conj())

    # ---Numpy ufunc and array_function interface---
    def _unify_device_ids_and_get_val(self, args, kwargs):
        # TODO: Leverage Cupy's support for peer-to-peer memory access.
        device_id = None

        args2 = []
        for vv in args:
            if isinstance(vv, AnyArray):
                if device_id is None:
                    device_id = vv._device_id
                vv = vv.at(device_id)._val
            if isinstance(vv, (tuple, list)):
                tmp = []
                for ww in vv:
                    if isinstance(ww, AnyArray):
                        if device_id is None:
                            device_id = ww._device_id
                        ww = ww.at(device_id)._val
                    tmp.append(ww)
                vv = tuple(tmp)
            args2.append(vv)

        kwargs2 = {}
        for ii, vv in kwargs.items():
            if ii == "out":
                for vvv in vv:
                    if isinstance(vv, AnyArray) and device_id is not None \
                       and vvv.device_id != device_id:
                        raise RuntimeError("Specifying 'out' on a different "
                                           "device is not supported (yet).")
            if isinstance(vv, AnyArray):
                if device_id is None:
                    device_id = vv._device_id
                vv = vv.at(device_id)._val
            if isinstance(vv, (tuple, list)):
                tmp = []
                for ww in vv:
                    if isinstance(ww, AnyArray):
                        if device_id is None:
                            device_id = ww._device_id
                        ww = ww.at(device_id)._val
                    tmp.append(ww)
                vv = tuple(tmp)
            kwargs2[ii] = vv

        return args2, kwargs2

    @staticmethod
    def _check_responsibility(lst):
        any_array_present = False
        for x in lst:
            if isinstance(x, ALLOWED_WRAPPEES):
                return False
            elif isinstance(x, (tuple, list)):
                for xx in x:
                    if isinstance(xx, ALLOWED_WRAPPEES):
                        return False
                    if isinstance(xx, AnyArray):
                        any_array_present = True
            elif isinstance(x, AnyArray):
                any_array_present = True
        return any_array_present

    @staticmethod
    def _wrap_result(obj, out=None):
        if obj is NotImplemented:
            warn("Falling back to inefficient way to determine responsibility "
                 "for AnyArray operations. Please report.")
            return NotImplemented
        # TODO: Add make_scalar_if_scalar here
        if out is not None:
            return out
        if isinstance(obj, ALLOWED_WRAPPEES):
            return AnyArray(obj)
        if isinstance(obj, tuple):
            return tuple(AnyArray(x) if isinstance(x, ALLOWED_WRAPPEES)
                         else x for x in obj)
        return obj

    def __array_ufunc__(self, ufunc, method, *args, **kwargs):
        if not self._check_responsibility(chain(args, kwargs.values())):
            return NotImplemented
        args2, kwargs2 = self._unify_device_ids_and_get_val(args, kwargs)
        return self._wrap_result(getattr(ufunc, method)(*args2, **kwargs2),
                                 out=kwargs.get("out", None))

    def __array_function__(self, func, types, args, kwargs):
        if any(not issubclass(t, AnyArray) for t in types):
            return NotImplemented
        if not self._check_responsibility(chain(args, kwargs.values())):
            return NotImplemented
        args2, kwargs2 = self._unify_device_ids_and_get_val(args, kwargs)
        return self._wrap_result(func(*args2, **kwargs2),
                                 out=kwargs.get("out", None))

    # ---Printing functions---
    def __repr__(self):
        return f"<nifty.cl.AnyArray: device_id={self._device_id}>"

    def __str__(self):
        return "\n".join(["nifty.cl.AnyArray instance",
                          f"- device_id = {self.device_id}",
                          f"- shape     = {self.shape}",
                          f"- strides   = {self.strides}",
                          f"- dtype     = {self.dtype}",
                          f"- size      = {self.size}",
                          f"- memory    = {self.nbytes*1e-6:.3f} MB",
                          ])

    def __bool__(self):
        return self._val.__bool__()


#---Operations with potential scalar output---
for op in ["sum", "prod", "mean", "var", "std", "all", "any", "min", "max",
           "conjugate", "reshape", "conj", "flatten"]:
    def func(op):
        def func2(self, *args, **kwargs):
            res = getattr(self._val, op)(*args, **kwargs)
            # Convention: simple types (float, int, complex, ...) are always on the host
            res = _make_scalar_if_scalar(res)
            return res if np.isscalar(res) else AnyArray(res)
        return func2
    setattr(AnyArray, op, func(op))


# ---Inplace operations---
for op in ["__iadd__", "__isub__", "__imul__", "__idiv__",
           "__itruediv__", "__ifloordiv__", "__ipow__"]:
    def func(op):
        def func2(self, other):
            if not isinstance(other, AnyArray):
                return NotImplemented
            if self.device_id != other.device_id:
                raise RuntimeError(f"Cannot compute {op} if AnyArrays are stored on different devices. "
                                   f"Got: {self.device_id} and {other.device_id}.")
            if self._writeable:
                return AnyArray(getattr(self._val, op)(other._val))
            raise TypeError("AnyArray is readonly")
        return func2
    setattr(AnyArray, op, func(op))
