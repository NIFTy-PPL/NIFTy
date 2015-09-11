# -*- coding: utf-8 -*-
# NIFTY (Numerical Information Field Theory) has been developed at the
# Max-Planck-Institute for Astrophysics.
#
# Copyright (C) 2015 Max-Planck-Society
#
# Author: Theo Steininger
# Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import numpy as np
from weakref import WeakValueDictionary as weakdict

from keepers import about,\
                    global_configuration as gc,\
                    global_dependency_injector as gdi

MPI = gdi[gc['mpi_module']]
h5py = gdi.get('h5py')
pyfftw = gdi.get('pyfftw')


## initialize the 'FOUND-packages'-dictionary
#FOUND = {}
#try:
#    from mpi4py import MPI
#    FOUND['MPI'] = True
#except(ImportError):
#    import mpi_dummy as MPI
#    FOUND['MPI'] = False
#
#try:
#    import pyfftw
#    FOUND['pyfftw'] = True
#except(ImportError):
#    FOUND['pyfftw'] = False
#
#try:
#    import h5py
#    FOUND['h5py'] = True
#    FOUND['h5py_parallel'] = h5py.get_config().mpi
#except(ImportError):
#    FOUND['h5py'] = False
#    FOUND['h5py_parallel'] = False

_maybe_fftw = ['fftw'] if ('pyfftw' in gdi) else []
STRATEGIES = {
                'all': ['not', 'equal', 'freeform'] + _maybe_fftw,
                'global': ['not', 'equal'] + _maybe_fftw,
                'local': ['freeform'],
                'slicing': ['equal', 'freeform'] + _maybe_fftw,
                'not': ['not'],
                'hdf5': ['equal'] + _maybe_fftw,
             }


class distributed_data_object(object):
    """

        NIFTY class for distributed data

        Parameters
        ----------
        global_data : {tuple, list, numpy.ndarray} *at least 1-dimensional*
            Initial data which will be casted to a numpy.ndarray and then
            stored according to the distribution strategy. The global_data's
            shape overwrites global_shape.
        global_shape : tuple of ints, *optional*
            If no global_data is supplied, global_shape can be used to
            initialize an empty distributed_data_object
        dtype : type, *optional*
            If an explicit dtype is supplied, the given global_data will be
            casted to it.
        distribution_strategy : {'fftw' (default), 'not'}, *optional*
            Specifies the way, how global_data will be distributed to the
            individual nodes.
            'fftw' follows the distribution strategy of pyfftw.
            'not' does not distribute the data at all.


        Attributes
        ----------
        data : numpy.ndarray
            The numpy.ndarray in which the individual node's data is stored.
        dtype : type
            Data type of the data object.
        distribution_strategy : string
            Name of the used distribution_strategy
        distributor : distributor
            The distributor object which takes care of all distribution and
            consolidation of the data.
        shape : tuple of int
            The global shape of the data

        Raises
        ------
        TypeError :
            If the supplied distribution strategy is not known.

    """

    def __init__(self, global_data=None, global_shape=None, dtype=None,
                 local_data=None, local_shape=None,
                 distribution_strategy='fftw', hermitian=False,
                 alias=None, path=None, comm=MPI.COMM_WORLD,
                 copy=True, *args, **kwargs):

        # TODO: allow init with empty shape

        if isinstance(global_data, tuple) or isinstance(global_data, list):
            global_data = np.array(global_data, copy=False)
        if isinstance(local_data, tuple) or isinstance(local_data, list):
            local_data = np.array(local_data, copy=False)

        self.distributor = distributor_factory.get_distributor(
                                distribution_strategy=distribution_strategy,
                                comm=comm,
                                global_data=global_data,
                                global_shape=global_shape,
                                local_data=local_data,
                                local_shape=local_shape,
                                alias=alias,
                                path=path,
                                dtype=dtype,
                                **kwargs)

        self.distribution_strategy = distribution_strategy
        self.dtype = self.distributor.dtype
        self.shape = self.distributor.global_shape
        self.local_shape = self.distributor.local_shape
        self.comm = self.distributor.comm

        self.init_args = args
        self.init_kwargs = kwargs

        (self.data, self.hermitian) = self.distributor.initialize_data(
            global_data=global_data,
            local_data=local_data,
            alias=alias,
            path=path,
            hermitian=hermitian,
            copy=copy)
        self.index = d2o_librarian.register(self)

    @property
    def real(self):
        new_data = self.get_local_data().real
        new_dtype = new_data.dtype
        new_d2o = self.copy_empty(dtype=new_dtype)
        new_d2o.set_local_data(data=new_data,
                               hermitian=self.hermitian)
        return new_d2o

    @property
    def imag(self):
        new_data = self.get_local_data().imag
        new_dtype = new_data.dtype
        new_d2o = self.copy_empty(dtype=new_dtype)
        new_d2o.set_local_data(data=new_data,
                               hermitian=self.hermitian)
        return new_d2o

    def copy(self, dtype=None, distribution_strategy=None, **kwargs):
        temp_d2o = self.copy_empty(dtype=dtype,
                                   distribution_strategy=distribution_strategy,
                                   **kwargs)
        if distribution_strategy is None or \
                distribution_strategy == self.distribution_strategy:
            temp_d2o.set_local_data(self.get_local_data(), copy=True)
        else:
            temp_d2o.inject((slice(None),), self, (slice(None),))
        temp_d2o.hermitian = self.hermitian
        return temp_d2o

    def copy_empty(self, global_shape=None, local_shape=None, dtype=None,
                   distribution_strategy=None, **kwargs):
        if self.distribution_strategy == 'not' and \
                distribution_strategy in STRATEGIES['local'] and \
                local_shape is None:
            result = self.copy_empty(global_shape=global_shape,
                                     local_shape=local_shape,
                                     dtype=dtype,
                                     distribution_strategy='equal',
                                     **kwargs)
            return result.copy_empty(distribution_strategy='freeform')

        if global_shape is None:
            global_shape = self.shape
        if local_shape is None:
            local_shape = self.local_shape
        if dtype is None:
            dtype = self.dtype
        if distribution_strategy is None:
            distribution_strategy = self.distribution_strategy

        kwargs.update(self.init_kwargs)

        temp_d2o = distributed_data_object(
                                   global_shape=global_shape,
                                   local_shape=local_shape,
                                   dtype=dtype,
                                   distribution_strategy=distribution_strategy,
                                   comm=self.comm,
                                   *self.init_args,
                                   **kwargs)
        return temp_d2o

    def apply_scalar_function(self, function, inplace=False, dtype=None):
        remember_hermitianQ = self.hermitian

        if inplace is True:
            temp = self
            if dtype is not None and self.dtype != np.dtype(dtype):
                about.warnings.cprint(
                    "WARNING: Inplace dtype conversion is not possible!")

        else:
            temp = self.copy_empty(dtype=dtype)

        if np.prod(self.local_shape) != 0:
            try:
                temp.data[:] = function(self.data)
            except:
                temp.data[:] = np.vectorize(function)(self.data)
        else:
            # Noting to do here. The value-empty array
            # is also geometrically empty
            pass

        if function in (np.exp, np.log):
            temp.hermitian = remember_hermitianQ
        else:
            temp.hermitian = False
        return temp

    def apply_generator(self, generator):
        self.set_local_data(generator(self.distributor.local_shape))
        self.hermitian = False

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return '<distributed_data_object>\n' + self.data.__repr__()

    def _compare_helper(self, other, op):
        result = self.copy_empty(dtype=np.bool_)
        # Case 1: 'other' is a scalar
        # -> make point-wise comparison
        if np.isscalar(other):
            result.set_local_data(
                getattr(self.get_local_data(copy=False), op)(other))
            return result

        # Case 2: 'other' is a numpy array or a distributed_data_object
        # -> extract the local data and make point-wise comparison
        elif isinstance(other, np.ndarray) or\
                isinstance(other, distributed_data_object):
            temp_data = self.distributor.extract_local_data(other)
            result.set_local_data(
                getattr(self.get_local_data(copy=False), op)(temp_data))
            return result

        # Case 3: 'other' is None
        elif other is None:
            return False

        # Case 4: 'other' is something different
        # -> make a numpy casting and make a recursive call
        else:
            temp_other = np.array(other)
            return getattr(self, op)(temp_other)

    def __ne__(self, other):
        return self._compare_helper(other, '__ne__')

    def __lt__(self, other):
        return self._compare_helper(other, '__lt__')

    def __le__(self, other):
        return self._compare_helper(other, '__le__')

    def __eq__(self, other):

        return self._compare_helper(other, '__eq__')

    def __ge__(self, other):
        return self._compare_helper(other, '__ge__')

    def __gt__(self, other):
        return self._compare_helper(other, '__gt__')

    def equal(self, other):
        if other is None:
            return False
        try:
            assert(self.dtype == other.dtype)
            assert(self.shape == other.shape)
            assert(self.init_args == other.init_args)
            assert(self.init_kwargs == other.init_kwargs)
            assert(self.distribution_strategy == other.distribution_strategy)
            assert(np.all(self.data == other.data))
        except(AssertionError, AttributeError):
            return False
        else:
            return True

    def __pos__(self):
        temp_d2o = self.copy_empty()
        temp_d2o.set_local_data(data=self.get_local_data(), copy=True)
        return temp_d2o

    def __neg__(self):
        temp_d2o = self.copy_empty()
        temp_d2o.set_local_data(data=self.get_local_data().__neg__(),
                                copy=True)
        return temp_d2o

    def __abs__(self):
        # translate complex dtypes
        if self.dtype == np.dtype('complex64'):
            new_dtype = np.dtype('float32')
        elif self.dtype == np.dtype('complex128'):
            new_dtype = np.dtype('float64')
        elif issubclass(self.dtype.type, np.complexfloating):
            new_dtype = np.dtype('float')
        else:
            new_dtype = self.dtype
        temp_d2o = self.copy_empty(dtype=new_dtype)
        temp_d2o.set_local_data(data=self.get_local_data().__abs__(),
                                copy=True)
        return temp_d2o

    def _builtin_helper(self, operator, other, inplace=False):
        if isinstance(other, distributed_data_object):
            other_is_real = other.isreal()
        else:
            other_is_real = np.isreal(other)

        # Case 1: other is not a scalar
        if not (np.isscalar(other) or np.shape(other) == (1,)):
            try:
                hermitian_Q = (other.hermitian and self.hermitian)
            except(AttributeError):
                hermitian_Q = False
            # extract the local data from the 'other' object
            temp_data = self.distributor.extract_local_data(other)
            temp_data = operator(temp_data)

        # Case 2: other is a real scalar -> preserve hermitianity
        elif other_is_real or (self.dtype not in (np.dtype('complex128'),
                                                  np.dtype('complex256'))):
            hermitian_Q = self.hermitian
            temp_data = operator(other)
        # Case 3: other is complex
        else:
            hermitian_Q = False
            temp_data = operator(other)
        # write the new data into a new distributed_data_object
        if inplace is True:
            temp_d2o = self
        else:
            # use common datatype for self and other
            new_dtype = np.dtype(np.find_common_type((self.dtype,),
                                                     (temp_data.dtype,)))
            temp_d2o = self.copy_empty(
                dtype=new_dtype)
        temp_d2o.set_local_data(data=temp_data)
        temp_d2o.hermitian = hermitian_Q
        return temp_d2o

    def __add__(self, other):
        return self._builtin_helper(self.get_local_data().__add__, other)

    def __radd__(self, other):
        return self._builtin_helper(self.get_local_data().__radd__, other)

    def __iadd__(self, other):
        return self._builtin_helper(self.get_local_data().__iadd__,
                                    other,
                                    inplace=True)

    def __sub__(self, other):
        return self._builtin_helper(self.get_local_data().__sub__, other)

    def __rsub__(self, other):
        return self._builtin_helper(self.get_local_data().__rsub__, other)

    def __isub__(self, other):
        return self._builtin_helper(self.get_local_data().__isub__,
                                    other,
                                    inplace=True)

    def __div__(self, other):
        return self._builtin_helper(self.get_local_data().__div__, other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __rdiv__(self, other):
        return self._builtin_helper(self.get_local_data().__rdiv__, other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __idiv__(self, other):
        return self._builtin_helper(self.get_local_data().__idiv__,
                                    other,
                                    inplace=True)

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __floordiv__(self, other):
        return self._builtin_helper(self.get_local_data().__floordiv__,
                                    other)

    def __rfloordiv__(self, other):
        return self._builtin_helper(self.get_local_data().__rfloordiv__,
                                    other)

    def __ifloordiv__(self, other):
        return self._builtin_helper(
            self.get_local_data().__ifloordiv__, other,
            inplace=True)

    def __mul__(self, other):
        return self._builtin_helper(self.get_local_data().__mul__, other)

    def __rmul__(self, other):
        return self._builtin_helper(self.get_local_data().__rmul__, other)

    def __imul__(self, other):
        return self._builtin_helper(self.get_local_data().__imul__,
                                    other,
                                    inplace=True)

    def __pow__(self, other):
        return self._builtin_helper(self.get_local_data().__pow__, other)

    def __rpow__(self, other):
        return self._builtin_helper(self.get_local_data().__rpow__, other)

    def __ipow__(self, other):
        return self._builtin_helper(self.get_local_data().__ipow__,
                                    other,
                                    inplace=True)

    def __mod__(self, other):
        return self._builtin_helper(self.get_local_data().__mod__, other)

    def __rmod__(self, other):
        return self._builtin_helper(self.get_local_data().__rmod__, other)

    def __imod__(self, other):
        return self._builtin_helper(self.get_local_data().__imod__,
                                    other,
                                    inplace=True)

    def __len__(self):
        return self.shape[0]

    def get_dim(self):
        return np.prod(self.shape)

    def vdot(self, other):
        other = self.distributor.extract_local_data(other)
        local_vdot = np.vdot(self.get_local_data(), other)
        local_vdot_list = self.distributor._allgather(local_vdot)
        global_vdot = np.sum(local_vdot_list)
        return global_vdot

    def __getitem__(self, key):
        return self.get_data(key)

    def __setitem__(self, key, data):
        self.set_data(data, key)

    def _contraction_helper(self, function, **kwargs):
        if np.prod(self.data.shape) == 0:
            local = 0
            include = False
        else:
            local = function(self.data, **kwargs)
            include = True

        local_list = self.distributor._allgather(local)
        local_list = np.array(local_list, dtype=np.dtype(local_list[0]))
        include_list = np.array(self.distributor._allgather(include))
        work_list = local_list[include_list]
        if work_list.shape[0] == 0:
            raise ValueError("ERROR: Zero-size array to reduction operation " +
                             "which has no identity")
        else:
            result = function(work_list, axis=0)
            return result

    def min(self, **kwargs):
        return self.amin(**kwargs)

    def amin(self, **kwargs):
        return self._contraction_helper(np.amin, **kwargs)

    def nanmin(self, **kwargs):
        return self._contraction_helper(np.nanmin, **kwargs)

    def max(self, **kwargs):
        return self.amax(**kwargs)

    def amax(self, **kwargs):
        return self._contraction_helper(np.amax, **kwargs)

    def nanmax(self, **kwargs):
        return self._contraction_helper(np.nanmax, **kwargs)

    def sum(self, **kwargs):
        return self._contraction_helper(np.sum, **kwargs)

    def prod(self, **kwargs):
        return self._contraction_helper(np.prod, **kwargs)

    def mean(self, power=1):
        # compute the local means and the weights for the mean-mean.
        if np.prod(self.data.shape) == 0:
            local_mean = 0
            include = False
        else:
            local_mean = np.mean(self.data**power)
            include = True

        local_weight = np.prod(self.data.shape)
        # collect the local means and cast the result to a ndarray
        local_mean_list = self.distributor._allgather(local_mean)
        local_weight_list = self.distributor._allgather(local_weight)

        local_mean_list = np.array(local_mean_list,
                                   dtype=np.dtype(local_mean_list[0]))
        local_weight_list = np.array(local_weight_list)
        # extract the parts from the non-empty nodes
        include_list = np.array(self.distributor._allgather(include))
        work_mean_list = local_mean_list[include_list]
        work_weight_list = local_weight_list[include_list]
        if work_mean_list.shape[0] == 0:
            raise ValueError("ERROR:  Mean of empty slice.")
        else:
            # compute the denominator for the weighted mean-mean
            global_weight = np.sum(work_weight_list)
            # compute the numerator
            numerator = np.sum(work_mean_list * work_weight_list)
            global_mean = numerator / global_weight
            return global_mean

    def var(self):
        mean_of_the_square = self.mean(power=2)
        square_of_the_mean = self.mean()**2
        return mean_of_the_square - square_of_the_mean

    def std(self):
        return np.sqrt(self.var())

    def argmin(self):
        if np.prod(self.data.shape) == 0:
            local_argmin = np.nan
            local_argmin_value = np.nan
            globalized_local_argmin = np.nan
        else:
            local_argmin = np.argmin(self.data)
            local_argmin_value = self.data[np.unravel_index(local_argmin,
                                                            self.data.shape)]

            globalized_local_argmin = self.distributor.globalize_flat_index(
                local_argmin)
        local_argmin_list = self.distributor._allgather(
                                                    (local_argmin_value,
                                                     globalized_local_argmin))
        local_argmin_list = np.array(local_argmin_list, dtype=[
            ('value', np.dtype('complex128')),
            ('index', np.dtype('float'))])
        local_argmin_list = np.sort(local_argmin_list,
                                    order=['value', 'index'])
        return np.int(local_argmin_list[0][1])

    def argmax(self):
        if np.prod(self.data.shape) == 0:
            local_argmax = np.nan
            local_argmax_value = np.nan
            globalized_local_argmax = np.nan
        else:
            local_argmax = np.argmax(self.data)
            local_argmax_value = -self.data[np.unravel_index(local_argmax,
                                                             self.data.shape)]
            globalized_local_argmax = self.distributor.globalize_flat_index(
                local_argmax)
        local_argmax_list = self.distributor._allgather(
                                                  (local_argmax_value,
                                                   globalized_local_argmax))
        local_argmax_list = np.array(local_argmax_list, dtype=[
            ('value', np.dtype('complex128')),
            ('index', np.dtype('float'))])
        local_argmax_list = np.sort(local_argmax_list,
                                    order=['value', 'index'])
        return np.int(local_argmax_list[0][1])

    def argmin_nonflat(self):
        return np.unravel_index(self.argmin(), self.shape)

    def argmax_nonflat(self):
        return np.unravel_index(self.argmax(), self.shape)

    def conjugate(self):
        temp_d2o = self.copy_empty()
        temp_data = np.conj(self.get_local_data())
        temp_d2o.set_local_data(temp_data)
        return temp_d2o

    def conj(self):
        return self.conjugate()

    def median(self):
        about.warnings.cprint(
            "WARNING: The current implementation of median is very expensive!")
        median = np.median(self.get_full_data())
        return median

    def _is_helper(self, function):
        temp_d2o = self.copy_empty(dtype=np.dtype('bool'))
        temp_d2o.set_local_data(function(self.data))
        return temp_d2o

    def iscomplex(self):
        return self._is_helper(np.iscomplex)

    def isreal(self):
        return self._is_helper(np.isreal)

    def isnan(self):
        return self._is_helper(np.isnan)

    def isinf(self):
        return self._is_helper(np.isinf)

    def isfinite(self):
        return self._is_helper(np.isfinite)

    def nan_to_num(self):
        temp_d2o = self.copy_empty()
        temp_d2o.set_local_data(np.nan_to_num(self.data))
        return temp_d2o

    def all(self):
        local_all = np.all(self.get_local_data())
        global_all = self.distributor._allgather(local_all)
        return np.all(global_all)

    def any(self):
        local_any = np.any(self.get_local_data())
        global_any = self.distributor._allgather(local_any)
        return np.any(global_any)

    def unique(self):
        local_unique = np.unique(self.get_local_data())
        global_unique = self.distributor._allgather(local_unique)
        global_unique = np.concatenate(global_unique)
        return np.unique(global_unique)

    def bincount(self, weights=None, minlength=None):
        if self.dtype not in [np.dtype('int16'), np.dtype('int32'),
                              np.dtype('int64'),  np.dtype('uint16'),
                              np.dtype('uint32'), np.dtype('uint64')]:
            raise TypeError(about._errors.cstring(
                "ERROR: Distributed-data-object must be of integer datatype!"))

        minlength = max(self.amax() + 1, minlength)

        if weights is not None:
            local_weights = self.distributor.extract_local_data(weights).\
                flatten()
        else:
            local_weights = None

        local_counts = np.bincount(self.get_local_data().flatten(),
                                   weights=local_weights,
                                   minlength=minlength)
        if self.distribution_strategy == 'not':
            return local_counts
        else:
            list_of_counts = self.distributor._allgather(local_counts)
            counts = np.sum(list_of_counts, axis=0)
            return counts

    def where(self):
        return self.distributor.where(self.data)

    def set_local_data(self, data, hermitian=False, copy=True):
        """
            Stores data directly in the local data attribute. No distribution
            is done. The shape of the data must fit the local data attributes
            shape.

            Parameters
            ----------
            data : tuple, list, numpy.ndarray
                The data which should be stored in the local data attribute.

            Returns
            -------
            None

        """
        self.hermitian = hermitian
        if copy is True:
            self.data[:] = data
        else:
            self.data = np.array(data,
                                 dtype=self.dtype,
                                 copy=False,
                                 order='C').reshape(self.local_shape)

    def set_data(self, data, to_key, from_key=None, local_keys=False,
                 hermitian=False, copy=True, **kwargs):
        """
            Stores the supplied data in the region which is specified by key.
            The data is distributed according to the distribution strategy. If
            the individual nodes get different key-arguments. Their data is
            processed one-by-one.

            Parameters
            ----------
            data : tuple, list, numpy.ndarray
                The data which should be distributed.
            key : int, slice, tuple of int or slice
                The key is the object which specifies the region, where data
                will be stored in.

            Returns
            -------
            None

        """
        self.hermitian = hermitian
        self.distributor.disperse_data(data=self.data,
                                       to_key=to_key,
                                       data_update=data,
                                       from_key=from_key,
                                       local_keys=local_keys,
                                       copy=copy,
                                       **kwargs)

    def set_full_data(self, data, hermitian=False, copy=True, **kwargs):
        """
            Distributes the supplied data to the nodes. The shape of data must
            match the shape of the distributed_data_object.

            Parameters
            ----------
            data : tuple, list, numpy.ndarray
                The data which should be distributed.

            Notes
            -----
            set_full_data(foo) is equivalent to set_data(foo,slice(None)) but
            faster.

            Returns
            -------
            None

        """
        self.hermitian = hermitian
        self.data = self.distributor.distribute_data(data=data, copy=copy,
                                                     **kwargs)

    def get_local_data(self, key=(slice(None),), copy=True):
        """
            Loads data directly from the local data attribute. No consolidation
            is done.

            Parameters
            ----------
            key : int, slice, tuple of int or slice
                The key which will be used to access the data.

            Returns
            -------
            self.data[key] : numpy.ndarray

        """
        if copy is True:
            return self.data[key]
        if copy is False:
            return self.data

    def get_data(self, key, local_keys=False, **kwargs):
        """
            Loads data from the region which is specified by key. The data is
            consolidated according to the distribution strategy. If the
            individual nodes get different key-arguments, they get individual
            data.

            Parameters
            ----------

            key : int, slice, tuple of int or slice
                The key is the object which specifies the region, where data
                will be loaded from.

            Returns
            -------
            global_data[key] : numpy.ndarray

        """
        if key is None:
            return self.copy()
        elif isinstance(key, slice):
            if key == slice(None):
                return self.copy()
        elif isinstance(key, tuple):
            try:
                if all(x == slice(None) for x in key):
                    return self.copy()
            except(ValueError):
                pass

        return self.distributor.collect_data(self.data,
                                             key,
                                             local_keys=local_keys,
                                             **kwargs)

    def get_full_data(self, target_rank='all'):
        """
            Fully consolidates the distributed data.

            Parameters
            ----------
            target_rank : 'all' (default), int *optional*
                If only one node should recieve the full data, it can be
                specified here.

            Notes
            -----
            get_full_data() is equivalent to get_data(slice(None)) but
            faster.

            Returns
            -------
            None
        """

        return self.distributor.consolidate_data(self.data,
                                                 target_rank=target_rank)

    def inject(self, to_key=(slice(None),), data=None,
               from_key=(slice(None),)):
        if data is None:
            return self
        self.distributor.inject(self.data, to_key, data, from_key)

    def flatten(self, inplace=False):
        flat_data = self.distributor.flatten(self.data, inplace=inplace)

        flat_global_shape = (np.prod(self.shape),)
        flat_local_shape = np.shape(flat_data)

        # Try to keep the distribution strategy. Therefore
        # create an empty copy of self which has the new shape
        temp_d2o = self.copy_empty(global_shape=flat_global_shape,
                                   local_shape=flat_local_shape)
        # Check if the local shapes match.
        if temp_d2o.local_shape == flat_local_shape:
            work_d2o = temp_d2o
        # if the shapes do not match, create a freeform d2o
        else:
            work_d2o = self.copy_empty(local_shape=flat_local_shape,
                                       distribution_strategy='freeform')

        # Feed the work_d2o with the flat data
        work_d2o.set_local_data(data=flat_data,
                                copy=False)

        if inplace is True:
            self = work_d2o
            return self
        else:
            return work_d2o

    def save(self, alias, path=None, overwriteQ=True):
        """
            Saves a distributed_data_object to disk utilizing h5py.

            Parameters
            ----------
            alias : string
                The name for the dataset which is saved within the hdf5 file.

            path : string *optional*
                The path to the hdf5 file. If no path is given, the alias is
                taken as filename in the current path.

            overwriteQ : Boolean *optional*
                Specifies whether a dataset may be overwritten if it is already
                present in the given hdf5 file or not.
        """
        self.distributor.save_data(self.data, alias, path, overwriteQ)

    def load(self, alias, path=None):
        """
            Loads a distributed_data_object from disk utilizing h5py.

            Parameters
            ----------
            alias : string
                The name of the dataset which is loaded from the hdf5 file.

            path : string *optional*
                The path to the hdf5 file. If no path is given, the alias is
                taken as filename in the current path.
        """
        self.data = self.distributor.load_data(alias, path)


class _distributor_factory(object):

    def __init__(self):
        self.distributor_store = {}

    def parse_kwargs(self, distribution_strategy, comm,
                     global_data=None, global_shape=None,
                     local_data=None, local_shape=None,
                     alias=None, path=None,
                     dtype=None, **kwargs):

        return_dict = {}

        # Check that all nodes got the same distribution_strategy
        strat_list = comm.allgather(distribution_strategy)
        if all(x == strat_list[0] for x in strat_list) == False:
            raise ValueError(about._errors.cstring(
                "ERROR: The distribution-strategy must be the same on " +
                "all nodes!"))

        # Check for an hdf5 file and open it if given
        if 'h5py' in gdi and alias is not None:
            # set file path
            file_path = path if (path is not None) else alias
            # open hdf5 file
            if h5py.get_config().mpi and gc['mpi_module'] == 'MPI':
                f = h5py.File(file_path, 'r', driver='mpio', comm=comm)
            else:
                f = h5py.File(file_path, 'r')
            # open alias in file
            dset = f[alias]
        else:
            dset = None

        # Parse the MPI communicator
        if comm is None:
            raise ValueError(about._errors.cstring(
                "ERROR: The distributor needs MPI-communicator object comm!"))
        else:
            return_dict['comm'] = comm

        # Parse the datatype
        if distribution_strategy in ['not', 'equal', 'fftw'] and \
                (dset is not None):
            dtype = dset.dtype

        elif distribution_strategy in ['not', 'equal', 'fftw']:
            if dtype is None:
                if global_data is None:
                    dtype = np.dtype('float64')
                    about.infos.cprint('INFO: dtype set was set to default.')
                else:
                    try:
                        dtype = global_data.dtype
                    except(AttributeError):
                        dtype = np.array(global_data).dtype
            else:
                dtype = np.dtype(dtype)

        elif distribution_strategy in ['freeform']:
            if dtype is None:
                if isinstance(global_data, distributed_data_object):
                    dtype = global_data.dtype
                elif local_data is not None:
                    try:
                        dtype = local_data.dtype
                    except(AttributeError):
                        dtype = np.array(local_data).dtype
                else:
                    dtype = np.dtype('float64')
                    about.infos.cprint('INFO: dtype set was set to default.')

            else:
                dtype = np.dtype(dtype)
        dtype_list = comm.allgather(dtype)
        if all(x == dtype_list[0] for x in dtype_list) == False:
            raise ValueError(about._errors.cstring(
                "ERROR: The given dtype must be the same on all nodes!"))
        return_dict['dtype'] = dtype

        # Parse the shape
        # Case 1: global-type slicer
        if distribution_strategy in ['not', 'equal', 'fftw']:
            if dset is not None:
                global_shape = dset.shape
            elif global_data is not None and np.isscalar(global_data) == False:
                global_shape = global_data.shape
            elif global_shape is not None:
                global_shape = tuple(global_shape)
            else:
                raise ValueError(about._errors.cstring(
                    "ERROR: Neither non-0-dimensional global_data nor " +
                    "global_shape nor hdf5 file supplied!"))
            if global_shape == ():
                raise ValueError(about._errors.cstring(
                    "ERROR: global_shape == () is not a valid shape!"))

            global_shape_list = comm.allgather(global_shape)
            if not all(x == global_shape_list[0] for x in global_shape_list):
                raise ValueError(about._errors.cstring(
                    "ERROR: The global_shape must be the same on all nodes!"))
            return_dict['global_shape'] = global_shape

        # Case 2: local-type slicer
        elif distribution_strategy in ['freeform']:
            if isinstance(global_data, distributed_data_object):
                local_shape = global_data.local_shape
            elif local_data is not None and np.isscalar(local_data) == False:
                local_shape = local_data.shape
            elif local_shape is not None:
                local_shape = tuple(local_shape)
            else:
                raise ValueError(about._errors.cstring(
                    "ERROR: Neither non-0-dimensional local_data nor " +
                    "local_shape nor global d2o supplied!"))
            if local_shape == ():
                raise ValueError(about._errors.cstring(
                    "ERROR: local_shape == () is not a valid shape!"))

            local_shape_list = comm.allgather(local_shape[1:])
            cleared_set = set(local_shape_list)
            cleared_set.discard(())
            if len(cleared_set) > 1:
                # if not any(x == () for x in map(np.shape, local_shape_list)):
                # if not all(x == local_shape_list[0] for x in
                # local_shape_list):
                raise ValueError(about._errors.cstring(
                    "ERROR: All but the first entry of local_shape must be " +
                    "the same on all nodes!"))
            return_dict['local_shape'] = local_shape

        # Add the name of the distributor if needed
        if distribution_strategy in ['equal', 'fftw', 'freeform']:
            return_dict['name'] = distribution_strategy

        # close the file-handle
        if dset is not None:
            f.close()

        return return_dict

    def hash_arguments(self, distribution_strategy, **kwargs):
        kwargs = kwargs.copy()

        comm = kwargs['comm']
        kwargs['comm'] = id(comm)

        if 'global_shape' in kwargs:
            kwargs['global_shape'] = kwargs['global_shape']
        if 'local_shape' in kwargs:
            local_shape = kwargs['local_shape']
            local_shape_list = comm.allgather(local_shape)
            kwargs['local_shape'] = tuple(local_shape_list)

        kwargs['dtype'] = self.dictionize_np(kwargs['dtype'])
        kwargs['distribution_strategy'] = distribution_strategy

        return frozenset(kwargs.items())

    def dictionize_np(self, x):
        dic = x.type.__dict__.items()
        if x is np.float:
            dic[24] = 0
            dic[29] = 0
            dic[37] = 0
        return frozenset(dic)

    def get_distributor(self, distribution_strategy, comm, **kwargs):
        # check if the distribution strategy is known
        if distribution_strategy not in STRATEGIES['all']:
            raise TypeError(about._errors.cstring(
                "ERROR: Unknown distribution strategy supplied."))

        # parse the kwargs
        parsed_kwargs = self.parse_kwargs(
            distribution_strategy=distribution_strategy,
            comm=comm,
            **kwargs)

        hashed_kwargs = self.hash_arguments(distribution_strategy,
                                            **parsed_kwargs)
        # check if the distributors has already been produced in the past
        if hashed_kwargs in self.distributor_store:
            return self.distributor_store[hashed_kwargs]
        else:
            # produce new distributor
            if distribution_strategy == 'not':
                produced_distributor = _not_distributor(**parsed_kwargs)

            elif distribution_strategy == 'equal':
                produced_distributor = _slicing_distributor(
                    slicer=_equal_slicer,
                    **parsed_kwargs)

            elif distribution_strategy == 'fftw':
                produced_distributor = _slicing_distributor(
                    slicer=_fftw_slicer,
                    **parsed_kwargs)
            elif distribution_strategy == 'freeform':
                produced_distributor = _slicing_distributor(
                    slicer=_freeform_slicer,
                    **parsed_kwargs)

            self.distributor_store[hashed_kwargs] = produced_distributor
            return self.distributor_store[hashed_kwargs]


distributor_factory = _distributor_factory()


def _infer_key_type(key):
    if key is None:
        return (None, None)
    found_boolean = False
    # Check which case we got:
    if isinstance(key, tuple) or isinstance(key, slice) or np.isscalar(key):
        # Check if there is something different in the array than
        # scalars and slices
        if isinstance(key, slice) or np.isscalar(key):
            key = [key]

        scalarQ = np.array(map(np.isscalar, key))
        sliceQ = np.array(map(lambda z: isinstance(z, slice), key))
        if np.all(scalarQ + sliceQ):
            found = 'slicetuple'
        else:
            found = 'indexinglist'
    elif isinstance(key, np.ndarray):
        found = 'ndarray'
        found_boolean = (key.dtype.type == np.bool_)
    elif isinstance(key, distributed_data_object):
        found = 'd2o'
        found_boolean = (key.dtype == np.bool_)
    elif isinstance(key, list):
        found = 'indexinglist'
    else:
        raise ValueError(about._errors.cstring("ERROR: Unknown keytype!"))
    return (found, found_boolean)


class distributor(object):

    def inject(self, data, to_slices, data_update, from_slices,
               **kwargs):
        # check if to_key and from_key are completely built of slices
        if not np.all(
                np.vectorize(lambda x: isinstance(x, slice))(to_slices)):
            raise ValueError(about._errors.cstring(
                "ERROR: The to_slices argument must be a list or " +
                "tuple of slices!")
            )

        if not np.all(
                np.vectorize(lambda x: isinstance(x, slice))(from_slices)):
            raise ValueError(about._errors.cstring(
                "ERROR: The from_slices argument must be a list or " +
                "tuple of slices!")
            )

        to_slices = tuple(to_slices)
        from_slices = tuple(from_slices)
        self.disperse_data(data=data,
                           to_key=to_slices,
                           data_update=data_update,
                           from_key=from_slices,
                           **kwargs)

    def disperse_data(self, data, to_key, data_update, from_key=None,
                      local_keys=False, copy=True, **kwargs):
        # Check which keys we got:
        (to_found, to_found_boolean) = _infer_key_type(to_key)
        (from_found, from_found_boolean) = _infer_key_type(from_key)

        comm = self.comm
        if local_keys is False:
            return self._disperse_data_primitive(
                                         data=data,
                                         to_key=to_key,
                                         data_update=data_update,
                                         from_key=from_key,
                                         copy=copy,
                                         to_found=to_found,
                                         to_found_boolean=to_found_boolean,
                                         from_found=from_found,
                                         from_found_boolean=from_found_boolean,
                                         **kwargs)

        else:
            # assert that all to_keys are from same type
            to_found_list = comm.allgather(to_found)
            assert(all(x == to_found_list[0] for x in to_found_list))
            to_found_boolean_list = comm.allgather(to_found_boolean)
            assert(all(x == to_found_boolean_list[0] for x in
                       to_found_boolean_list))
            from_found_list = comm.allgather(from_found)
            assert(all(x == from_found_list[0] for x in from_found_list))
            from_found_boolean_list = comm.allgather(from_found_boolean)
            assert(all(x == from_found_boolean_list[0] for
                       x in from_found_boolean_list))

            # gather the local to_keys into a global to_key_list
            # Case 1: the to_keys are not distributed_data_objects
            # -> allgather does the job
            if to_found != 'd2o':
                to_key_list = comm.allgather(to_key)
            # Case 2: if the to_keys are distributed_data_objects, gather
            # the index of the array and build the to_key_list with help
            # from the librarian
            else:
                to_index_list = comm.allgather(to_key.index)
                to_key_list = map(lambda z: d2o_librarian[z], to_index_list)

            # gather the local from_keys. It is the same procedure as above
            if from_found != 'd2o':
                from_key_list = comm.allgather(to_key)
            else:
                from_index_list = comm.allgather(from_key.index)
                from_key_list = map(lambda z: d2o_librarian[z],
                                    from_index_list)

            local_data_update_is_scalar = np.isscalar(data_update)
            local_scalar_list = comm.allgather(local_data_update_is_scalar)
            for i in xrange(len(to_key_list)):
                if np.all(np.array(local_scalar_list) == True):
                    scalar_list = comm.allgather(data_update)
                    temp_data_update = scalar_list[i]
                elif isinstance(data_update, distributed_data_object):
                    data_update_index_list = comm.allgather(data_update.index)
                    data_update_list = map(lambda z: d2o_librarian[z],
                                           data_update_index_list)
                    temp_data_update = data_update_list[i]
                else:
                    # build a temporary freeform d2o which only contains data
                    # from node i
                    if comm.rank == i:
                        temp_shape = np.shape(data_update)
                        try:
                            temp_dtype = np.dtype(data_update).type
                        except(TypeError):
                            temp_dtype = np.array(data_update).dtype.type
                    else:
                        temp_shape = None
                        temp_dtype = None
                    temp_shape = comm.bcast(temp_shape, root=i)
                    temp_dtype = comm.bcast(temp_dtype, root=i)

                    if comm.rank != i:
                        temp_shape = list(temp_shape)
                        temp_shape[0] = 0
                        temp_shape = tuple(temp_shape)
                        temp_data = np.empty(temp_shape, dtype=temp_dtype)
                    else:
                        temp_data = data_update
                    temp_data_update = distributed_data_object(
                        local_data=temp_data,
                        distribution_strategy='freeform',
                        comm=self.comm)
                # disperse the data one after another
                self._disperse_data_primitive(
                                      data=data,
                                      to_key=to_key_list[i],
                                      data_update=temp_data_update,
                                      from_key=from_key_list[i],
                                      copy=copy,
                                      to_found=to_found,
                                      to_found_boolean=to_found_boolean,
                                      from_found=from_found,
                                      from_found_boolean=from_found_boolean,
                                      **kwargs)
                i += 1


class _slicing_distributor(distributor):

    def __init__(self, slicer, name, dtype, comm, **remaining_parsed_kwargs):

        self.comm = comm
        self.distribution_strategy = name
        self.dtype = np.dtype(dtype)

        self._my_dtype_converter = global_dtype_converter

        if not self._my_dtype_converter.known_np_Q(self.dtype):
            raise TypeError(about._errors.cstring(
                "ERROR: The datatype " + str(self.dtype.__repr__()) +
                " is not known to mpi4py."))

        self.mpi_dtype = self._my_dtype_converter.to_mpi(self.dtype)

        self.slicer = slicer
        self._local_size = self.slicer(comm=comm, **remaining_parsed_kwargs)
        self.local_start = self._local_size[0]
        self.local_end = self._local_size[1]
        self.global_shape = self._local_size[2]

        self.local_length = self.local_end - self.local_start
        self.local_shape = (self.local_length,) + tuple(self.global_shape[1:])
        self.local_dim = np.product(self.local_shape)
        self.local_dim_list = np.empty(comm.size, dtype=np.int)
        comm.Allgather([np.array(self.local_dim, dtype=np.int), MPI.INT],
                       [self.local_dim_list, MPI.INT])
        self.local_dim_offset = np.sum(self.local_dim_list[0:comm.rank])

        self.local_slice = np.array([self.local_start, self.local_end,
                                     self.local_length, self.local_dim,
                                     self.local_dim_offset],
                                    dtype=np.int)
        # collect all local_slices
        self.all_local_slices = np.empty((comm.size, 5), dtype=np.int)
        comm.Allgather([np.array((self.local_slice,), dtype=np.int), MPI.INT],
                       [self.all_local_slices, MPI.INT])

    def initialize_data(self, global_data, local_data, alias, path, hermitian,
                        copy, **kwargs):
        if 'h5py' in gdi and alias is not None:
            local_data = self.load_data(alias=alias, path=path)
            return (local_data, hermitian)

        if self.distribution_strategy in ['equal', 'fftw']:
            if np.isscalar(global_data):
                local_data = np.empty(self.local_shape, dtype=self.dtype)
                local_data.fill(global_data)
                hermitian = True
            else:
                local_data = self.distribute_data(data=global_data,
                                                  copy=copy)
        elif self.distribution_strategy in ['freeform']:
            if isinstance(global_data, distributed_data_object):
                local_data = global_data.get_local_data()
            elif np.isscalar(local_data):
                temp_local_data = np.empty(self.local_shape,
                                           dtype=self.dtype)
                temp_local_data.fill(local_data)
                local_data = temp_local_data
                hermitian = True
            elif local_data is None:
                local_data = np.empty(self.local_shape, dtype=self.dtype)
            else:
                local_data = np.array(local_data).astype(
                    self.dtype, copy=copy).reshape(self.local_shape)
        else:
            raise TypeError(about._errors.cstring(
                "ERROR: Unknown istribution strategy"))
        return (local_data, hermitian)

    def globalize_flat_index(self, index):
        return int(index) + self.local_dim_offset

    def globalize_index(self, index):
        index = np.array(index, dtype=np.int).flatten()
        if index.shape != (len(self.global_shape),):
            raise TypeError(about._errors.cstring("ERROR: Length\
                of index tuple does not match the array's shape!"))
        globalized_index = index
        globalized_index[0] = index[0] + self.local_start
        # ensure that the globalized index list is within the bounds
        global_index_memory = globalized_index
        globalized_index = np.clip(globalized_index,
                                   -np.array(self.global_shape),
                                   np.array(self.global_shape) - 1)
        if np.any(global_index_memory != globalized_index):
            about.warnings.cprint("WARNING: Indices were clipped!")
        globalized_index = tuple(globalized_index)
        return globalized_index

    def _allgather(self, thing, comm=None):
        if comm is None:
            comm = self.comm
        gathered_things = comm.allgather(thing)
        return gathered_things

    def distribute_data(self, data=None, alias=None,
                        path=None, copy=True, **kwargs):
        '''
        distribute data checks
        - whether the data is located on all nodes or only on node 0
        - that the shape of 'data' matches the global_shape
        '''

        comm = self.comm

        if 'h5py' in gdi and alias is not None:
            data = self.load_data(alias=alias, path=path)

        local_data_available_Q = (data is not None)
        data_available_Q = np.array(comm.allgather(local_data_available_Q))

        if np.all(data_available_Q == False):
            return np.empty(self.local_shape, dtype=self.dtype, order='C')
        # if all nodes got data, we assume that it is the right data and
        # store it individually.
        elif np.all(data_available_Q == True):
            if isinstance(data, distributed_data_object):
                temp_d2o = data.get_data((slice(self.local_start,
                                                self.local_end),),
                                         local_keys=True)
                return temp_d2o.get_local_data().astype(self.dtype,
                                                        copy=copy)
            else:
                return data[self.local_start:self.local_end].astype(
                    self.dtype,
                    copy=copy)
        else:
            raise ValueError(
                "ERROR: distribute_data must get data on all nodes!")

    def _disperse_data_primitive(self, data, to_key, data_update, from_key,
                                 copy, to_found, to_found_boolean, from_found,
                                 from_found_boolean, **kwargs):

        if np.isscalar(data_update):
            from_key = None

        # Case 1: to_key is a slice-tuple. Hence, the basic indexing/slicing
        # machinery will be used
        if to_found == 'slicetuple':
            if from_found == 'slicetuple':
                return self.disperse_data_to_slices(data=data,
                                                    to_slices=to_key,
                                                    data_update=data_update,
                                                    from_slices=from_key,
                                                    copy=copy,
                                                    **kwargs)
            else:
                if from_key is not None:
                    about.infos.cprint(
                        "INFO: Advanced injection is not available for this " +
                        "combination of to_key and from_key.")
                    prepared_data_update = data_update[from_key]
                else:
                    prepared_data_update = data_update
                return self.disperse_data_to_slices(
                                            data=data,
                                            to_slices=to_key,
                                            data_update=prepared_data_update,
                                            copy=copy,
                                            **kwargs)

        # Case 2: key is an array
        elif (to_found == 'ndarray' or to_found == 'd2o'):
            # Case 2.1: The array is boolean.
            if to_found_boolean:
                if from_key is not None:
                    about.infos.cprint(
                        "INFO: Advanced injection is not available for this " +
                        "combination of to_key and from_key.")
                    prepared_data_update = data_update[from_key]
                else:
                    prepared_data_update = data_update
                return self.disperse_data_to_bool(
                                              data=data,
                                              to_boolean_key=to_key,
                                              data_update=prepared_data_update,
                                              copy=copy,
                                              **kwargs)
            # Case 2.2: The array is not boolean. Only 1-dimensional
            # advanced slicing is supported.
            else:
                if len(to_key.shape) != 1:
                    raise ValueError(about._errors.cstring(
                        "WARNING: Only one-dimensional advanced indexing " +
                        "is supported"))
                # Make a recursive call in order to trigger the 'list'-section
                return self.disperse_data(data=data, to_key=[to_key],
                                          data_update=data_update,
                                          from_key=from_key, copy=copy,
                                          **kwargs)

        # Case 3 : to_key is a list. This list is interpreted as
        # one-dimensional advanced indexing list.
        elif to_found == 'indexinglist':
            if from_key is not None:
                about.infos.cprint(
                    "INFO: Advanced injection is not available for this " +
                    "combination of to_key and from_key.")
                prepared_data_update = data_update[from_key]
            else:
                prepared_data_update = data_update
            return self.disperse_data_to_list(data=data,
                                              to_list_key=to_key,
                                              data_update=prepared_data_update,
                                              copy=copy,
                                              **kwargs)

    def disperse_data_to_list(self, data, to_list_key, data_update,
                              copy=True, **kwargs):

        if to_list_key == []:
            return data

        local_to_list_key = self._advanced_index_decycler(to_list_key)
        return self._disperse_data_to_list_and_bool_helper(
            data=data,
            local_to_key=local_to_list_key,
            data_update=data_update,
            copy=copy,
            **kwargs)

    def disperse_data_to_bool(self, data, to_boolean_key, data_update,
                              copy=True, **kwargs):
        # Extract the part of the to_boolean_key which corresponds to the
        # local data
        local_to_boolean_key = self.extract_local_data(to_boolean_key)
        return self._disperse_data_to_list_and_bool_helper(
            data=data,
            local_to_key=local_to_boolean_key,
            data_update=data_update,
            copy=copy,
            **kwargs)

    def _disperse_data_to_list_and_bool_helper(self, data, local_to_key,
                                               data_update, copy, **kwargs):
        comm = self.comm
        rank = comm.rank
        # Infer the length and offset of the locally affected data
        locally_affected_data = data[local_to_key]
        data_length = np.shape(locally_affected_data)[0]
        data_length_list = comm.allgather(data_length)
        data_length_offset_list = np.append([0],
                                            np.cumsum(data_length_list)[:-1])

        # Update the local data object with its very own portion
        o = data_length_offset_list
        l = data_length

        if isinstance(data_update, distributed_data_object):
            data[local_to_key] = data_update.get_data(
                                          slice(o[rank], o[rank] + l),
                                          local_keys=True
                                          ).get_local_data().astype(self.dtype)
        elif np.isscalar(data_update):
            data[local_to_key] = data_update
        else:
            data[local_to_key] = np.array(data_update[o[rank]:o[rank] + l],
                                          copy=copy).astype(self.dtype)
        return data

    def disperse_data_to_slices(self, data, to_slices,
                                data_update, from_slices=None, copy=True):

        comm = self.comm
        (to_slices, sliceified) = self._sliceify(to_slices)

        # parse the to_slices object
        localized_to_start, localized_to_stop = self._backshift_and_decycle(
            to_slices[0], self.local_start, self.local_end,
            self.global_shape[0])
        local_to_slice = (slice(localized_to_start, localized_to_stop,
                                to_slices[0].step),) + to_slices[1:]
        local_to_slice_shape = data[local_to_slice].shape

        to_step = to_slices[0].step
        if to_step is None:
            to_step = 1
        elif to_step == 0:
            raise ValueError(about._errors.cstring(
                "ERROR: to_step size == 0!"))

        # Compute the offset of the data the individual node will take.
        # The offset is free of stepsizes. It is the offset in terms of
        # the purely transported data. If to_step < 0, the offset will
        # be calculated in reverse order
        order = np.sign(to_step)

        local_affected_data_length = local_to_slice_shape[0]
        local_affected_data_length_list = np.empty(comm.size, dtype=np.int)
        comm.Allgather(
            [np.array(local_affected_data_length, dtype=np.int), MPI.INT],
            [local_affected_data_length_list, MPI.INT])
        local_affected_data_length_offset_list = np.append([0],
                                                           np.cumsum(
            local_affected_data_length_list[::order])[:-1])[::order]

        if np.isscalar(data_update):
            data[local_to_slice] = data_update
        else:
            # construct the locally adapted from_slice object
            r = comm.rank
            o = local_affected_data_length_offset_list
            l = local_affected_data_length

            data_update = self._enfold(data_update, sliceified)

            # parse the from_slices object
            if from_slices is None:
                from_slices = (slice(None, None, None),)
            (from_slices_start, from_slices_stop) = \
                self._backshift_and_decycle(
                                            slice_object=from_slices[0],
                                            shifted_start=0,
                                            shifted_stop=data_update.shape[0],
                                            global_length=data_update.shape[0])
            if from_slices_start is None:
                raise ValueError(about._errors.cstring(
                    "ERROR: _backshift_and_decycle should never return " +
                    "None for local_start!"))

            # parse the step sizes
            from_step = from_slices[0].step
            if from_step is None:
                from_step = 1
            elif from_step == 0:
                raise ValueError(about._errors.cstring(
                    "ERROR: from_step size == 0!"))

            localized_from_start = from_slices_start + from_step * o[r]
            localized_from_stop = localized_from_start + from_step * l
            if localized_from_stop < 0:
                localized_from_stop = None

            localized_from_slice = (slice(localized_from_start,
                                          localized_from_stop,
                                          from_step),)

            update_slice = localized_from_slice + from_slices[1:]

            if isinstance(data_update, distributed_data_object):
                local_data_update = data_update.get_data(
                                 key=update_slice,
                                 local_keys=True
                                 ).get_local_data(copy=copy).astype(self.dtype)
                if np.prod(np.shape(local_data_update)) != 0:
                    data[local_to_slice] = local_data_update
            # elif np.isscalar(data_update):
            #    data[local_to_slice] = data_update
            else:
                local_data_update = np.array(data_update)[update_slice]
                if np.prod(np.shape(local_data_update)) != 0:
                    data[local_to_slice] = np.array(
                                                local_data_update,
                                                copy=copy).astype(self.dtype)

    def collect_data(self, data, key, local_keys=False, **kwargs):
        # collect_data supports three types of keys
        # Case 1: key is a slicing/index tuple
        # Case 2: key is a boolean-array of the same shape as self
        # Case 3: key is a list of shape (n,), where n is
        #         0<n<len(self.shape). The entries of the list must be a
        #         scalar/list/tuple/ndarray. If not scalar the length must be
        #         the same for all of the lists. This is essentially
        #         numpy advanced indexing in one dimension, only.

        # Check which case we got:
        (found, found_boolean) = _infer_key_type(key)

        comm = self.comm

        if local_keys is False:
            return self._collect_data_primitive(data, key, found,
                                                found_boolean, **kwargs)
        else:
            # assert that all keys are from same type
            found_list = comm.allgather(found)
            assert(all(x == found_list[0] for x in found_list))
            found_boolean_list = comm.allgather(found_boolean)
            assert(all(x == found_boolean_list[0] for x in found_boolean_list))

            # gather the local_keys into a global key_list
            # Case 1: the keys are no distributed_data_objects
            # -> allgather does the job
            if found != 'd2o':
                key_list = comm.allgather(key)
            # Case 2: if the keys are distributed_data_objects, gather
            # the index of the array and build the key_list with help
            # from the librarian
            else:
                index_list = comm.allgather(key.index)
                key_list = map(lambda z: d2o_librarian[z], index_list)

            i = 0
            for temp_key in key_list:
                # build the locally fed d2o
                temp_d2o = self._collect_data_primitive(data, temp_key, found,
                                                        found_boolean,
                                                        **kwargs)
                # collect the data stored in the d2o to the individual target
                # rank
                temp_data = temp_d2o.get_full_data(target_rank=i)
                if comm.rank == i:
                    individual_data = temp_data
                i += 1
            return_d2o = distributed_data_object(
                local_data=individual_data,
                distribution_strategy='freeform',
                comm=self.comm)
            return return_d2o

    def _collect_data_primitive(self, data, key, found, found_boolean,
                                **kwargs):

        # Case 1: key is a slice-tuple. Hence, the basic indexing/slicing
        # machinery will be used
        if found == 'slicetuple':
            return self.collect_data_from_slices(data=data,
                                                 slice_objects=key,
                                                 **kwargs)
        # Case 2: key is an array
        elif (found == 'ndarray' or found == 'd2o'):
            # Case 2.1: The array is boolean.
            if found_boolean:
                return self.collect_data_from_bool(data=data,
                                                   boolean_key=key,
                                                   **kwargs)
            # Case 2.2: The array is not boolean. Only 1-dimensional
            # advanced slicing is supported.
            else:
                if len(key.shape) != 1:
                    raise ValueError(about._errors.cstring(
                        "WARNING: Only one-dimensional advanced indexing " +
                        "is supported"))
                # Make a recursive call in order to trigger the 'list'-section
                return self.collect_data(data=data, key=[key], **kwargs)

        # Case 3 : key is a list. This list is interpreted as one-dimensional
        # advanced indexing list.
        elif found == 'indexinglist':
            return self.collect_data_from_list(data=data,
                                               list_key=key,
                                               **kwargs)

    def collect_data_from_list(self, data, list_key, **kwargs):
        if list_key == []:
            raise ValueError(about._errors.cstring(
                "ERROR: key == [] is an unsupported key!"))

        local_list_key = self._advanced_index_decycler(list_key)
        local_result = data[local_list_key]
        global_result = distributed_data_object(
                                            local_data=local_result,
                                            distribution_strategy='freeform',
                                            comm=self.comm)
        return global_result

    def _advanced_index_decycler(self, from_list_key):
        global_length = self.global_shape[0]
        local_length = self.local_length
        shift = self.local_start
        rank = self.comm.rank

        zeroth_key = from_list_key[0]
        # Check if from_list_key is a scalar
        if np.isscalar(zeroth_key):
            # decycle negative index
            if zeroth_key < 0:
                zeroth_key += global_length
            # if the index is still negative, or it is greater than
            # global_length the index is ill-choosen
            if zeroth_key < 0 or zeroth_key >= global_length:
                raise ValueError(about._errors.cstring(
                    "ERROR: Index out of bounds!"))
            # shift the index
            local_zeroth_key = zeroth_key - shift
            # if the index lies within the local nodes' data-range
            # take the shifted index, combined with rest of from_list_key
            result = [local_zeroth_key]
            for ii in xrange(1, len(from_list_key)):
                current = from_list_key[ii]
                if isinstance(current, distributed_data_object):
                    result.append(current.get_full_data())
                else:
                    result.append(current)
            if (local_zeroth_key < 0) or (local_zeroth_key >= local_length):
                result = (np.array([], dtype=np.dtype('int')),) * \
                    len(from_list_key)

        elif isinstance(zeroth_key, distributed_data_object):
            zeroth_key = zeroth_key.copy()
            # decycle negative indices
            zeroth_key[zeroth_key < 0] = zeroth_key[zeroth_key < 0] + \
                global_length
            # if there are still negative indices, or indices greater than
            # global_length the indices are ill-choosen
            if (zeroth_key < 0).any() or (zeroth_key >= global_length).any():
                raise ValueError(about._errors.cstring(
                    "ERROR: Index out of bounds!"))
            # shift the indices according to shift
            shift_list = self.comm.allgather(shift)
            local_zeroth_key_list = map(lambda z: zeroth_key - z, shift_list)
            # discard all entries where the indices are negative or larger
            # than local_length
            greater_than_lower_list = map(lambda z: z >= 0,
                                          local_zeroth_key_list)
            # -> build up a list with the local selection d2o's
            local_length_list = self.comm.allgather(local_length)
            less_than_upper_list = map(lambda z, zz: z < zz,
                                       local_zeroth_key_list,
                                       local_length_list)
            local_selection_list = map(lambda z, zz: z * zz,
                                       less_than_upper_list,
                                       greater_than_lower_list)

            for j in xrange(len(local_zeroth_key_list)):
                temp_result = local_zeroth_key_list[j].\
                    get_data(local_selection_list[j]).\
                    get_full_data(target_rank=j)
                if j == rank:
                    result = temp_result

            if not all(result[i] <= result[i + 1]
                       for i in xrange(len(result) - 1)):
                raise ValueError(about._errors.cstring(
                    "ERROR: The first dimemnsion of list_key must be sorted!"))
            result = [result]

            for ii in xrange(1, len(from_list_key)):
                current = from_list_key[ii]
                if np.isscalar(current):
                    result.append(current)
                elif isinstance(current, distributed_data_object):
                    result.append(current.get_data(
                                   local_selection_list[rank],
                                   local_keys=True).get_local_data(copy=False))
                else:
                    for j in xrange(len(local_selection_list)):
                        temp_select = local_selection_list[j].\
                            get_full_data(target_rank=j)
                        if j == rank:
                            temp_result = current[temp_select]
                    result.append(temp_result)

        else:
            zeroth_key = zeroth_key.copy()
            # decycle negative indices
            zeroth_key[zeroth_key < 0] = zeroth_key[zeroth_key < 0] + \
                global_length
            # if there are still negative indices, or indices greater than
            # global_length the indices are ill-choosen
            if (zeroth_key < 0).any() or (zeroth_key >= global_length).any():
                raise ValueError(about._errors.cstring(
                    "ERROR: Index out of bounds!"))
            # shift the indices according to shift
            local_zeroth_key = zeroth_key - shift
            # discard all entries where the indices are negative or larger
            # than local_length
            greater_than_lower = (local_zeroth_key >= 0)
            less_than_upper = (local_zeroth_key < local_length)
            local_selection = greater_than_lower * less_than_upper

            result = [local_zeroth_key[local_selection]]
            if not all(result[0][i] <= result[0][i + 1]
                       for i in xrange(len(result[0]) - 1)):
                raise ValueError(about._errors.cstring(
                    "ERROR: The first dimemnsion of list_key must be sorted!"))

            for ii in xrange(1, len(from_list_key)):
                current = from_list_key[ii]
                if np.isscalar(current):
                    result.append(current)
                elif isinstance(current, distributed_data_object):
                    result.append(current.get_data(
                                   local_selection,
                                   local_keys=True).get_local_data(copy=False))
                else:
                    result.append(current[local_selection])

        return result

    def collect_data_from_bool(self, data, boolean_key, **kwargs):
        local_boolean_key = self.extract_local_data(boolean_key)
        local_result = data[local_boolean_key]
        global_result = distributed_data_object(
                                            local_data=local_result,
                                            distribution_strategy='freeform',
                                            comm=self.comm)
        return global_result

    def _invert_mpi_data_ordering(self, data):
        comm = self.comm
        s = comm.size
        r = comm.rank
        if s == 1:
            return data

        partner = s - 1 - r
        new_data = comm.sendrecv(sendobj=data,
                                 dest=partner,
                                 source=partner)
        comm.barrier()
        return new_data

    def collect_data_from_slices(self, data, slice_objects,
                                 target_rank='all'):

        (slice_objects, sliceified) = self._sliceify(slice_objects)

        localized_start, localized_stop = self._backshift_and_decycle(
            slice_objects[0],
            self.local_start,
            self.local_end,
            self.global_shape[0])
        first_step = slice_objects[0].step
        local_slice = (slice(localized_start,
                             localized_stop,
                             first_step),) + slice_objects[1:]

        # if directly_to_np_Q == False:
        local_result = data[local_slice]
        if (first_step is not None) and (first_step < 0):
            local_result = self._invert_mpi_data_ordering(local_result)

        global_result = distributed_data_object(
                                            local_data=local_result,
                                            distribution_strategy='freeform',
                                            comm=self.comm)

        return self._defold(global_result, sliceified)

    def _backshift_and_decycle(self, slice_object, shifted_start, shifted_stop,
                               global_length):

        # Reformulate negative indices
        if slice_object.start < 0 and slice_object.start is not None:
            temp_start = slice_object.start + global_length
            if temp_start < 0:
                temp_start = 0

            slice_object = slice(temp_start, slice_object.stop,
                                 slice_object.step)

        if slice_object.stop < 0 and slice_object.stop is not None:
            temp_stop = slice_object.stop + global_length
            if temp_stop < 0:
                temp_stop = None

            slice_object = slice(slice_object.start, temp_stop,
                                 slice_object.step)

        # initialize the step
        if slice_object.step is None:
            step = 1
        else:
            step = slice_object.step

        # compute local_length
        local_length = shifted_stop - shifted_start
        if step > 0:
            shift = shifted_start
            # calculate the start index
            if slice_object.start is None:
                local_start = (-shift) % step  # step size compensation
            else:
                local_start = slice_object.start - shift
                # if the local_start is negative, pull it up to zero
                local_start = \
                    local_start % step if local_start < 0 else local_start

            if local_start >= local_length:
                return (0, 0)

            # calculate the stop index
            if slice_object.stop is None:
                local_stop = None
            else:
                local_stop = slice_object.stop - shift
                # if local_stop is negative, the slice is empty
                if local_stop < 0:
                    return (0, 0)
                if local_stop > local_length:
                    local_stop = None

        else:  # if step < 0
            step = -step
            # calculate the start index. (Here, local_start > local_stop!)
            if slice_object.start is None:
                local_start = (local_length - 1) -\
                    (-(global_length - shifted_stop)
                     ) % step  # stepsize compensation
                # if local_start becomes negative here, it means, that the
                # step size is bigger than the length of the local slice and
                # that no relevant data is in this slice
                if local_start < 0:
                    return (0, 0)
            else:
                if slice_object.start > global_length - 1:
                    slice_object = slice(global_length - 1,
                                         slice_object.stop,
                                         slice_object.step)
                local_start = slice_object.start - shifted_start
                # if the local_start is negative, immediately return the
                # values for an empty slice
                if local_start < 0:
                    return (0, 0)

                # if the local_start is greater than the local length, pull
                # it down
                if local_start > local_length - 1:
                    overhead = local_start - (local_length - 1)
                    overhead = overhead - overhead % (-step)
                    local_start = local_start - overhead
                    # if local_start becomes negative here, it means, that the
                    # step size is bigger than the length of the localslice and
                    # that no relevant data is in this slice
                    if local_start < 0:
                        return (0, 0)

            # calculate the stop index
            if slice_object.stop is None:
                local_stop = None
            else:
                local_stop = slice_object.stop - shifted_start
                # if local_stop is negative, pull it up to None
                local_stop = None if local_stop < 0 else local_stop
        # Note: if start or stop are greater than the array length,
        # numpy will automatically cut the index value down into the
        # array's range
#        if local_start > local_length:
#            local_start = local_length
#        if local_stop > local_length:
#            local_stop = local_length
        return (local_start, local_stop)

    def extract_local_data(self, data_object):
        # if data_object is not a ndarray or a d2o, cast it to a ndarray
        if not (isinstance(data_object, np.ndarray) or
                isinstance(data_object, distributed_data_object)):
            data_object = np.array(data_object)
        # check if the shapes are remotely compatible, reshape if possible
        # and determine which dimensions match only via broadcasting
        try:
            (data_object, matching_dimensions) = \
                self._reshape_foreign_data(data_object)
        # if the shape-casting fails, try to fix things via local data
        # matching
        except(ValueError):
            # Check if all the local shapes match the supplied data
            local_matchQ = (self.local_shape == data_object.shape)
            global_matchQ = self._allgather(local_matchQ)
            # if the local shapes match, simply return the data_object
            if np.all(global_matchQ):
                extracted_data = data_object[:]
            else:
                raise ValueError(about._errors.cstring(
                    "ERROR: supplied shapes do neither match globally " +
                    "nor locally"))

        # if shape-casting was successfull, extract the data
        else:
            # If the first dimension matches only via broadcasting...
            # Case 1: ...do broadcasting. This procedure does not depend on the
            # array type (ndarray or d2o)
            if matching_dimensions[0] == False:
                extracted_data = data_object[0:1]

            # Case 2: First dimension fits directly and data_object is a d2o
            elif isinstance(data_object, distributed_data_object):
                # Check if the distributor and the comm match
                # the own ones. Checking equality via 'is' is ok, as the
                # distributor factory caches simmilar distributors
                if self is data_object.distributor and\
                        self.comm is data_object.distributor.comm:
                    # Case 1: yes. Simply take the local data
                    extracted_data = data_object.data
                else:
                    # Case 2: no. All nodes extract their local slice from the
                    # data_object
                    extracted_data =\
                        data_object.get_data(slice(self.local_start,
                                                   self.local_end),
                                             local_keys=True)
                    extracted_data = extracted_data.get_local_data()

            # Case 3: First dimension fits directly and data_object is an
            # generic array
            else:
                extracted_data =\
                    data_object[self.local_start:self.local_end]

        return extracted_data

    def _reshape_foreign_data(self, foreign):
        # Case 1:
        # check if the shapes match directly
        if self.global_shape == foreign.shape:
            matching_dimensions = [True, ] * len(self.global_shape)
            return (foreign, matching_dimensions)
        # Case 2:
        # if not, try to reshape the input data.
        # In particular, this will fail when foreign is a d2o as long as
        # reshaping is not implemented
        try:
            output = foreign.reshape(self.global_shape)
            matching_dimensions = [True, ] * len(self.global_shape)
            return (output, matching_dimensions)
        except(ValueError, AttributeError):
            pass
        # Case 3:
        # if this does not work, try to broadcast the shape
        # check if the dimensions match
        if len(self.global_shape) != len(foreign.shape):
            raise ValueError(
                about._errors.cstring("ERROR: unequal number of dimensions!"))
        # check direct matches
        direct_match = (np.array(self.global_shape) == np.array(foreign.shape))
        # check broadcast compatibility
        broadcast_match = (np.ones(len(self.global_shape), dtype=int) ==
                           np.array(foreign.shape))
        # combine the matches and assert that all are true
        combined_match = (direct_match | broadcast_match)
        if not np.all(combined_match):
            raise ValueError(
                about._errors.cstring("ERROR: incompatible shapes!"))
        matching_dimensions = tuple(direct_match)
        return (foreign, matching_dimensions)

    def consolidate_data(self, data, target_rank='all', comm=None):
        if comm is None:
            comm = self.comm
        _gathered_data = np.empty(self.global_shape, dtype=self.dtype)
        _dim_list = self.all_local_slices[:, 3]
        _dim_offset_list = self.all_local_slices[:, 4]
        if target_rank == 'all':
            comm.Allgatherv([data, self.mpi_dtype],
                            [_gathered_data, _dim_list, _dim_offset_list,
                             self.mpi_dtype])
        else:
            comm.Gatherv([data, self.mpi_dtype],
                         [_gathered_data, _dim_list, _dim_offset_list,
                          self.mpi_dtype],
                         root=target_rank)
        return _gathered_data

    def flatten(self, data, inplace=False):
        if inplace:
            return data.ravel()
        else:
            return data.flatten()

    def where(self, data):
        # compute np.where result from the node's local data
        local_where = list(np.where(data))
        # shift the first dimension
        local_where[0] = local_where[0] + self.local_start
        local_where = tuple(local_where)

        global_where = map(lambda z: distributed_data_object(
                                             local_data=z,
                                             distribution_strategy='freeform'),
                           local_where)
        return global_where

    def _sliceify(self, inp):
        sliceified = []
        result = []
        if isinstance(inp, tuple):
            x = inp
        elif isinstance(inp, list):
            x = tuple(inp)
        else:
            x = (inp, )

        for i in range(len(x)):
            if isinstance(x[i], slice):
                result += [x[i], ]
                sliceified += [False, ]
            else:
                if x[i] >= self.global_shape[i]:
                    raise IndexError('Index out of bounds!')
                result += [slice(x[i], x[i] + 1), ]
                sliceified += [True, ]

        return (tuple(result), sliceified)

    def _enfold(self, in_data, sliceified):
        # TODO: Implement a reshape functionality in order to avoid this
        # low level mess!!

        if isinstance(in_data, distributed_data_object):
            local_data = in_data.data
        elif isinstance(in_data, np.ndarray) == False:
            local_data = np.array(in_data, copy=False)
            in_data = local_data
        else:
            local_data = in_data

        temp_local_shape = ()
        temp_global_shape = ()
        j = 0
        for i in sliceified:
            if i is False:
                try:
                    temp_local_shape += (local_data.shape[j],)
                    temp_global_shape += (in_data.shape[j],)
                except(IndexError):
                    temp_local_shape += (1,)
                    temp_global_shape += (1,)
                j += 1
            else:
                temp_local_shape += (1,)
                temp_global_shape += (1,)
                try:
                    if in_data.shape[j] == 1:
                        j += 1
                except(IndexError):
                    pass

        # take into account that the sliceified tuple may be too short,
        # because of a non-exaustive list of slices
        for i in range(len(local_data.shape) - j):
            temp_local_shape += (local_data.shape[j],)
            temp_global_shape += (in_data.shape[j],)
            j += 1

        if isinstance(in_data, distributed_data_object):
            # in case of leading scalars, indenify the node with data
            # and broadcast the shape to the others
            if sliceified[0]:
                local_has_data = (np.prod(
                    np.shape(in_data.get_local_data())
                ) != 0)
                local_has_data_list = np.array(self.comm.allgather(
                    local_has_data))
                nodes_with_data = np.where(local_has_data_list == True)[0]
                if np.shape(nodes_with_data)[0] > 1:
                    raise ValueError(
                        "ERROR: scalar index on first dimension, but more " +
                        "than one node has data!")
                elif np.shape(nodes_with_data)[0] == 1:
                    node_with_data = nodes_with_data[0]
                else:
                    node_with_data = -1

                if node_with_data == -1:
                    broadcasted_shape = (0,) * len(temp_local_shape)
                else:
                    broadcasted_shape = self.comm.bcast(temp_local_shape,
                                                        root=node_with_data)
                if self.comm.rank != node_with_data:
                    temp_local_shape = np.array(broadcasted_shape)
                    temp_local_shape[0] = 0
                    temp_local_shape = tuple(temp_local_shape)

            if in_data.distribution_strategy != 'freeform':
                new_data = in_data.copy_empty(global_shape=temp_global_shape)
                new_data.set_local_data(local_data, copy=False)
            else:
                reshaped_data = local_data.reshape(temp_local_shape)
                new_data = distributed_data_object(
                                           local_data=reshaped_data,
                                           distribution_strategy='freeform',
                                           comm=self.comm)
            return new_data
        else:
            return local_data.reshape(temp_local_shape)

    def _defold(self, in_data, sliceified):
        # TODO: Implement a reshape functionality in order to avoid this
        # low level mess!!
        if isinstance(in_data, distributed_data_object):
            local_data = in_data.data
        elif isinstance(in_data, np.ndarray) == False:
            local_data = np.array(in_data, copy=False)
            in_data = local_data
        else:
            local_data = in_data
        temp_local_shape = ()
        temp_global_shape = ()
        j = 0
        for i in sliceified:
            if i is False:
                try:
                    temp_local_shape += (local_data.shape[j],)
                    temp_global_shape += (in_data.shape[j],)
                except(IndexError):
                    temp_local_shape += (1,)
                    temp_global_shape += (1,)
            j += 1

        # take into account that the sliceified tuple may be too short,
        # because of a non-exaustive list of slices
        for i in range(len(local_data.shape) - j):
            temp_local_shape += (local_data.shape[j],)
            temp_global_shape += (in_data.shape[j],)
            j += 1

        if isinstance(in_data, distributed_data_object):
            if temp_global_shape == ():
                new_data = in_data.get_full_data().flatten()[0]
            elif in_data.distribution_strategy != 'freeform':
                new_data = in_data.copy_empty(global_shape=temp_global_shape)
                if np.any(np.array(local_data.shape)[np.array(sliceified)] ==
                          0):
                    new_data.data[:] = np.empty((0,) * len(temp_local_shape),
                                                dtype=in_data.dtype)
                else:
                    new_data.data[:] = local_data.reshape(temp_local_shape)
            else:
                if np.any(np.array(local_data.shape)[np.array(sliceified)] ==
                          0):
                    temp = np.array(temp_local_shape)
                    temp[0] = 0
                    temp_local_shape = tuple(temp)
                    reshaped_data = np.empty(temp_local_shape,
                                             dtype=in_data.dtype)
                else:
                    reshaped_data = local_data.reshape(temp_local_shape)

                new_data = distributed_data_object(
                                           local_data=reshaped_data,
                                           distribution_strategy='freeform',
                                           comm=self.comm)
            return new_data
        else:
            if temp_global_shape == ():
                return local_data.flatten()[0]
            else:
                return local_data.reshape(temp_local_shape)

    if 'h5py' in gdi:
        def save_data(self, data, alias, path=None, overwriteQ=True):
            comm = self.comm
            h5py_parallel = h5py.get_config().mpi
            if comm.size > 1 and not h5py_parallel:
                raise RuntimeError("ERROR: Programm is run with MPI " +
                                   "size > 1 but non-parallel version of " +
                                   "h5py is loaded.")
            # if no path and therefore no filename was given, use the alias
            # as filename
            use_path = alias if path is None else path

            # create the file-handle
            if h5py_parallel and gc['mpi_module'] == 'MPI':
                f = h5py.File(use_path, 'a', driver='mpio', comm=comm)
            else:
                f = h5py.File(use_path, 'a')
            # check if dataset with name == alias already exists
            try:
                f[alias]
                # if yes, and overwriteQ is set to False, raise an Error
                if overwriteQ is False:
                    raise ValueError(about._errors.cstring(
                        "ERROR: overwriteQ is False, but alias already " +
                        "in use!"))
                else:  # if yes, remove the existing dataset
                    del f[alias]
            except(KeyError):
                pass

            # create dataset
            dset = f.create_dataset(alias,
                                    shape=self.global_shape,
                                    dtype=self.dtype)
            # write the data
            dset[self.local_start:self.local_end] = data
            # close the file
            f.close()

        def load_data(self, alias, path):
            comm = self.comm
            # parse the path
            file_path = path if (path is not None) else alias
            # create the file-handle
            if h5py.get_config().mpi and gc['mpi_module'] == 'MPI':
                f = h5py.File(file_path, 'r', driver='mpio', comm=comm)
            else:
                f = h5py.File(file_path, 'r')
            dset = f[alias]
            # check shape
            if dset.shape != self.global_shape:
                raise TypeError(about._errors.cstring(
                    "ERROR: The shape of the given dataset does not match " +
                    "the distributed_data_object."))
            # check dtype
            if dset.dtype != self.dtype:
                print ('dsets', dset.dtype, self.dtype)
                raise TypeError(about._errors.cstring(
                    "ERROR: The datatype of the given dataset does not " +
                    "match the one of the distributed_data_object."))
            # if everything seems to fit, load the data
            data = dset[self.local_start:self.local_end]
            # close the file
            f.close()
            return data
    else:
        def save_data(self, *args, **kwargs):
            raise ImportError(about._errors.cstring(
                "ERROR: h5py is not available"))

        def load_data(self, *args, **kwargs):
            raise ImportError(about._errors.cstring(
                "ERROR: h5py is not available"))


def _equal_slicer(comm, global_shape):
    rank = comm.rank
    size = comm.size

    global_length = global_shape[0]
    # compute the smallest number of rows the node will get
    local_length = global_length // size
    # calculate how many nodes will get an extra row
    number_of_extras = global_length - local_length * size

    # calculate the individual offset
    offset = rank * local_length + min(rank, number_of_extras) * 1

    # check if local node will get an extra row or not
    if number_of_extras > rank:
        # if yes, increase the local_length by one
        local_length += 1

    return (offset, offset + local_length, global_shape)


def _freeform_slicer(comm, local_shape):
    rank = comm.rank
    size = comm.size
    # Check that all but the first dimensions of local_shape are the same
    local_sub_shape = local_shape[1:]
    local_sub_shape_list = comm.allgather(local_sub_shape)

    cleared_set = set(local_sub_shape_list)
    cleared_set.discard(())

    if len(cleared_set) > 1:
        raise ValueError(about._errors.cstring("ERROR: All but the first " +
                                               "dimensions of local_shape " +
                                               "must be the same!"))
    if local_shape == ():
        first_shape_index = 0
    else:
        first_shape_index = local_shape[0]
    first_shape_index_list = comm.allgather(first_shape_index)
    first_shape_index_cumsum = (0,) + tuple(np.cumsum(first_shape_index_list))
    local_offset = first_shape_index_cumsum[rank]
    global_shape = (first_shape_index_cumsum[size],) + local_shape[1:]
    return (local_offset, local_offset + first_shape_index, global_shape)


if 'pyfftw' in gdi:
    def _fftw_slicer(comm, global_shape):
        if gc['mpi_module'] != 'MPI':
            comm = None
        # pyfftw.local_size crashes if any of the entries of global_shape
        working_shape = np.array(global_shape)
        mask = (working_shape == 0)
        if mask[0] == True:
            start = 0
            end = 0
            return (start, end, global_shape)

        if np.any(mask):
            working_shape[mask] = 1

        local_size = pyfftw.local_size(working_shape, comm=comm)
        start = local_size[2]
        end = start + local_size[1]
        return (start, end, global_shape)


class _not_distributor(distributor):

    def __init__(self, global_shape, dtype, comm, *args,  **kwargs):
        self.comm = comm
        self.dtype = dtype
        self.global_shape = global_shape

        self.local_shape = self.global_shape
        self.distribution_strategy = 'not'

    def initialize_data(self, global_data, alias, path, hermitian, copy,
                        **kwargs):
        if np.isscalar(global_data):
            local_data = np.empty(self.local_shape, dtype=self.dtype)
            local_data.fill(global_data)
            hermitian = True
        else:
            local_data = self.distribute_data(data=global_data,
                                              alias=alias,
                                              path=path,
                                              copy=copy)
        return (local_data, hermitian)

    def globalize_flat_index(self, index):
        return index

    def globalize_index(self, index):
        return index

    def _allgather(self, thing):
        return [thing, ]

    def distribute_data(self, data, alias=None, path=None, copy=True,
                        **kwargs):
        if 'h5py' in gdi and alias is not None:
            data = self.load_data(alias=alias, path=path)

        if data is None:
            return np.empty(self.global_shape, dtype=self.dtype)
        elif isinstance(data, distributed_data_object):
            new_data = data.get_full_data()
        else:
            new_data = np.array(data)
        return new_data.astype(self.dtype,
                               copy=copy).reshape(self.global_shape)

    def _disperse_data_primitive(self, data, to_key, data_update, from_key,
                                 copy, to_found, to_found_boolean, from_found,
                                 from_found_boolean, **kwargs):
        if to_found == 'd2o':
            to_key = to_key.get_full_data()
        if from_found is None:
            from_key = slice(None)

        if np.isscalar(data_update):
            update = data_update
        elif isinstance(data_update, distributed_data_object):
            update = data_update[from_key].get_full_data().\
                astype(self.dtype)
        else:
            if isinstance(from_key, distributed_data_object):
                from_key = from_key.get_full_data()
            update = np.array(data_update,
                              copy=copy)[from_key].astype(self.dtype)
        data[to_key] = update

    def collect_data(self, data, key, local_keys=False, **kwargs):
        if isinstance(key, distributed_data_object):
            key = key.get_full_data()
        new_data = data[key]
        if isinstance(new_data, np.ndarray):
            if local_keys:
                return distributed_data_object(
                                           local_data=new_data,
                                           distribution_strategy='freeform',
                                           comm=self.comm)
            else:
                return distributed_data_object(global_data=new_data,
                                               distribution_strategy='not',
                                               comm=self.comm)
        else:
            return new_data

    def consolidate_data(self, data, **kwargs):
        return data.copy()

    def extract_local_data(self, data_object):
        if isinstance(data_object, distributed_data_object):
            return data_object.get_full_data().reshape(self.global_shape)
        else:
            return np.array(data_object)[:].reshape(self.global_shape)

    def flatten(self, data, inplace=False):
        if inplace:
            return data.ravel()
        else:
            return data.flatten()

    def where(self, data):
        # compute the result from np.where
        local_where = np.where(data)
        global_where = map(lambda z: distributed_data_object(
                                                 global_data=z,
                                                 distribution_strategy='not'),
                           local_where)
        return global_where

    if 'h5py' in gdi:
        def save_data(self, data, alias, path=None, overwriteQ=True):
            comm = self.comm
            h5py_parallel = h5py.get_config().mpi
            if comm.size > 1 and not h5py_parallel:
                raise RuntimeError("ERROR: Programm is run with MPI " +
                                   "size > 1 but non-parallel version of " +
                                   "h5py is loaded.")
            # if no path and therefore no filename was given, use the alias
            # as filename
            use_path = alias if path is None else path

            # create the file-handle
            if h5py_parallel and gc['mpi_module'] == 'MPI':
                f = h5py.File(use_path, 'a', driver='mpio', comm=comm)
            else:
                f = h5py.File(use_path, 'a')
            # check if dataset with name == alias already exists
            try:
                f[alias]
                # if yes, and overwriteQ is set to False, raise an Error
                if overwriteQ is False:
                    raise ValueError(about._errors.cstring(
                        "ERROR: overwriteQ == False, but alias already " +
                        "in use!"))
                else:  # if yes, remove the existing dataset
                    del f[alias]
            except(KeyError):
                pass

            # create dataset
            dset = f.create_dataset(alias,
                                    shape=self.global_shape,
                                    dtype=self.dtype)
            # write the data
            dset[:] = data
            # close the file
            f.close()

        def load_data(self, alias, path):
            comm = self.comm
            # parse the path
            file_path = path if (path is not None) else alias
            # create the file-handle
            if h5py.get_config().mpi and gc['mpi_module'] == 'MPI':
                f = h5py.File(file_path, 'r', driver='mpio', comm=comm)
            else:
                f = h5py.File(file_path, 'r')
            dset = f[alias]
            # check shape
            if dset.shape != self.global_shape:
                raise TypeError(about._errors.cstring(
                    "ERROR: The shape of the given dataset does not match " +
                    "the distributed_data_object."))
            # check dtype
            if dset.dtype != self.dtype:
                raise TypeError(about._errors.cstring(
                    "ERROR: The datatype of the given dataset does not " +
                    "match the distributed_data_object."))
            # if everything seems to fit, load the data
            data = dset[:]
            # close the file
            f.close()
            return data
    else:
        def save_data(self, *args, **kwargs):
            raise ImportError(about._errors.cstring(
                "ERROR: h5py is not available"))

        def load_data(self, *args, **kwargs):
            raise ImportError(about._errors.cstring(
                "ERROR: h5py is not available"))


class _dtype_converter(object):
    """
        NIFTY class for dtype conversion between python/numpy dtypes and MPI
        dtypes.
    """

    def __init__(self):
        pre_dict = [
            # [, MPI_CHAR],
            # [, MPI_SIGNED_CHAR],
            # [, MPI_UNSIGNED_CHAR],
            [np.dtype('bool'), MPI.BYTE],
            [np.dtype('int16'), MPI.SHORT],
            [np.dtype('uint16'), MPI.UNSIGNED_SHORT],
            [np.dtype('uint32'), MPI.UNSIGNED_INT],
            [np.dtype('int32'), MPI.INT],
            [np.dtype('int'), MPI.LONG],
            [np.dtype(np.long), MPI.LONG],
            [np.dtype('int64'), MPI.LONG_LONG],
            [np.dtype('longlong'), MPI.LONG],
            [np.dtype('uint'), MPI.UNSIGNED_LONG],
            [np.dtype('uint64'), MPI.UNSIGNED_LONG_LONG],
            [np.dtype('ulonglong'), MPI.UNSIGNED_LONG_LONG],
            [np.dtype('float32'), MPI.FLOAT],
            [np.dtype('float64'), MPI.DOUBLE],
            [np.dtype('float128'), MPI.LONG_DOUBLE],
            [np.dtype('complex64'), MPI.COMPLEX],
            [np.dtype('complex128'), MPI.DOUBLE_COMPLEX]]

        to_mpi_pre_dict = np.array(pre_dict)
        to_mpi_pre_dict[:, 0] = map(self.dictionize_np, to_mpi_pre_dict[:, 0])
        self._to_mpi_dict = dict(to_mpi_pre_dict)

        to_np_pre_dict = np.array(pre_dict)[:, ::-1]
        to_np_pre_dict[:, 0] = map(self.dictionize_mpi, to_np_pre_dict[:, 0])
        self._to_np_dict = dict(to_np_pre_dict)

    def dictionize_np(self, x):
        dic = x.type.__dict__.items()
        if x.type is np.float:
            dic[24] = 0
            dic[29] = 0
            dic[37] = 0
        return frozenset(dic)

    def dictionize_mpi(self, x):
        return x.name

    def to_mpi(self, dtype):
        return self._to_mpi_dict[self.dictionize_np(dtype)]

    def to_np(self, dtype):
        return self._to_np_dict[self.dictionize_mpi(dtype)]

    def known_mpi_Q(self, dtype):
        return (self.dictionize_mpi(dtype) in self._to_np_dict)

    def known_np_Q(self, dtype):
        return (self.dictionize_np(np.dtype(dtype)) in self._to_mpi_dict)

global_dtype_converter = _dtype_converter()


class _d2o_librarian(object):

    def __init__(self):
        self.library = weakdict()
        self.counter = 0

    def register(self, d2o):
        self.counter += 1
        self.library[self.counter] = d2o
        return self.counter

    def __getitem__(self, key):
        return self.library[key]

d2o_librarian = _d2o_librarian()
