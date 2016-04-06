# -*- coding: utf-8 -*-


import numpy as np

from nifty.keepers import about,\
                          global_configuration as gc,\
                          global_dependency_injector as gdi

from d2o_librarian import d2o_librarian

from strategies import STRATEGIES

MPI = gdi[gc['mpi_module']]


class distributed_data_object(object):
    """A multidimensional array with modular MPI-based distribution schemes.

    The purpose of a distributed_data_object (d2o) is to provide the user
    with a numpy.ndarray like interface while storing the data on an arbitrary
    number of MPI nodes. The logic of a certain distribution strategy is
    implemented by an associated distributor.

    Parameters
    ----------
    global_data : array-like, at least 1-dimensional
        Used with global-type distribution strategies in order to fill the
        d2o with data during initialization.
    global_shape : tuple of ints
        Used with global-type distribution strategies. If no global_data is
        supplied, it will be used.
    dtype : {np.dtype, type}
        Used as the d2o's datatype. Overwrites the data-type of any init data.
    local_data : array-like, at least 1-dimensional
        Used with local-type distribution strategies in order to fill the
        d2o with data during initialization.
    local_shape : tuple of ints
        Used with local-type distribution strategies. If no local_data is
        supplied, local_shape will be used.
    distribution_strategy : optional[{'fftw', 'equal', 'not', 'freeform'}]
        Specifies which distributor will be created and used.
        'fftw'      uses the distribution strategy of pyfftw,
        'equal'     tries to  distribute the data as uniform as possible
        'not'       does not distribute the data at all
        'freeform'  distribute the data according to the given local data/shape
    hermitian : boolean
        Specifies if the given init-data is hermitian or not. The
        self.hermitian attribute will be set accordingly.
    alias : String
        Used in order to initialize the d2o from a hdf5 file.
    path : String
        Used in order to initialize the d2o from a hdf5 file. If no path is
        given, '$working_directory/alias' is used.
    comm : mpi4py.MPI.Intracomm
        The MPI communicator on which the d2o lives.
    copy : boolean
        If true it is guaranteed that the input data will be copied. If false
        copying is tried to be avoided.
    *args
        Although not directly used during the init process, further parameters
        are stored in the self.init_args attribute.
    **kwargs
        Additional keyword arguments are passed to the distributor_factory and
        furthermore get stored in the self.init_kwargs attribute.
    skip_parsing : boolean (optional keyword argument)
        If true, the distribution_factory will skip all sanity checks and
        completions of the given (keyword-)arguments. It just uses what it
        gets. Hence the user is fully responsible for supplying complete and
        consistent parameters. This can be used in order to speed up the init
        process. Also see notes section.

    Attributes
    ----------
    data : numpy.ndarray
        The numpy.ndarray in which the individual node's data is stored.
    dtype : type
        Data type of the data object.
    distribution_strategy : string
        Name of the used distribution_strategy.
    distributor : distributor
        The distributor object which takes care of all distribution and
        consolidation of the data.
    shape : tuple of int
        The global shape of the data.
    local_shape : tuple of int
        The nodes individual local shape of the stored data.
    comm : mpi4py.MPI.Intracomm
        The MPI communicator on which the d2o lives.
    hermitian : boolean
        Specfies whether the d2o's data definitely possesses hermitian
        symmetry.
    index : int
        The d2o's registration index it got from the d2o_librarian.
    init_args : list
        Any additional initialization arguments are stored here.
    init_kwargs : dict
        Any additional initialization keyword arguments are stored here.

    Raises
    ------
    ValueError
        Raised if
            * the supplied distribution strategy is not known
            * comm is None
            * different distribution strategies where given on the
              individual nodes
            * different dtypes where given on the individual nodes
            * neither a non-0-dimensional global_data nor global_shape nor
              hdf5 file supplied
            * global_shape == ()
            * different global_shapes where given on the individual nodes
            * neither non-0-dimensional local_data nor local_shape nor
              global d2o supplied
            * local_shape == ()
            * the first entry of local_shape is not the same on all nodes

    Notes
    -----
    The index is the d2o's global unique indentifier. One may use it in order
    to assemble the corresponding local d2o objects on different nodes if
    only one local object on a specific node is given.

    In order to speed up the init process the distributor_factory checks
    if the global_configuration object gc yields gc['d2o_init_checks'] == True.
    If yes, all checks expensive checks are skipped; namely those which  need
    mpi communication. Use this in order to get a fast init speed without
    loosing d2o's init parsing logic.

    Examples
    --------
    >>> a = np.arange(16, dtype=np.float).reshape((4,4))
    >>> obj = distributed_data_object(a, dtype=np.complex)
    >>> obj
    <distributed_data_object>
    array([[  0.+0.j,   1.+0.j,   2.+0.j,   3.+0.j],
           [  4.+0.j,   5.+0.j,   6.+0.j,   7.+0.j],
           [  8.+0.j,   9.+0.j,  10.+0.j,  11.+0.j],
           [ 12.+0.j,  13.+0.j,  14.+0.j,  15.+0.j]])

    See Also
    --------
    distributor
    """

    def __init__(self, global_data=None, global_shape=None, dtype=None,
                 local_data=None, local_shape=None,
                 distribution_strategy=None, hermitian=False,
                 alias=None, path=None, comm=MPI.COMM_WORLD,
                 copy=True, *args, **kwargs):

        # TODO: allow init with empty shape

        if isinstance(global_data, tuple) or isinstance(global_data, list):
            global_data = np.array(global_data, copy=False)
        if isinstance(local_data, tuple) or isinstance(local_data, list):
            local_data = np.array(local_data, copy=False)

        if distribution_strategy is None:
            distribution_strategy = gc['default_distribution_strategy']

        from distributor_factory import distributor_factory
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
        """ Returns a d2o containing the real part of the d2o's elements.

        Returns
        -------
        out : distributed_data_object
            The output object. The new datatype is the one numpy yields when
            taking the real part on the local data.
        """

        new_data = self.get_local_data(copy=False).real
        new_dtype = new_data.dtype
        new_d2o = self.copy_empty(dtype=new_dtype)
        new_d2o.set_local_data(data=new_data,
                               copy=False,
                               hermitian=self.hermitian)
        return new_d2o

    @property
    def imag(self):
        """ Returns a d2o containing the imaginary part of the d2o's elements.

        Returns
        -------
        out : distributed_data_object
            The output object. The new datatype is the one numpy yields when
            taking the imaginary part on the local data.
        """

        new_data = self.get_local_data(copy=False).imag
        new_dtype = new_data.dtype
        new_d2o = self.copy_empty(dtype=new_dtype)
        new_d2o.set_local_data(data=new_data,
                               copy=False,
                               hermitian=self.hermitian)
        return new_d2o

    @property
    def hermitian(self):
        return self._hermitian

    @hermitian.setter
    def hermitian(self, value):
        self._hermitian = bool(value)

    def _fast_copy_empty(self):
        """ Make a very fast low level copy of the d2o without its data.

        This function is fast, because it uses EmptyD2o - a derived class from
        distributed_data_object and then copies the __dict__ directly. Unlike
        copy_empty, _fast_copy_empty will copy all attributes unchanged.
        """
        # make an empty d2o
        new_copy = EmptyD2o()
        # repair its class
        new_copy.__class__ = self.__class__
        # now copy everthing in the __dict__ except for the data array
        for key, value in self.__dict__.items():
            if key != 'data':
                new_copy.__dict__[key] = value
            else:
                new_copy.__dict__[key] = np.empty_like(value)
        # Register the new d2o at the librarian in order to get a unique index
        new_copy.index = d2o_librarian.register(new_copy)
        return new_copy

    def copy(self, dtype=None, distribution_strategy=None, **kwargs):
        """ Returns a full copy of the distributed data object.

        If no keyword arguments are given, the returned object will be an
        identical copy of the original d2o. By explicit specification one is
        able to define the dtype and the distribution_strategy of the returned
        d2o.

        Parameters
        ----------
        dtype : type
            The dtype that the new d2o will have. The data of the primary
            d2o will be casted.
        distribution_strategy : all supported distribution strategies
            The distribution strategy the new d2o should have. If not None and
            different from the original one, there will certainly be inter-node
            communication.
        **kwargs
            Additional keyword arguments get passed to the used copy_empty
            routine.

        Returns
        -------
        out : distributed_data_object
            The output object. It containes the old data, possibly casted to a
            new datatype and distributed according to a new distribution
            strategy

        See Also
        --------
        copy_empty

        """
        temp_d2o = self.copy_empty(dtype=dtype,
                                   distribution_strategy=distribution_strategy,
                                   **kwargs)
        if distribution_strategy is None or \
                distribution_strategy == self.distribution_strategy:
            temp_d2o.set_local_data(self.get_local_data(copy=False), copy=True)
        else:
            temp_d2o.set_full_data(self, hermitian=self.hermitian)
        temp_d2o.hermitian = self.hermitian
        return temp_d2o

    def copy_empty(self, global_shape=None, local_shape=None, dtype=None,
                   distribution_strategy=None, **kwargs):
        """ Returns an empty copy of the distributed data object.

        If no keyword arguments are given, the returned object will be an
        identical copy of the original d2o containing random data. By explicit
        specification one is able to define the new dtype and
        distribution_strategy of the returned d2o and to modify the new shape.

        Parameters
        ----------
        global_shape : tuple of ints
            The global shape that the new d2o shall have. Relevant for
            global-type distribution strategies like 'equal' or 'fftw'.
        local_shape : tuple of ints
            The local shape that the new d2o shall have. Relevant for
            local-type distribution strategies like 'freeform'.
        dtype : type
            The dtype that the new d2o will have.
        distribution_strategy : all supported distribution strategies
            The distribution strategy the new d2o should have.
        **kwargs
            Additional keyword arguments get passed to the init-call if the
            full initialization of a new distributed_data_object is necessary

        Returns
        -------
        out : distributed_data_object
            The output object. It contains random data.

        See Also
        --------
        copy

        """
        if self.distribution_strategy == 'not' and \
                distribution_strategy in STRATEGIES['local'] and \
                local_shape is None:
            result = self.copy_empty(global_shape=global_shape,
                                     local_shape=local_shape,
                                     dtype=dtype,
                                     distribution_strategy='equal',
                                     **kwargs)
            return result.copy_empty(
                distribution_strategy=distribution_strategy)

        if global_shape is None:
            global_shape = self.shape
        if local_shape is None:
            local_shape = self.local_shape
        if dtype is None:
            dtype = self.dtype
        else:
            dtype = np.dtype(dtype)
        if distribution_strategy is None:
            distribution_strategy = self.distribution_strategy

        # check if all parameters remain the same -> use the _fast_copy_empty
        if (global_shape == self.shape and
                local_shape == self.local_shape and
                dtype == self.dtype and
                distribution_strategy == self.distribution_strategy and
                kwargs == self.init_kwargs):
            return self._fast_copy_empty()

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
        """ Maps a scalar function on each entry of an array.

        The result of the function evaluation may be stored in the original
        array or in a new array (default). Furthermore the dtype of the
        returned array can be specified explicitly if inplace is set to False.

        Parameters
        ----------
        function : callable
            Will be applied to the array's entries. It will be the node's local
            data array into function as a whole. If this fails, the numpy
            vectorize function will be used.
        inplace : boolean
            Specifies if the result of the function evaluation should be stored
            in the original array or not.
        dtype : type
            If inplace is set to False, it is possible to specify the return
            d2o's dtype explicitly.

        Returns
        -------
        out : distributed_data_object
            Resulting d2o. This is either a newly created array or the primary
            d2o itself.
        """
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
                about.warnings.cprint(
                    "WARNING: Trying to use np.vectorize!")
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

    def apply_generator(self, generator, copy=False):
        """ Evaluates generator(local_shape) and stores the result locally.

        Parameters
        ----------
        generator : callable
            This function must be able to process the node's local data shape
            and return a numpy.ndarray of this very shape. This array is then
            stored as the local data array on each node.
        copy : boolean
            Specifies whether the self.set_local_data method is instructed to
            copy the result from generator or not.

        Notes
        -----
        The generator function yields node-local results. Therefore it is
        assumed that the resulting overall d2o does not possess hermitian
        symmetry anymore. Therefore self.hermitian is set to False.

        """
        self.set_local_data(generator(self.distributor.local_shape), copy=copy)
        self.hermitian = False

    def __array__(self, dtype=None):
        """ Returns the d2o's full data. """
        return self.get_full_data()

    def __str__(self):
        """ x.__str__() <==> str(x)"""
        return self.data.__str__()

    def __repr__(self):
        """ x.__repr__() <==> repr(x)"""
        return '<distributed_data_object>\n' + self.data.__repr__()

    def _compare_helper(self, other, op):
        """ _compare_helper is used for <, <=, ==, !=, >= and >.

        It checks the class of `other` and then utilizes the appropriate
        methods of self. If `other` is not a scalar, numpy.ndarray or
        distributed_data_object this method will use numpy casting.

        Parameters
        ----------
        other : scalar, numpy.ndarray, distributed_data_object, array_like
            This is the object that will be compared to self.
        op : string
            The name of the comparison function, e.g. '__ne__'.

        Returns
        -------
        result : boolean, distributed_data_object
            If `other` was None, False will be returned. This follows the
            behaviour of numpy but will changed as soon as numpy changed their
            convention. In every other case a distributed_data_object with
            element-wise comparison results will be returned.

        """

        if other is not None:
            result = self.copy_empty(dtype=np.bool_)

        # Case 1: 'other' is a scalar
        # -> make element-wise comparison
        if np.isscalar(other):
            result.set_local_data(
                getattr(self.get_local_data(copy=False), op)(other))
            return result

        # Case 2: 'other' is a numpy array or a distributed_data_object
        # -> extract the local data and make element-wise comparison
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
        """ x.__ne__(y) <==> x != y

        See Also
        --------
        _compare_helper

        """
        return self._compare_helper(other, '__ne__')

    def __lt__(self, other):
        """ x.__lt__(y) <==> x < y

        See Also
        --------
        _compare_helper

        """

        return self._compare_helper(other, '__lt__')

    def __le__(self, other):
        """ x.__le__(y) <==> x <= y

        See Also
        --------
        _compare_helper

        """

        return self._compare_helper(other, '__le__')

    def __eq__(self, other):
        """ x.__eq__(y) <==> x == y

        See Also
        --------
        _compare_helper

        """

        return self._compare_helper(other, '__eq__')

    def __ge__(self, other):
        """ x.__ge__(y) <==> x >= y

        See Also
        --------
        _compare_helper

        """

        return self._compare_helper(other, '__ge__')

    def __gt__(self, other):
        """ x.__gt__(y) <==> x > y

        See Also
        --------
        _compare_helper

        """

        return self._compare_helper(other, '__gt__')

    def __iter__(self):
        """ x.__iter__() <==> iter(x)

        The __iter__ call returns an iterator it got from self.distributor.

        See Also
        --------
        distributor.get_iter

        """
        return self.distributor.get_iter(self)

    def equal(self, other):
        """  Checks if `other` and `self` are structurally the same.

        In contrast to the element-wise comparison with `__eq__`, `equal`
        checks more than only the equality of the array data.
        It checks the equality of
            * shape
            * dtype
            * init_args
            * init_kwargs
            * distribution_strategy
            * node's local data

        Parameters
        ----------
        other : object
            The object that will be compared to `self`.

        Returns
        -------
        result : boolean
            True if above conditions are met, False otherwise.

        """

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
        """ x.__pos__() <==> +x

        Returns a (positive) copy of `self`.
        """

        temp_d2o = self.copy_empty()
        temp_d2o.set_local_data(data=self.get_local_data().__pos__(),
                                copy=False)
        return temp_d2o

    def __neg__(self):
        """ x.__neg__() <==> -x

        Returns a negative copy of `self`.
        """

        temp_d2o = self.copy_empty()
        temp_d2o.set_local_data(data=self.get_local_data().__neg__(),
                                copy=False)
        return temp_d2o

    def __abs__(self):
        """ x.__abs__() <==> abs(x)

        Returns an absolute valued copy of `self`.
        """

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
                                copy=False)
        return temp_d2o

    def _builtin_helper(self, operator, other, inplace=False):
        """ Used for various binary operations like +, -, *, /, **, *=, +=,...

        _builtin_helper checks whether `other` is a scalar or an array and
        based on that extracts the locally relevant data from it. If `self`
        is hermitian, _builtin_helper tries to conserve this flag; but without
        checking hermitianity explicitly.

        Parameters
        ----------
        operator : callable

        other : scalar, array-like

        inplace : boolean
            If the result shall be saved in the data array of `self`. Used for
            +=, -=, etc...
        Returns
        -------
        out : distributed_data_object
            The distributed_data_object containing the computation's result.
            Equals `self` if `inplace is True`.

        """
        # Case 1: other is not a scalar
        if not (np.isscalar(other) or np.shape(other) == (1,)):
            try:
                hermitian_Q = (other.hermitian and self.hermitian)
            except(AttributeError):
                hermitian_Q = False
            # extract the local data from the 'other' object
            input_data = self.distributor.extract_local_data(other)

        # Case 2: other is a scalar
        else:
            # if other is a scalar packed in a d2o, extract its value.
            if isinstance(other, distributed_data_object):
                input_data = other[0]
            else:
                input_data = other

            if np.isrealobj(other):
                hermitian_Q = self.hermitian
            else:
                hermitian_Q = False

        local_data = self.get_local_data(copy=False)

        result_data = getattr(local_data, operator)(input_data)

        # select the return-distributed_data_object
        if inplace is True:
            temp_d2o = self
        else:
            # use common datatype for self and other
            new_dtype = np.dtype(np.find_common_type((self.dtype,),
                                                     (result_data.dtype,)))
            temp_d2o = self.copy_empty(dtype=new_dtype)

        # write the new data into the return-distributed_data_object
        temp_d2o.set_local_data(data=result_data, copy=False)
        temp_d2o.hermitian = hermitian_Q
        return temp_d2o

    def __add__(self, other):
        """ x.__add__(y) <==> x+y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__add__', other)

    def __radd__(self, other):
        """ x.__radd__(y) <==> y+x

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__radd__', other)

    def __iadd__(self, other):
        """ x.__iadd__(y) <==> x+=y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__iadd__',
                                    other,
                                    inplace=True)

    def __sub__(self, other):
        """ x.__sub__(y) <==> x-y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__sub__', other)

    def __rsub__(self, other):
        """ x.__rsub__(y) <==> y-x

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__rsub__', other)

    def __isub__(self, other):
        """ x.__isub__(y) <==> x-=y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__isub__',
                                    other,
                                    inplace=True)

    def __div__(self, other):
        """ x.__div__(y) <==> x/y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__div__', other)

    def __truediv__(self, other):
        """ x.__truediv__(y) <==> x/y

        See Also
        --------
        _builtin_helper
        """

        return self.__div__(other)

    def __rdiv__(self, other):
        """ x.__rdiv__(y) <==> y/x

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__rdiv__', other)

    def __rtruediv__(self, other):
        """ x.__rtruediv__(y) <==> y/x

        See Also
        --------
        _builtin_helper
        """

        return self.__rdiv__(other)

    def __idiv__(self, other):
        """ x.__idiv__(y) <==> x/=y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__idiv__',
                                    other,
                                    inplace=True)

    def __itruediv__(self, other):
        """ x.__itruediv__(y) <==> x/=y

        See Also
        --------
        _builtin_helper
        """

        return self.__idiv__(other)

    def __floordiv__(self, other):
        """ x.__floordiv__(y) <==> x//y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__floordiv__',
                                    other)

    def __rfloordiv__(self, other):
        """ x.__rfloordiv__(y) <==> y//x

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__rfloordiv__',
                                    other)

    def __ifloordiv__(self, other):
        """ x.__ifloordiv__(y) <==> x//=y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper(
            '__ifloordiv__', other,
            inplace=True)

    def __mul__(self, other):
        """ x.__mul__(y) <==> x*y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__mul__', other)

    def __rmul__(self, other):
        """ x.__rmul__(y) <==> y*x

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__rmul__', other)

    def __imul__(self, other):
        """ x.__imul__(y) <==> x*=y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__imul__',
                                    other,
                                    inplace=True)

    def __pow__(self, other):
        """ x.__pow__(y) <==> x**y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__pow__', other)

    def __rpow__(self, other):
        """ x.__rpow__(y) <==> y**x

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__rpow__', other)

    def __ipow__(self, other):
        """ x.__ipow__(y) <==> x**=y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__ipow__',
                                    other,
                                    inplace=True)

    def __mod__(self, other):
        """ x.__mod__(y) <==> x%y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__mod__', other)

    def __rmod__(self, other):
        """ x.__rmod__(y) <==> y%x

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__rmod__', other)

    def __imod__(self, other):
        """ x.__imod__(y) <==> x%=y

        See Also
        --------
        _builtin_helper
        """

        return self._builtin_helper('__imod__',
                                    other,
                                    inplace=True)

    def __len__(self):
        """ Returns the length of the first axis."""

        return self.shape[0]

    def get_dim(self):
        """" Returns the total number of entries in the array.

        This is equivalent to the product of the shape.
        """

        return np.prod(self.shape)

    def vdot(self, other):
        """ Returns the numpy.vdot analogous product of two arrays.

        If `self` is a complex array, the complex conjugate of it will be used.
        Internally the numpy.vdot function is used for the d2o's local data,
        and the individual results get MPI-reduced.

        See Also
        --------
        numpy.vdot
        """

        other = self.distributor.extract_local_data(other)
        local_vdot = np.array([np.vdot(self.get_local_data(), other)])
        global_vdot = np.empty_like(local_vdot)
        self.distributor._Allreduce_sum(sendbuf=local_vdot,
                                        recvbuf=global_vdot)

#        local_vdot = np.vdot(self.get_local_data(), other)
#        local_vdot_list = self.distributor._allgather(local_vdot)
#        global_vdot = np.result_type(self.dtype,
#                                     other.dtype).type(np.sum(local_vdot_list))
        return global_vdot[0]

    def __getitem__(self, key):
        """ x.__getitem__(y) <==> x[y] <==> x.get_data(y) """
        return self.get_data(key)

    def __setitem__(self, key, data):
        """ x.__setitem__(i, y) <==> x[i]=y <==> x.set_data(y, i) """
        self.set_data(data, key)

    def _contraction_helper(self, function, **kwargs):
        """ Used for various operations like min, max, sum, prod, mean,...

        _builtin_helper checks whether the local node's data array is empty,
        then applies the given function on the local data, collects the
        results, and then applies the function to this with the keyword axis=0.

        Parameters
        ----------
        function : callable
            This object will be applied to the local data array.

        **kwargs
            Additional keyword arguments will be passed to the first `function`
            call.

        Returns
        -------
        out :
            The return object of `function`.

        Raises
        ------
        ValueError
            Raised if the d2o's shape equals (0,).
        """

        if self.shape == (0,):
            raise ValueError("ERROR: Zero-size array to reduction operation " +
                             "which has no identity")
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
        """ x.min() <==> x.amin() """
        return self.amin(**kwargs)

    def amin(self, **kwargs):
        """ Returns the minimum of an array.

        See Also
        --------
        numpy.amin
        """

        return self._contraction_helper(np.amin, **kwargs)

    def nanmin(self, **kwargs):
        """ Returns the minimum of an array ignoring all NaNs.

        See Also
        --------
        numpy.nanmin
        """

        return self._contraction_helper(np.nanmin, **kwargs)

    def max(self, **kwargs):
        """ x.max() <==> x.amax() """
        return self.amax(**kwargs)

    def amax(self, **kwargs):
        """ Returns the maximum of an array.

        See Also
        --------
        numpy.amax
        """

        return self._contraction_helper(np.amax, **kwargs)

    def nanmax(self, **kwargs):
        """ Returns the maximum of an array ignoring all NaNs.

        See Also
        --------
        numpy.nanmax
        """

        return self._contraction_helper(np.nanmax, **kwargs)

    def sum(self, **kwargs):
        """ Sums the array elements.

        See Also
        --------
        numpy.sum
        """

        if self.shape == (0,):
            return self.dtype.type(0)
        return self._contraction_helper(np.sum, **kwargs)

    def prod(self, **kwargs):
        """ Multiplies the array elements.

        See Also
        --------
        numpy.prod
        """

        if self.shape == (0,):
            return self.dtype.type(1)
        return self._contraction_helper(np.prod, **kwargs)

    def mean(self, power=1):
        """ Returns the mean of the d2o's elements.

        Parameters
        ----------
        power : scalar
            Used for point-wise exponentiation of the array's elements before
            computing the mean: mean(data**power)

        Returns
        -------
        mean : scalar
            The (pre-powered) mean-value of the array.
        """
        if self.shape == (0,):
            return np.mean(np.array([], dtype=self.dtype))
        # compute the local means and the weights for the mean-mean.
        if np.prod(self.data.shape) == 0:
            local_mean = 0
            include = False
        else:
            if power != 1:
                local_mean = np.mean(self.data**power)
            else:
                local_mean = np.mean(self.data)
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
        """ Returns the variance of the d2o's elements.

        Internally the formula <x**2> - <x>**2 is used.
        """

        if self.shape == (0,):
            return np.var(np.array([], dtype=self.dtype))

        if issubclass(self.dtype.type, np.complexfloating):
            mean_of_the_square = abs(self**2).mean()
            square_of_the_mean = abs(self.mean())**2
        else:
            mean_of_the_square = self.mean(power=2)
            square_of_the_mean = self.mean()**2
        return mean_of_the_square - square_of_the_mean

    def std(self):
        """ Returns the standard deviation of the d2o's elements. """
        if self.shape == (0,):
            return np.std(np.array([], dtype=self.dtype))
        if self.shape == (0,):
            return np.nan
        return np.sqrt(self.var())

    def argmin(self):
        """ Returns the (flat) index of the d2o's smallest value.

        See Also:
        argmax, argmin_nonflat, argmax_nonflat
        """

        if self.shape == (0,):
            raise ValueError(
                "ERROR: attempt to get argmin of an empty object")
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
        """ Returns the (flat) index of the d2o's biggest value.

        See Also:
        argmin, argmin_nonflat, argmax_nonflat
        """

        if self.shape == (0,):
            raise ValueError(
                "ERROR: attempt to get argmax of an empty object")
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
        """ Returns the unraveld index of the d2o's smallest value.

        See Also:
        argmin, argmax, argmax_nonflat
        """

        return np.unravel_index(self.argmin(), self.shape)

    def argmax_nonflat(self):
        """ Returns the unraveld index of the d2o's biggest value.

        See Also:
        argmin, argmax, argmin_nonflat
        """

        return np.unravel_index(self.argmax(), self.shape)

    def conjugate(self):
        """ Returns the element-wise complex conjugate. """

        temp_d2o = self.copy_empty()
        temp_data = np.conj(self.get_local_data())
        temp_d2o.set_local_data(temp_data)
        temp_d2o.hermitian = self.hermitian
        return temp_d2o

    def conj(self):
        """ Returns the element-wise complex conjugate.

        This function essentially calls the `d2o.conjugate` method.
        """

        return self.conjugate()

    def median(self):
        """ Returns the d2o element's median.

        The median is computed by collecting the full d2o data and then passing
        it to the numpy.median function. Hence this implementation is very
        expensive.
        """

        about.warnings.cprint(
            "WARNING: The current implementation of median is very expensive!")
        median = np.median(self.get_full_data())
        return median

    def _is_helper(self, function):
        """ _is_helper is used for functions like isreal, isinf, isfinite,...

        Parameters
        ----------
        function : callable
            The function that will be applied to the node's local data.

        Returns
        -------
        out : distributed_data_object
            A copy of `self` of datatype boolean containing the result of
            `function(self.data)`.
        """

        temp_d2o = self.copy_empty(dtype=np.dtype('bool'))
        temp_d2o.set_local_data(function(self.data))
        return temp_d2o

    def iscomplex(self):
        """ Returns a boolean copy of `self`, where True if element is complex.

        See Also
        --------
        isreal
        """

        return self._is_helper(np.iscomplex)

    def isreal(self):
        """ Returns a boolean copy of `self`, where True if element is real.

        See Also
        --------
        iscomplex
        """

        return self._is_helper(np.isreal)

    def isnan(self):
        """ Returns a boolean copy of `self`, where True if element is NaN.

        See Also
        --------
        isinf
        isfinite
        """

        return self._is_helper(np.isnan)

    def isinf(self):
        """ Returns a boolean copy of `self`, where True if element is +/-inf.

        See Also
        --------
        isnan
        isfinite
        """
        return self._is_helper(np.isinf)

    def isfinite(self):
        """ Returns a boolean copy of `self`, where True if element != +/-inf.

        See Also
        --------
        isnan
        isinf
        """
        return self._is_helper(np.isfinite)

    def nan_to_num(self):
        """ Replace nan with zero and inf with finite numbers.

        Returns a copy of `self` replacing NaN-entries with zero, (positive)
        infinity with a very large number and negative infinity with a very
        small (or negative) number.

        See Also
        --------
        isnan
        isinf
        isfinite
        """
        temp_d2o = self.copy_empty()
        temp_d2o.set_local_data(np.nan_to_num(self.data))
        return temp_d2o

    def all(self):
        """ Returns True if all elements of an array evaluate to True. """

        local_all = np.all(self.get_local_data())
        global_all = self.distributor._allgather(local_all)
        return np.all(global_all)

    def any(self):
        """ Returns True if any element of an array evaluate to True. """

        local_any = np.any(self.get_local_data())
        global_any = self.distributor._allgather(local_any)
        return np.any(global_any)

    def unique(self):
        """ Returns a `numpy.ndarray` holding the d2o's unique elements. """

        local_unique = np.unique(self.get_local_data())
        global_unique = self.distributor._allgather(local_unique)
        global_unique = np.concatenate(global_unique)
        return np.unique(global_unique)

    def bincount(self, weights=None, minlength=None):
        """ Count weighted number of occurrences of each value in the d2o.

        The number of integer bins is `max(self.amax()+1, minlength)`.

        Parameters
        ----------
        weights : optional[array-like]
            An array of the same shape as `self`.
        minlength : optional[int]
            A minimum number of bins for the output array.

        Returns
        -------
        out : numpy.ndarray of ints
            The result of binning `self`.

        Raises
        ------
        TypeError
            If the type of `self` is float or complex.

        See Also
        --------
        numpy.bincount
        """

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

        local_counts = np.bincount(self.get_local_data(copy=False).flatten(),
                                   weights=local_weights,
                                   minlength=minlength)

        if self.distribution_strategy == 'not':
            return local_counts
        else:
            counts = np.empty_like(local_counts)
            # self.distributor._Allreduce_sum(local_counts, counts)
            # Potentially faster, but buggy. <- If np.binbount yields
            # inconsistent datatypes because of empty arrays on certain nodes,
            # the Allreduce produces non-sense results.

            list_of_counts = self.distributor._allgather(local_counts)
            counts = np.sum(list_of_counts, axis=0)
            return counts

    def where(self):
        """ Return the indices where `self` is True.

        Returns
        -------
        out : list of d2os
            The length of the list equals the number of axes `self` has. The
            elements of the list are d2o's containing the x_i'th coordinate
            of the elments of `self`, which were non-zero.
        """

        return self.distributor.where(self.data)

    def set_local_data(self, data, hermitian=False, copy=True):
        """ Writes data directly to the node's local data array.

        No distribution is done. The shape of the input data must fit the
        local data's shape exactly.

        Parameters
        ----------
        data : array-like
            The data that will be stored in `self.data`. The input data will be
            casted to the d2o's dtype and to C-order.

        hermitian : optional[boolean]
            The d2o's hermitian attribute will be set to this value.

        copy : optional[boolean]
            If False, the copying of `data` will be tried to be avoided. If
            True, it is guaranteed, that `data` will be copied.

        Returns
        -------
        None

        See Also
        --------
        get_local_data
        set_data
        set_full_data
        """

        self.hermitian = hermitian
        casted_data = np.array(data,
                               dtype=self.dtype,
                               copy=False,
                               order='C').reshape(self.local_shape)

        if copy is True:
            self.data[:] = casted_data
        else:
            self.data = casted_data

    def set_data(self, data, to_key, from_key=None, local_keys=False,
                 hermitian=False, copy=True, **kwargs):
        """ Takes the supplied `data` and distributes it to the nodes.

        Essentially this method behaves like `d2o[to_key] = data[from_key]`.
        In order to makes this process efficient, the built-in distributors
        do not evaluate the object `d2o[from_key]` explicitly. Instead, the
        individual nodes check for the self-affecting part of `to_key`, then
        compute the corresponding part of `from_key` and extract this
        localized part from `data`.

        By default it is assumed that all nodes got the same `data`-objects:
        either the same integer/list/tuple/ndarray or the individual local
        instance of the same distributed_data_object. Also they assume, that
        the `key` objects are the same on all nodes. In case of d2o's as data-
        and/or key-objects this is important, otherwise MPI-calls from
        different d2os will be mixed and therefore produce randomly wrong
        results or a deadlock. If one likes to use node-individual data- and
        key-objects, the switch `local_keys` must be set to True. Then the
        individual objects will be process one by one and the relevant parts
        transported to the respective nodes.

        Parameters
        ----------
        data : scalar or array-like
            Will be distributed to the individual nodes. If scalar, all entries
            specified by `to_key` will be set this this value.
        to_key : indexing-key like
            Specifies where the data should be stored to. Follows the
            conventions of numpy indexing. Therefore allowed types are
            `integer`, `slice`, `tuple of integers and slices`, `boolean
            array-likes` and `list of index array-like`.
        from_key : optional[indexing-key like]
            The key which specifies the source-data via `data[from_key]`.
        local_keys : optional[boolean]
            Specifies whether all nodes got the same data- and key-objects or
            not. See the descripion above.
        hermitian : optional[boolean]
            The `hermitian` attribute of `self` is set to this value. As the
            default is False, a d2o will lose its potentential hermitianity.
            The behaviour is like that, as a write operation in general
            will violate hermitian symmetry.
        copy : optional[boolean]
            If False, it will be tried to avoid data copying. If True, it is
            guaranteed that `data` will be copied.
        **kwargs
            Additional keyword-arguments are passed to the `disperse_data`
            method of the distributor.

        Returns
        -------
        None

        See Also
        --------
        get_data
        set_local_data
        set_full_data
        d2o_librarian
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
        """ Distributes `data` among the nodes.

        The shapes of `data` and `self` must match.

        This method behaves similar to set_data(data, to_key=slice(None)), but
        as no slice- and/or indexing-arithmetic must be done it is faster.

        Parameters
        ----------
        data : array-like
            The full data set, that will be written into `self`.
        hermitian : optional[boolean]
            The `hermitian` attribute of `self` is set to this value. As the
            default is False, a d2o will lose its potentential hermitianity.
            The behaviour is like that, as the supplied `data` is not
            guaranteed to have hermitian symmetry.
        copy : optional[boolean]
            If True it is guaranteed that the input data will be copied. If
            False copying is tried to be avoided.
        **kwargs
            Additional keyword-arguments are passed to the distributor's
            `distribute_data` method.

        Returns
        -------
        None

        See Also
        --------
        get_full_data
        set_data
        set_local_data
        """

        self.hermitian = hermitian
        self.data = self.distributor.distribute_data(data=data, copy=copy,
                                                     **kwargs)

    def get_local_data(self, copy=True):
        """ Returns the node's local data array.

        Parameters
        ----------
        copy : optional[boolean]
            If True, a copy of `self.data` is returned, else `self.data`
            itself.

        Returns
        -------
        data : numpy.ndarray
            The node's local data array (or a copy of it).

        See Also
        --------
        set_local_data
        get_data
        get_full_data
        """

        if copy is True:
            return np.copy(self.data)
        if copy is False:
            return self.data

    def get_data(self, key, local_keys=False, **kwargs):
        """ Returns data from the d2o specified by `key`.

        Essentially this method corresponds to `d2o[key]`.

        By default it is assumed that all nodes got the same `key`-objects:
        either the same integer/list/tuple/ndarray or the individual local
        instance of the same distributed_data_object. In order to avoid
        inter-node communication as much as possible, the result is then
        returned as a d2o which contains the node's local part of `d2o[key]`.
        There the distributor decides, which distribution strategy the
        return-d2o should have: in case of slicing distribution strategies,
        the return-d2o will have a 'freeform'-distributor; the
        'not'-distributor will return a 'not'-distributed d2o. If `local-keys`
        is set to True, the return-d2o will be 'freeform'-distributed and
        every node will possess the data which was particularized by its
        local key. Naturally this involves more inter-node
        communication if a node requests some data, that was not located on
        itself.


        Parameters
        ----------
        key : indexing-key like
            Loads data from the region which is specified by key. The data is
            consolidated according to the distribution strategy. If the
            individual nodes get different key-arguments, they get individual
            data.
        local_keys : optional[boolean]
            Specifies whether all nodes got the same key-object or not. See the
            description above.
        **kwargs
            Additional keyword-arguments are passed to the `collect_data`
            method of the distributor.

        Returns
        -------
        out : distributed_data_object
            The d2o containing the data specified by `key`.

        See Also
        --------
        set_data
        get_local_data
        get_full_data
        d2o_librarian
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
        """ Consolidates the d2o's data and returns it as a numpy.ndarray.

        This method behaves similar to get_data(key=slice(None)) but is faster
        as no slice- and/or indexing-arithmetic must be done.

        Parameters
        ----------
        target_rank : optional[{'all', int}]
            Specifies if all or only one specific node should recieve the
            result of data consolidation.

        Returns
        -------
        out : numpy.ndarray
            Contains the entire data of the distributed_data_object.

        See Also
        --------
        set_full_data
        get_local_data
        get_data
        """

        return self.distributor.consolidate_data(self.data,
                                                 target_rank=target_rank)

    def flatten(self, inplace=False):
        """ Returns a flat copy of the d2o collapsed into one dimension.

        Copying data will be avoided if possible (regardless of `inplace`).

        Parameters
        ----------
        inplace : optional[boolean]
            If set to True, `self` will be replaced by the result of the
            flattening.

        Returns
        -------
        out : distributed_data_object
            The flatted version of the original distributed_data_object.
        """

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

    def cumsum(self, axis=None):
        """ Return the cumulative sum of the elements along the given axis.

        Parameters
        ----------
        axis : optional[int]
            Axis along which the cumulative sum is computed. The default (None)
            is to compute the cumsum over the flattened d2o.

        Returns
        -------
        out : distributed_data_object
            Contains the results of the cummulative sum.
        """

        cumsum_data = self.distributor.cumsum(self.data, axis=axis)

        if axis is None:
            flat_global_shape = (np.prod(self.shape),)
            flat_local_shape = np.shape(cumsum_data)
            result_d2o = self.copy_empty(global_shape=flat_global_shape,
                                         local_shape=flat_local_shape)
        else:
            result_d2o = self.copy_empty()

        result_d2o.set_local_data(cumsum_data)

        return result_d2o

    def save(self, alias, path=None, overwriteQ=True):
        """ Saves the distributed_data_object to disk utilizing h5py.

        Parameters
        ----------
        alias : string
            The name for the dataset which is saved within the hdf5 file.

        path : optional[str]
            The path to the hdf5 file. If no path is given, the alias is
            taken as filename in the current working directory.

        overwriteQ : optional[boolean]
            Specifies whether a dataset may be overwritten if it is already
            present in the given hdf5 file or not.
        """

        self.distributor.save_data(self.data, alias, path, overwriteQ)

    def load(self, alias, path=None):
        """ Loads a distributed_data_object from disk utilizing h5py.

        Parameters
        ----------
        alias : string
            The name of the dataset which is loaded from the hdf5 file.

        path : optional[str]
            The path to the hdf5 file. If no path is given, the alias is
            taken as filename in the current path.
        """

        self.data = self.distributor.load_data(alias, path)


class EmptyD2o(distributed_data_object):
    def __init__(self):
        pass
