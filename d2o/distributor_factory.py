# -*- coding: utf-8 -*-

import numbers

import numpy as np

from nifty.keepers import about,\
                          global_configuration as gc,\
                          global_dependency_injector as gdi

from distributed_data_object import distributed_data_object

from d2o_iter import d2o_slicing_iter,\
                     d2o_not_iter
from d2o_librarian import d2o_librarian
from dtype_converter import dtype_converter
from cast_axis_to_tuple import cast_axis_to_tuple
from translate_to_mpi_operator import op_translate_dict

from strategies import STRATEGIES

MPI = gdi[gc['mpi_module']]
h5py = gdi.get('h5py')
pyfftw = gdi.get('pyfftw')


class _distributor_factory(object):

    def __init__(self):
        self.distributor_store = {}

    def parse_kwargs(self, distribution_strategy, comm,
                     global_data=None, global_shape=None,
                     local_data=None, local_shape=None,
                     alias=None, path=None,
                     dtype=None, skip_parsing=False, **kwargs):

        if skip_parsing:
            return_dict = {'comm': comm,
                           'dtype': dtype,
                           'name': distribution_strategy
                           }
            if distribution_strategy in STRATEGIES['global']:
                return_dict['global_shape'] = global_shape
            elif distribution_strategy in STRATEGIES['local']:
                return_dict['local_shape'] = local_shape
            return return_dict

        return_dict = {}

        expensive_checks = gc['d2o_init_checks']

        # Parse the MPI communicator
        if comm is None:
            raise ValueError(about._errors.cstring(
                "ERROR: The distributor needs MPI-communicator object comm!"))
        else:
            return_dict['comm'] = comm

        if expensive_checks:
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

        elif distribution_strategy in STRATEGIES['local']:
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
        if expensive_checks:
            dtype_list = comm.allgather(dtype)
            if all(x == dtype_list[0] for x in dtype_list) == False:
                raise ValueError(about._errors.cstring(
                    "ERROR: The given dtype must be the same on all nodes!"))
        return_dict['dtype'] = dtype

        # Parse the shape
        # Case 1: global-type slicer
        if distribution_strategy in STRATEGIES['global']:
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

            if expensive_checks:
                global_shape_list = comm.allgather(global_shape)
                if not all(x == global_shape_list[0]
                           for x in global_shape_list):
                    raise ValueError(about._errors.cstring(
                        "ERROR: The global_shape must be the same on all " +
                        "nodes!"))
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

            if expensive_checks:
                local_shape_list = comm.allgather(local_shape[1:])
                cleared_set = set(local_shape_list)
                cleared_set.discard(())
                if len(cleared_set) > 1:
                    raise ValueError(about._errors.cstring(
                        "ERROR: All but the first entry of local_shape " +
                        "must be the same on all nodes!"))
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
            raise ValueError(about._errors.cstring(
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
                from_key_list = comm.allgather(from_key)
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
                            temp_dtype = np.dtype(data_update.dtype)
                        except(TypeError):
                            temp_dtype = np.array(data_update).dtype
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
                                        copy=False,
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

        self._my_dtype_converter = dtype_converter

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
        self.global_dim = reduce(lambda x, y: x*y, self.global_shape)

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
                local_data = global_data.get_local_data(copy=copy)
            elif np.isscalar(local_data):
                temp_local_data = np.empty(self.local_shape,
                                           dtype=self.dtype)
                temp_local_data.fill(local_data)
                local_data = temp_local_data
                hermitian = True
            elif local_data is None:
                local_data = np.empty(self.local_shape, dtype=self.dtype)
            elif isinstance(local_data, np.ndarray):
                local_data = local_data.astype(
                               self.dtype, copy=copy).reshape(self.local_shape)
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

    def _Allreduce_helper(self, sendbuf, recvbuf, op):
        send_dtype = self._my_dtype_converter.to_mpi(sendbuf.dtype)
        recv_dtype = self._my_dtype_converter.to_mpi(recvbuf.dtype)
        self.comm.Allreduce([sendbuf, send_dtype],
                            [recvbuf, recv_dtype],
                            op=op)
        return recvbuf

    def _selective_allreduce(self, data, op, bufferQ=False):
        size = self.comm.size
        rank = self.comm.rank

        if size == 1:
            if data is None:
                raise ValueError("ERROR: No process with non-None data.")
            result_data = data
        else:

            # infer which data should be included in the allreduce and if its
            # array data
            if data is None:
                got_array = np.array([0])
            elif not isinstance(data, np.ndarray):
                got_array = np.array([2])
            elif reduce(lambda x, y: x*y, data.shape) == 0:
                got_array = np.array([1])
            elif np.issubdtype(data.dtype, np.complexfloating):
                # MPI.MAX and MPI.MIN do not support complex data types
                got_array = np.array([3])
            else:
                got_array = np.array([4])

            got_array_list = np.empty(size, dtype=np.int)
            self.comm.Allgather([got_array, MPI.INT],
                                [got_array_list, MPI.INT])

            if reduce(lambda x, y: x & y, got_array_list == 1):
                return data

            # get first node with non-None data
            try:
                start = next(i for i in xrange(size) if got_array_list[i] > 1)
            except(StopIteration):
                raise ValueError("ERROR: No process with non-None data.")

            # check if the Uppercase function can be used or not
            # -> check if op supports buffers and if we got real array-data
            if bufferQ and got_array[start] == 4:
                # Send the dtype and shape from the start process to the others
                (new_dtype,
                 new_shape) = self.comm.bcast((data.dtype,
                                               data.shape), root=start)
                mpi_dtype = self._my_dtype_converter.to_mpi(new_dtype)
                if rank == start:
                    result_data = data
                else:
                    result_data = np.empty(new_shape, dtype=new_dtype)

                self.comm.Bcast([result_data, mpi_dtype], root=start)

                for i in xrange(start+1, size):
                    if got_array_list[i] > 1:
                        if rank == i:
                            temp_data = data
                        else:
                            temp_data = np.empty(new_shape, dtype=new_dtype)
                        self.comm.Bcast([temp_data, mpi_dtype], root=i)
                        result_data = op(result_data, temp_data)

            else:
                result_data = self.comm.bcast(data, root=start)
                for i in xrange(start+1, size):
                    if got_array_list[i] > 1:
                        temp_data = self.comm.bcast(data, root=i)
                        result_data = op(result_data, temp_data)
        return result_data

    def contraction_helper(self, parent, function, allow_empty_contractions,
                           axis=None, **kwargs):
        if axis == ():
            return parent.copy()

        old_shape = parent.shape
        axis = cast_axis_to_tuple(axis)
        if axis is None:
            new_shape = ()
        else:
            new_shape = tuple([old_shape[i] for i in xrange(len(old_shape))
                               if i not in axis])

        local_data = parent.data

        try:
            contracted_local_data = function(local_data, axis=axis, **kwargs)
        except(ValueError):
            contracted_local_data = None

        # check if additional contraction along the first axis must be done
        if axis is None or 0 in axis:
            (mpi_op, bufferQ) = op_translate_dict[function]
            contracted_global_data = self._selective_allreduce(
                                        contracted_local_data,
                                        mpi_op,
                                        bufferQ)
            new_dist_strategy = 'not'
        else:
            if contracted_local_data is None:
                # raise the exception implicitly
                function(local_data, axis=axis, **kwargs)
            contracted_global_data = contracted_local_data
            new_dist_strategy = parent.distribution_strategy

        new_dtype = contracted_global_data.dtype

        if new_shape == ():
            result = contracted_global_data
        else:
            # try to store the result in a distributed_data_object with the
            # distribution_strategy as parent
            result = parent.copy_empty(global_shape=new_shape,
                                       dtype=new_dtype,
                                       distribution_strategy=new_dist_strategy)

            # However, there are cases where the contracted data does not any
            # longer follow the prior distribution scheme.
            # Example: FFTW distribution on 4 MPI processes
            # Contracting (4, 4) to (4,).
            # (4, 4) was distributed (1, 4)...(1, 4)
            # (4, ) is not distributed like (1,)...(1,) but like (2,)(2,)()()!
            if result.local_shape != contracted_global_data.shape:
                result = parent.copy_empty(
                                    local_shape=contracted_global_data.shape,
                                    dtype=new_dtype,
                                    distribution_strategy='freeform')
            result.set_local_data(contracted_global_data, copy=False)

        return result

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
                                         local_keys=True,
                                         copy=copy)
                return temp_d2o.get_local_data(copy=False).astype(self.dtype,
                                                                  copy=False)
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
            local_data_update = data_update.get_data(
                                          slice(o[rank], o[rank] + l),
                                          local_keys=True
                                          ).get_local_data(copy=False)
            data[local_to_key] = local_data_update.astype(self.dtype,
                                                          copy=False)
        elif np.isscalar(data_update):
            data[local_to_key] = data_update
        else:
            data[local_to_key] = np.array(data_update[o[rank]:o[rank] + l],
                                          copy=copy).astype(self.dtype,
                                                            copy=False)
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
                selected_update = data_update.get_data(
                                 key=update_slice,
                                 local_keys=True)
                local_data_update = selected_update.get_local_data(copy=False)
                local_data_update = local_data_update.astype(self.dtype,
                                                             copy=False)
                if np.prod(np.shape(local_data_update)) != 0:
                    data[local_to_slice] = local_data_update
            # elif np.isscalar(data_update):
            #    data[local_to_slice] = data_update
            else:
                local_data_update = np.array(data_update)[update_slice]
                if np.prod(np.shape(local_data_update)) != 0:
                    data[local_to_slice] = np.array(
                                                local_data_update,
                                                copy=copy).astype(self.dtype,
                                                                  copy=False)

    def collect_data(self, data, key, local_keys=False, copy=True, **kwargs):
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
                                                found_boolean, copy=copy,
                                                **kwargs)
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
                                                        copy=copy, **kwargs)
                # collect the data stored in the d2o to the individual target
                # rank
                temp_data = temp_d2o.get_full_data(target_rank=i)
                if comm.rank == i:
                    individual_data = temp_data
                i += 1
            return_d2o = distributed_data_object(
                            local_data=individual_data,
                            distribution_strategy='freeform',
                            copy=False,
                            comm=self.comm)
            return return_d2o

    def _collect_data_primitive(self, data, key, found, found_boolean,
                                copy=True, **kwargs):

        # Case 1: key is a slice-tuple. Hence, the basic indexing/slicing
        # machinery will be used
        if found == 'slicetuple':
            return self.collect_data_from_slices(data=data,
                                                 slice_objects=key,
                                                 copy=copy,
                                                 **kwargs)
        # Case 2: key is an array
        elif (found == 'ndarray' or found == 'd2o'):
            # Case 2.1: The array is boolean.
            if found_boolean:
                return self.collect_data_from_bool(data=data,
                                                   boolean_key=key,
                                                   copy=copy,
                                                   **kwargs)
            # Case 2.2: The array is not boolean. Only 1-dimensional
            # advanced slicing is supported.
            else:
                if len(key.shape) != 1:
                    raise ValueError(about._errors.cstring(
                        "WARNING: Only one-dimensional advanced indexing " +
                        "is supported"))
                # Make a recursive call in order to trigger the 'list'-section
                return self.collect_data(data=data, key=[key], copy=copy,
                                         **kwargs)

        # Case 3 : key is a list. This list is interpreted as one-dimensional
        # advanced indexing list.
        elif found == 'indexinglist':
            return self.collect_data_from_list(data=data,
                                               list_key=key,
                                               copy=copy,
                                               **kwargs)

    def collect_data_from_list(self, data, list_key, copy=True, **kwargs):
        if list_key == []:
            raise ValueError(about._errors.cstring(
                "ERROR: key == [] is an unsupported key!"))
        local_list_key = self._advanced_index_decycler(list_key)
        local_result = data[local_list_key]
        global_result = distributed_data_object(
                                            local_data=local_result,
                                            distribution_strategy='freeform',
                                            copy=copy,
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
# TODO: Implement fast check!
#            if not all(result[i] <= result[i + 1]
#                       for i in xrange(len(result) - 1)):
#                raise ValueError(about._errors.cstring(
#                   "ERROR: The first dimemnsion of list_key must be sorted!"))

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
# TODO: Implement fast check!
#            if not all(result[0][i] <= result[0][i + 1]
#                       for i in xrange(len(result[0]) - 1)):
#                raise ValueError(about._errors.cstring(
#                   "ERROR: The first dimemnsion of list_key must be sorted!"))

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

    def collect_data_from_bool(self, data, boolean_key, copy=True, **kwargs):
        local_boolean_key = self.extract_local_data(boolean_key)
        local_result = data[local_boolean_key]
        global_result = distributed_data_object(
                                            local_data=local_result,
                                            distribution_strategy='freeform',
                                            copy=copy,
                                            comm=self.comm)
        return global_result

    def _invert_mpi_data_ordering(self, data):
        data = np.ascontiguousarray(data)

        comm = self.comm
        s = comm.size
        r = comm.rank
        if s == 1:
            return data

        partner = s - 1 - r

        new_shape = comm.sendrecv(sendobj=data.shape,
                                  dest=partner,
                                  source=partner)
        new_data = np.empty(new_shape,
                            dtype=self.dtype)

        comm.Sendrecv(sendbuf=[data, self.mpi_dtype],
                      recvbuf=[new_data, self.mpi_dtype],
                      dest=partner,
                      source=partner)

        return new_data

    def collect_data_from_slices(self, data, slice_objects, copy=True,
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

        local_result = data[local_slice]
        if (first_step is not None) and (first_step < 0):
            local_result = self._invert_mpi_data_ordering(local_result)

        global_result = distributed_data_object(
                                            local_data=local_result,
                                            distribution_strategy='freeform',
                                            copy=copy,
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
                # Check if both d2os have the same slicing
                # If the distributor is exactly the same, extract the data
                if self is data_object.distributor:
                    # Simply take the local data
                    extracted_data = data_object.data
                # If the distributor is not exactly the same, check if the
                # geometry matches if it is a slicing distributor
                # -> comm and local shapes
                elif (isinstance(data_object.distributor,
                                 _slicing_distributor) and
                      (self.comm is data_object.distributor.comm) and
                      (np.all(self.all_local_slices ==
                              data_object.distributor.all_local_slices))):
                        extracted_data = data_object.data

                else:
                    # Case 2: no. All nodes extract their local slice from the
                    # data_object
                    extracted_data =\
                        data_object.get_data(slice(self.local_start,
                                                   self.local_end),
                                             local_keys=True)
                    extracted_data = extracted_data.get_local_data()


#                # Check if the distributor and the comm match
#                # the own ones. Checking equality via 'is' is ok, as the
#                # distributor factory caches simmilar distributors
#                if self is data_object.distributor and\
#                        self.comm is data_object.distributor.comm:
#                    # Case 1: yes. Simply take the local data
#                    extracted_data = data_object.data
#                # If the distributors do not match directly, check
#                else:
#                    # Case 2: no. All nodes extract their local slice from the
#                    # data_object
#                    extracted_data =\
#                        data_object.get_data(slice(self.local_start,
#                                                   self.local_end),
#                                             local_keys=True)
#                    extracted_data = extracted_data.get_local_data()
#
#

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

    def unique(self, data):
        # if the size of the MPI communicator is equal to 1, the
        # reduce operator will not be applied. -> Cover this case directly.
        comm = self.comm
        size = self.comm.size
        rank = self.comm.rank
        if size == 1:
            global_unique_data = np.unique(data)
        else:
            local_unique_data = np.unique(data)
            local_data_length = np.array([local_unique_data.shape[0]])
            local_data_length_list = np.empty(size, dtype=np.int)
            comm.Allgather([local_data_length, MPI.INT],
                           [local_data_length_list, MPI.INT])

            global_unique_data = np.array([], dtype=self.dtype)
            for i in xrange(size):
                # broadcast data to the other nodes
                # prepare the recv array
                if rank != i:
                    work_shape = local_data_length_list[i]
                    work_array = np.empty(work_shape, dtype=self.dtype)
                else:
                    work_array = local_unique_data
                # do the actual broadcasting
                comm.Bcast([work_array, self.mpi_dtype], root=i)
                global_unique_data = np.unique(
                                        np.concatenate([work_array.flatten(),
                                                        global_unique_data]))
        return global_unique_data

    def bincount(self, local_data, local_weights, minlength):
        if local_weights is None:
            result_dtype = np.int
        else:
            result_dtype = np.float

        local_counts = np.bincount(local_data,
                                   weights=local_weights,
                                   minlength=minlength)

        # cast the local_counts to the right dtype while avoiding copying
        local_counts = np.array(local_counts, copy=False, dtype=result_dtype)
        global_counts = np.empty_like(local_counts)
        self._Allreduce_helper(local_counts,
                               global_counts,
                               MPI.SUM)
        return global_counts

    def cumsum(self, parent, axis):
        data = parent.data
        # compute the local np.cumsum
        local_cumsum = np.cumsum(data, axis=axis)
        if axis is None or axis == 0:
            # communicate the highest value from node to node
            rank = self.comm.rank
            if local_cumsum.shape[0] == 0:
                local_shift = np.zeros((), dtype=local_cumsum.dtype)
            else:
                local_shift = local_cumsum[-1]
            local_shift_list = self.comm.allgather(local_shift)
            local_sum_of_shift = np.sum(local_shift_list[:rank],
                                        axis=0)
            local_cumsum += local_sum_of_shift

        # create the return d2o
        if axis is None:
            # try to preserve the distribution_strategy
            flat_global_shape = (self.global_dim, )
            flat_local_shape = np.shape(local_cumsum)
            result_d2o = parent.copy_empty(global_shape=flat_global_shape,
                                           local_shape=flat_local_shape)
            # check if the original distribution strategy yielded a suitable
            # local_shape
            if result_d2o.local_shape != flat_local_shape:
                # if it does not fit, construct a freeform d2o
                result_d2o = parent.copy_empty(
                                            global_shape=flat_global_shape,
                                            local_shape=flat_local_shape,
                                            distribution_strategy='freeform')
        else:
            result_d2o = parent.copy_empty()

        result_d2o.set_local_data(local_cumsum, copy=False)

        return result_d2o

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
                if x[i] == -1:
                    result += [slice(-1, None)]
                else:
                    result += [slice(x[i], x[i] + 1), ]
                sliceified += [True, ]

        return (tuple(result), sliceified)

    def _enfold(self, in_data, sliceified):
        # TODO: Implement a reshape functionality in order to avoid this
        # low level mess!!

        if isinstance(in_data, distributed_data_object):
            local_data = in_data.get_local_data(copy=False)
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
                # Case 1: The in_data d2o has more than one dimension
                if len(in_data.shape) > 1 and \
                  in_data.distribution_strategy in STRATEGIES['slicing']:
                    local_in_data = in_data.get_local_data(copy=False)
                    local_has_data = (np.prod(local_in_data.shape) != 0)
                    local_has_data_list = np.array(
                                       self.comm.allgather(local_has_data))
                    nodes_with_data = np.where(local_has_data_list)[0]
                    if np.shape(nodes_with_data)[0] > 1:
                        raise ValueError(
                            "ERROR: scalar index on first dimension, " +
                            "but more than one node has data!")
                    elif np.shape(nodes_with_data)[0] == 1:
                        node_with_data = nodes_with_data[0]
                    else:
                        node_with_data = -1

                    if node_with_data == -1:
                        broadcasted_shape = (0,) * len(temp_local_shape)
                    else:
                        broadcasted_shape = self.comm.bcast(
                                                    temp_local_shape,
                                                    root=node_with_data)
                    if self.comm.rank != node_with_data:
                        temp_local_shape = np.array(broadcasted_shape)
                        temp_local_shape[0] = 0
                        temp_local_shape = tuple(temp_local_shape)

                # Case 2: The in_data d2o is only onedimensional
                else:
                    # The data contained in the d2o must be stored on one
                    # single node at the end. Hence it is ok to consolidate
                    # the data and to make a recursive call.
                    temp_data = in_data.get_full_data()
                    return self._enfold(temp_data, sliceified)

            if in_data.distribution_strategy in STRATEGIES['global']:
                new_data = in_data.copy_empty(global_shape=temp_global_shape)
                new_data.set_local_data(local_data, copy=False)
            elif in_data.distribution_strategy in STRATEGIES['local']:
                reshaped_data = local_data.reshape(temp_local_shape)
                new_data = distributed_data_object(
                           local_data=reshaped_data,
                           distribution_strategy=in_data.distribution_strategy,
                           copy=False,
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
                                           copy=False,
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

    def get_iter(self, d2o):
        return d2o_slicing_iter(d2o)


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

    def _Allreduce_helper(self, sendbuf, recvbuf, op):
        recvbuf[:] = sendbuf
        return recvbuf

    def contraction_helper(self, parent, function, allow_empty_contractions,
                           axis=None, **kwargs):
        if axis == ():
            return parent.copy()

        local_result = function(parent.data, axis=axis, **kwargs)

        if isinstance(local_result, np.ndarray):
            result_object = parent.copy_empty(global_shape=local_result.shape,
                                              dtype=local_result.dtype)
            result_object.set_local_data(local_result, copy=False)
        else:
            result_object = local_result

        return result_object

    def distribute_data(self, data, alias=None, path=None, copy=True,
                        **kwargs):
        if 'h5py' in gdi and alias is not None:
            data = self.load_data(alias=alias, path=path)

        if data is None:
            return np.empty(self.global_shape, dtype=self.dtype)
        elif isinstance(data, distributed_data_object):
            new_data = data.get_full_data()
        elif isinstance(data, np.ndarray):
            new_data = data
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
            update = data_update[from_key].get_full_data()
            update = update.astype(self.dtype,
                                   copy=False)
        else:
            if isinstance(from_key, distributed_data_object):
                from_key = from_key.get_full_data()
            elif isinstance(from_key, list):
                try:
                    from_key = [item.get_full_data() for item in from_key]
                except(AttributeError):
                    pass
            update = np.array(data_update,
                              copy=copy)[from_key]
            update = update.astype(self.dtype,
                                   copy=False)
        data[to_key] = update

    def collect_data(self, data, key, local_keys=False, **kwargs):
        if isinstance(key, distributed_data_object):
            key = key.get_full_data()
        elif isinstance(key, list):
            try:
                key = [item.get_full_data() for item in key]
            except(AttributeError):
                pass

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

    def unique(self, data):
        return np.unique(data)

    def bincount(self, local_data, local_weights, minlength):
        counts = np.bincount(local_data,
                             weights=local_weights,
                             minlength=minlength)
        return counts

    def cumsum(self, parent, axis):
        data = parent.data
        # compute the local results from np.cumsum
        local_cumsum = np.cumsum(data, axis=axis)
        result_d2o = parent.copy_empty(global_shape=local_cumsum.shape)
        result_d2o.set_local_data(local_cumsum, copy=False)
        return result_d2o

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

    def get_iter(self, d2o):
        return d2o_not_iter(d2o)
