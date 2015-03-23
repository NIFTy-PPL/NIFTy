# -*- coding: utf-8 -*-

##initialize the 'found-packages'-dictionary 
found = {}

import numpy as np
import nifty_core

try:
    from mpi4py import MPI
    found[MPI] = True
except(ImportError): 
#    from mpi4py_dummy import MPI
    found[MPI] = False

try:
    import pyfftw
    found['pyfftw'] = True
except(ImportError):       
    found['pyfftw'] = False

try:
    import h5py
    found['h5py'] = True
    found['h5py_parallel'] = h5py.get_config().mpi
except(ImportError):
    found['h5py'] = False
    found['h5py_parallel'] = False

   


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
    def __init__(self,  global_data=None, global_shape=None, dtype=None, distribution_strategy='fftw', *args, **kwargs):
        if global_data != None:
            global_data_input = np.array(global_data, copy=False)
        else:
            global_data_input = None
            
        self.distributor = self._get_distributor(distribution_strategy)(global_data=global_data_input, global_shape=global_shape, dtype=dtype, *args, **kwargs)
        self.set_full_data(data=global_data_input, *args, **kwargs)
        
        self.distribution_strategy = distribution_strategy
        self.dtype = self.distributor.dtype
        self.shape = self.distributor.global_shape
        
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return '<distributed_data_object>\n'+self.data.__repr__()
    
    def __neg__(self):
        temp_d2o = distributed_data_object(global_shape=self.shape, 
                                           dtype=self.dtype,
                                           distribution_strategy=self.distribution_strategy)
        temp_d2o.set_local_data(data = self.get_local_data().__neg__()) 
        return temp_d2o
    
            
    def __builtin_helper__(self, operator, other):
        temp_d2o = distributed_data_object(global_shape=self.shape, 
                                           dtype=self.dtype, 
                                           distribution_strategy=self.distribution_strategy)
        if isinstance(other, distributed_data_object):        
            temp_data = operator(other.get_local_data())
        else:
            temp_data = operator(other)
        temp_d2o.set_local_data(data=temp_data)
        return temp_d2o

    def __add__(self, other):
        return self.__builtin_helper__(self.get_local_data().__add__, other)

    def __radd__(self, other):
        return self.__builtin_helper__(self.get_local_data().__radd__, other)
    
    def __sub__(self, other):
        return self.__builtin_helper__(self.get_local_data().__sub__, other)
    
    def __rsub__(self, other):
        return self.__builtin_helper__(self.get_local_data().__rsub__, other)
    
    def __isub__(self, other):
        return self.__builtin_helper__(self.get_local_data().__isub__, other)
        
    def __div__(self, other):
        return self.__builtin_helper__(self.get_local_data().__div__, other)
    
    def __rdiv__(self, other):
        return self.__builtin_helper__(self.get_local_data().__rdiv__, other)

    def __floordiv__(self, other):
        return self.__builtin_helper__(self.get_local_data().__floordiv__, other)
    
    def __rfloordiv__(self, other):
        return self.__builtin_helper__(self.get_local_data().__rfloordiv__, other)
    
    def __mul__(self, other):
        return self.__builtin_helper__(self.get_local_data().__mul__, other)
    
    def __rmul__(self, other):
        return self.__builtin_helper__(self.get_local_data().__rmul__, other)

    def __imul__(self, other):
        return self.__builtin_helper__(self.get_local_data().__imul__, other)
    
    def __pow__(self, other):
        return self.__builtin_helper__(self.get_local_data().__pow__, other)
 
    def __rpow__(self, other):
        return self.__builtin_helper__(self.get_local_data().__rpow__, other)

    def __ipow__(self, other):
        return self.__builtin_helper__(self.get_local_data().__ipow__, other)

    def __getitem__(self, key):
        return self.get_data(key)
    
    def __setitem__(self, key, data):
        self.set_data(data, key)
        
    def set_local_data(self, data):
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
        self.data = np.array(data).astype(self.dtype, copy=False)
    
    def set_data(self, data, key, *args, **kwargs):
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
        (slices, sliceified) = self.__sliceify__(key)        
        self.distributor.disperse_data(self.data, self.__enfold__(data, sliceified), slices, *args, **kwargs)        
    
    def set_full_data(self, data, *args, **kwargs):
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
        self.data = self.distributor.distribute_data(data=data, *args, **kwargs)
    

    def get_local_data(self, key=(slice(None),)):
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
        return self.data[key]        
        
    def get_data(self, key, *args, **kwargs):
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
        (slices, sliceified) = self.__sliceify__(key)        
        result= self.distributor.collect_data(self.data, slices, *args, **kwargs)        
        return self.__defold__(result, sliceified)
        
    
    
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

        return self.distributor.consolidate_data(self.data, target_rank)

  
    def _get_distributor(self, distribution_strategy):
        '''
            Comments:
              - The distributor's get_data and set_data functions MUST be 
                supplied with a tuple of slice objects. In case that there was 
                a direct integer involved, the unfolding will be done by the
                helper functions __sliceify__, __enfold__ and __unfold__.
        '''
        
        distributor_dict={
            'fftw':     _fftw_distributor,
            'not':      _not_distributor
        }
        if not distributor_dict.has_key(distribution_strategy):
            raise TypeError(nifty_core.about._errors.cstring("ERROR: Unknown distribution strategy supplied."))
        return distributor_dict[distribution_strategy]
      
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
           
    def __sliceify__(self, inp):
        sliceified = []
        result = []
        if isinstance(inp, tuple):
            x = inp
        else:
            x = (inp, )
        
        for i in range(len(x)):
            if isinstance(x[i], slice):
                result += [x[i], ]
                sliceified += [False, ]
            else:
                result += [slice(x[i], x[i]+1), ]
                sliceified += [True, ]
    
        return (tuple(result), sliceified)
                
                
    def __enfold__(self, in_data, sliceified):
        data = np.array(in_data, copy=False)    
        temp_shape = ()
        j=0
        for i in sliceified:
            if i == True:
                temp_shape += (1,)
            else:
                temp_shape += (data.shape[j],)
                j += 1
        ## take into account that the sliceified tuple may be too short, because 
        ## of a non-exaustive list of slices
        for i in range(len(data.shape)-j):
            temp_shape += (data.shape[j],)
            j += 1
            
        return data.reshape(temp_shape)
    
    def __defold__(self, data, sliceified):
        temp_slice = ()
        for i in sliceified:
            if i == True:
                temp_slice += (0,)
            else:
                temp_slice += (slice(None),)
        return data[temp_slice]

    

   
class _fftw_distributor(object):
    def __init__(self, global_data=None, global_shape=None, dtype=None, comm=MPI.COMM_WORLD, alias=None, path=None):
        
        if alias != None:
            file_path = path if path != None else alias 
            if found['h5py_parallel']:
                f = h5py.File(file_path, 'r', driver='mpio', comm=comm)
            else:
                f= h5py.File(file_path, 'r')        
            dset = f[alias]        

        
        if comm.rank == 0:        
            if alias != None:
                self.global_shape = dset.shape
            else:                
                if global_data == None:
                    if global_shape == None:
                        raise TypeError(nifty_core.about._errors.cstring("ERROR: Neither data nor shape supplied!"))
                    else:
                        self.global_shape = global_shape
                else:
                    self.global_shape = global_data.shape


        else:
            self.global_shape = None
        
        
        self.global_shape = comm.bcast(self.global_shape, root = 0)
        self.global_shape = tuple(self.global_shape)
        
        if comm.rank == 0:        
            if alias != None:
                self.dtype = dset.dtype.type
            else:    
                if dtype != None:        
                    self.dtype = dtype
                elif global_data != None:
                    self.dtype = np.array(global_data).dtype.type
                else:
                    raise TypeError(nifty_core.about._errors.cstring("ERROR: Failed setting datatype. Neither data, nor datatype supplied."))
        else:
            self.dtype=None
        
        self.dtype = comm.bcast(self.dtype, root=0)
        if alias != None:        
            f.close()        
        
        self._my_dtype_converter = dtype_converter()
        
        if not self._my_dtype_converter.known_np_Q(self.dtype):
            raise TypeError(nifty_core.about._errors.cstring("ERROR: The data type is not known to mpi4py."))

        self.mpi_dtype  = self._my_dtype_converter.to_mpi(self.dtype)
        
        self._local_size = pyfftw.local_size(self.global_shape)
        self.local_start = self._local_size[2]
        self.local_end = self.local_start + self._local_size[1]
        self.local_length = self.local_end-self.local_start        
        self.local_shape = (self.local_length,) + tuple(self.global_shape[1:])
        self.local_dim = np.product(self.local_shape)
        self.local_dim_list = np.empty(comm.size, dtype=np.int)
        comm.Allgather([np.array(self.local_dim,dtype=np.int), MPI.INT], [self.local_dim_list, MPI.INT])
        self.local_dim_offset = np.sum(self.local_dim_list[0:comm.rank])
        
        self.local_slice = np.array([self.local_start, self.local_end, self.local_length, self.local_dim, self.local_dim_offset], dtype=np.int)
        ## collect all local_slices 
        ## [start, stop, length=stop-start, dimension, dimension_offset]
        self.all_local_slices = np.empty((comm.size,5),dtype=np.int)
        comm.Allgather([np.array((self.local_slice,),dtype=np.int), MPI.INT], [self.all_local_slices, MPI.INT])
        
        
    def distribute_data(self, data=None, comm = MPI.COMM_WORLD, alias=None, path=None):
        '''
        distribute data checks 
        - whether the data is located on all nodes or only on node 0
        - that the shape of 'data' matches the global_shape
        '''
        if data == None and found['h5py']:
            try: 
                file_path = path if path != None else alias 
                if found['h5py_parallel']:
                    f = h5py.File(file_path, 'r', driver='mpio', comm=comm)
                else:
                    f= h5py.File(file_path, 'r')        
                dset = f[alias]
                if dset.shape == self.global_shape and dset.dtype.type == self.dtype:
                    temp_data = dset[self.local_start:self.local_end]
                    f.close()
                    return temp_data
                else:
                    raise TypeError(nifty_core.about._errors.cstring("ERROR: Input data has the wrong shape or wrong dtype!"))                 
            except(IOError, AttributeError):
                pass
            
        rank = comm.Get_rank()
        size = comm.Get_size()        
        local_data_available_Q = np.array((int(data != None), ))
        data_available_Q = np.empty(size,dtype=int)
        comm.Allgather([local_data_available_Q, MPI.INT], [data_available_Q, MPI.INT])
        ## if all nodes got data, we assume that it is the right data and 
        ## store it individually. If not, take the data on node 0 and scatter it.
        if np.all(data_available_Q):
            return data[self.local_start:self.local_end].astype(self.dtype, copy=False)
        else:
            if data == None:
                data = np.empty(self.global_shape)            
            if rank == 0:
                if np.all(data.shape != self.global_shape):
                    raise TypeError(nifty_core.about._errors.cstring("ERROR: Input data has the wrong shape!"))
            ## Scatter the data!            
            _scattered_data = np.zeros(self.local_shape, dtype = self.dtype)
            _dim_list = self.all_local_slices[:,3]
            _dim_offset_list = self.all_local_slices[:,4]
            #comm.Scatterv([data.astype(np.float64, copy=False), _dim_list, _dim_offset_list, MPI.DOUBLE], [_scattered_data, MPI.DOUBLE], root=0)
            comm.Scatterv([data, _dim_list, _dim_offset_list, self.mpi_dtype], [_scattered_data, self.mpi_dtype], root=0)
            return _scattered_data
        return None
    
    def _disperse_data_primitive(self, data, data_update, slice_objects, source_rank='all', comm=MPI.COMM_WORLD):
        ## compute the part of the slice which is relevant for the individual node      
        localized_start, localized_stop = self._backshift_and_decycle(
            slice_objects[0], self.local_start)
        local_slice = (slice(localized_start,localized_stop,slice_objects[0].step),) + slice_objects[1:]
        
        ## compute the parameter sets and list for the data splitting
        local_slice_shape = data[local_slice].shape        
        local_affected_data_length = local_slice_shape[0]
        local_affected_data_length_list=np.empty(comm.size, dtype=np.int)        
        comm.Allgather([np.array(local_affected_data_length, dtype=np.int), MPI.INT], [local_affected_data_length_list, MPI.INT])        
        local_affected_data_length_offset_list = np.append([0],np.cumsum(local_affected_data_length_list)[:-1])
        
        
        if source_rank == 'all':
            ## only take the relevant part out of data_update and plug it into 
            ## data[local_slice]
            r = comm.rank
            o = local_affected_data_length_offset_list
            l = local_affected_data_length
            update_slice = (slice(o[r], o[r]+l),) 
            data[local_slice] = np.array(data_update[update_slice], copy=False).astype(self.dtype)
            
        else:
            ## Scatterv the relevant part from the source_rank to the others 
            ## and plug it into data[local_slice]
            local_affected_data_dim_list= np.array(local_affected_data_length_list) * np.product(local_slice_shape[1:])                    
            local_affected_data_dim_offset_list = np.append([0],np.cumsum(local_affected_data_dim_list)[:-1])
            local_dispersed_data = np.zeros(local_slice_shape, dtype=self.dtype)
            comm.Scatterv([np.array(data_update, copy=False).astype(self.dtype), local_affected_data_dim_list, local_affected_data_dim_offset_list, self.mpi_dtype],
                          [local_dispersed_data, self.mpi_dtype], 
                          root=source_rank)                            
            print (comm.rank, local_dispersed_data, local_slice)
            data[local_slice] = local_dispersed_data
        return None
        
    
    
    def disperse_data(self, data, data_update, slice_objects, comm=MPI.COMM_WORLD):
        
        slice_objects_list = comm.allgather(slice_objects)
        ## check if all slices are the same. 
        if all(x == slice_objects_list[0] for x in slice_objects_list):
            ## in this case, the _disperse_data_primitive can simply be called 
            ##with target_rank = 'all'
            self._disperse_data_primitive(data=data, data_update=data_update, slice_objects=slice_objects, source_rank='all', comm=comm)
        ## if the different nodes got different slices, disperse the data individually
        else:
            i = 0        
            for temp_slices in slice_objects_list:
                ## make the collect_data call on all nodes            
                self._disperse_data_primitive(data=data, data_update=data_update, slice_objects=temp_slices, source_rank=i, comm=comm)
                i += 1
                 
        
    def _collect_data_primitive(self, data, slice_objects, target_rank='all', comm=MPI.COMM_WORLD):
        localized_start, localized_stop = self._backshift_and_decycle(
            slice_objects[0], self.local_start)
        local_slice = (slice(localized_start,localized_stop,slice_objects[0].step),)+slice_objects[1:]
        local_collected_data = np.ascontiguousarray(data[local_slice])

        local_collected_data_length = local_collected_data.shape[0]
        local_collected_data_length_list=np.empty(comm.size, dtype=np.int)        
        comm.Allgather([np.array(local_collected_data_length, dtype=np.int), MPI.INT], [local_collected_data_length_list, MPI.INT])        
             
        collected_data_length = np.sum(local_collected_data_length_list) 
        collected_data_shape = (collected_data_length,)+local_collected_data.shape[1:]
        local_collected_data_dim_list= np.array(local_collected_data_length_list) * np.product(local_collected_data.shape[1:])        
        
        local_collected_data_dim_offset_list = np.append([0],np.cumsum(local_collected_data_dim_list)[:-1])
        collected_data = np.empty(collected_data_shape, dtype=self.dtype)

        if target_rank == 'all':
            comm.Allgatherv([local_collected_data, self.mpi_dtype], 
                         [collected_data, local_collected_data_dim_list, local_collected_data_dim_offset_list, self.mpi_dtype])                
        else:
            comm.Gatherv([local_collected_data, self.mpi_dtype], 
                         [collected_data, local_collected_data_dim_list, local_collected_data_dim_offset_list, self.mpi_dtype], root=target_rank)                            
        return collected_data

    def collect_data(self, data, slice_objects, comm=MPI.COMM_WORLD):
        slice_objects_list = comm.allgather(slice_objects)
        ## check if all slices are the same. 
        if all(x == slice_objects_list[0] for x in slice_objects_list):
            ## in this case, the _collect_data_primitive can simply be called 
            ##with target_rank = 'all'
            return self._collect_data_primitive(data=data, slice_objects=slice_objects, target_rank='all', comm=comm)
        
        ## if the different nodes got different slices, collect the data individually
        i = 0        
        for temp_slices in slice_objects_list:
            ## make the collect_data call on all nodes            
            temp_data = self._collect_data_primitive(data=data, slice_objects=temp_slices, target_rank=i, comm=comm)
            ## save the result only on the pulling node            
            if comm.rank == i:
                individual_data = temp_data
            i += 1
        return individual_data
        
    
    def _backshift_and_decycle(self, slice_object, shift):
        ## initialize the step and step_size_compensation
        if slice_object.step == None:
            step = 1
        else:
            step = slice_object.step
            
        ## calculate the start index
        if slice_object.start == None:
            local_start = (-shift)%step ## step size compensation
        else:
            local_start = slice_object.start - shift
            ## if the local_start is negative, pull it up to zero
            local_start = local_start%step if local_start < 0 else local_start
        ## calculate the stop index
        if slice_object.stop == None:
            local_stop = None
        else:
            local_stop = slice_object.stop - shift
            ## if local_stop is negative, pull it up to zero
            local_stop = 0 if local_stop < 0 else local_stop
                    ## Note: if start or stop are greater than the array length,
        ## numpy will automatically cut the index value down into the 
        ## array's range 
        return local_start, local_stop        
        
    def consolidate_data(self, data, target_rank='all', comm = MPI.COMM_WORLD):
        _gathered_data = np.empty(self.global_shape, dtype=self.dtype)
        _dim_list = self.all_local_slices[:,3]
        _dim_offset_list = self.all_local_slices[:,4]
        if target_rank == 'all':
            comm.Allgatherv([data, self.mpi_dtype], 
                         [_gathered_data, _dim_list, _dim_offset_list, self.mpi_dtype])                
        else:
            comm.Gatherv([data, self.mpi_dtype], 
                         [_gathered_data, _dim_list, _dim_offset_list, self.mpi_dtype],
                         root=target_rank)
        return _gathered_data
    
    if found['h5py']:
        def save_data(self, data, alias, path=None, overwriteQ=True, comm=MPI.COMM_WORLD):
            ## if no path and therefore no filename was given, use the alias as filename        
            use_path = alias if path==None else path
            
            ## create the file-handle
            if found['h5py_parallel']:
                f = h5py.File(use_path, 'a', driver='mpio', comm=comm)
            else:
                f= h5py.File(use_path, 'a')
            ## check if dataset with name == alias already exists
            try: 
                f[alias]
                if overwriteQ == False: #if yes, and overwriteQ is set to False, raise an Error
                    raise KeyError(nifty_core.about._errors.cstring("ERROR: overwriteQ == False, but alias already in use!"))
                else: # if yes, remove the existing dataset
                    del f[alias]
            except(KeyError):
                pass
            
            ## create dataset
            dset = f.create_dataset(alias, shape=self.global_shape, dtype=self.dtype)
            ## write the data
            dset[self.local_start:self.local_end] = data
            ## close the file
            f.close()
        
        def load_data(self, alias, path):
            ## create the file-handle
            if found['h5py_parallel']:
                f = h5py.File(path, 'r', driver='mpio', comm=comm)
            else:
                f= h5py.File(path, 'r')        
            dset = f[alias]        
            ## check shape
            if dset.shape != self.global_shape:
                raise TypeError(nifty_core.about._errors.cstring("ERROR: The shape of the given dataset does not match the distributed_data_object."))
            ## check dtype
            if dset.dtype.type != self.dtype:
                raise TypeError(nifty_core.about._errors.cstring("ERROR: The datatype of the given dataset does not match the distributed_data_object."))
            ## if everything seems to fit, load the data
            data = dset[self.local_start:self.local_end]
            ## close the file
            f.close()
            return data
    else:
        def save_data(self, *args, **kwargs):
            raise ImportError(nifty_core.about._errors.cstring("ERROR: h5py was not imported")) 
        def load_data(self, *args, **kwargs):
            raise ImportError(nifty_core.about._errors.cstring("ERROR: h5py was not imported")) 
        
        
        
        

class _not_distributor(object):
    def __init__(self, global_data=None, global_shape=None, dtype=None, *args,  **kwargs):
        if dtype != None:        
            self.dtype = dtype
        elif global_data != None:
            self.dtype = np.array(global_data).dtype.type
            
        if global_data != None:
            self.global_shape = np.array(global_data).shape
        elif global_shape != None:
            self.global_shape = global_shape
        else:
            raise TypeError(nifty_core.about._errors.cstring("ERROR: Neither data nor shape supplied!")) 
    def distribute_data(self, data, **kwargs):
        return np.array(data).astype(self.dtype, copy=False)
    
    def disperse_data(self, data, data_update, key):
        
        data[key] = np.array(data_update, copy=False).astype(self.dtype)
                     
    def collect_data(self, data, slice_object, *args,  **kwargs):
        return data[slice_object]
        
    def consolidate_data(self, data, *args, **kwargs):
        return data





class dtype_converter:
    """
        NIFTY class for dtype conversion between python/numpy dtypes and MPI
        dtypes.
    """
    
    def __init__(self):
        pre_dict = [
                    #[, MPI_CHAR],
                    #[, MPI_SIGNED_CHAR],
                    #[, MPI_UNSIGNED_CHAR],                                        
                    [np.int16, MPI.SHORT],
                    [np.uint16, MPI.UNSIGNED_SHORT],
                    [np.uint32, MPI.UNSIGNED_INT],                    
                    [np.int, MPI.INT],                    
                    [np.int32, MPI.INT],
                    [np.int64, MPI.LONG],
                    [np.uint64, MPI.UNSIGNED_LONG],
                    [np.int64, MPI.LONG_LONG],
                    [np.uint64, MPI.UNSIGNED_LONG_LONG],
                    [np.float32, MPI.FLOAT],
                    [np.float, MPI.DOUBLE],
                    [np.float64, MPI.DOUBLE],
                    [np.float128, MPI.LONG_DOUBLE],
                    [np.complex64, MPI.COMPLEX],
                    [np.complex, MPI.DOUBLE_COMPLEX],
                    [np.complex128, MPI.DOUBLE_COMPLEX]]
                    
        to_mpi_pre_dict = np.array(pre_dict)
        to_mpi_pre_dict[:,0] = map(self.dictionize_np, to_mpi_pre_dict[:,0])
        self._to_mpi_dict = dict(to_mpi_pre_dict)
        
        to_np_pre_dict = np.array(pre_dict)[:,::-1]
        to_np_pre_dict[:,0] = map(self.dictionize_mpi, to_np_pre_dict[:,0])
        self._to_np_dict = dict(to_np_pre_dict)

    def dictionize_np(self, x):
        return frozenset(x.__dict__.items())
        
    def dictionize_mpi(self, x):
        return x.name
    
    def to_mpi(self, dtype):
        return self._to_mpi_dict[self.dictionize_np(dtype)]

    def to_np(self, dtype):
        return self._to_np_dict[self.dictionize_mpi(dtype)]
    
    def known_mpi_Q(self, dtype):
        return self._to_np_dict.has_key(self.dictionize_mpi(dtype))
    
    def known_np_Q(self, dtype):
        return self._to_mpi_dict.has_key(self.dictionize_np(dtype))

class test(object):
    def __init__(self,x=None):
        self.x =x
    @property
    def val(self):
        return self.x
    
    @val.setter
    def val(self, x):
        self.x = x


if __name__ == '__main__':    
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        x = np.arange(48).reshape((12,4)).astype(np.float64)
        print x
        #x = np.arange(3)
    else:
        x = None
    obj = distributed_data_object(global_data=x, distribution_strategy='fftw')
    
    
    #obj.load('myalias', 'mpitest.hdf5')
    if MPI.COMM_WORLD.rank==0:
        print ('rank', rank, vars(obj.distributor))
    MPI.COMM_WORLD.Barrier()
    print ('rank', rank, vars(obj))
    
    MPI.COMM_WORLD.Barrier()
    temp_erg =obj.get_full_data(target_rank='all')
    print ('rank', rank, 'full data', temp_erg, temp_erg.shape)
    
    MPI.COMM_WORLD.Barrier()
    if rank == 0:    
        print ('erwuenscht', x[slice(1,10,2)])
    sl = slice(1,2+rank,1)
    print ('slice', rank, sl, obj[sl,2])
    print obj[1:5:2,1:3]
    if rank == 0:
        sl = (slice(1,9,2), slice(1,5,2))
        d = [[111, 222],[333,444],[111, 222],[333,444]]
    else:
        sl = (slice(6,10,2), slice(1,5,2))
        d = [[555, 666],[777,888]]
    obj[sl] = d
    print obj.get_full_data()    
    
   
