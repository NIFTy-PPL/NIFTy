# -*- coding: utf-8 -*-
## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2015 Max-Planck-Society
##
## Author: Theo Steininger
## Project homepage: <http://www.mpa-garching.mpg.de/ift/nifty/>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
## See the GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program. If not, see <http://www.gnu.org/licenses/>.



##initialize the 'found-packages'-dictionary 
found = {}

import numpy as np
from nifty_about import about

try:
    from mpi4py import MPI
    found[MPI] = True
except(ImportError): 
    import mpi_dummy as MPI
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
    def __init__(self,  global_data=None, global_shape=None, dtype=None, 
                 distribution_strategy='fftw', hermitian=False, 
                 *args, **kwargs):
        if global_data != None:
            if np.isscalar(global_data):
                global_data_input = None
                dtype = np.array(global_data).dtype.type
            else:
                global_data_input = np.array(global_data, copy=True, order='C')
        else:
            global_data_input = None

        self.hermitian = False

        self.distributor = self._get_distributor(distribution_strategy)(
                            global_data=global_data_input, 
                            global_shape=global_shape, 
                            dtype=dtype, **kwargs)

        self.set_full_data(data=global_data_input, hermitian=hermitian, 
                           **kwargs)
        
            
        self.distribution_strategy = distribution_strategy
        self.dtype = self.distributor.dtype
        self.shape = self.distributor.global_shape
        
        self.init_args = args 
        self.init_kwargs = kwargs
        
        ## If the input data was a scalar, set the whole array to this value
        if global_data != None and np.isscalar(global_data):
            temp = np.empty(self.distributor.local_shape)
            temp.fill(global_data)
            self.set_local_data(temp)
            self.hermitian = True
        
    def copy(self, dtype=None, distribution_strategy=None, **kwargs):
        temp_d2o = self.copy_empty(dtype=dtype, 
                                   distribution_strategy=distribution_strategy, 
                                   **kwargs)     
        if distribution_strategy == None or \
            distribution_strategy == self.distribution_strategy:
            temp_d2o.set_local_data(self.get_local_data(), copy=True)
        else:
            temp_d2o.set_full_data(self.get_full_data())
        temp_d2o.hermitian = self.hermitian
        return temp_d2o
    
    def copy_empty(self, global_shape=None, dtype=None, 
                   distribution_strategy=None, **kwargs):
        if global_shape == None:
            global_shape = self.shape
        if dtype == None:
            dtype = self.dtype
        if distribution_strategy == None:
            distribution_strategy = self.distribution_strategy

        kwargs.update(self.init_kwargs)
        
        temp_d2o = distributed_data_object(global_shape=global_shape,
                                           dtype=dtype,
                                           distribution_strategy=distribution_strategy,
                                           *self.init_args,
                                           **kwargs)
        return temp_d2o
    
    def apply_scalar_function(self, function, inplace=False, dtype=None):
        if inplace == True:        
            temp = self
        else:
            temp = self.copy_empty(dtype=dtype)

        try: 
            temp.data[:] = function(self.data)
        except:
            temp.data[:] = np.vectorize(function)(self.data)
        
        temp.hermitian = False
        return temp
    
    def apply_generator(self, generator):
        self.set_local_data(generator(self.distributor.local_shape))
        self.hermitian = False
            
    def __str__(self):
        return self.data.__str__()
    
    def __repr__(self):
        return '<distributed_data_object>\n'+self.data.__repr__()
    
    def __eq__(self, other):
        result = self.copy_empty(dtype = np.bool)
        ## Case 1: 'other' is a scalar
        ## -> make point-wise comparison
        if np.isscalar(other):
            result.set_local_data(self.get_local_data(copy = False) == other)
            return result        

        ## Case 2: 'other' is a numpy array or a distributed_data_object
        ## -> extract the local data and make point-wise comparison
        elif isinstance(other, np.ndarray) or\
        isinstance(other, distributed_data_object):
            temp_data = self.distributor.extract_local_data(other)
            result.set_local_data(self.get_local_data(copy=False) == temp_data)
            return result
        
        ## Case 3: 'other' is None
        elif other == None:
            return False
        
        ## Case 4: 'other' is something different
        ## -> make a numpy casting and make a recursion
        else:
            temp_other = np.array(other)
            return self.__eq__(temp_other)
            
            
        
    
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
        temp_d2o.set_local_data(data = self.get_local_data())
        return temp_d2o
        
    def __neg__(self):
        temp_d2o = self.copy_empty()
        temp_d2o.set_local_data(data = self.get_local_data().__neg__()) 
        return temp_d2o
    
    def __abs__(self):
        ## translate complex dtypes
        if self.dtype == np.complex64:
            new_dtype = np.float32
        elif self.dtype == np.complex128:
            new_dtype = np.float64
        elif self.dtype == np.complex:
            new_dtype = np.float
        elif issubclass(self.dtype, np.complexfloating):
            new_dtype = np.float
        else:
            new_dtype = self.dtype
        temp_d2o = self.copy_empty(dtype = new_dtype)
        temp_d2o.set_local_data(data = self.get_local_data().__abs__()) 
        return temp_d2o
            
    def __builtin_helper__(self, operator, other, inplace=False):
        ## Case 1: other is not a scalar
        if not (np.isscalar(other) or np.shape(other) == (1,)):
##            if self.shape != other.shape:            
##                raise AttributeError(about._errors.cstring(
##                    "ERROR: Shapes do not match!")) 
            try:            
                hermitian_Q = other.hermitian
            except(AttributeError):
                hermitian_Q = False
            ## extract the local data from the 'other' object
            temp_data = self.distributor.extract_local_data(other)
            temp_data = operator(temp_data)
            
        ## Case 2: other is a real scalar -> preserve hermitianity
        elif np.isreal(other) or (self.dtype not in (np.complex, np.complex128,
                                                np.complex256)):
            hermitian_Q = self.hermitian
            temp_data = operator(other)
        ## Case 3: other is complex
        else:
            hermitian_Q = False
            temp_data = operator(other)        
        ## write the new data into a new distributed_data_object        
        if inplace == True:
            temp_d2o = self
        else:
            temp_d2o = self.copy_empty()        
        temp_d2o.set_local_data(data=temp_data)
        temp_d2o.hermitian = hermitian_Q
        return temp_d2o
    """
    def __inplace_builtin_helper__(self, operator, other):
        ## Case 1: other is not a scalar
        if not (np.isscalar(other) or np.shape(other) == (1,)):        
            temp_data = self.distributor.extract_local_data(other)
            temp_data = operator(temp_data)
        ## Case 2: other is a real scalar -> preserve hermitianity
        elif np.isreal(other):
            hermitian_Q = self.hermitian
            temp_data = operator(other)
        ## Case 3: other is complex
        else:
            temp_data = operator(other)        
        self.set_local_data(data=temp_data)
        self.hermitian = hermitian_Q
        return self
    """ 
    
    def __add__(self, other):
        return self.__builtin_helper__(self.get_local_data().__add__, other)

    def __radd__(self, other):
        return self.__builtin_helper__(self.get_local_data().__radd__, other)

    def __iadd__(self, other):
        return self.__builtin_helper__(self.get_local_data().__iadd__, 
                                               other,
                                               inplace = True)

    def __sub__(self, other):
        return self.__builtin_helper__(self.get_local_data().__sub__, other)
    
    def __rsub__(self, other):
        return self.__builtin_helper__(self.get_local_data().__rsub__, other)
    
    def __isub__(self, other):
        return self.__builtin_helper__(self.get_local_data().__isub__, 
                                               other,
                                               inplace = True)
        
    def __div__(self, other):
        return self.__builtin_helper__(self.get_local_data().__div__, other)
    
    def __rdiv__(self, other):
        return self.__builtin_helper__(self.get_local_data().__rdiv__, other)

    def __idiv__(self, other):
        return self.__builtin_helper__(self.get_local_data().__idiv__, 
                                               other,
                                               inplace = True)

    def __floordiv__(self, other):
        return self.__builtin_helper__(self.get_local_data().__floordiv__, 
                                       other)    
    def __rfloordiv__(self, other):
        return self.__builtin_helper__(self.get_local_data().__rfloordiv__, 
                                       other)
    def __ifloordiv__(self, other):
        return self.__builtin_helper__(
                    self.get_local_data().__ifloordiv__, other,
                                               inplace = True)
    
    def __mul__(self, other):
        return self.__builtin_helper__(self.get_local_data().__mul__, other)
    
    def __rmul__(self, other):
        return self.__builtin_helper__(self.get_local_data().__rmul__, other)

    def __imul__(self, other):
        return self.__builtin_helper__(self.get_local_data().__imul__, 
                                               other,
                                               inplace = True)

    def __pow__(self, other):
        return self.__builtin_helper__(self.get_local_data().__pow__, other)
 
    def __rpow__(self, other):
        return self.__builtin_helper__(self.get_local_data().__rpow__, other)

    def __ipow__(self, other):
        return self.___builtin_helper__(self.get_local_data().__ipow__, 
                                               other,
                                               inplace = True)
   
    def __len__(self):
        return self.shape[0]
    
    def dim(self):
        return np.prod(self.shape)
        
    def vdot(self, other):
        if isinstance(other, distributed_data_object):        
            other = other.get_local_data()
        local_vdot = np.vdot(self.get_local_data(), other)
        local_vdot_list = self.distributor._allgather(local_vdot)
        global_vdot = np.sum(local_vdot_list)
        return global_vdot
            

    
    def __getitem__(self, key):
        ## Case 1: key is a boolean array.
        ## -> take the local data portion from key, use this for data 
        ## extraction, and then merge the result in a flat numpy array
        if isinstance(key, np.ndarray):
            found = 'ndarray'
            found_boolean = (key.dtype.type == np.bool)
        elif isinstance(key, distributed_data_object):
            found = 'd2o'
            found_boolean = (key.dtype == np.bool)
        else:
            found = 'other'
                
        if (found == 'ndarray' or found == 'd2o') and found_boolean == True:
            ## extract the data of local relevance
            local_bool_array = self.distributor.extract_local_data(key)
            local_results = self.get_local_data(copy=False)[local_bool_array]
            global_results = self.distributor._allgather(local_results)
            global_results = np.concatenate(global_results)
            return global_results            
            
        else:
            return self.get_data(key)
    
    def __setitem__(self, key, data):
        self.set_data(data, key)
        
    def _contraction_helper(self, function, **kwargs):
        local = function(self.data, **kwargs)
        local_list = self.distributor._allgather(local)
        global_ = function(local_list, axis=0)
        return global_
        
    def amin(self, **kwargs):
        return self._contraction_helper(np.amin, **kwargs)

    def nanmin(self, **kwargs):
        return self._contraction_helper(np.nanmin, **kwargs)
        
    def amax(self, **kwargs):
        return self._contraction_helper(np.amax, **kwargs)
    
    def nanmax(self, **kwargs):
        return self._contraction_helper(np.nanmax, **kwargs)
    
    def sum(self, **kwargs):
        return self._contraction_helper(np.sum, **kwargs)

    def prod(self, **kwargs):
        return self._contraction_helper(np.prod, **kwargs)        
        
    def mean(self, power=1):
        ## compute the local means and the weights for the mean-mean. 
        local_mean = np.mean(self.data**power)
        local_weight = np.prod(self.data.shape)
        ## collect the local means and cast the result to a ndarray
        local_mean_weight_list = self.distributor._allgather((local_mean, 
                                                              local_weight))
        local_mean_weight_list =np.array(local_mean_weight_list)   
        ## compute the denominator for the weighted mean-mean                                                           
        global_weight = np.sum(local_mean_weight_list[:,1])
        ## compute the numerator
        numerator = np.sum(local_mean_weight_list[:,0]*\
            local_mean_weight_list[:,1])
        global_mean = numerator/global_weight
        return global_mean

    def var(self):
        mean_of_the_square = self.mean(power=2)
        square_of_the_mean = self.mean()**2
        return mean_of_the_square - square_of_the_mean
    
    def std(self):
        return np.sqrt(self.var())
        
    def _argmin_argmax_flat_helper(self, function):
        local_argmin = function(self.data)
        local_argmin_value = self.data[np.unravel_index(local_argmin, 
                                                        self.data.shape)]
        globalized_local_argmin = self.distributor.globalize_flat_index(local_argmin)                                                       
        local_argmin_list = self.distributor._allgather((local_argmin_value, 
                                                         globalized_local_argmin))
        local_argmin_list = np.array(local_argmin_list, dtype=[('value', int),
                                                               ('index', int)])    
        return local_argmin_list
        
    def argmin_flat(self):
        local_argmin = np.argmin(self.data)
        local_argmin_value = self.data[np.unravel_index(local_argmin, 
                                                        self.data.shape)]
        globalized_local_argmin = self.distributor.globalize_flat_index(local_argmin)                                                       
        local_argmin_list = self.distributor._allgather((local_argmin_value, 
                                                         globalized_local_argmin))
        local_argmin_list = np.array(local_argmin_list, dtype=[('value', int),
                                                               ('index', int)])    
        local_argmin_list = np.sort(local_argmin_list, order=['value', 'index'])        
        return local_argmin_list[0][1]
    
    def argmax_flat(self):
        local_argmax = np.argmax(self.data)
        local_argmax_value = -self.data[np.unravel_index(local_argmax, 
                                                        self.data.shape)]
        globalized_local_argmax = self.distributor.globalize_flat_index(local_argmax)                                                       
        local_argmax_list = self.distributor._allgather((local_argmax_value, 
                                                         globalized_local_argmax))
        local_argmax_list = np.array(local_argmax_list, dtype=[('value', int),
                                                               ('index', int)])         
        return local_argmax_list[0][1]
        

    def argmin(self):    
        return np.unravel_index(self.argmin_flat(), self.shape)
    
    def argmax(self):
        return np.unravel_index(self.argmax_flat(), self.shape)
    
    def conjugate(self):
        temp_d2o = self.copy_empty()
        temp_data = np.conj(self.get_local_data())
        temp_d2o.set_local_data(temp_data)
        return temp_d2o

    
    def conj(self):
        return self.conjugate()      
        
    def median(self):
        about.warnings.cprint(\
            "WARNING: The current implementation of median is very expensive!")
        median = np.median(self.get_full_data())
        return median
        
    def iscomplex(self):
        temp_d2o = self.copy_empty(dtype=bool)
        temp_d2o.set_local_data(np.iscomplex(self.data))
        return temp_d2o
    
    def isreal(self):
        temp_d2o = self.copy_empty(dtype=bool)
        temp_d2o.set_local_data(np.isreal(self.data))
        return temp_d2o
    
    def is_completely_real(self):
        local_realiness = np.all(self.isreal())
        global_realiness = self.distributor._allgather(local_realiness)
        return np.all(global_realiness)
    
    def set_local_data(self, data, hermitian=False, copy=False):
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
        self.data = np.array(data, dtype=self.dtype, copy=copy, order='C')
    
    def set_data(self, data, key, hermitian=False, *args, **kwargs):
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
        (slices, sliceified) = self.__sliceify__(key)        
        self.distributor.disperse_data(data=self.data, 
                        to_slices = slices,
                        data_update = self.__enfold__(data, sliceified), 
                        *args, **kwargs)        
    
    def set_full_data(self, data, hermitian=False, **kwargs):
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
        self.data = self.distributor.distribute_data(data=data, **kwargs)
    

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
        if copy == True:
            return self.data[key]        
        if copy == False:
            return self.data
        
    def get_data(self, key, **kwargs):
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
        result = self.distributor.collect_data(self.data, slices, **kwargs)        
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

    def inject(self, to_slices=(slice(None),), data=None, 
               from_slices=(slice(None),)):
        if data == None:
            return self
        
        self.distributor.inject(self.data, to_slices, data, from_slices)
        
        
    def _get_distributor(self, distribution_strategy):
        '''
            Comments:
              - The distributor's get_data and set_data functions MUST be 
                supplied with a tuple of slice objects. In case that there was 
                a direct integer involved, the unfolding will be done by the
                helper functions __sliceify__, __enfold__ and __defold__.
        '''
        
        distributor_dict={
            'fftw':     _fftw_distributor,
            'not':      _not_distributor
        }
        if not distributor_dict.has_key(distribution_strategy):
            raise TypeError(about._errors.cstring("ERROR: Unknown distribution strategy supplied."))
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
        elif isinstance(inp, list):
            x = tuple(inp)
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
                if data.shape[j] == 1:
                    j +=1
            else:
                try:
                    temp_shape += (data.shape[j],)
                except(IndexError):
                    temp_shape += (1,)
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
    def __init__(self, global_data=None, global_shape=None, dtype=None, 
                 comm=MPI.COMM_WORLD, alias=None, path=None):
        
        if alias != None:
            file_path = path if path != None else alias 
            if found['h5py_parallel']:
                f = h5py.File(file_path, 'r', driver='mpio', comm=comm)
            else:
                f= h5py.File(file_path, 'r')        
            dset = f[alias]        

        
        if comm.rank == 0:        
            ## Case 1: hdf5 path supplied
            if alias != None:
                self.global_shape = dset.shape
            ## Case 2: no hdf5 path supplied
            else:           
                ## subcase 1: input data is scalar or None
                if global_data == None or np.isscalar(global_data):
                    if global_shape == None:
                        raise TypeError(about._errors.\
                cstring("ERROR: Neither non-scalar data nor shape supplied!"))
                    else:
                        self.global_shape = global_shape
                ## subcase 2: input data is non-scalar 
                ## -> Take the shape of the input data
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
                    raise TypeError(about._errors.\
                    cstring("ERROR: Failed setting datatype. Neither data, "+\
                     "nor datatype supplied."))
        else:
            self.dtype=None
        self.dtype = comm.bcast(self.dtype, root=0)
        if alias != None:        
            f.close()        
        
        self._my_dtype_converter = dtype_converter()
        
        if not self._my_dtype_converter.known_np_Q(self.dtype):
            raise TypeError(about._errors.cstring(\
            "ERROR: The datatype "+str(self.dtype)+" is not known to mpi4py."))

        self.mpi_dtype  = self._my_dtype_converter.to_mpi(self.dtype)
        
        self._local_size = pyfftw.local_size(self.global_shape)
        self.local_start = self._local_size[2]
        self.local_end = self.local_start + self._local_size[1]
        self.local_length = self.local_end-self.local_start        
        self.local_shape = (self.local_length,) + tuple(self.global_shape[1:])
        self.local_dim = np.product(self.local_shape)
        self.local_dim_list = np.empty(comm.size, dtype=np.int)
        comm.Allgather([np.array(self.local_dim,dtype=np.int), MPI.INT],\
            [self.local_dim_list, MPI.INT])
        self.local_dim_offset = np.sum(self.local_dim_list[0:comm.rank])
        
        self.local_slice = np.array([self.local_start, self.local_end,\
            self.local_length, self.local_dim, self.local_dim_offset],\
            dtype=np.int)
        ## collect all local_slices 
        ## [start, stop, length=stop-start, dimension, dimension_offset]
        self.all_local_slices = np.empty((comm.size,5),dtype=np.int)
        comm.Allgather([np.array((self.local_slice,),dtype=np.int), MPI.INT],\
            [self.all_local_slices, MPI.INT])
        
        self.comm = comm
        
    def globalize_flat_index(self, index):
        return int(index)+self.local_dim_offset
        
    def globalize_index(self, index):
        index = np.array(index, dtype=np.int).flatten()
        if index.shape != (len(self.global_shape),):
            raise TypeError(about._errors.cstring("ERROR: Length\
                of index tuple does not match the array's shape!"))                 
        globalized_index = index
        globalized_index[0] = index[0] + self.local_start
        ## ensure that the globalized index list is within the bounds
        global_index_memory = globalized_index
        globalized_index = np.clip(globalized_index, 
                                   -np.array(self.global_shape),
                                    np.array(self.global_shape)-1)
        if np.any(global_index_memory != globalized_index):
            about.warnings.cprint("WARNING: Indices were clipped!")
        globalized_index = tuple(globalized_index)
        return globalized_index
    
    def _allgather(self, thing, comm=None):
        if comm == None:
            comm = self.comm            
        gathered_things = comm.allgather(thing)
        return gathered_things
    
    def distribute_data(self, data=None, comm = None, alias=None,
                        path=None, **kwargs):
        '''
        distribute data checks 
        - whether the data is located on all nodes or only on node 0
        - that the shape of 'data' matches the global_shape
        '''
        if comm == None:
            comm = self.comm            
        rank = comm.Get_rank()
        size = comm.Get_size()        
        local_data_available_Q = np.array((int(data != None), ))
        data_available_Q = np.empty(size,dtype=int)
        comm.Allgather([local_data_available_Q, MPI.INT], 
                       [data_available_Q, MPI.INT])        
        
        if data_available_Q[0]==False and found['h5py']:
            try: 
                file_path = path if path != None else alias 
                if found['h5py_parallel']:
                    f = h5py.File(file_path, 'r', driver='mpio', comm=comm)
                else:
                    f= h5py.File(file_path, 'r')        
                dset = f[alias]
                if dset.shape == self.global_shape and \
                 dset.dtype.type == self.dtype:
                    temp_data = dset[self.local_start:self.local_end]
                    f.close()
                    return temp_data
                else:
                    raise TypeError(about._errors.cstring("ERROR: \
                    Input data has the wrong shape or wrong dtype!"))                 
            except(IOError, AttributeError):
                pass
            
        if np.all(data_available_Q==False):
            return np.empty(self.local_shape, dtype=self.dtype, order='C')
        ## if all nodes got data, we assume that it is the right data and 
        ## store it individually. If not, take the data on node 0 and scatter 
        ## it...
        if np.all(data_available_Q):
            return data[self.local_start:self.local_end].astype(self.dtype,\
                copy=False)    
        ## ... but only if node 0 has actually data!
        elif data_available_Q[0] == False:# or np.all(data_available_Q==False):
            return np.empty(self.local_shape, dtype=self.dtype, order='C')
        
        else:
            if data == None:
                data = np.empty(self.global_shape)            
            if rank == 0:
                if np.all(data.shape != self.global_shape):
                    raise TypeError(about._errors.cstring(\
                        "ERROR: Input data has the wrong shape!"))
            ## Scatter the data!            
            _scattered_data = np.empty(self.local_shape, dtype = self.dtype)
            _dim_list = self.all_local_slices[:,3]
            _dim_offset_list = self.all_local_slices[:,4]
            comm.Scatterv([data, _dim_list, _dim_offset_list, self.mpi_dtype],\
                [_scattered_data, self.mpi_dtype], root=0)
            return _scattered_data
        return None
    
    def _disperse_data_primitive(self, data, to_slices, data_update, 
                                 from_slices, source_rank='all', comm=None):
        if comm == None:
            comm = self.comm            
        ## compute the part of the slice which is relevant for the 
        ## individual node      
        localized_start, localized_stop = self._backshift_and_decycle(
            to_slices[0], self.local_start, self.local_end,\
                self.global_shape[0])
        local_slice = (slice(localized_start, localized_stop,\
                        to_slices[0].step),) + to_slices[1:]
        
        ## compute the parameter sets and list for the data splitting
        local_slice_shape = data[local_slice].shape        
        local_affected_data_length = local_slice_shape[0]
        local_affected_data_length_list=np.empty(comm.size, dtype=np.int)        
        comm.Allgather(\
            [np.array(local_affected_data_length, dtype=np.int), MPI.INT],\
            [local_affected_data_length_list, MPI.INT])        
        local_affected_data_length_offset_list = np.append([0],\
                            np.cumsum(local_affected_data_length_list)[:-1])
        
        
        if source_rank == 'all':
            ## only take the relevant part out of data_update and plug it into 
            ## data[local_slice]
            r = comm.rank
            o = local_affected_data_length_offset_list
            l = local_affected_data_length
            
            ## if the from_slices object is not None, i.e. only a part from
            ## the data source is used, form the update_slice accordingly
            if from_slices == None:
                update_slice = (slice(o[r], o[r]+l),)
            else:
                    
                f_step = from_slices[0].step
                if f_step == None:
                    f_step = 1
                    
                f_direction = np.sign(f_step)

                f_relative_start = from_slices[0].start
                if f_relative_start != None:
                    f_start = f_relative_start + f_direction*o[r]
                else:
                    f_start = None
                    f_relative_start = 0
                    
                f_stop = f_relative_start + f_direction*(o[r]+l*np.abs(f_step))
                if f_stop < 0:
                    f_stop = None


                ## combine the slicing for the first dimension 
                update_slice = (slice(f_start,
                                      f_stop,
                                      f_step),
                                )
                ## add the rest of the from_slicing
                update_slice += from_slices[1:]

            data[local_slice] = np.array(data_update[update_slice],\
                                    copy=False).astype(self.dtype)
            
        else:
            ## Scatterv the relevant part from the source_rank to the others 
            ## and plug it into data[local_slice]
            
            ## if the first slice object has a negative step size, the ordering 
            ## of the Scatterv function must be reversed         
            order = to_slices[0].step
            if order == None:
                order = 1
            else:
                order = np.sign(order)

            local_affected_data_dim_list = \
                np.array(local_affected_data_length_list) *\
                    np.product(local_slice_shape[1:])                    

            local_affected_data_dim_offset_list = np.append([0],\
                np.cumsum(local_affected_data_dim_list[::order])[:-1])[::order]
                
            local_dispersed_data = np.zeros(local_slice_shape,\
                dtype=self.dtype)
            comm.Scatterv(\
                [np.array(data_update[from_slices],copy=False).\
                                                        astype(self.dtype),\
                    local_affected_data_dim_list,\
                    local_affected_data_dim_offset_list, self.mpi_dtype],
                          [local_dispersed_data, self.mpi_dtype], 
                          root=source_rank)                            
            data[local_slice] = local_dispersed_data
        return None
        
    
    
    def disperse_data(self, data, to_slices, data_update, from_slices=None,
                      comm=None, **kwargs):
        if comm == None:
            comm = self.comm            
        to_slices_list = comm.allgather(to_slices)
        ## check if all slices are the same. 
        if all(x == to_slices_list[0] for x in to_slices_list):
            ## in this case, the _disperse_data_primitive can simply be called 
            ##with target_rank = 'all'
            self._disperse_data_primitive(data = data, 
                                          to_slices = to_slices,
                                          data_update=data_update,
                                          from_slices=from_slices, 
                                          source_rank='all', 
                                          comm=comm)
        ## if the different nodes got different slices, disperse the data 
        ## individually
        else:
            i = 0        
            for temp_to_slices in to_slices_list:
                ## make the collect_data call on all nodes            
                self._disperse_data_primitive(data=data,
                                              to_slices=temp_to_slices,
                                              data_update=data_update,
                                              from_slices=from_slices,
                                              source_rank=i, 
                                              comm=comm)
                i += 1
                 
        
    def _collect_data_primitive(self, data, slice_objects, target_rank='all', comm=None):
        if comm == None:
            comm = self.comm            
            
        localized_start, localized_stop = self._backshift_and_decycle(
            slice_objects[0], self.local_start, self.local_end, self.global_shape[0])
        local_slice = (slice(localized_start,localized_stop,slice_objects[0].step),)+slice_objects[1:]
        local_collected_data = np.ascontiguousarray(data[local_slice])

        local_collected_data_length = local_collected_data.shape[0]
        local_collected_data_length_list=np.empty(comm.size, dtype=np.int)        
        comm.Allgather([np.array(local_collected_data_length, dtype=np.int), MPI.INT], [local_collected_data_length_list, MPI.INT])        
             
        collected_data_length = np.sum(local_collected_data_length_list) 
        collected_data_shape = (collected_data_length,)+local_collected_data.shape[1:]
        local_collected_data_dim_list= np.array(local_collected_data_length_list) * np.product(local_collected_data.shape[1:])        
        
        ## if the first slice object has a negative step size, the ordering 
        ## of the Gatherv functions must be reversed         
        order = slice_objects[0].step
        if order == None:
            order = 1
        else:
            order = np.sign(order)
            
        local_collected_data_dim_offset_list = np.append([0],np.cumsum(local_collected_data_dim_list[::order])[:-1])[::order]

        local_collected_data_dim_offset_list = local_collected_data_dim_offset_list
        collected_data = np.empty(collected_data_shape, dtype=self.dtype)
        

        if target_rank == 'all':
            comm.Allgatherv([local_collected_data, self.mpi_dtype], 
                         [collected_data, local_collected_data_dim_list, local_collected_data_dim_offset_list, self.mpi_dtype])                
        else:
            comm.Gatherv([local_collected_data, self.mpi_dtype], 
                         [collected_data, local_collected_data_dim_list, local_collected_data_dim_offset_list, self.mpi_dtype], root=target_rank)                            
        return collected_data

    def collect_data(self, data, slice_objects, comm=None, **kwargs):
        if comm == None:
            comm = self.comm                    
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
        
    
    def _backshift_and_decycle(self, slice_object, shifted_start, shifted_stop, global_length):
        ## Crop the start value
        if slice_object.start > global_length-1:
            slice_object = slice(global_length-1, slice_object.stop,
                                 slice_object.step)
                                 
        ## Reformulate negative indices                                  
        if slice_object.start < 0 and slice_object.start != None:
            temp_start = slice_object.start + global_length
            if temp_start < 0:
                raise ValueError(about._errors.cstring(\
                "ERROR: Index is out of bounds!"))
            slice_object = slice(temp_start, slice_object.stop,\
            slice_object.step) 

        if slice_object.stop < 0 and slice_object.stop != None:
            temp_stop = slice_object.stop + global_length
            if temp_stop < 0:
                raise ValueError(about._errors.cstring(\
                "ERROR: Index is out of bounds!"))
            slice_object = slice(slice_object.start, temp_stop,\
            slice_object.step) 
                
        ## initialize the step
        if slice_object.step == None:
            step = 1
        else:
            step = slice_object.step
        
        if step > 0:
            shift = shifted_start
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
                
        else: # if step < 0
            step = -step
            local_length = shifted_stop - shifted_start
            ## calculate the start index. (Here, local_start > local_stop!)
            if slice_object.start == None:
                local_start = (local_length-1) -\
                    (global_length-shifted_stop)%step #stepsize compensation
            else:
                local_start = slice_object.start - shifted_start
                ## if the local_start is negative, pull it up to zero
                local_start = 0 if local_start < 0 else local_start                
                ## if the local_start is greater than the local length, pull
                ## it down 
                if local_start > local_length-1:
                    overhead = local_start - (local_length-1)
                    overhead = overhead - overhead%(-step)
                    local_start = local_start - overhead
            ## calculate the stop index
            if slice_object.stop == None:
                local_stop = None
            else:
                local_stop = slice_object.stop - shifted_start
                ## if local_stop is negative, pull it up to zero
                local_stop = 0 if local_stop < 0 else local_stop    
        ## Note: if start or stop are greater than the array length,
        ## numpy will automatically cut the index value down into the 
        ## array's range 
        return local_start, local_stop        
    
    def inject(self, data, to_slices, data_update, from_slices, comm=None, 
               **kwargs):
        ## check if to_key and from_key are completely built of slices 
        if not np.all(
            np.vectorize(lambda x: isinstance(x, slice))(to_slices)):
            raise ValueError(about._errors.cstring(
            "ERROR: The to_slices argument must be a list or tuple of slices!")
            )

        if not np.all(
            np.vectorize(lambda x: isinstance(x, slice))(from_slices)):
            raise ValueError(about._errors.cstring(
            "ERROR: The from_slices argument must be a list or tuple of slices!")
            )
            
        to_slices = tuple(to_slices)
        from_slices = tuple(from_slices)
        self.disperse_data(data = data, 
                           to_slices = to_slices,
                           data_update = data_update,
                           from_slices = from_slices,
                           comm=comm,
                           **kwargs)

    def extract_local_data(self, data_object):
        ## if data_object is not a ndarray or a d2o, cast it to a ndarray
        if not (isinstance(data_object, np.ndarray) or 
                isinstance(data_object, distributed_data_object)):
            data_object = np.array(data_object)
        ## check if the shapes are remotely compatible, reshape if possible
        ## and determine which dimensions match only via broadcasting
        try:
            (data_object, matching_dimensions) = \
                self._reshape_foreign_data(data_object)
        ## if the shape-casting fails, try to fix things via locall data
        ## matching
        except(ValueError):
            ## Check if all the local shapes match the supplied data
            local_matchQ = (self.local_shape == data_object.shape)
            global_matchQ = self._allgather(local_matchQ)            
            ## if the local shapes match, simply return the data_object            
            if np.all(global_matchQ):
                extracted_data = data_object[:] 
            ## if not, allgather the local data pieces and extract from this
            else:
                allgathered_data = self._allgather(data_object)
                allgathered_data = np.concatenate(allgathered_data)
                if allgathered_data.shape != self.global_shape:
                    raise ValueError(
                            about._errors.cstring(
            "ERROR: supplied shapes do neither match globally nor locally"))
                return self.extract_local_data(allgathered_data)
            
        ## if shape-casting was successfull, extract the data
        else:
            ## If the first dimension matches only via broadcasting...
            ## Case 1: ...do broadcasting. This procedure does not depend on the
            ## array type (ndarray or d2o)
            if matching_dimensions[0] == False:
                extracted_data = data_object[0:1]
    
    
            ## Case 2: First dimension fits directly and data_object is a d2o
            elif isinstance(data_object, distributed_data_object):
                ## Check if the distribution_strategy and the comm match 
                ## the own ones.            
                if type(self) == type(data_object.distributor) and\
                    self.comm == data_object.distributor.comm:
                    ## Case 1: yes. Simply take the local data
                    extracted_data = data_object.data
                else:            
                    ## Case 2: no. All nodes extract their local slice from the 
                    ## data_object
                    extracted_data =\
                        data_object[self.local_start:self.local_end]
            
            ## Case 3: First dimension fits directly and data_object is an generic
            ## array        
            else:
                extracted_data =\
                    data_object[self.local_start:self.local_end]
            
        return extracted_data

    def _reshape_foreign_data(self, foreign):
        ## Case 1:        
        ## check if the shapes match directly 
        if self.global_shape == foreign.shape:
            matching_dimensions = [True,]*len(self.global_shape)            
            return (foreign, matching_dimensions)
        ## Case 2:
        ## if not, try to reshape the input data
        ## in particular, this will fail when foreign is a d2o as long as 
        ## reshaping is not implemented
        try:
            output = foreign.reshape(self.global_shape)
            matching_dimensions = [True,]*len(self.global_shape)
            return (output, matching_dimensions)
        except(ValueError, AttributeError):
            pass
        ## Case 3:
        ## if this does not work, try to broadcast the shape
        ## check if the dimensions match
        if len(self.global_shape) != len(foreign.shape):
           raise ValueError(
               about._errors.cstring("ERROR: unequal number of dimensions!")) 
        ## check direct matches
        direct_match = (np.array(self.global_shape) == np.array(foreign.shape))
        ## check broadcast compatibility
        broadcast_match = (np.ones(len(self.global_shape), dtype=int) ==\
                            np.array(foreign.shape))
        ## combine the matches and assert that all are true
        combined_match = (direct_match | broadcast_match)
        if not np.all(combined_match):
            raise ValueError(
                about._errors.cstring("ERROR: incompatible shapes!")) 
        matching_dimensions = tuple(direct_match)
        return (foreign, matching_dimensions)
        
                
    def consolidate_data(self, data, target_rank='all', comm = None):
        if comm == None:
            comm = self.comm            
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
        def save_data(self, data, alias, path=None, overwriteQ=True, comm=None):
            if comm == None:
                comm = self.comm            
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
                    raise KeyError(about._errors.cstring("ERROR: overwriteQ == False, but alias already in use!"))
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
        
        def load_data(self, alias, path, comm=None):
            if comm == None:
                comm = self.comm            
            ## create the file-handle
            if found['h5py_parallel']:
                f = h5py.File(path, 'r', driver='mpio', comm=comm)
            else:
                f= h5py.File(path, 'r')        
            dset = f[alias]        
            ## check shape
            if dset.shape != self.global_shape:
                raise TypeError(about._errors.cstring("ERROR: The shape of the given dataset does not match the distributed_data_object."))
            ## check dtype
            if dset.dtype.type != self.dtype:
                raise TypeError(about._errors.cstring("ERROR: The datatype of the given dataset does not match the distributed_data_object."))
            ## if everything seems to fit, load the data
            data = dset[self.local_start:self.local_end]
            ## close the file
            f.close()
            return data
    else:
        def save_data(self, *args, **kwargs):
            raise ImportError(about._errors.cstring("ERROR: h5py was not imported")) 
        def load_data(self, *args, **kwargs):
            raise ImportError(about._errors.cstring("ERROR: h5py was not imported")) 
        
        
        
        

class _not_distributor(object):
    def __init__(self, global_data=None, global_shape=None, dtype=None, *args,  **kwargs):
        if dtype != None:        
            self.dtype = dtype
        elif global_data != None:
            self.dtype = np.array(global_data).dtype.type
            
        if global_data != None and np.array(global_data).shape != ():
            self.global_shape = np.array(global_data).shape
        elif global_shape != None:
            self.global_shape = global_shape
        else:
            raise TypeError(about._errors.cstring("ERROR: Neither data nor shape supplied!")) 
    
    def globalize_flat_index(self, index):
        return index
    
    def globalize_index(self, index):
        return index
    
    def _allgather(self, thing):
        return [thing,]
        
    def distribute_data(self, data, **kwargs):
        if data == None:        
            return np.zeros(self.global_shape, dtype=self.dtype)
        else:
            return np.array(data).astype(self.dtype, copy=False).\
                    reshape(self.global_shape)
    
    def disperse_data(self, data, data_update, key, **kwargs):
        data[key] = np.array(data_update, copy=False).astype(self.dtype)
                     
    def collect_data(self, data, slice_objects,  **kwargs):
        return data[slice_objects]
        
    def consolidate_data(self, data, **kwargs):
        return data
        
    def inject(self, data, to_slices = (slice(None),), data_update = None, 
               from_slices = (slice(None),)):
        data[to_slices] = data_update[from_slices]
    
    def extract_local_data(self, data_object):
        return data_object.get_full_data()
        
    def save_data(self, *args, **kwargs):
        raise AttributeError(about._errors.cstring(
                                        "ERROR: save_data not implemented")) 
    def load_data(self, *args, **kwargs):
        raise AttributeError(about._errors.cstring(
                                        "ERROR: load_data not implemented")) 
                                        



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
                    [np.bool, MPI.BYTE],
                    [np.int16, MPI.SHORT],
                    [np.uint16, MPI.UNSIGNED_SHORT],
                    [np.uint32, MPI.UNSIGNED_INT],
                    [np.int32, MPI.INT],
                    [np.int, MPI.LONG],  
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
#
#class test(object):
#    def __init__(self,x=None, *args, **kwargs):
#        self.x =x
#        print args
#        print kwargs
#    @property
#    def val(self):
#        return self.x
#    
#    @val.setter
#    def val(self, x):
#        self.x = x
#
#
#if __name__ == '__main__':    
#    comm = MPI.COMM_WORLD
#    rank = comm.rank
#    if True:
#    #if rank == 0:
#        x = np.arange(100).reshape((10,10)).astype(np.int)
#        #x = x**2
#        #x = x[::-1,::-1] + x
#        
#        #print x
#        #x = np.arange(3)
#
#
#    else:
#        x = None
#    obj = distributed_data_object(global_data=x, distribution_strategy='fftw')
#    
#    
#    #obj.load('myalias', 'mpitest.hdf5')
#    if MPI.COMM_WORLD.rank==0:
#        print ('rank', rank, vars(obj.distributor))
#    MPI.COMM_WORLD.Barrier()
#    #print ('rank', rank, vars(obj))
#    
#    MPI.COMM_WORLD.Barrier()
#    temp_erg =obj.get_full_data(target_rank='all')
#    print ('rank', rank, 'full data', np.all(temp_erg == x), temp_erg.shape)
#    #print ('rank', rank, ' local flat index: ', 1000, ' globalized: ', obj.distributor.globalize_flat_index(1000))    
#    #temp_index= (80,80,666)    
#    #print ('rank', rank, ' local index: ', temp_index, ' globalized: ', obj.distributor.globalize_index(temp_index)) 
#    
#
#    MPI.COMM_WORLD.Barrier()
#    sl = slice(13,1,-3)
#    if rank == 0:    
#        print ('erwuenscht', x[sl])
#    print obj[sl]
#    """
#    sl = slice(1,2+rank,1)
#    print ('slice', rank, sl, obj[sl,2])
#    print obj[1:5:2,1:3]
#    if rank == 0:
#        sl = (slice(1,9,2), slice(1,5,2))
#        d = [[111, 222],[333,444],[111, 222],[333,444]]
#    else:
#        sl = (slice(6,10,2), slice(1,5,2))
#        d = [[555, 666],[777,888]]
#    obj[sl] = d
#    print obj.get_full_data()    
#   """
