# -*- coding: utf-8 -*-

from numpy.testing import assert_equal, assert_array_equal, assert_raises
from nose_parameterized import parameterized
import unittest
from time import sleep

import itertools
import os
import imp
import numpy as np
import nifty
from nifty.nifty_mpi_data import distributed_data_object, d2o_librarian

FOUND = {}
try:
    imp.find_module('h5py')
    FOUND['h5py'] = True
except(ImportError):
    FOUND['h5py'] = False

try:
    from mpi4py import MPI
    FOUND['MPI'] = True
except(ImportError): 
    import mpi_dummy as MPI
    FOUND['MPI'] = False
    
###############################################################################

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

###############################################################################

np.random.seed(123)

###############################################################################    

#all_datatypes = [np.bool_, np.int16, np.uint16, np.uint32, np.int32, np.int_, 
#             np.int, np.int64, np.uint64, np.float32, np.float_, np.float, 
#             np.float64, np.float128, np.complex64, np.complex_, 
#             np.complex, np.complex128]
all_datatypes = [np.dtype('bool'), np.dtype('int16'), np.dtype('uint16'), 
                np.dtype('uint32'), np.dtype('int32'), np.dtype('int'), 
                np.dtype('int64'), np.dtype('uint'), np.dtype('uint64'), 
                np.dtype('float32'), np.dtype('float64'), np.dtype('float128'), 
                np.dtype('complex64'), np.dtype('complex128')] 
             
###############################################################################              

all_distribution_strategies = ['not', 'equal', 'fftw', 'freeform']
global_distribution_strategies = ['not', 'equal', 'fftw']
local_distribution_strategies = ['freeform']

###############################################################################

binary_non_inplace_operators = ['__add__', '__radd__', '__sub__', '__rsub__', 
                                '__div__', '__truediv__', '__rdiv__', 
                                '__rtruediv__', '__floordiv__', 
                                '__rfloordiv__', '__mul__', '__rmul__', 
                                '__pow__', '__rpow__']
binary_inplace_operators = ['__iadd__', '__isub__', '__idiv__', '__itruediv__',
                            '__ifloordiv__', '__imul__', '__ipow__'] 

###############################################################################

hdf5_test_paths = [#('hdf5_init_test.hdf5', None),
    ('hdf5_init_test.hdf5', os.path.join(os.path.dirname(nifty.__file__),
                                   'test/hdf5_init_test.hdf5')),
        ('hdf5_init_test.hdf5', os.path.join(os.path.dirname(nifty.__file__),
                                'test//hdf5_test_folder/hdf5_init_test.hdf5'))]

###############################################################################
 
def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" %(
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )


###############################################################################
###############################################################################

                             
class Test_Globaltype_Initialization(unittest.TestCase):

###############################################################################    
    @parameterized.expand(
    itertools.product([(1,), (7,), (78,11), (256,256)],
                      all_datatypes,
                      global_distribution_strategies,
                      [True, False]), 
                        testcase_func_name=custom_name_func)
    def test_successful_init_via_global_shape_and_dtype(self,
                                                        global_shape, 
                                                        dtype,
                                                        distribution_strategy,
                                                        hermitian):
        obj = distributed_data_object(global_shape = global_shape, 
                                dtype = dtype,
                                distribution_strategy = distribution_strategy,
                                hermitian = hermitian)        
                                
        assert_equal(obj.dtype, dtype)
        assert_equal(obj.shape, global_shape)
        assert_equal(obj.distribution_strategy, distribution_strategy)
        assert_equal(obj.hermitian, hermitian)
        assert_equal(obj.data.dtype, np.dtype(dtype))
    
                                              
###############################################################################
                                              
    @parameterized.expand(
    itertools.product([(1,), (7,), (77,11), (256,256)],
                       all_datatypes,
                       global_distribution_strategies), 
                       testcase_func_name=custom_name_func)
    def test_successful_init_via_global_data(self, 
                                             global_shape,
                                             dtype,
                                             distribution_strategy):
        
        a = (np.random.rand(*global_shape)*100-50).astype(dtype)
        obj = distributed_data_object(global_data = a, 
                            distribution_strategy = distribution_strategy)
        assert_equal(obj.dtype, np.dtype(dtype))
        assert_equal(obj.shape, global_shape)
        assert_equal(obj.distribution_strategy, distribution_strategy)
        assert_equal(obj.data.dtype, np.dtype(dtype))

###############################################################################

    @parameterized.expand(
    itertools.product([(1,), (7,), (77,11), (256,256)],
                       ['tuple', 'list'],
                       all_datatypes,
                       global_distribution_strategies), 
                       testcase_func_name=custom_name_func)
    def test_successful_init_via_tuple_and_list(self, 
                                       global_shape,
                                       global_data_type,
                                       dtype,
                                       distribution_strategy):
        
        a = (np.random.rand(*global_shape)*100-50).astype(dtype)
        if global_data_type == 'list':
            a = a.tolist()
        elif global_data_type == 'tuple':
            a = tuple(a.tolist())
            
        obj = distributed_data_object(global_data = a, 
                            distribution_strategy = distribution_strategy)
        assert_equal(obj.shape, global_shape)
        assert_equal(obj.distribution_strategy, distribution_strategy)



###############################################################################
    
    @parameterized.expand(itertools.product([
         [1, (13,7), np.float64, (13,7), np.float64],
         [np.array([1]), (13,7), np.float64, (1,), np.float64],
         [np.array([[1.,2.],[3.,4.]]), (13,7), np.int_, (2,2), np.int_]
    ], global_distribution_strategies), 
        testcase_func_name=custom_name_func)    
    def test_special_init_cases(self, 
                                (global_data, 
                                global_shape, 
                                dtype,
                                expected_shape,
                                expected_dtype),
                                distribution_strategy):
        obj = distributed_data_object(global_data = global_data,
                                      global_shape = global_shape,
                                      dtype = dtype,
                                distribution_strategy = distribution_strategy) 
        assert_equal(obj.shape, expected_shape)
        assert_equal(obj.dtype, expected_dtype)
                                              
###############################################################################
                                              
    if FOUND['h5py'] == True:
        @parameterized.expand(itertools.product(hdf5_test_paths,
                                            global_distribution_strategies), 
                                        testcase_func_name=custom_name_func)
        def test_hdf5_init(self, (alias, path), distribution_strategies):
            obj = distributed_data_object(global_data = 1.,
                                          global_shape = (12,6),
                                          alias = alias,
                                          path = path)
            assert_equal(obj.dtype, np.complex128)
            assert_equal(obj.shape, (13,7))

###############################################################################
    
    @parameterized.expand(
        itertools.product(
            [(None, None, None, None, None),
            (None, (8,8), None, None, None),
            (None, None, np.int_, None, None),
            (1, None, None, None, None),
            (None, None, None, np.array([1,2,3]), (3,)),
            (None, (3*size,), None, np.array([1,2,3]), None),
            (None, None, np.int_, None, (3,)),],
            global_distribution_strategies), 
    testcase_func_name=custom_name_func)   
    def test_failed_init_on_unsufficient_parameters(self, 
                (global_data, global_shape, dtype, local_data, local_shape), 
                distribution_strategy):
        assert_raises(ValueError,
                      lambda: distributed_data_object(
                              global_data = global_data,
                              global_shape = global_shape,
                              dtype = dtype,
                              local_data = local_data,
                              local_shape = local_shape,
                              distribution_strategy = distribution_strategy))

###############################################################################

    @parameterized.expand(
    itertools.product([(0,), (1,0), (0,1), (25,0,10), (0,0)],
                        global_distribution_strategies),
                        testcase_func_name=custom_name_func)
    def test_init_with_zero_type_shape(self, global_shape, 
                                       distribution_strategy):
        obj = distributed_data_object(global_shape = global_shape,
                                dtype = np.int,
                                distribution_strategy = distribution_strategy)
        assert_equal(obj.shape, global_shape)                                


            
###############################################################################
###############################################################################
                             
class Test_Localtype_Initialization(unittest.TestCase):

###############################################################################    
    @parameterized.expand(
    itertools.product([(1,), (7,), (78,11), (256,256)],
                      [False, True],
                      all_datatypes,
                      local_distribution_strategies,
                      [True, False]), 
                        testcase_func_name=custom_name_func)
    def test_successful_init_via_local_shape_and_dtype(self,
                                                        local_shape, 
                                                        different_shapes,
                                                        dtype,
                                                        distribution_strategy,
                                                        hermitian):

        if different_shapes == True:                                                 
            expected_global_shape = np.array(local_shape)
            expected_global_shape[0] *= size*(size-1)/2
            expected_global_shape = tuple(expected_global_shape)            
            local_shape = list(local_shape)
            local_shape[0] *= rank
            local_shape = tuple(local_shape)
        else:
            expected_global_shape = np.array(local_shape)
            expected_global_shape[0] *= size
            expected_global_shape = tuple(expected_global_shape)

        obj = distributed_data_object(local_shape = local_shape, 
                                dtype = dtype,
                                distribution_strategy = distribution_strategy,
                                hermitian = hermitian)        

        assert_equal(obj.dtype, dtype)
        assert_equal(obj.shape, expected_global_shape)
        assert_equal(obj.distribution_strategy, distribution_strategy)
        assert_equal(obj.hermitian, hermitian)
        assert_equal(obj.data.dtype, np.dtype(dtype))
    
                                              
###############################################################################
                                              
    @parameterized.expand(
    itertools.product([(1,), (7,), (77,11), (256,256)],
                       [False, True],
                       all_datatypes,
                       local_distribution_strategies), 
                       testcase_func_name=custom_name_func)
    def test_successful_init_via_local_data(self, 
                                             local_shape,
                                             different_shapes,
                                             dtype,
                                             distribution_strategy):
        if different_shapes == True:                                                 
            expected_global_shape = np.array(local_shape)
            expected_global_shape[0] *= size*(size-1)/2
            expected_global_shape = tuple(expected_global_shape)
            local_shape = list(local_shape)
            local_shape[0] *= rank
            local_shape = tuple(local_shape)
        else:
            expected_global_shape = np.array(local_shape)
            expected_global_shape[0] *= size
            expected_global_shape = tuple(expected_global_shape)
            
        a = (np.random.rand(*local_shape)*100-50).astype(dtype)
        obj = distributed_data_object(local_data = a, 
                            distribution_strategy = distribution_strategy)
                            
        assert_equal(obj.dtype, np.dtype(dtype))
        assert_equal(obj.shape, expected_global_shape)
        assert_equal(obj.distribution_strategy, distribution_strategy)
        assert_equal(obj.data.dtype, np.dtype(dtype))

###############################################################################

    @parameterized.expand(
    itertools.product([(1,)],#, (7,), (77,11), (256,256)],
                       ['tuple', 'list'],
                       all_datatypes,
                       local_distribution_strategies), 
                       testcase_func_name=custom_name_func)
    def test_successful_init_via_tuple_and_list(self, 
                                       local_shape,
                                       local_data_type,
                                       dtype,
                                       distribution_strategy):
        
        a = (np.random.rand(*local_shape)*100).astype(dtype)
        if local_data_type == 'list':
            a = a.tolist()
        elif local_data_type == 'tuple':
            a = tuple(a.tolist())
        sleep(0.01)
        obj = distributed_data_object(local_data = a, 
                            distribution_strategy = distribution_strategy)

        expected_global_shape = np.array(local_shape)
        expected_global_shape[0] *= size
        expected_global_shape = tuple(expected_global_shape)

        assert_equal(obj.shape, expected_global_shape)
        assert_equal(obj.distribution_strategy, distribution_strategy)



###############################################################################
    
    @parameterized.expand(itertools.product([
         [1, (13,7), np.float64, (13*size,7), np.float64],
         [np.array([1]), (13,7), np.float64, (1*size,), np.float64],
         [np.array([[1.,2.],[3.,4.]]), (13,7), np.int, (2*size,2), np.int]
    ], local_distribution_strategies), 
        testcase_func_name=custom_name_func)    
    def test_special_init_cases(self, 
                                (local_data, 
                                local_shape, 
                                dtype,
                                expected_shape,
                                expected_dtype),
                                distribution_strategy):
        obj = distributed_data_object(local_data = local_data,
                              local_shape = local_shape,
                              dtype = dtype,
                              distribution_strategy = distribution_strategy) 
                              
        assert_equal(obj.shape, expected_shape)
        assert_equal(obj.dtype, expected_dtype)
                                              
################################################################################
#                                              
#    if FOUND['h5py'] == True:
#        @parameterized.expand(itertools.product(hdf5_test_paths,
#                                            global_distribution_strategies), 
#                                        testcase_func_name=custom_name_func)
#        def test_hdf5_init(self, (alias, path), distribution_strategies):
#            obj = distributed_data_object(global_data = 1.,
#                                          global_shape = (12,6),
#                                          alias = alias,
#                                          path = path)
#            assert_equal(obj.dtype, np.complex128)
#            assert_equal(obj.shape, (13,7))
#
################################################################################
#    
#    @parameterized.expand(
#        itertools.product(
#            [(None, None, None, None, None),
#            (None, (8,8), None, None, None),
#            (None, None, np.int_, None, None),
#            (1, None, None, None, None),
#            (None, None, None, np.array([1,2,3]), (3,)),
#            (None, (3*size,), None, np.array([1,2,3]), None),
#            (None, None, np.int_, None, (3,)),],
#            global_distribution_strategies), 
#    testcase_func_name=custom_name_func)   
#    def test_failed_init_on_unsufficient_parameters(self, 
#                (global_data, global_shape, dtype, local_data, local_shape), 
#                distribution_strategy):
#        assert_raises(ValueError,
#                      lambda: distributed_data_object(
#                              global_data = global_data,
#                              global_shape = global_shape,
#                              dtype = dtype,
#                              local_data = local_data,
#                              local_shape = local_shape,
#                              distribution_strategy = distribution_strategy))
#
################################################################################
#
#    @parameterized.expand(
#    itertools.product([(0,), (1,0), (0,1), (25,0,10), (0,0)],
#                        global_distribution_strategies),
#                        testcase_func_name=custom_name_func)
#    def test_init_with_zero_type_shape(self, global_shape, 
#                                       distribution_strategy):
#        obj = distributed_data_object(global_shape = global_shape,
#                                dtype = np.int,
#                                distribution_strategy = distribution_strategy)
#        assert_equal(obj.shape, global_shape)                                
#

            
###############################################################################
###############################################################################
                                              
class Test_set_get_full_and_local_data(unittest.TestCase):
    @parameterized.expand(
    itertools.product([(1,), (7,), (2,7), (77,11), (256,256)],
                      all_datatypes,
                      all_distribution_strategies), 
                    testcase_func_name=custom_name_func)
    def test_initializing_data_and_get_full_and_local_data(self,
                                                           global_shape,
                                                           dtype,
                                                        distribution_strategy):
        a = np.arange(np.prod(global_shape)).astype(dtype).\
                                                        reshape(global_shape)
        obj = distributed_data_object(global_data = a,
                                distribution_strategy = distribution_strategy)
        assert_equal(obj.get_full_data(), a)

    if FOUND['h5py']:  
        @parameterized.expand(hdf5_test_paths)
        def test_loading_hdf5_file(self, alias, path):
            a = np.arange(13*7).reshape((13,7)).astype(np.float)
            b = a[::-1, ::-1]
            a = a+1j*b
            obj = distributed_data_object(alias = alias, 
                                          path = path)
            assert_equal(obj.get_full_data(), a)


    @parameterized.expand(
    itertools.product([(1,), (7,), (2,7), (77,11), (256,256)],
                      all_datatypes,
                      all_distribution_strategies), 
                    testcase_func_name=custom_name_func)            
    def test_set_get_full_data(self, 
                           global_shape,
                           dtype,
                           distribution_strategy):
        a = np.arange(np.prod(global_shape)).astype(dtype).\
                                                        reshape(global_shape)
        obj = distributed_data_object(global_shape = global_shape,
                                dtype = dtype,
                                distribution_strategy = distribution_strategy)
        obj.set_full_data(a)
        assert_equal(obj.get_full_data(), a)
                             
    @parameterized.expand(
    itertools.product([(1,), (7,), (2,7), (77,11), (256,256)],
                      all_datatypes,
                      all_distribution_strategies), 
                    testcase_func_name=custom_name_func)            
                      
    def test_get_set_local_data(self, 
                                 global_shape,
                                 dtype,
                                 distribution_strategy):
        obj = distributed_data_object(global_shape = global_shape,
                                dtype = dtype,
                                distribution_strategy = distribution_strategy)
        b = obj.get_local_data()
        c = (np.random.random(b.shape)*100).astype(np.dtype(dtype))
        obj.set_local_data(data = c)                                                     
        assert_equal(obj.get_local_data(), c)

                                              
###############################################################################
###############################################################################
                                              
class Test_slicing_get_set_data(unittest.TestCase):
    @parameterized.expand(
    itertools.product(
                      [(4,4),(20,21), (77,11), (256,256)],
                      all_datatypes,
                      all_distribution_strategies, 
                      [slice(None, None, None), 
                       slice(5, 18),
                       slice(5, 18, 4),
                       slice(18, 5, -3),
                       slice(6, 14),
                       slice(6, 14, 4),
                       slice(14, 6, -4),
                       slice(5, None),
                       slice(5, None, 3),
                       slice(None, 5, -2),
                       slice(None, 10),
                       slice(None, 10, 3),
                       slice(10, None, -2),
                       slice(None, None, 3),
                       slice(None, None, -3),
                       slice(2, 2),
                       slice(2, 2, 4),
                       slice(2, 2, -1),
                       slice(2, 2, -4),
                       slice(-5, -2),
                       slice(-1000, 300),
                       slice(5,300),
                       slice(1000, -300, -1),
                       slice(1000, -300, -3),
                       (1,),
                       (2, slice(5,18)),
                       (slice(None), 2),
                       (slice(5,18), slice(18,5,-1))]), 
                    testcase_func_name=custom_name_func)
    def test_get_set_data(self,
                      global_shape,
                      dtype,
                      distribution_strategy,
                      slice_tuple):
        a = np.arange(np.prod(global_shape)).astype(dtype).\
                                                        reshape(global_shape)
        obj = distributed_data_object(global_data = a, 
                                distribution_strategy = distribution_strategy)
        
        assert_array_equal(obj[slice_tuple].get_full_data(), a[slice_tuple])                                      
        
        b = 100*np.copy(a)
        obj[slice_tuple] = b[slice_tuple]
        a[slice_tuple] = b[slice_tuple]
        
        assert_equal(obj.get_full_data(), a)
    
###############################################################################
###############################################################################
    
    
class Test_inject(unittest.TestCase):    
    @parameterized.expand(
            itertools.product([
    ((10,10), (slice(2,8),slice(3,5)), (9,11), (slice(1,7), slice(4,6))),
    ((10,10), (slice(8,2,-1),slice(3,5)), (9,11), (slice(1,7), slice(4,6))),
    ((10,10), (slice(2,8),slice(3,5)), (9,11), (slice(7,1,-1), slice(4,6))),
    ((10,10), (slice(2,8,3),slice(3,5)), (9,11), (slice(1,5,2), slice(4,6))),
    ((10,10), (slice(8,2,-3),slice(3,5)), (9,11), (slice(1,5,2), slice(4,6))),
    ((10,10), (slice(2,8,3),slice(3,5)), (9,11), (slice(5,1,-2), slice(4,6))),
    ((10,10), (slice(None,None,3),slice(3,5)), (9,11), (slice(None,4), slice(4,6))),
    ((10,10), (slice(None,None,-3),slice(3,5)), (9,11), (slice(3,None,-1), slice(4,6)))
        ], 
    all_distribution_strategies
    ), testcase_func_name=custom_name_func)
    def test_inject(self, (global_shape_1, slice_tuple_1,
                    global_shape_2, slice_tuple_2),
                    distribution_strategy):
        a = np.arange(np.prod(global_shape_1)).reshape(global_shape_1)
        obj = distributed_data_object(global_data = a, 
                                distribution_strategy = distribution_strategy)

        b = np.arange(np.prod(global_shape_2)).reshape(global_shape_2)
        p = distributed_data_object(global_data = b, 
                                distribution_strategy = distribution_strategy)
        
        obj.inject(to_slices = slice_tuple_1, 
                   data = p,
                   from_slices = slice_tuple_2)
        a[slice_tuple_1] = b[slice_tuple_2]                   
        assert_equal(obj.get_full_data(), a)
        
            

def scalar_only_square(x):
    if np.isscalar(x):
        return x*x
    else:
        raise ValueError
###############################################################################
###############################################################################
           
    
class Test_copy_and_copy_empty(unittest.TestCase):
    @parameterized.expand([
            ((8,7), None, None),
            (None, np.float_, None),
            (None, None, 'not')
        ], testcase_func_name=custom_name_func)
    def test_copy_empty(self, 
                        new_global_shape,
                        new_dtype,
                        new_distribution_strategy):
        old_shape = (2,2)
        old_dtype = np.int
        old_distribution_strategy = 'fftw'
        obj = distributed_data_object(global_shape = old_shape, 
                                      dtype = old_dtype,
                            distribution_strategy = old_distribution_strategy)
        p = obj.copy_empty(global_shape = new_global_shape,
                           dtype = new_dtype,
                           distribution_strategy = new_distribution_strategy)
        if new_global_shape is not None:
            assert_equal(p.shape, new_global_shape)
        if new_dtype is not None:
            assert_equal(p.dtype, new_dtype)
        if new_distribution_strategy is not None:
            assert_equal(p.distribution_strategy, new_distribution_strategy)

    @parameterized.expand([
            (np.float_, None),
            (None, 'not')
        ], testcase_func_name=custom_name_func)    
    def test_copy(self,
                  new_dtype,
                  new_distribution_strategy):
        old_shape = (2,2)
        old_dtype = np.int
        old_distribution_strategy = 'fftw'
        obj = distributed_data_object(global_shape = old_shape, 
                                      dtype = old_dtype,
                            distribution_strategy = old_distribution_strategy)
        p = obj.copy(dtype = new_dtype,
                     distribution_strategy = new_distribution_strategy)
        if new_dtype is not None:
            assert_equal(p.dtype, new_dtype)
        if new_distribution_strategy is not None:
            assert_equal(p.distribution_strategy, new_distribution_strategy)
        
        assert_equal(p.get_full_data(), obj.get_full_data())
    
    @parameterized.expand([
        (scalar_only_square, False, None),
        (lambda x: x*x, False, None),
        (lambda x: x*x, True, None),
        (lambda x: x*x, True, np.int),
        ], testcase_func_name=custom_name_func)    
    def test_apply_scalar_function(self, square_function, inplace, dtype):
        global_shape = (8,8)                 
        dtype_old = np.float64
        a = np.arange(np.prod(global_shape)).reshape(global_shape).\
                                                              astype(dtype_old)
        obj = distributed_data_object(global_data = a)
        p = obj.apply_scalar_function(function = square_function,
                                      inplace = inplace,
                                      dtype = dtype)
                                              
        if inplace == True:
            assert_equal(p.get_full_data(), a*a)
            assert_equal(id(p), id(obj))
            assert_equal(p.dtype, dtype_old)
        else:
            assert_equal(p.get_full_data(), (a*a).astype(dtype))
            assert_raises(AssertionError, 
                          lambda: assert_equal(id(p), id(obj)))    

        
    def test_conserve_hermitianity_apply_scalar(self):
        obj = distributed_data_object(global_shape = (2,2), 
                                      dtype = np.float,
                                      hermitian = True)
        obj.apply_scalar_function(np.exp, inplace = True)
        assert_equal(obj.hermitian, True)
        obj.apply_scalar_function(np.log, inplace = True)
        assert_equal(obj.hermitian, True)       

    def test_apply_generator(self):
        temp_shape = (7,5)
        desired_dtype = np.float64
        obj = distributed_data_object(global_shape = temp_shape, 
                                      dtype = desired_dtype)
        generator = lambda shape: np.ones(shape = shape, dtype=np.int) 
        obj.apply_generator(generator)
        assert_equal(obj.get_full_data(), np.ones(shape=temp_shape))        
        assert_equal(obj.dtype, desired_dtype)
        
###############################################################################
###############################################################################
    
    
class Test_unary_and_binary_operations(unittest.TestCase):
    @parameterized.expand(['__pos__', '__neg__', '__abs__'], 
                          testcase_func_name=custom_name_func)    
    def test_unary_operations(self, function):
        temp_shape = (8,8)
        a = np.arange(np.prod(temp_shape)).reshape(temp_shape)-30
        obj = distributed_data_object(a)
        assert_equal(getattr(obj, function)().get_full_data(),
                     getattr(a, function)())
    
    def test_len(self):
        temp_shape = (7,13)
        a = np.arange(np.prod(temp_shape)).reshape(temp_shape)
        obj = distributed_data_object(a)
        assert_equal(len(obj), len(a))
    
    def test_conjugate(self):
        temp_shape = (8,8)
        a = ((np.arange(np.prod(temp_shape)).reshape(temp_shape)+1)*2-\
            (np.prod(temp_shape)+1))/3
        a = a+2*a*1j        
        obj = distributed_data_object(a)
        assert_equal(obj.conjugate().get_full_data(), np.conjugate(a))
        assert_equal(obj.conj().get_full_data(), np.conj(a))
    
    @parameterized.expand(binary_non_inplace_operators, 
                          testcase_func_name=custom_name_func)    
    def test_binary_operations_with_d2o(self, function):
        temp_shape = (8,8)
        a = ((np.arange(np.prod(temp_shape)).reshape(temp_shape)+1)*2-\
            (np.prod(temp_shape)+1))/3
        a = a+2*a*1j
        b = a**3
        obj = distributed_data_object(a)
        p = distributed_data_object(b)        
        
        assert_equal(getattr(obj, function)(p).get_full_data(),
                     getattr(a, function)(b))
                     
    @parameterized.expand(binary_non_inplace_operators, 
                          testcase_func_name=custom_name_func)    
    def test_binary_operations_with_nparray(self, function):
        temp_shape = (8,8)
        a = ((np.arange(np.prod(temp_shape)).reshape(temp_shape)+1)*2-\
            (np.prod(temp_shape)+1))/3
        a = a+2*a*1j
        b = a**3
        obj = distributed_data_object(a)
        
        assert_equal(getattr(obj, function)(b).get_full_data(),
                     getattr(a, function)(b))       
                     
    @parameterized.expand(binary_non_inplace_operators, 
                          testcase_func_name=custom_name_func)    
    def test_binary_operations_with_scalar(self, function):
        temp_shape = (8,8)
        a = ((np.arange(np.prod(temp_shape)).reshape(temp_shape)+1)*2-\
            (np.prod(temp_shape)+1))/3
        a = a+2*a*1j
        b = 2+0.5j
        obj = distributed_data_object(a)
        
        assert_equal(getattr(obj, function)(b).get_full_data(),
                     getattr(a, function)(b))                         
    
    def test_binary_operation_with_dtype_conversion(self):
        temp_shape = (8,8)
        a = np.arange(np.prod(temp_shape)).reshape(temp_shape)
        obj = distributed_data_object(a)
        assert_equal((obj+1j).get_full_data(), a+1j)
        
    @parameterized.expand(binary_non_inplace_operators, 
                          testcase_func_name=custom_name_func)    
    def test_binary_operations_with_one_dimensional_nparray(self, function):
        temp_shape = (8,8)
        a = ((np.arange(np.prod(temp_shape)).reshape(temp_shape)+1)*2-\
            (np.prod(temp_shape)+1))/3
        a = a+2*a*1j
        b = np.array([2+0.5j,])
        obj = distributed_data_object(a)
        
        assert_equal(getattr(obj, function)(b).get_full_data(),
                     getattr(a, function)(b))    
    
    @parameterized.expand(binary_inplace_operators, 
                          testcase_func_name=custom_name_func)                         
    def test_inplace_binary_operations_with_d2o(self, function):
        temp_shape = (8,8)
        a = ((np.arange(np.prod(temp_shape)).reshape(temp_shape)+1)*2-\
            (np.prod(temp_shape)+1))/3
        a = a+2*a*1j
        b = a**3
        obj = distributed_data_object(a)
        old_id = id(obj)        
        p = distributed_data_object(b)        
        
        assert_equal(getattr(obj, function)(p).get_full_data(),
                     getattr(a, function)(b))                     
        assert_equal(old_id, id(obj))
        
        
    def test_double_underscore_equal(self):
        temp_shape = (8,8)
        a = np.arange(np.prod(temp_shape)).reshape(temp_shape)
        obj = distributed_data_object(a)
        ## Check with scalar
        assert_equal((obj == 0).get_full_data(), a==0)
        ## Check with numpy array
        b = np.copy(a)
        b[0,0]=111
        assert_equal((obj == b).get_full_data(), a == b)
        ## Check with None
        assert_equal(obj == None, a == None)
        ## Check with something different, e.g. a list
        t = [[3,]*temp_shape[1],]*temp_shape[0]
        assert_equal((obj == t).get_full_data(), a == t)
    
    def test_equal(self):
        temp_shape = (8,8)
        a = np.arange(np.prod(temp_shape)).reshape(temp_shape)
        obj = distributed_data_object(a)
        p = obj.copy()
        assert_equal(obj.equal(p), True)        
        assert_equal(obj.equal(p+1), False)        
        assert_equal(obj.equal(None), False)

###############################################################################
###############################################################################
      
class Test_contractions(unittest.TestCase):
    def test_vdot(self):
        temp_shape = (8,8)
        a = np.arange(np.prod(temp_shape)).reshape(temp_shape)
        obj = distributed_data_object(a)
        assert_equal(obj.vdot(2*obj), np.vdot(a,2*a))
        assert_equal(obj.vdot(2*a), np.vdot(a,2*a))
    
    
    @parameterized.expand(['amin', 'nanmin', 'amax', 'nanmax', 'sum',
                           'prod', 'mean', 'var', 'std', 'median'], 
                          testcase_func_name=custom_name_func)      
    def test_compatible_contractions(self, function):
        temp_shape = (8,8)
        a = ((np.arange(np.prod(temp_shape)).reshape(temp_shape)+1)*2-\
            (np.prod(temp_shape)+1))/3
        obj = distributed_data_object(a)
        assert_equal(getattr(obj, function)(),
                     getattr(np, function)(a))   
    
    def test_argmin_argmax(self):
        temp_shape = (8,8)
        a = 1/(
   (np.arange(np.prod(temp_shape)).reshape(temp_shape).astype(np.float)+1)*2 -\
            (np.prod(temp_shape)+1)
            )/3
        obj = distributed_data_object(a)
        assert_equal(obj.argmax_flat(), np.argmax(a))
        assert_equal(obj.argmin_flat(), np.argmin(a))
        assert_equal(obj.argmin(), np.unravel_index(np.argmin(a), temp_shape))
        assert_equal(obj.argmax(), np.unravel_index(np.argmax(a), temp_shape))
        
    def test_is_real_is_complex(self):
        temp_shape = (8,7)
        a = np.arange(np.prod(temp_shape)).reshape(temp_shape)
        a = a + ((-1)**a + 1) * 1j
        obj = distributed_data_object(a)
        assert_equal(obj.isreal().get_full_data(), np.isreal(a))
        assert_equal(obj.iscomplex().get_full_data(), np.iscomplex(a))

    @parameterized.expand([(lambda x: x + ((-1)**x + 1) * 1j,),
                           (lambda x: x,),
                           (lambda x: x*1j,)], 
                          testcase_func_name=custom_name_func)     
    def test_any_all(self, function):
        shape = (8,7)
        a = np.arange(np.prod(shape)).reshape(shape)
        a = function(a)
        obj = distributed_data_object(a)
        assert_equal(obj.isreal().all(), 
                     np.all(np.isreal(a)))
        assert_equal(obj.isreal().any(), 
                     np.any(np.isreal(a)))
        
## Todo: Assert that data is copied, when copy flag is set        
## Todo: Assert that set, get and injection work, if there is different data
## on the nodes
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                                              
                                              