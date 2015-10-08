# -*- coding: utf-8 -*-

from numpy.testing import assert_equal,\
    assert_almost_equal,\
    assert_raises

from nose_parameterized import parameterized
import unittest

import itertools
import os
import numpy as np
import warnings
import tempfile

import nifty
from nifty.nifty_mpi_data import distributed_data_object,\
                                 STRATEGIES

FOUND = {}
try:
    import h5py
    FOUND['h5py'] = True
    FOUND['h5py_parallel'] = h5py.get_config().mpi
except(ImportError):
    FOUND['h5py'] = False
    FOUND['h5py_parallel'] = False

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

# all_datatypes = [np.bool_, np.int16, np.uint16, np.uint32, np.int32, np.int_,
#             np.int, np.int64, np.uint64, np.float32, np.float_, np.float,
#             np.float64, np.float128, np.complex64, np.complex_,
#             np.complex, np.complex128]
all_datatypes = [np.dtype('bool'), np.dtype('int16'), np.dtype('uint16'),
                 np.dtype('uint32'), np.dtype('int32'), np.dtype('int'),
                 np.dtype('int64'), np.dtype('uint'), np.dtype('uint64'),
                 np.dtype('float32'), np.dtype(
                     'float64'), np.dtype('float128'),
                 np.dtype('complex64'), np.dtype('complex128')]

###############################################################################

all_distribution_strategies = STRATEGIES['all']
global_distribution_strategies = STRATEGIES['global']
local_distribution_strategies = STRATEGIES['local']
hdf5_distribution_strategies = STRATEGIES['hdf5']

###############################################################################

binary_non_inplace_operators = ['__add__', '__radd__', '__sub__', '__rsub__',
                                '__div__', '__truediv__', '__rdiv__',
                                '__rtruediv__', '__floordiv__',
                                '__rfloordiv__', '__mul__', '__rmul__',
                                '__pow__', '__rpow__']
binary_inplace_operators = ['__iadd__', '__isub__', '__idiv__', '__itruediv__',
                            '__ifloordiv__', '__imul__', '__ipow__']
comparison_operators = ['__ne__', '__lt__', '__le__', '__eq__', '__ge__',
                        '__gt__', ]

###############################################################################

hdf5_test_paths = [  # ('hdf5_init_test.hdf5', None),
    ('hdf5_init_test.hdf5', os.path.join(os.path.dirname(nifty.__file__),
                                         'test/hdf5_init_test.hdf5')),
    ('hdf5_init_test.hdf5',
     os.path.join(os.path.dirname(nifty.__file__),
                  'test//hdf5_test_folder/hdf5_init_test.hdf5'))]

###############################################################################


def custom_name_func(testcase_func, param_num, param):
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name("_".join(str(x) for x in param.args)),
    )


###############################################################################
###############################################################################

def generate_data(global_shape, dtype, distribution_strategy,
                  strictly_positive=False):
    if distribution_strategy in global_distribution_strategies:
        a = np.arange(np.prod(global_shape))
        a -= np.prod(global_shape) // 2

        a = a * (1 + 1j)
        if strictly_positive:
            a = abs(a)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = a.reshape(global_shape).astype(dtype)
        obj = distributed_data_object(
                                a, distribution_strategy=distribution_strategy)
        global_a = a

    elif distribution_strategy in local_distribution_strategies:
        local_shape = list(global_shape)
        if rank % 2 == 1:
            local_shape[0] = 0
        else:
            local_shape[0] = global_shape[0] // np.ceil(size / 2.)
            number_of_extras = global_shape[
                0] - local_shape[0] * np.ceil(size / 2.)
            if number_of_extras > rank:
                local_shape[0] += 1
        local_shape = tuple(local_shape)

        a = np.arange(np.prod(local_shape))
        a -= np.prod(local_shape) // 2

        a = a * (1 + 1j)
        if strictly_positive:
            a = abs(a)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = a.reshape(local_shape).astype(dtype)
        a *= (rank + 1)
        obj = distributed_data_object(
                    local_data=a, distribution_strategy=distribution_strategy)
        a_list = comm.allgather(a)
        global_a = np.concatenate(a_list)

    return (global_a, obj)


###############################################################################
###############################################################################


class Test_Globaltype_Initialization(unittest.TestCase):

    @parameterized.expand(
        itertools.product([(1,), (7,), (78, 11), (256, 256)],
                          all_datatypes,
                          global_distribution_strategies,
                          [True, False]),
        testcase_func_name=custom_name_func)
    def test_successful_init_via_global_shape_and_dtype(self,
                                                        global_shape,
                                                        dtype,
                                                        distribution_strategy,
                                                        hermitian):
        obj = distributed_data_object(
                                  global_shape=global_shape,
                                  dtype=dtype,
                                  distribution_strategy=distribution_strategy,
                                  hermitian=hermitian)

        assert_equal(obj.dtype, dtype)
        assert_equal(obj.shape, global_shape)
        assert_equal(obj.distribution_strategy, distribution_strategy)
        assert_equal(obj.hermitian, hermitian)
        assert_equal(obj.data.dtype, np.dtype(dtype))


###############################################################################

    @parameterized.expand(
        itertools.product([(1,), (7,), (77, 11), (256, 256)],
                          all_datatypes,
                          global_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_successful_init_via_global_data(self,
                                             global_shape,
                                             dtype,
                                             distribution_strategy):

        a = (np.random.rand(*global_shape) * 100 - 50).astype(dtype)
        obj = distributed_data_object(
                                  global_data=a,
                                  distribution_strategy=distribution_strategy)
        assert_equal(obj.dtype, np.dtype(dtype))
        assert_equal(obj.shape, global_shape)
        assert_equal(obj.distribution_strategy, distribution_strategy)
        assert_equal(obj.data.dtype, np.dtype(dtype))

###############################################################################

    @parameterized.expand(
        itertools.product([(1,), (7,), (77, 11), (256, 256)],
                          ['tuple', 'list'],
                          all_datatypes,
                          global_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_successful_init_via_tuple_and_list(self,
                                                global_shape,
                                                global_data_type,
                                                dtype,
                                                distribution_strategy):

        a = (np.random.rand(*global_shape) * 100 - 50).astype(dtype)
        if global_data_type == 'list':
            a = a.tolist()
        elif global_data_type == 'tuple':
            a = tuple(a.tolist())

        obj = distributed_data_object(
                                global_data=a,
                                distribution_strategy=distribution_strategy)
        assert_equal(obj.shape, global_shape)
        assert_equal(obj.distribution_strategy, distribution_strategy)

###############################################################################

    @parameterized.expand(itertools.product([
        [1, (13, 7), np.dtype('float64'),
         (13, 7), np.dtype('float64')],
        [np.array([1]), (13, 7), np.dtype('float64'),
         (1,), np.dtype('float64')],
        [np.array([[1., 2.], [3., 4.]]), (13, 7), np.dtype('int'),
         (2, 2), np.dtype('int')],
        [None, (10, 10), None,
         (10, 10), np.dtype('float64')],
    ], global_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_special_init_cases(self,
                                (global_data,
                                 global_shape,
                                 dtype,
                                 expected_shape,
                                 expected_dtype),
                                distribution_strategy):
        obj = distributed_data_object(
                                  global_data=global_data,
                                  global_shape=global_shape,
                                  dtype=dtype,
                                  distribution_strategy=distribution_strategy)
        assert_equal(obj.shape, expected_shape)
        assert_equal(obj.dtype, expected_dtype)

###############################################################################

    if FOUND['h5py'] == True:
        @parameterized.expand(itertools.product(hdf5_test_paths,
                                                hdf5_distribution_strategies),
                              testcase_func_name=custom_name_func)
        def test_hdf5_init(self, (alias, path), distribution_strategy):
            obj = distributed_data_object(
                                  global_data=1.,
                                  global_shape=(12, 6),
                                  alias=alias,
                                  path=path,
                                  distribution_strategy=distribution_strategy)
            assert_equal(obj.dtype, np.complex128)
            assert_equal(obj.shape, (13, 7))

###############################################################################

    @parameterized.expand(
        itertools.product(
            [(None, None, None, None, None),
             (None, None, np.int_, None, None),
                (None, (), np.dtype('int'), None, None),
                (1, None, None, None, None),
             (None, None, None, np.array([1, 2, 3]), (3,)),
                (None, None, np.int_, None, (3,))],
            global_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_failed_init_on_unsufficient_parameters(self,
                                                    (global_data,
                                                     global_shape,
                                                     dtype,
                                                     local_data, local_shape),
                                                    distribution_strategy):
        assert_raises(ValueError,
                      lambda: distributed_data_object(
                          global_data=global_data,
                          global_shape=global_shape,
                          dtype=dtype,
                          local_data=local_data,
                          local_shape=local_shape,
                          distribution_strategy=distribution_strategy))

###############################################################################

    if size > 1:
        @parameterized.expand(
            itertools.product(
                [(None, (10, rank, 10), np.dtype('int'), None, None),
                 (None, (2, 2),
                  np.dtype('int') if (rank == 0) else np.dtype('float'),
                  None, None), ],
                global_distribution_strategies),
            testcase_func_name=custom_name_func)
        def test_failed_init_unsufficient_params_mpi(self,
                                                     (global_data,
                                                      global_shape,
                                                      dtype,
                                                      local_data,
                                                      local_shape),
                                                     distribution_strategy):
            assert_raises(ValueError,
                          lambda: distributed_data_object(
                              global_data=global_data,
                              global_shape=global_shape,
                              dtype=dtype,
                              local_data=local_data,
                              local_shape=local_shape,
                              distribution_strategy=distribution_strategy))


###############################################################################

    @parameterized.expand(
        itertools.product([(0,), (1, 0), (0, 1), (25, 0, 10), (0, 0)],
                          global_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_init_with_zero_type_shape(self, global_shape,
                                       distribution_strategy):
        obj = distributed_data_object(
                                  global_shape=global_shape,
                                  dtype=np.int,
                                  distribution_strategy=distribution_strategy)
        assert_equal(obj.shape, global_shape)


###############################################################################
###############################################################################

class Test_Localtype_Initialization(unittest.TestCase):

    ##########################################################################
    @parameterized.expand(
        itertools.product([(1,), (7,), (78, 11), (256, 256)],
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

        if different_shapes is True:
            expected_global_shape = np.array(local_shape)
            expected_global_shape[0] *= size * (size - 1) / 2
            expected_global_shape = tuple(expected_global_shape)
            local_shape = list(local_shape)
            local_shape[0] *= rank
            local_shape = tuple(local_shape)
        else:
            expected_global_shape = np.array(local_shape)
            expected_global_shape[0] *= size
            expected_global_shape = tuple(expected_global_shape)

        obj = distributed_data_object(
                                  local_shape=local_shape,
                                  dtype=dtype,
                                  distribution_strategy=distribution_strategy,
                                  hermitian=hermitian)

        assert_equal(obj.dtype, dtype)
        assert_equal(obj.shape, expected_global_shape)
        assert_equal(obj.distribution_strategy, distribution_strategy)
        assert_equal(obj.hermitian, hermitian)
        assert_equal(obj.data.dtype, np.dtype(dtype))


###############################################################################

    @parameterized.expand(
        itertools.product([(1,), (7,), (77, 11), (256, 256)],
                          [False, True],
                          all_datatypes,
                          local_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_successful_init_via_local_data(self,
                                            local_shape,
                                            different_shapes,
                                            dtype,
                                            distribution_strategy):
        if different_shapes is True:
            expected_global_shape = np.array(local_shape)
            expected_global_shape[0] *= size * (size - 1) / 2
            expected_global_shape = tuple(expected_global_shape)
            local_shape = list(local_shape)
            local_shape[0] *= rank
            local_shape = tuple(local_shape)
        else:
            expected_global_shape = np.array(local_shape)
            expected_global_shape[0] *= size
            expected_global_shape = tuple(expected_global_shape)

        a = (np.random.rand(*local_shape) * 100 - 50).astype(dtype)
        obj = distributed_data_object(
                                  local_data=a,
                                  distribution_strategy=distribution_strategy)

        assert_equal(obj.dtype, np.dtype(dtype))
        assert_equal(obj.shape, expected_global_shape)
        assert_equal(obj.distribution_strategy, distribution_strategy)
        assert_equal(obj.data.dtype, np.dtype(dtype))

###############################################################################

    @parameterized.expand(
        itertools.product([(1,), (7,), (77, 11)],
                          ['tuple', 'list'],
                          all_datatypes,
                          local_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_successful_init_via_tuple_and_list(self,
                                                local_shape,
                                                local_data_type,
                                                dtype,
                                                distribution_strategy):

        a = (np.random.rand(*local_shape) * 100).astype(dtype)
        if local_data_type == 'list':
            a = a.tolist()
        elif local_data_type == 'tuple':
            a = tuple(a.tolist())
        obj = distributed_data_object(
                                local_data=a,
                                distribution_strategy=distribution_strategy)

        expected_global_shape = np.array(local_shape)
        expected_global_shape[0] *= size
        expected_global_shape = tuple(expected_global_shape)

        assert_equal(obj.shape, expected_global_shape)
        assert_equal(obj.distribution_strategy, distribution_strategy)


###############################################################################

    @parameterized.expand(itertools.product([
        [1, (13, 7), np.float64, (13 * size, 7), np.float64],
        [np.array([1]), (13, 7), np.float64, (1 * size,), np.float64],
        [np.array([[1., 2.], [3., 4.]]), (13, 7),
         np.int, (2 * size, 2), np.int]
    ], local_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_special_init_cases(self,
                                (local_data,
                                 local_shape,
                                 dtype,
                                 expected_shape,
                                 expected_dtype),
                                distribution_strategy):
        obj = distributed_data_object(
                                  local_data=local_data,
                                  local_shape=local_shape,
                                  dtype=dtype,
                                  distribution_strategy=distribution_strategy)

        assert_equal(obj.shape, expected_shape)
        assert_equal(obj.dtype, expected_dtype)

###############################################################################

    def test_special_init_from_d2o_cases(self):
        global_shape = (8, 8)
        dtype = np.dtype('int')
        (a, obj) = generate_data(global_shape, dtype, 'equal')
        # Given dtype overwrites the one from data
        p = distributed_data_object(global_data=obj,
                                    dtype=np.dtype('float'),
                                    distribution_strategy='freeform')
        assert_equal(p.dtype, np.dtype('float'))
        # Global d2o overwrites local data
        p = distributed_data_object(global_data=obj,
                                    local_data=np.array([1, 2, 3]),
                                    distribution_strategy='freeform')
        assert_equal(obj.get_full_data(), p.get_full_data())
        # Global d2o overwrites local shapes
        p = distributed_data_object(global_data=obj,
                                    local_shape=(4, 4),
                                    distribution_strategy='freeform')
        assert_equal(obj.get_full_data(), p.get_full_data())


###############################################################################

    @parameterized.expand(
        itertools.product(
            [(None, None, None, None, None),
             (None, (8, 8), None, None, None),
                (None, None, np.int_, None, None),
                (1, None, None, None, None),
             (np.array([1, 2, 3]), (3,), None, None, None),
                (None, (4, 4), np.int_, None, None)],
            local_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_failed_init_on_unsufficient_parameters(self,
                                                    (global_data,
                                                     global_shape,
                                                     dtype,
                                                     local_data,
                                                     local_shape),
                                                    distribution_strategy):
        assert_raises(ValueError,
                      lambda: distributed_data_object(
                          global_data=global_data,
                          global_shape=global_shape,
                          dtype=dtype,
                          local_data=local_data,
                          local_shape=local_shape,
                          distribution_strategy=distribution_strategy))

###############################################################################

    if size > 1:
        @parameterized.expand(
            itertools.product(
                [(None, None, np.dtype('int'), None, (2, rank, 3)),
                 (None, None, None, np.arange(2 * rank).reshape((2, rank)),
                  None),
                 (None, None,
                  np.dtype('int') if (rank == 0) else np.dtype('float'),
                  None, (2, 2))],
                local_distribution_strategies),
            testcase_func_name=custom_name_func)
        def test_failed_init_unsufficient_params_mpi(self,
                                                     (global_data,
                                                      global_shape,
                                                      dtype,
                                                      local_data,
                                                      local_shape),
                                                     distribution_strategy):
            assert_raises(ValueError,
                          lambda: distributed_data_object(
                              global_data=global_data,
                              global_shape=global_shape,
                              dtype=dtype,
                              local_data=local_data,
                              local_shape=local_shape,
                              distribution_strategy=distribution_strategy))

##########################################################################

    @parameterized.expand(
        itertools.product([(0,), (1, 0), (0, 1), (25, 0, 10), (0, 0)],
                          local_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_init_with_zero_type_shape(self, local_shape,
                                       distribution_strategy):

        obj = distributed_data_object(
                                   local_shape=local_shape,
                                   dtype=np.int,
                                   distribution_strategy=distribution_strategy)

        expected_global_shape = np.array(local_shape)
        expected_global_shape[0] *= size
        expected_global_shape = tuple(expected_global_shape)

        assert_equal(obj.shape, expected_global_shape)


###############################################################################
###############################################################################

class Test_init_from_existing_d2o(unittest.TestCase):

    @parameterized.expand(
        itertools.product(all_distribution_strategies,
                          all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_simple_init_from_existing_d2o(self, old_strat, new_strat):
        global_shape = (8, 8)
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype, old_strat)
        p = distributed_data_object(global_data=obj,
                                    distribution_strategy=new_strat)
        if old_strat == 'not' and new_strat in local_distribution_strategies:
            old_blown_up = np.concatenate([obj.get_full_data(), ] * size)
            assert_equal(old_blown_up, p.get_full_data())
        else:
            assert_equal(obj.get_full_data(), p.get_full_data())
        assert_equal(obj.distribution_strategy, old_strat)
        assert_equal(p.distribution_strategy, new_strat)


###############################################################################
###############################################################################

class Test_set_get_full_and_local_data(unittest.TestCase):

    @parameterized.expand(
        itertools.product([(1,), (7,), (2, 7), (77, 11), (256, 256)],
                          all_datatypes,
                          all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_init_data_and_get_full_and_local_data(self,
                                                   global_shape,
                                                   dtype,
                                                   distribution_strategy):
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        assert_equal(obj.get_full_data(), a)

###############################################################################

    if FOUND['h5py']:
        @parameterized.expand(
            itertools.product(hdf5_test_paths, hdf5_distribution_strategies),
            testcase_func_name=custom_name_func)
        def test_loading_hdf5_file(self, (alias, path), distribution_strategy):
            a = np.arange(13 * 7).reshape((13, 7)).astype(np.float)
            b = a[::-1, ::-1]
            a = a + 1j * b
            obj = distributed_data_object(
                                  alias=alias,
                                  path=path,
                                  distribution_strategy=distribution_strategy)
            assert_equal(obj.get_full_data(), a)

###############################################################################

    @parameterized.expand(
        itertools.product([(1,), (7,), (2, 7), (77, 11), (256, 256)],
                          all_datatypes,
                          all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_set_get_full_data(self,
                               global_shape,
                               dtype,
                               distribution_strategy):
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        new_a = np.array(5 * a, copy=True, dtype=dtype)
        obj.set_full_data(new_a)
        assert_equal(obj.get_full_data(), new_a)

###############################################################################

    @parameterized.expand(
        itertools.product([(1,), (7,), (2, 7), (77, 11), (256, 256)],
                          all_datatypes,
                          all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_get_set_local_data(self,
                                global_shape,
                                dtype,
                                distribution_strategy):
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        b = obj.get_local_data()
        c = (np.random.random(b.shape) * 100).astype(np.dtype(dtype))
        obj.set_local_data(data=c)
        assert_equal(obj.get_local_data(), c)


##########################################################################
##########################################################################

class Test_slicing_get_set_data(unittest.TestCase):

    @parameterized.expand(
        itertools.product(
            [(4, 4, 4)],  # (20,21), (256,256)],
            [np.dtype('uint'), np.dtype('int'), np.dtype('float')],
            # all_datatypes,
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
             slice(5, 300),
             slice(1000, -300, -1),
             slice(1000, -300, -3),
             (3,),
             (2, slice(5, 18)),
             (slice(None), 2),
             (slice(5, 18), slice(18, 5, -1))],
            ['equal', 'np']),
        testcase_func_name=custom_name_func)
    def test_get_set_slicing_data(self,
                                  global_shape,
                                  dtype,
                                  distribution_strategy,
                                  slice_tuple,
                                  from_distribution_strategy):

        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)

        assert_equal(obj[slice_tuple].get_full_data(), a[slice_tuple])

        if from_distribution_strategy == 'np':
            b = a.copy() * 100
            obj[slice_tuple] = b[slice_tuple]
            a[slice_tuple] = b[slice_tuple]
        else:
            (b, p) = generate_data(global_shape, dtype,
                                   from_distribution_strategy)
            b *= 100
            p *= 100
            obj[slice_tuple] = p[slice_tuple]
            a[slice_tuple] = b[slice_tuple]

        assert_equal(obj.get_full_data(), a)

        a[slice_tuple] = 111
        obj[slice_tuple] = 111

        assert_equal(obj.get_full_data(), a)


###############################################################################

    @parameterized.expand(all_distribution_strategies,
                          testcase_func_name=custom_name_func)
    def test_get_single_value_from_d2o(self, distribution_strategy):
        (a, obj) = generate_data((4,), np.dtype('float'),
                                 distribution_strategy)
        assert_equal(obj[0], a[0])


###############################################################################

    @parameterized.expand(
        itertools.product(all_distribution_strategies,
                          all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_single_row_from_d2o(self, distribution_strategy1,
                                 distribution_strategy2):
        (a, obj) = generate_data((8, 8), np.dtype('float'),
                                 distribution_strategy1)
        (b, p) = generate_data((8,), np.dtype('float'),
                               distribution_strategy2)
        a[4] = b
        obj[4] = p
        assert_equal(obj.get_full_data(), a)


###############################################################################
###############################################################################

class Test_boolean_get_set_data(unittest.TestCase):

    @parameterized.expand(
        itertools.product(
            [(4, 4)],  # (20,21)],
            all_datatypes,
            all_distribution_strategies,
            all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_boolean_get_data(self,
                              global_shape,
                              dtype,
                              distribution_strategy_1,
                              distribution_strategy_2):

        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy_1)

        b = ((a < -5) + (a > 4))

        p = obj.copy_empty(dtype=np.dtype('bool'),
                           distribution_strategy=distribution_strategy_2)
        p[:] = b

        assert_equal(obj[b].get_full_data(), a[b])
        assert_equal(obj[p].get_full_data(), a[b])

###############################################################################

    @parameterized.expand(
        itertools.product(
            [(4, 4), (20, 21)],
            all_datatypes,
            all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_boolean_set_data(self,
                              global_shape,
                              dtype,
                              distribution_strategy):

        (a, obj_mem) = generate_data(global_shape, dtype,
                                     distribution_strategy)

        p = ((obj_mem < -5) + (obj_mem > 4))
        b = ((a < -5) + (a > 4))
        c = a[b] * 100
        q = obj_mem[p] * 100

        a[b] = c

        obj = obj_mem.copy()
        obj[b] = c
        assert_equal(obj.get_full_data(), a)

        obj = obj_mem.copy()
        obj[b] = q
        assert_equal(obj.get_full_data(), a)

        obj = obj_mem.copy()
        obj[p] = q
        assert_equal(obj.get_full_data(), a)

###############################################################################

    @parameterized.expand(all_distribution_strategies)
    def test_boolean_set_data_of_a_scalar(self, distribution_strategy):
        global_shape = (8, 8)
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)

        b = a.copy()
        b[a < 3] = 14

        p = obj.copy()
        p[obj < 3] = 14
        assert_equal(p.get_full_data(), b)

        p = obj.copy()
        p[a < 3] = 14
        assert_equal(p.get_full_data(), b)


###############################################################################
###############################################################################

class Test_list_get_set_data(unittest.TestCase):

    @parameterized.expand(
        itertools.product(
            [(4, 4, 4)],
            all_datatypes,
            all_distribution_strategies,
            all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_list_get_data(self,
                           global_shape,
                           dtype,
                           distribution_strategy_1,
                           distribution_strategy_2):

        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy_1)

        w = np.where(a > 30)
        p = obj.copy(distribution_strategy=distribution_strategy_2)
        wo = (p > 30).where()

        assert_equal(obj[w].get_full_data(), a[w])
        assert_equal(obj[wo].get_full_data(), a[w])

        w0 = 3
        w1 = np.array([0, 1, 3])
        w1o = distributed_data_object(global_data=w1,
                                      distribution_strategy='equal')

        w = [w0, w1, w1]
        wo = [w0, w1, w1o]
        assert_equal(obj[w].get_full_data(), a[w])
        assert_equal(obj[wo].get_full_data(), a[w])

        w = [w1, w0, w1]
        wo = [w1, w0, w1o]
        assert_equal(obj[w].get_full_data(), a[w])
        assert_equal(obj[wo].get_full_data(), a[w])

        w = [w1, w0, w1]
        wo = [w1o, w0, w1]
        assert_equal(obj[w].get_full_data(), a[w])
        assert_equal(obj[wo].get_full_data(), a[w])


##############################################################################

    @parameterized.expand(
        itertools.product(
            [(4, 4), (20, 21)],
            all_datatypes,
            all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_list_set_data(self,
                           global_shape,
                           dtype,
                           distribution_strategy):

        (a, obj_mem) = generate_data(global_shape, dtype,
                                     distribution_strategy)

        p = ((obj_mem < -5) + (obj_mem > 4)).where()
        b = np.where((a < -5) + (a > 4))
        c = a[b] * 100
        q = obj_mem[p] * 100

        a[b] = c

        obj = obj_mem.copy()
        obj[b] = c
        assert_equal(obj.get_full_data(), a)

        obj = obj_mem.copy()
        obj[b] = q
        assert_equal(obj.get_full_data(), a)

        obj = obj_mem.copy()
        obj[p] = q
        assert_equal(obj.get_full_data(), a)

###############################################################################

    @parameterized.expand(all_distribution_strategies)
    def test_list_set_data_of_a_scalar(self, distribution_strategy):
        global_shape = (8, 8)
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)

        b = a.copy()
        b[np.where(a < 3)] = 14

        p = obj.copy()
        p[(obj < 3).where()] = 14
        assert_equal(p.get_full_data(), b)

        p = obj.copy()
        p[np.where(a < 3)] = 14
        assert_equal(p.get_full_data(), b)


###############################################################################
###############################################################################

class Test_non_local_keys(unittest.TestCase):

    @parameterized.expand(all_distribution_strategies)
    def test_get_from_non_local_keys(self, distribution_strategy):
        global_shape = (2 * size, 8)
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        local_scalar = rank
        p = obj.get_data(local_scalar, local_keys=True)
        assert_equal(p.get_local_data(), a[local_scalar])

        local_slice = (slice(0, rank),)
        p = obj.get_data(local_slice, local_keys=True)
        assert_equal(p.get_local_data(), a[local_slice])

        local_bool = (obj < rank)
        p = obj.get_data(local_bool, local_keys=True)
        assert_equal(p.get_local_data(), a[local_bool.get_full_data()])

        local_list = (np.array([rank, 2 * rank]), np.array([1, 3]))
        p = obj.get_data(local_list, local_keys=True)
        assert_equal(p.get_local_data(), a[local_list])

###############################################################################

    @parameterized.expand(all_distribution_strategies)
    def test_set_from_non_local_keys(self, distribution_strategy):
        global_shape = (2 * size, 8)
        dtype = np.dtype('float')
        (a_backup, obj_backup) = generate_data(global_shape, dtype,
                                               distribution_strategy)

        local_scalar_key = rank

        a = a_backup.copy()
        b = np.expand_dims(np.arange(size), axis=1).repeat(8, axis=1)**2
        a[np.arange(size)] = b

        local_data_update = rank**2
        obj = obj_backup.copy()
        obj.set_data(local_data_update, local_scalar_key, local_keys=True)
        assert_equal(obj.get_full_data(), a)

        local_data_update = np.ones((8,)) * (rank**2)
        obj = obj_backup.copy()
        obj.set_data(local_data_update, local_scalar_key, local_keys=True)
        assert_equal(obj.get_full_data(), a)

        local_data_update_list = [np.ones((8,)) * (z**2) for z in xrange(size)]
        local_data_update_list = map(lambda z: distributed_data_object(
                                        z, distribution_strategy='equal'),
                                     local_data_update_list)
        obj = obj_backup.copy()
        obj.set_data(local_data_update_list[rank], local_scalar_key,
                     local_keys=True)
        assert_equal(obj.get_full_data(), a)

###############################################################################


###############################################################################
###############################################################################

class Test_inject(unittest.TestCase):

    @parameterized.expand(
        itertools.product([
            ((10, 10), (slice(2, 8), slice(3, 5)),
             (9, 11), (slice(1, 7), slice(4, 6))),
            ((10, 10), (slice(8, 2, -1), slice(3, 5)),
             (9, 11), (slice(1, 7), slice(4, 6))),
            ((10, 10), (slice(2, 8), slice(3, 5)),
             (9, 11), (slice(7, 1, -1), slice(4, 6))),
            ((10, 10), (slice(2, 8, 3), slice(3, 5)),
             (9, 11), (slice(1, 5, 2), slice(4, 6))),
            ((10, 10), (slice(8, 2, -3), slice(3, 5)),
             (9, 11), (slice(1, 5, 2), slice(4, 6))),
            ((10, 10), (slice(2, 8, 3), slice(3, 5)),
             (9, 11), (slice(5, 1, -2), slice(4, 6))),
            ((10, 10), (slice(None, None, 3), slice(3, 5)),
             (9, 11), (slice(None, 4), slice(4, 6))),
            ((10, 10), (slice(None, None, -3), slice(3, 5)),
             (9, 11), (slice(3, None, -1), slice(4, 6)))
        ],
            all_distribution_strategies
        ), testcase_func_name=custom_name_func)
    def test_inject(self, (global_shape_1, slice_tuple_1,
                           global_shape_2, slice_tuple_2),
                    distribution_strategy):
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape_1, dtype,
                                 distribution_strategy)

        (b, p) = generate_data(global_shape_2, dtype,
                               distribution_strategy)

        obj.inject(to_key=slice_tuple_1,
                   data=p,
                   from_key=slice_tuple_2)
        a[slice_tuple_1] = b[slice_tuple_2]
        assert_equal(obj.get_full_data(), a)


###############################################################################
###############################################################################

def scalar_only_square(x):
    if np.isscalar(x):
        return x * x
    else:
        raise ValueError


class Test_copy_and_copy_empty(unittest.TestCase):

    @parameterized.expand(
        itertools.chain(
            itertools.product([None, (8, 7)],
                              [None],
                              [None],
                              global_distribution_strategies,
                              global_distribution_strategies),

            itertools.product([None],
                              all_datatypes,
                              all_datatypes,
                              global_distribution_strategies,
                              global_distribution_strategies)),
        testcase_func_name=custom_name_func)
    def test_copy_empty(self,
                        new_shape,
                        old_dtype,
                        new_dtype,
                        new_distribution_strategy,
                        old_distribution_strategy):
        old_shape = (10, 10)
        (a, obj) = generate_data(old_shape, old_dtype,
                                 old_distribution_strategy)

        if new_distribution_strategy in global_distribution_strategies:
            p = obj.copy_empty(global_shape=new_shape,
                               dtype=new_dtype,
                               distribution_strategy=new_distribution_strategy)
            if new_shape is not None:
                assert_equal(p.shape, new_shape)
        elif new_distribution_strategy in local_distribution_strategies:
            p = obj.copy_empty(local_shape=new_shape,
                               dtype=new_dtype,
                               distribution_strategy=new_distribution_strategy)
            if new_shape is not None:
                assert_equal(p.local_shape, new_shape)
        else:
            raise ValueError(
                "ERROR: distribution_strategy neither in local nor global.")

        if new_dtype is not None:
            assert_equal(p.dtype, new_dtype)
        if new_distribution_strategy is not None:
            assert_equal(p.distribution_strategy, new_distribution_strategy)

    def test_copy_empty_from_not_to_freeform(self):
        global_shape = (10, 10)
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype, 'not')
        p = obj.copy_empty(distribution_strategy='freeform')
        assert_equal(p.shape, global_shape)


###############################################################################

    @parameterized.expand([
        (np.float_, None),
        (None, 'freeform')
    ], testcase_func_name=custom_name_func)
    def test_copy(self,
                  new_dtype,
                  new_distribution_strategy):
        old_shape = (2, 2)
        old_dtype = np.int
        old_distribution_strategy = 'equal'
        obj = distributed_data_object(
                              global_shape=old_shape,
                              dtype=old_dtype,
                              distribution_strategy=old_distribution_strategy)
        p = obj.copy(dtype=new_dtype,
                     distribution_strategy=new_distribution_strategy)
        if new_dtype is not None:
            assert_equal(p.dtype, new_dtype)
        if new_distribution_strategy is not None:
            assert_equal(p.distribution_strategy, new_distribution_strategy)

        assert_equal(p.get_full_data(), obj.get_full_data())

###############################################################################

    @parameterized.expand(
        itertools.product([
            (scalar_only_square, False, None),
            (lambda x: x * x, False, None),
            (lambda x: x * x, True, None),
            (lambda x: x * x, True, np.int),
        ], all_distribution_strategies), testcase_func_name=custom_name_func)
    def test_apply_scalar_function(self, (square_function, inplace, dtype),
                                   distribution_strategy):
        global_shape = (8, 8)
        old_dtype = np.float64
        (a, obj) = generate_data(global_shape, old_dtype,
                                 distribution_strategy)
        p = obj.apply_scalar_function(function=square_function,
                                      inplace=inplace,
                                      dtype=dtype)

        if inplace is True:
            assert_equal(p.get_full_data(), a * a)
            assert_equal(id(p), id(obj))
            assert_equal(p.dtype, old_dtype)
        else:
            assert_equal(p.get_full_data(), (a * a).astype(dtype))
            if dtype is not None:
                assert_equal(p.dtype, dtype)
            assert_raises(AssertionError,
                          lambda: assert_equal(id(p), id(obj)))

###############################################################################

    def test_conserve_hermitianity_apply_scalar(self):
        obj = distributed_data_object(global_shape=(2, 2),
                                      dtype=np.dtype('float'),
                                      hermitian=True,
                                      distribution_strategy='equal')
        obj.apply_scalar_function(np.exp, inplace=True)
        assert_equal(obj.hermitian, True)
        obj.apply_scalar_function(np.log, inplace=True)
        assert_equal(obj.hermitian, True)

###############################################################################
    @parameterized.expand(all_distribution_strategies)
    def test_apply_generator(self, distribution_strategy):
        global_shape = (7, 5)
        start_dtype = np.dtype('complex')
        generator_dtype = np.dtype('int')
        (a, obj) = generate_data(global_shape, start_dtype,
                                 distribution_strategy)
        obj.apply_generator(lambda shape: np.ones(shape=shape,
                                                  dtype=generator_dtype))
        assert_equal(obj.get_full_data(), np.ones(shape=global_shape))
        assert_equal(obj.dtype, start_dtype)

###############################################################################
###############################################################################


class Test_unary_and_binary_operations(unittest.TestCase):

    @parameterized.expand(
        itertools.product(['__pos__', '__neg__', '__abs__'],
                          [np.dtype('int'), np.dtype('complex64'),
                           np.dtype('complex128')],
                          all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_unary_operations(self, function, dtype, distribution_strategy):
        global_shape = (8, 8)
        a, obj = generate_data(global_shape, dtype, distribution_strategy)
        p = getattr(obj, function)()
        b = getattr(a, function)()
        assert_equal(p.get_full_data(), b)
        assert_equal(p.dtype, b.dtype)

###############################################################################

    def test_len(self):
        temp_shape = (7, 13)
        a = np.arange(np.prod(temp_shape)).reshape(temp_shape)
        obj = distributed_data_object(a,
                                      distribution_strategy='equal')
        assert_equal(len(obj), len(a))

##############################################################################

    def test_conjugate(self):
        temp_shape = (8, 8)
        a = ((np.arange(np.prod(temp_shape)).reshape(temp_shape) + 1) * 2 -
             (np.prod(temp_shape) + 1)) / 3
        a = a + 2 * a * 1j
        obj = distributed_data_object(a,
                                      distribution_strategy='equal')
        assert_equal(obj.conjugate().get_full_data(), np.conjugate(a))
        assert_equal(obj.conj().get_full_data(), np.conj(a))

###############################################################################

    @parameterized.expand(all_distribution_strategies)
    def test_is_real_is_complex(self, distribution_strategy):
        if distribution_strategy in local_distribution_strategies:
            stored_distribution_strategy = distribution_strategy
            distribution_strategy = 'equal'
        else:
            stored_distribution_strategy = None

        global_shape = (8, 7)
        a = np.arange(np.prod(global_shape)).reshape(global_shape)
        a = a + ((-1)**a + 1) * 1j
        obj = distributed_data_object(
                                a, distribution_strategy=distribution_strategy)

        if stored_distribution_strategy is not None:
            (b, p) = generate_data(
                            global_shape,
                            np.dtype('complex'),
                            distribution_strategy=stored_distribution_strategy)
            p.inject((slice(None),), a, (slice(None),))
            obj = p

        assert_equal(obj.isreal().get_full_data(), np.isreal(a))
        assert_equal(obj.iscomplex().get_full_data(), np.iscomplex(a))

###############################################################################

    @parameterized.expand(all_distribution_strategies)
    def test_is_nan_inf(self, distribution_strategy):
        global_shape = (8, 8)
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype, distribution_strategy)
        a[0, 0] = np.nan
        a[0, 1] = np.inf
        obj[0, 0] = np.nan
        obj[0, 1] = np.inf

        assert_equal(obj.isnan().get_full_data(), np.isnan(a))
        assert_equal(obj.isinf().get_full_data(), np.isinf(a))
        assert_equal(obj.isfinite().get_full_data(), np.isfinite(a))
        assert_equal(obj.nan_to_num().get_full_data(), np.nan_to_num(a))

###############################################################################

    @parameterized.expand(
        itertools.product(binary_non_inplace_operators,
                          all_distribution_strategies,
                          all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_binary_operations_with_d2o(self, function, strat1, strat2):
        global_shape = (8, 8)
        (a, obj) = generate_data(global_shape, np.dtype('float'), strat1)
        (b, p) = generate_data(global_shape, np.dtype('float'), strat2)

        b **= 3
        p **= 3
        assert_equal(getattr(obj, function)(p).get_full_data(),
                     getattr(a, function)(b))

###############################################################################

    @parameterized.expand(binary_non_inplace_operators,
                          testcase_func_name=custom_name_func)
    def test_binary_operations_with_nparray(self, function):
        temp_shape = (8, 8)
        a = ((np.arange(np.prod(temp_shape)).reshape(temp_shape) + 1) * 2 -
             (np.prod(temp_shape) + 1)) / 3
        a = a + 2 * a * 1j
        b = a**3
        obj = distributed_data_object(a, distribution_strategy='equal')

        assert_equal(getattr(obj, function)(b).get_full_data(),
                     getattr(a, function)(b))

###############################################################################

    @parameterized.expand(binary_non_inplace_operators,
                          testcase_func_name=custom_name_func)
    def test_binary_operations_with_scalar(self, function):
        temp_shape = (8, 8)
        a = ((np.arange(np.prod(temp_shape)).reshape(temp_shape) + 1) * 2 -
             (np.prod(temp_shape) + 1)) / 3
        a = a + 2 * a * 1j
        b = 2 + 0.5j
        obj = distributed_data_object(a, distribution_strategy='equal')

        assert_equal(getattr(obj, function)(b).get_full_data(),
                     getattr(a, function)(b))

###############################################################################

    def test_binary_operation_with_dtype_conversion(self):
        temp_shape = (8, 8)
        a = np.arange(np.prod(temp_shape)).reshape(temp_shape)
        obj = distributed_data_object(a, distribution_strategy='equal')
        assert_equal((obj + 1j).get_full_data(), a + 1j)

###############################################################################

    @parameterized.expand(binary_non_inplace_operators,
                          testcase_func_name=custom_name_func)
    def test_binary_operations_with_one_dimensional_d2o(self, function):
        temp_shape = (8, 8)
        a = ((np.arange(np.prod(temp_shape)).reshape(temp_shape) + 1) * 2 -
             (np.prod(temp_shape) + 1)) / 3
        a = a + 2 * a * 1j
        b = distributed_data_object(global_data=[2 + 0.5j],
                                    distribution_strategy='not')
        obj = distributed_data_object(a, distribution_strategy='equal')

        assert_equal(getattr(obj, function)(b).get_full_data(),
                     getattr(a, function)(b))

###############################################################################

    @parameterized.expand(binary_non_inplace_operators,
                          testcase_func_name=custom_name_func)
    def test_binary_operations_with_one_dimensional_nparray(self, function):
        temp_shape = (8, 8)
        a = ((np.arange(np.prod(temp_shape)).reshape(temp_shape) + 1) * 2 -
             (np.prod(temp_shape) + 1)) / 3
        a = a + 2 * a * 1j
        b = np.array([2 + 0.5j, ])
        obj = distributed_data_object(a, distribution_strategy='equal')

        assert_equal(getattr(obj, function)(b).get_full_data(),
                     getattr(a, function)(b))

###############################################################################

    @parameterized.expand(binary_inplace_operators,
                          testcase_func_name=custom_name_func)
    def test_inplace_binary_operations_with_d2o(self, function):
        temp_shape = (8, 8)
        a = ((np.arange(np.prod(temp_shape)).reshape(temp_shape) + 1) * 2 -
             (np.prod(temp_shape) + 1)) / 3
        a = a + 2 * a * 1j
        b = a**3
        obj = distributed_data_object(a, distribution_strategy='equal')
        old_id = id(obj)
        p = distributed_data_object(b)

        assert_equal(getattr(obj, function)(p).get_full_data(),
                     getattr(a, function)(b))
        assert_equal(old_id, id(obj))

###############################################################################

    def test_mod(sel):
        global_shape = (8, 8)
        dtype = np.dtype('float')
        distribution_strategy = 'equal'
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        a[a == 0] = 1
        obj[obj == 0] = 1

        b = a / 3
        p = obj / 3

        assert_almost_equal((obj % b).get_full_data(), a % b)
        assert_almost_equal((obj % p).get_full_data(), a % b)
        assert_almost_equal((obj % 4).get_full_data(), a % 4)

        assert_almost_equal(p.__rmod__(obj).get_full_data(), a % b)
        assert_almost_equal((4 % obj).get_full_data(), 4 % a)

        q = obj.__imod__(p)
        assert_almost_equal(q, a % b)
        assert_equal(id(q), id(obj))


###############################################################################

    def test_double_underscore_equal(self):
        temp_shape = (8, 8)
        a = np.arange(np.prod(temp_shape)).reshape(temp_shape)
        obj = distributed_data_object(a)
        # Check with scalar
        assert_equal((obj == 0).get_full_data(), a == 0)
        # Check with numpy array
        b = np.copy(a)
        b[0, 0] = 111
        assert_equal((obj == b).get_full_data(), a == b)
        # Check with None
        assert_equal(obj is None, a is None)
        # Check with something different, e.g. a list
        t = [[3, ] * temp_shape[1], ] * temp_shape[0]
        assert_equal((obj == t).get_full_data(), a == t)

###############################################################################

    def test_equal(self):
        temp_shape = (8, 8)
        a = np.arange(np.prod(temp_shape)).reshape(temp_shape)
        obj = distributed_data_object(a)
        p = obj.copy()
        assert_equal(obj.equal(p), True)
        assert_equal(obj.equal(p + 1), False)
        assert_equal(obj.equal(None), False)

###############################################################################
    @parameterized.expand(all_distribution_strategies)
    def test_shape_casting(self, distribution_strategy):
        global_shape = (8, 8)
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        assert_equal((obj + a.flatten()).get_full_data(), 2 * a)


###############################################################################
###############################################################################

class Test_contractions(unittest.TestCase):

    @parameterized.expand(
        itertools.product([np.dtype('int'), np.dtype('float'),
                           np.dtype('complex')],
                          [(0,), (4, 4)],
                          all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_vdot(self, dtype, global_shape, distribution_strategy):
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        assert_equal(obj.vdot(2 * obj), np.vdot(a, 2 * a))
        assert_equal(obj.vdot(2 * a), np.vdot(a, 2 * a))

###############################################################################

    @parameterized.expand(
        itertools.product(['sum', 'prod', 'mean', 'var', 'std', 'median'],
                          all_datatypes,
                          [(0,), (6, 6)],
                          all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_compatible_contractions_with_zeros(self, function, dtype,
                                                global_shape,
                                                distribution_strategy):
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy,
                                 strictly_positive=True)
        assert_almost_equal(getattr(obj, function)(), getattr(np, function)(a),
                            decimal=4)

###############################################################################

    @parameterized.expand(
        itertools.product(['min', 'amin', 'nanmin', 'max', 'amax', 'nanmax'],
                          all_datatypes,
                          [(6, 6)],
                          all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_compatible_contractions_without_zeros(self, function, dtype,
                                                   global_shape,
                                                   distribution_strategy):
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy,
                                 strictly_positive=True)
        assert_almost_equal(getattr(obj, function)(), getattr(np, function)(a),
                            decimal=4)

###############################################################################

    @parameterized.expand(
        itertools.product(all_datatypes,
                          all_distribution_strategies
                          ))
    def test_argmin_argmax(self, dtype, distribution_strategy):
        global_shape = (8, 8)
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        assert_equal(obj.argmax(), np.argmax(a))
        assert_equal(obj.argmin(), np.argmin(a))
        assert_equal(obj.argmin_nonflat(),
                     np.unravel_index(np.argmin(a), global_shape))
        assert_equal(obj.argmax_nonflat(),
                     np.unravel_index(np.argmax(a), global_shape))


###############################################################################

    @parameterized.expand([(lambda x: x + ((-1)**x + 1) * 1j,),
                           (lambda x: x,),
                           (lambda x: x * 1j,)],
                          testcase_func_name=custom_name_func)
    def test_any_all(self, function):
        shape = (8, 7)
        a = np.arange(np.prod(shape)).reshape(shape)
        a = function(a)
        obj = distributed_data_object(a)
        assert_equal(obj.isreal().all(),
                     np.all(np.isreal(a)))
        assert_equal(obj.isreal().any(),
                     np.any(np.isreal(a)))


###############################################################################
###############################################################################

class Test_comparisons(unittest.TestCase):

    @parameterized.expand(
        itertools.product(comparison_operators, all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_comparison_with_scalar(self, comp, distribution_strategy):
        global_shape = (8, 8)
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        scalar = 1
        assert_equal(getattr(obj, comp)(1).get_full_data(),
                     getattr(a, comp)(scalar))

    @parameterized.expand(
        itertools.product(comparison_operators, all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_comparison_with_ndarray(self, comp, distribution_strategy):
        global_shape = (7, 8)
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        b = a[::-1]
        assert_equal(getattr(obj, comp)(b).get_full_data(),
                     getattr(a, comp)(b))
        assert_equal(getattr(b, comp)(obj),
                     getattr(b, comp)(a))

    @parameterized.expand(
        itertools.product(comparison_operators, all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_comparison_with_d2o(self, comp, distribution_strategy):
        global_shape = (7, 8)
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        b = a[::-1]
        p = obj[::-1]
        assert_equal(getattr(obj, comp)(p).get_full_data(),
                     getattr(a, comp)(b))


###############################################################################
###############################################################################

class Test_special_methods(unittest.TestCase):

    @parameterized.expand(all_distribution_strategies,
                          testcase_func_name=custom_name_func)
    def test_bincount(self, distribution_strategy):
        global_shape = (80,)
        dtype = np.dtype('int')
        dtype_weights = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        a = abs(a)
        obj = abs(obj)

        (b, p) = generate_data(global_shape, dtype_weights,
                               distribution_strategy)
        b **= 2
        p **= 2

        assert_equal(obj.bincount(weights=p),
                     np.bincount(a, weights=b))

###############################################################################

    @parameterized.expand(all_distribution_strategies,
                          testcase_func_name=custom_name_func)
    def Test_unique(self, distribution_strategy):
        global_shape = (40, 40)
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        assert_equal(obj.unique(), np.unique(a))

###############################################################################

    @parameterized.expand(
        itertools.product([(4,), (8, 8), (0, 4), (4, 0, 8)],
                          all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_flatten(self, global_shape, distribution_strategy):
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        assert_equal(obj.flatten().get_full_data(), a.flatten())
        p = obj.flatten(inplace=True)
        if np.prod(global_shape) != 0:
            p[0] = 2222
            assert_equal(obj[(0,) * len(global_shape)], 2222)


###############################################################################

    @parameterized.expand(
        itertools.product([(4,), (8, 8)],
                          all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_where(self, global_shape, distribution_strategy):
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)

        b = ((a < 4) + (a > 30))
        p = ((obj < 4) + (obj > 30))

        assert_equal(map(lambda z: z.get_full_data(), p.where()), np.where(b))

###############################################################################

    @parameterized.expand(all_distribution_strategies,
                          testcase_func_name=custom_name_func)
    def test_real_imag(self, distribution_strategy):
        global_shape = (8, 8)
        dtype = np.dtype('complex')
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        assert_equal(obj.real.get_full_data(), a.real)
        assert_equal(obj.imag.get_full_data(), a.imag)

###############################################################################

    @parameterized.expand(
        itertools.product([None, 0, 1, 2],
                          all_distribution_strategies),
        testcase_func_name=custom_name_func)
    def test_cumsum(self, axis, distribution_strategy):
        global_shape = (3, 4, 5)
        dtype = np.dtype('float')
        (a, obj) = generate_data(global_shape, dtype,
                                 distribution_strategy)
        assert_equal(obj.cumsum(axis=axis).get_full_data(),
                     a.cumsum(axis=axis))

###############################################################################
###############################################################################


if FOUND['h5py'] == True:
    class Test_load_save(unittest.TestCase):

        @parameterized.expand(
            itertools.product(all_datatypes,
                              all_distribution_strategies),
            testcase_func_name=custom_name_func)
        def test_load_save(self, dtype, distribution_strategy):
            if dtype == np.dtype('float128'):
                dtype = np.dtype('float')

            global_shape = (8, 8)
            (a, obj) = generate_data(global_shape, dtype,
                                     distribution_strategy)
            alias = 'test_alias'

            path = os.path.join(tempfile.gettempdir(),
                                'temp_hdf5_file.hdf5')
            if size > 1 and FOUND['h5py_parallel'] == False:
                assert_raises(RuntimeError,
                              lambda: obj.save(alias=alias, path=path))
            else:
                obj.save(alias=alias, path=path)
                p = distributed_data_object(alias=alias,
                                            path=path)
                assert_equal(obj.get_full_data(), p.get_full_data())

                assert_raises(ValueError,
                              lambda: obj.save(alias=alias,
                                               path=path,
                                               overwriteQ=False))

                obj *= 3
                obj.save(alias=alias, path=path)
                p = distributed_data_object(alias=alias,
                                            path=path)
                assert_equal(obj.get_full_data(), p.get_full_data())
                if rank == 0:
                    os.remove(path)


# Todo: Assert that data is copied, when copy flag is set
# Todo: Assert that set, get and injection work, if there is different data
# on the nodes
