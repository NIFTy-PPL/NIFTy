# -*- coding: utf-8 -*-

import unittest
import numpy as np
from nifty.nifty_mpi_data import distributed_data_object

found = {}

try:
    from mpi4py import MPI
    found[MPI] = True
except(ImportError): 
#    from mpi4py_dummy import MPI
    found[MPI] = False



class TestDistributedData(unittest.TestCase):
    def test_full_data_wr(self):
        temp_data = np.array(np.arange(1000), dtype=int).reshape((200,5))
        obj = distributed_data_object(global_data = temp_data)
        np.testing.assert_equal(temp_data, obj.get_full_data())
    

if __name__ == '__main__':
    unittest.main()
    
    

    comm = MPI.COMM_WORLD
    rank = comm.rank
    if True:
    #if rank == 0:
        x = np.arange(10100000).reshape((101,100,1000)).astype(np.complex128)
        #print x
        #x = np.arange(3)
    else:
        x = None
    obj = distributed_data_object(global_data=x, distribution_strategy='fftw')
    
    
    #obj.load('myalias', 'mpitest.hdf5')
    if MPI.COMM_WORLD.rank==0:
        print ('rank', rank, vars(obj.distributor))
    MPI.COMM_WORLD.Barrier()
    #print ('rank', rank, vars(obj))
    
    MPI.COMM_WORLD.Barrier()
    temp_erg =obj.get_full_data(target_rank='all')
    print ('rank', rank, 'full data', np.all(temp_erg == x), temp_erg.shape)
    """
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
    """
   
