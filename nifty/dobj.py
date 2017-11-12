#from __future__ import print_function

try:
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size()==1:
        #print ("MPI found, but only with one task, using numpy_do...")
        from .data_objects.numpy_do import *
    else:
        #if MPI.COMM_WORLD.Get_rank() == 0:
        #    print ("MPI with multiple tasks found, using distributed_do...")
        from .data_objects.distributed_do import *
except ImportError:
    #print ("MPI not found, using numpy_do...")
    from .data_objects.numpy_do import *
