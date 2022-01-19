import numpy as np

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    master = comm.Get_rank() == 0
except ImportError:
    comm = None
    master = True


work = np.arange(100)
validation_cumsum = np.cumsum(work)
print(validation_cumsum)

