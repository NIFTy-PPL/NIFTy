try:
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() == 1:
        from .data_objects.numpy_do import *
        # mprint("MPI found, but only with one task, using numpy_do...")
    else:
        from .data_objects.distributed_do import *
        # mprint("MPI with multiple tasks found, using distributed_do...")
except ImportError:
    from .data_objects.numpy_do import *
    # mprint("MPI not found, using numpy_do...")

__all__ = ["ntask", "rank", "master", "local_shape", "data_object", "full",
           "empty", "zeros", "ones", "empty_like", "vdot", "abs", "exp",
           "log", "sqrt", "from_object", "from_random",
           "local_data", "ibegin", "np_allreduce_sum", "distaxis",
           "from_local_data", "from_global_data", "to_global_data",
           "redistribute", "default_distaxis", "mprint"]
