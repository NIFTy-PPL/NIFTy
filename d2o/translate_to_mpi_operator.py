# -*- coding: utf-8 -*-

import numpy as np

from nifty.keepers import global_configuration as gc,\
                          global_dependency_injector as gdi

MPI = gdi[gc['mpi_module']]

custom_NANMIN = MPI.Op.Create(lambda x, y, datatype:
                              np.nanmin(np.vstack(x, y), axis=0))

custom_NANMAX = MPI.Op.Create(lambda x, y, datatype:
                              np.nanmax(np.vstack(x, y), axis=0))

custom_UNIQUE = MPI.Op.Create(lambda x, y, datatype:
                              np.unique(np.concatenate([x, y])))

op_translate_dict = {}

# the value tuple contains the operator and a boolean which specifies
# if the operator is compatible to buffers (for Allreduce instead of allreduce)
op_translate_dict[np.sum] = (MPI.SUM, True)
op_translate_dict[np.prod] = (MPI.PROD, True)
op_translate_dict[np.amin] = (MPI.MIN, True)
op_translate_dict[np.amax] = (MPI.MAX, True)
op_translate_dict[np.all] = (MPI.LAND, True)
op_translate_dict[np.any] = (MPI.LOR, True)
op_translate_dict[np.nanmin] = (custom_NANMIN, False)
op_translate_dict[np.nanmax] = (custom_NANMAX, False)
op_translate_dict[np.unique] = (custom_UNIQUE, False)
