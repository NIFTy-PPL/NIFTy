# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

try:
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() == 1:
        from .data_objects.numpy_do import *
    else:
        from .data_objects.distributed_do import *
except ImportError:
    from .data_objects.numpy_do import *

__all__ = ["ntask", "rank", "master", "local_shape", "data_object", "full",
           "empty", "zeros", "ones", "empty_like", "vdot", "exp",
           "log", "tanh", "sqrt", "from_object", "from_random",
           "local_data", "ibegin", "ibegin_from_shape", "np_allreduce_sum",
           "distaxis", "from_local_data", "from_global_data", "to_global_data",
           "redistribute", "default_distaxis", "is_numpy",
           "lock", "locked"]
