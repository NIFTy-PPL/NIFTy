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
# Copyright(C) 2013-2017 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from __future__ import division

from .version import __version__

# initialize the logger instance
from keepers import MPILogger
logger = MPILogger()

# it is important to import config before d2o such that NIFTy is able to
# pre-create d2o's configuration object with the corrected path
from .config import dependency_injector,\
                   nifty_configuration,\
                   d2o_configuration

from d2o import distributed_data_object, d2o_librarian

from .energies import *

from .field import Field

from .random import Random

from .basic_arithmetics import *

from .nifty_utilities import *

from .field_types import *

from .minimization import *

from .spaces import *

from .operators import *

from .probing import *

from .sugar import *

from . import plotting
