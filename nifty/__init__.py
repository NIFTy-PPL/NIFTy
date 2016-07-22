## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2013 Max-Planck-Society
##
## Author: Marco Selig
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

from __future__ import division

import matplotlib as mpl
mpl.use('Agg')

from .version import __version__

import dummys

# it is important to import config before d2o such that NIFTy is able to
# pre-create d2o's configuration object with the corrected path
from config import about,\
                   dependency_injector,\
                   nifty_configuration,\
                   d2o_configuration

from d2o import distributed_data_object, d2o_librarian

from nifty_cmaps import ncmap
from field import Field

# this line exists for compatibility reasons
# TODO: Remove this once the transition to field types is done.
from spaces.space import Space as point_space

from nifty_random import random
from nifty_simple_math import *
from nifty_utilities import *

from field_types import FieldType,\
                        FieldArray

from operators import *

from spaces import *

from demos import get_demo_dir

#import pyximport; pyximport.install(pyimport = True)
from transformations import *