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
from nifty_core import * ## imports `about`
from nifty_cmaps import *
from nifty_power import *
from nifty_tools import *
from nifty_explicit import *
from nifty_mpi_data import distributed_data_object

## optional submodule `rg`
try:
    from rg import *
except(ImportError):
    pass

## optional submodule `lm`
try:
    from lm import *
except(ImportError):
    print 'asdf'    
    pass

from demos import *
from pickling import *

#import pyximport; pyximport.install(pyimport = True)