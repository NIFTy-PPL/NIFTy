## NIFTY (Numerical Information Field Theory) has been developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2015 Max-Planck-Society
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

"""
    NIFTY provides a number of demonstrations that you can run either using
    ``execfile`` provided with (absolute or relative) file names or using
    ``run -m`` providing the module name. You can retrieve the directory of
    the NIFTY demos calling :py:func:`get_demo_dir`.

"""
import os
import nifty

def get_demo_dir():
    """
        Returns the path of the NIFTY demos directory.

    """
    return os.path.split(nifty.demos.__file__)[0]

