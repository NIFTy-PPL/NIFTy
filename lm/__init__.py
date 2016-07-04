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

from __future__ import division
from nifty.config import about

from distutils.version import LooseVersion as lv

try:
    import libsharp_wrapper_gl as gl
except(ImportError):
    try:
        import healpy as hp
        if lv(hp.__version__) < lv('1.8.1'):
            raise ImportError(
                about._errors.cprint(
                "ERROR: installed healpy version is older than 1.8.1!"))
    except(ImportError):
        about.infos.cprint(
            "INFO: neither libsharp_wrapper_gl nor healpy available.")
        pass ## import nothing
    else:
        from lm_space import LMSpace ## import lm & hp
        from hp_space import HPSpace
        ## TODO: change about
else:
    try:
        import healpy as hp
        if lv(hp.__version__) < lv('1.8.1'):
            raise ImportError(
                about._errors.cprint(
                "ERROR: installed healpy version is older than 1.8.1!"))
    except(ImportError):
        from gl_space import GLSpace ## import lm & gl
        from lm_space import LMSpace
    else:
        from gl_space import GLSpace ##import all
        from lm_space import LMSpace
        from hp_space import HPSpace

from nifty.lm.nifty_power_conversion_lm import power_backward_conversion_lm,\
                                               power_forward_conversion_lm

