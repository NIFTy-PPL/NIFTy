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

from distutils.core import setup
#from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import sys
import os
import numpy

include_dirs = [numpy.get_include()]

#os.environ["CC"] = "g++-4.8"
#os.environ["CXX"] = "g++-4.8"

ext_modules=[Extension(
                   "line_integrator",
                   ["operators/line_integrator.pyx"],
                   include_dirs=include_dirs)]# "vector.pxd"],
                   #language='c++')]


setup(name="ift_nifty",
      version="1.0.7",
      author="Marco Selig",
      author_email="mselig@mpa-garching.mpg.de",
      maintainer="Theo Steininger",
      maintainer_email="theos@mpa-garching.mpg.de",
      description="Numerical Information Field Theory",
      url="http://www.mpa-garching.mpg.de/ift/nifty/",
      packages=["nifty", "nifty.demos", "nifty.rg", "nifty.lm",
                "nifty.operators", "nifty.dummys", "nifty.config"],
      cmdclass={'build_ext': build_ext},
      ext_modules = ext_modules,
      #ext_modules=cythonize(["operators/line_integrator_vector.pyx"]),

      package_dir={"nifty": ""},
      data_files=[(os.path.expanduser('~') + "/.nifty", ["nifty_config"])],
      package_data={'nifty.demos' : ['demo_faraday_map.npy'],
                    },
      license="GPLv3")



