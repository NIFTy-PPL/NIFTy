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

from setuptools import setup, find_packages
import os

from Cython.Build import cythonize
import numpy

exec(open('nifty/version.py').read())


setup(name="ift_nifty",
      version=__version__,
      author="Theo Steininger",
      author_email="theos@mpa-garching.mpg.de",
      description="Numerical Information Field Theory",
      url="http://www.mpa-garching.mpg.de/ift/nifty/",
      packages=find_packages(),
      package_dir={"nifty": "nifty"},
      zip_safe=False,
      ext_modules=cythonize([
          "nifty/operators/fft_operator/transformations/"
          "lm_transformation_factory.pyx",
          "nifty/spaces/lm_space/lm_helper.pyx"
      ]),
      include_dirs=[numpy.get_include()],
      dependency_links=[
        'git+https://gitlab.mpcdf.mpg.de/ift/keepers.git#egg=keepers',
        'git+https://gitlab.mpcdf.mpg.de/ift/d2o.git#egg=d2o-1.0.4'],
      install_requires=['keepers', 'd2o>=1.0.4'],
      package_data={'nifty.demos': ['demo_faraday_map.npy'],
                    },
      license="GPLv3",
      classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 "
        "or later (GPLv3+)"],
      )
