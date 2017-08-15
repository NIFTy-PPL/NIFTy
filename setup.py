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

from setuptools import setup, find_packages

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
      include_dirs=[numpy.get_include()],
      dependency_links=[
        'git+https://gitlab.mpcdf.mpg.de/ift/keepers.git#egg=keepers-0.3.7',
        'git+https://gitlab.mpcdf.mpg.de/ift/d2o.git#egg=d2o-1.1.1'],
      install_requires=['keepers>=0.3.7', 'd2o>=1.1.1'],
      package_data={'nifty.demos': ['demo_faraday_map.npy'],
                    },
      license="GPLv3",
      classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 "
        "or later (GPLv3+)"],
      )
