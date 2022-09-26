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
# Copyright(C) 2013-2021 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from functools import reduce
import operator
import os
import site
import sys

from setuptools import find_packages, setup

# Workaround until https://github.com/pypa/pip/issues/7953 is fixed
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

exec(open('nifty8/version.py').read())

with open("README.md") as f:
    long_description = f.read()
description = "Library for signal inference algorithms that operate regardless of the underlying grids and their resolutions."

extras_require = {
    "re": ("jax", ),
    "native": ("ducc0>=0.24.0", ),
    "doc": ("sphinx", "pydata-sphinx-theme", "jupyter", "jupytext"),
    "util": ("astropy", "healpy"),
}
extras_require["full"] = reduce(operator.add, extras_require.values())

setup(name="nifty8",
      version=__version__,
      author="Martin Reinecke",
      author_email="martin@mpa-garching.mpg.de",
      description=description,
      long_description=long_description,
      long_description_content_type="text/markdown",
      url="https://ift.pages.mpcdf.de/nifty/",
      project_urls={
          "Bug Tracker": "https://gitlab.mpcdf.mpg.de/ift/nifty/issues",
          "Documentation": "https://ift.pages.mpcdf.de/nifty/",
          "Source Code": "https://gitlab.mpcdf.mpg.de/ift/nifty",
          "Changelog": "https://gitlab.mpcdf.mpg.de/ift/nifty/-/blob/NIFTy_8/ChangeLog",
      },
      packages=find_packages(include=["nifty8", "nifty8.*"]),
      license="GPLv3",
      setup_requires=['scipy>=1.4.1', 'numpy>=1.17'],
      install_requires=['scipy>=1.4.1', 'numpy>=1.17'],
      extras_require=extras_require,
      python_requires='>=3.7',
      classifiers=[
          "Development Status :: 5 - Production/Stable",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Scientific/Engineering :: Physics",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
          "Intended Audience :: Science/Research"],
      )
