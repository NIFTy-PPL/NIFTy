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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from setuptools import find_packages, setup
import os


def write_version():
    import subprocess
    try:
        p = subprocess.Popen(["git", "describe", "--dirty", "--tags", "--always"],
                             stdout=subprocess.PIPE)
        res = p.communicate()[0].strip().decode('utf-8')
    except FileNotFoundError:
        print("Could not determine version string from git history")
        res = "unknown"
    with open(os.path.join("nifty7","git_version.py"), "w") as file:
        file.write('gitversion = "{}"\n'.format(res))


write_version()
exec(open('nifty7/version.py').read())

setup(name="nifty7",
      version=__version__,
      author="Theo Steininger, Martin Reinecke",
      author_email="martin@mpa-garching.mpg.de",
      description="Numerical Information Field Theory",
      url="http://www.mpa-garching.mpg.de/ift/nifty/",
      packages=find_packages(include=["nifty7", "nifty7.*"]),
      zip_safe=True,
      license="GPLv3",
      setup_requires=['scipy>=1.4.1', 'numpy>=1.17'],
      install_requires=['scipy>=1.4.1', 'numpy>=1.17'],
      python_requires='>=3.6',
      classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 "
        "or later (GPLv3+)"],
      )
