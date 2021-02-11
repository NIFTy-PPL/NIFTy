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
# Copyright(C) 2021 Max-Planck-Society
#
# jax_nifty is being developed at the Max-Planck-Institut fuer Astrophysik.

from setuptools import find_packages, setup


def write_version():
    import subprocess
    p = subprocess.Popen(["git", "describe", "--dirty", "--tags", "--always"],
                         stdout=subprocess.PIPE)
    res = p.communicate()[0].strip().decode('utf-8')
    with open("jifty1/git_version.py", "w") as file:
        file.write('gitversion = "{}"\n'.format(res))


write_version()
exec(open('jifty1/version.py').read())

setup(name="jifty1",
      version=__version__,
      author="Reimar Leike, Gordian Edenhofer",
      author_email="reimar@mpa-garching.mpg.de",
      description="Numerical Information Field Theory with JAX",
      packages=find_packages(include=["jifty1", "jifty1.*"]),
      zip_safe=True,
      license="GPLv3",
      setup_requires=['scipy>=1.4.1', 'numpy>=1.17', 'jax>=0.2.1'],
      install_requires=['scipy>=1.4.1', 'numpy>=1.17', 'jax>=0.2.1'],
      python_requires='>=3.8',
      classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 "
        "or later (GPLv3+)"],
      )
