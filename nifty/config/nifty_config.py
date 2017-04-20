# NIFTy
# Copyright (C) 2017  Theo Steininger
#
# Author: Theo Steininger
#
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

import os
from distutils.version import LooseVersion as lv

import numpy as np
import keepers

# Setup the dependency injector
dependency_injector = keepers.DependencyInjector(
                                   [('mpi4py.MPI', 'MPI'),
                                    'pyHealpix'])

dependency_injector.register('pyfftw', lambda z: hasattr(z, 'FFTW_MPI'))

# Initialize the variables
variable_fft_module = keepers.Variable(
                               'fft_module',
                               ['fftw', 'numpy'],
                               lambda z: (('pyfftw' in dependency_injector)
                                          if z == 'fftw' else True))

def _pyHealpix_validator(use_pyHealpix):
    if not isinstance(use_pyHealpix, bool):
        return False
    if not use_pyHealpix:
        return True
    if 'pyHealpix' not in dependency_injector:
        return False
    pyHealpix = dependency_injector['pyHealpix']
    return True


variable_use_pyHealpix = keepers.Variable(
                          'use_pyHealpix',
                          [True, False],
                          _pyHealpix_validator,
                          genus='boolean')

def _dtype_validator(dtype):
    try:
        np.dtype(dtype)
    except(TypeError):
        return False
    else:
        return True

variable_default_field_dtype = keepers.Variable(
                              'default_field_dtype',
                              ['float'],
                              _dtype_validator,
                              genus='str')

variable_default_distribution_strategy = keepers.Variable(
                              'default_distribution_strategy',
                              ['fftw', 'equal'],
                              lambda z: (('pyfftw' in dependency_injector)
                                         if z == 'fftw' else True),
                              genus='str')

nifty_configuration = keepers.get_Configuration(
                 name='NIFTy',
                 variables=[variable_fft_module,
                            variable_use_pyHealpix,
                            variable_default_field_dtype,
                            variable_default_distribution_strategy],
                 file_name='NIFTy.conf',
                 search_paths=[os.path.expanduser('~') + "/.config/nifty/",
                               os.path.expanduser('~') + "/.config/",
                               './'])

########

try:
    nifty_configuration.load()
except:
    pass
