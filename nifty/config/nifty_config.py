# -*- coding: utf-8 -*-

import os
from distutils.version import LooseVersion as lv

import numpy as np
import keepers

# Setup the dependency injector
dependency_injector = keepers.DependencyInjector(
                                   [('mpi4py.MPI', 'MPI'),
                                    'gfft',
                                    ('nifty.dummys.gfft_dummy', 'gfft_dummy'),
                                    'healpy',
                                    'libsharp_wrapper_gl'])

dependency_injector.register('pyfftw', lambda z: hasattr(z, 'FFTW_MPI'))


# Initialize the variables
variable_fft_module = keepers.Variable(
                               'fft_module',
                               ['pyfftw', 'gfft', 'gfft_dummy'],
                               lambda z: z in dependency_injector)


def _healpy_validator(use_healpy):
    if not isinstance(use_healpy, bool):
        return False
    if not use_healpy:
        return True
    if 'healpy' not in dependency_injector:
        return False
    healpy = dependency_injector['healpy']
    if lv(healpy.__version__) < lv('1.8.1'):
        return False
    return True


variable_use_healpy = keepers.Variable(
                          'use_healpy',
                          [True, False],
                          _healpy_validator,
                          genus='boolean')

variable_use_libsharp = keepers.Variable(
                         'use_libsharp',
                         [True, False],
                         lambda z: (('libsharp_wrapper_gl' in
                                     dependency_injector)
                                    if z else True) and isinstance(z, bool),
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
                            variable_use_healpy,
                            variable_use_libsharp,
                            variable_default_field_dtype,
                            variable_default_distribution_strategy],
                 file_name='NIFTy.conf',
                 search_pathes=[os.path.expanduser('~') + "/.config/nifty/",
                                os.path.expanduser('~') + "/.config/",
                                './'])

########
########

try:
    nifty_configuration.load()
except:
    pass
