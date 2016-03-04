# -*- coding: utf-8 -*-

import os

from nifty_dependency_injector import dependency_injector
from nifty_configuration import variable,\
                                configuration

global_dependency_injector = dependency_injector(
                                   ['h5py',
                                    ('mpi4py.MPI', 'MPI'),
                                    ('nifty.dummys.MPI_dummy', 'MPI_dummy'),
                                    'pyfftw',
                                    'gfft',
                                    ('nifty.dummys.gfft_dummy', 'gfft_dummy'),
                                    'healpy',
                                    'libsharp_wrapper_gl'])


variable_fft_module = variable('fft_module',
                               ['pyfftw', 'gfft', 'gfft_dummy'],
                               lambda z: z in global_dependency_injector)

# gl_space needs libsharp
variable_lm2gl = variable('lm2gl',
                          [True, False],
                          lambda z: (('libsharp_wrapper_gl' in
                                      global_dependency_injector)
                                     if z else True) and isinstance(z, bool),
                          genus='boolean')

variable_use_healpy = variable(
                          'use_healpy',
                          [True, False],
                          lambda z: (('healpy' in global_dependency_injector)
                                     if z else True) and isinstance(z, bool),
                          genus='boolean')

variable_use_libsharp = variable('use_libsharp',
                                 [True, False],
                                 lambda z: (('libsharp_wrapper_gl' in
                                             global_dependency_injector)
                                            if z else True) and
                                            isinstance(z, bool),
                                 genus='boolean')

variable_verbosity = variable('verbosity',
                              [1],
                              lambda z: z == abs(int(z)),
                              genus='int')

variable_mpi_module = variable('mpi_module',
                               ['MPI', 'MPI_dummy'],
                               lambda z: z in global_dependency_injector)

variable_default_distribution_strategy = variable(
                            'default_distribution_strategy',
                            ['fftw', 'equal', 'not'],
                            lambda z: (('pyfftw' in global_dependency_injector)
                                       if (z == 'pyfftw') else True)
                                                  )

variable_d2o_init_checks = variable('d2o_init_checks',
                                    [True, False],
                                    lambda z: isinstance(z, bool),
                                    genus='boolean')

global_configuration = configuration(
                     [variable_fft_module,
                      variable_lm2gl,
                      variable_use_healpy,
                      variable_use_libsharp,
                      variable_verbosity,
                      variable_mpi_module,
                      variable_default_distribution_strategy,
                      variable_d2o_init_checks
                      ],
                     path=os.path.expanduser('~') + "/.nifty/global_config")


variable_default_comm = variable(
                     'default_comm',
                     ['COMM_WORLD'],
                     lambda z: hasattr(global_dependency_injector[
                                       global_configuration['mpi_module']], z))

global_configuration.register(variable_default_comm)

try:
    global_configuration.load()
except:
    pass
