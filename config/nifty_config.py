# -*- coding: utf-8 -*-

import os

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
# gl_space needs libsharp
variable_lm2gl = keepers.Variable(
                          'lm2gl',
                          [True, False],
                          lambda z: (('libsharp_wrapper_gl' in
                                      dependency_injector)
                                     if z else True) and isinstance(z, bool),
                          genus='boolean')

variable_use_healpy = keepers.Variable(
                          'use_healpy',
                          [True, False],
                          lambda z: (('healpy' in dependency_injector)
                                     if z else True) and isinstance(z, bool),
                          genus='boolean')

variable_use_libsharp = keepers.Variable(
                                 'use_libsharp',
                                 [True, False],
                                 lambda z: (('libsharp_wrapper_gl' in
                                             dependency_injector)
                                            if z else True) and
                                            isinstance(z, bool),
                                 genus='boolean')

variable_verbosity = keepers.Variable('verbosity',
                                      [1],
                                      lambda z: z == abs(int(z)),
                                      genus='int')

variable_default_datamodel = keepers.Variable(
                              'default_datamodel',
                              ['fftw', 'equal'],
                              lambda z: (('pyfftw' in dependency_injector)
                                         if z == 'fftw' else True),
                              genus='str')

nifty_configuration = keepers.get_Configuration(
                     'NIFTy',
                     [variable_fft_module,
                      variable_lm2gl,
                      variable_use_healpy,
                      variable_use_libsharp,
                      variable_verbosity,
                      variable_default_datamodel,
                      ],
                     path=os.path.expanduser('~') + "/.nifty/nifty_config")
########
### Compatibility variables
########
variable_mpi_module = keepers.Variable('mpi_module',
                                       ['MPI'],
                                       lambda z: z in dependency_injector)


nifty_configuration.register(variable_mpi_module)

# register the default comm variable as the 'mpi_module' variable is now
# available
variable_default_comm = keepers.Variable(
                     'default_comm',
                     ['COMM_WORLD'],
                     lambda z: hasattr(dependency_injector[
                                       nifty_configuration['mpi_module']], z))

nifty_configuration.register(variable_default_comm)


########
########

try:
    nifty_configuration.load()
except:
    pass
