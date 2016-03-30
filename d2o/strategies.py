# -*- coding: utf-8 -*-

from nifty.keepers import global_dependency_injector as gdi

pyfftw = gdi.get('pyfftw')

_maybe_fftw = ['fftw'] if ('pyfftw' in gdi) else []
STRATEGIES = {
                'all': ['not', 'equal', 'freeform'] + _maybe_fftw,
                'global': ['not', 'equal'] + _maybe_fftw,
                'local': ['freeform'],
                'slicing': ['equal', 'freeform'] + _maybe_fftw,
                'not': ['not'],
                'hdf5': ['equal'] + _maybe_fftw,
             }
