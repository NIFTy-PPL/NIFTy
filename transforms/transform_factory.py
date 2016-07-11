import numpy as np

from nifty.rg import RGSpace
from nifty.lm import GLSpace, HPSpace, LMSpace
from nifty.config import about, dependency_injector as gdi
from gfft import GFFT
from fftw import FFTW


class TransformFactory(object):
    """
        Transform factory which generates transform objects
    """

    def __init__(self):
        # cache for storing the transform objects
        self.cache = {}

    def _get_transform(self, domain, codomain, module):
        if isinstance(domain, RGSpace):
            # fftw -> gfft -> gfft_dummy
            if module is None:
                if gdi.get('pyfftw') is None:
                    if gdi.get('gfft') is None:
                        return GFFT(domain, codomain, gdi.get('gfft_dummy'))
                    else:
                        return GFFT(domain, codomain, gdi.get('gfft'))
                return FFTW(domain, codomain)
            else:
                if module == 'pyfftw':
                    if gdi.get('pyfftw') is not None:
                        return FFTW(domain, codomain)
                    else:
                        raise RuntimeError("ERROR: pyfftw is not available.")
                elif module == 'gfft':
                    if gdi.get('gfft') is not None:
                        return GFFT(domain, codomain, gdi.get('gfft'))
                    else:
                        raise RuntimeError("ERROR: gfft is not available.")
                elif module == 'gfft_dummy':
                    return GFFT(domain, codomain, gdi.get('gfft_dummy'))
                else:
                    raise ValueError('Given FFT module is not known: ' +
                                     str(module))

    def create(self, domain, codomain, module=None):
        key = domain.__hash__() ^ ((111 * codomain.__hash__()) ^
                                   (179 * module.__hash__()))

        if key not in self.cache:
            self.cache[key] = self._get_transform(domain, codomain, module)

        return self.cache[key]