from nifty.rg import RGSpace
from nifty.lm import GLSpace, HPSpace, LMSpace
from nifty.config import dependency_injector as gdi
from gfft import GFFT
from fftw import FFTW


class TransformFactory(object):
    """
        Transform factory which generates transform objects
    """

    def __init__(self):
        # cache for storing the transform objects
        self.cache = {}

    def _get_transform_override(self, domain, codomain, module):
        if module == 'gfft':
            return GFFT(domain, codomain, gdi.get('gfft'))
        elif module == 'fftw':
            return FFTW(domain, codomain)
        elif module == 'gfft_dummmy':
            return GFFT(domain, codomain, gdi.get('gfft_dummy'))

    def _get_transform(self, domain, codomain):
        if isinstance(domain, RGSpace) and isinstance(codomain, RGSpace):
            # fftw -> gfft -> gfft_dummy
            if gdi.get('fftw') is None:
                if gdi.get('gfft') is None:
                    return GFFT(domain, codomain, gdi.get('gfft_dummy'))
                else:
                    return GFFT(domain, codomain, gdi.get('gfft'))
            return FFTW(domain, codomain)

    def create(self, domain, codomain, module=None):
        key = domain.__hash__() ^ ((111 * codomain.__hash__()) ^
                                   (179 * module.__hash__()))

        if key not in self.cache:
            if module is None:
                self.cache[key] = self._get_transform(domain, codomain)
            else:
                self.cache[key] = self._get_transform_override(domain,
                                                               codomain,
                                                               module)
        return self.cache[key]
