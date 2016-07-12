from fftw import FFTW
from gfft import  GFFT

from nifty.config import about, dependency_injector as gdi
from nifty import RGSpace

import numpy as np


class Transformation(object):
    """
        A generic transformation which defines a static check_codomain
        method for all transforms.
    """

    @staticmethod
    def check_codomain(domain, codomain):
        if codomain is None:
            return False

        if isinstance(domain, RGSpace):
            if not isinstance(codomain, RGSpace):
                raise TypeError(about._errors.cstring(
                    "ERROR: The given codomain must be a rg_space."
                ))

            if not np.all(np.array(domain.paradict['shape']) ==
                                  np.array(codomain.paradict['shape'])):
                return False

            if domain.harmonic == codomain.harmonic:
                return False

            # Check complexity
            # Prepare shorthands
            dcomp = domain.paradict['complexity']
            cocomp = codomain.paradict['complexity']

            # Case 1: if domain is completely complex, the codomain
            # must be complex too
            if dcomp == 2:
                if cocomp != 2:
                    return False
            # Case 2: if domain is hermitian, the codomain can be
            # real, a warning is raised otherwise
            elif dcomp == 1:
                if cocomp > 0:
                    about.warnings.cprint(
                        "WARNING: Unrecommended codomain! " +
                        "The domain is hermitian, hence the" +
                        "codomain should be restricted to real values."
                    )
            # Case 3: if domain is real, the codomain should be hermitian
            elif dcomp == 0:
                if cocomp == 2:
                    about.warnings.cprint(
                        "WARNING: Unrecommended codomain! " +
                        "The domain is real, hence the" +
                        "codomain should be restricted to" +
                        "hermitian configuration."
                    )
                elif cocomp == 0:
                    return False

            # Check if the distances match, i.e. dist' = 1 / (num * dist)
            if not np.all(
                np.absolute(np.array(domain.paradict['shape']) *
                            np.array(domain.distances) *
                            np.array(codomain.distances) - 1) < domain.epsilon):
                return False
        else:
            return False

        return True

    def __init__(self, domain, codomain, module=None):
        pass

    def transform(self, val, axes=None, **kwargs):
        raise NotImplementedError


class RGRGTransformation(Transformation):
    def __init__(self, domain, codomain, module=None):
        if Transformation.check_codomain(domain, codomain):
            if module is None:
                if gdi.get('pyfftw') is None:
                    if gdi.get('gfft') is None:
                        self._transform =\
                            GFFT(domain, codomain, gdi.get('gfft_dummy'))
                    else:
                        self._transform =\
                            GFFT(domain, codomain, gdi.get('gfft'))
                self._transform = FFTW(domain, codomain)
            else:
                if module == 'pyfftw':
                    if gdi.get('pyfftw') is not None:
                        self._transform = FFTW(domain, codomain)
                    else:
                        raise RuntimeError("ERROR: pyfftw is not available.")
                elif module == 'gfft':
                    if gdi.get('gfft') is not None:
                        self._transform =\
                            GFFT(domain, codomain, gdi.get('gfft'))
                    else:
                        raise RuntimeError("ERROR: gfft is not available.")
                elif module == 'gfft_dummy':
                    self._transform =\
                        GFFT(domain, codomain, gdi.get('gfft_dummy'))
                else:
                    raise ValueError('Given FFT module is not known: ' +
                                     str(module))
        else:
            raise ValueError("ERROR: Incompatible codomain!")

    def transform(self, val, axes=None, **kwargs):
        return self._transform.transform(val, axes, **kwargs)