import numpy as np
from transformation import Transformation
from rg_transforms import FFTW, GFFT
from nifty.config import about, dependency_injector as gdi
from nifty import RGSpace, nifty_configuration


class RGRGTransformation(Transformation):
    def __init__(self, domain, codomain, module=None):
        if self.check_codomain(domain, codomain):
            if module is None:
                if nifty_configuration['fft_module'] == 'pyfftw':
                    self._transform = FFTW(domain, codomain)
                elif nifty_configuration['fft_module'] == 'gfft' or \
                    nifty_configuration['fft_module'] == 'gfft_dummy':
                    self._transform = \
                        GFFT(domain,
                             codomain,
                             gdi.get(nifty_configuration['fft_module']))
                else:
                    raise ValueError('ERROR: unknow default FFT module:' +
                                     nifty_configuration['fft_module'])
            else:
                if module == 'pyfftw':
                    self._transform = FFTW(domain, codomain)
                elif module == 'gfft':
                    self._transform = \
                            GFFT(domain, codomain, gdi.get('gfft'))
                elif module == 'gfft_dummy':
                    self._transform = \
                        GFFT(domain, codomain, gdi.get('gfft_dummy'))
        else:
            raise ValueError("ERROR: incompatible codomain!")

    @staticmethod
    def check_codomain(domain, codomain):
        if not isinstance(domain, RGSpace):
            raise TypeError('ERROR: domain must be a RGSpace')

        if codomain is None:
            return False

        if not isinstance(codomain, RGSpace):
            raise TypeError(about._errors.cstring(
                "ERROR: codomain must be a RGSpace."
            ))

        if not np.all(np.array(domain.paradict['shape']) ==
                              np.array(codomain.paradict['shape'])):
            return False

        if domain.harmonic == codomain.harmonic:
            return False

        # check complexity
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

        return True

    def transform(self, val, axes=None, **kwargs):
        """
        RG -> RG transform method.

        Parameters
        ----------
        val : np.ndarray or distributed_data_object
            The value array which is to be transformed

        axes : None or tuple
            The axes along which the transformation should take place

        """
        if self._transform.codomain.harmonic:
            # correct for forward fft
            val = self._transform.domain.calc_weight(val, power=1)

        # Perform the transformation
        Tval = self._transform.transform(val, axes, **kwargs)

        if not self._transform.codomain.harmonic:
            # correct for inverse fft
            Tval = self._transform.codomain.calc_weight(Tval, power=-1)

        return Tval
