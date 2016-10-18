import numpy as np
from transformation import Transformation
from rg_transforms import FFTW, GFFT
from nifty.config import dependency_injector as gdi
from nifty import RGSpace, nifty_configuration


class RGRGTransformation(Transformation):
    def __init__(self, domain, codomain=None, module=None):
        super(RGRGTransformation, self).__init__(domain, codomain, module)

        if module is None:
            if nifty_configuration['fft_module'] == 'pyfftw':
                self._transform = FFTW(domain, codomain)
            elif (nifty_configuration['fft_module'] == 'gfft' or
                  nifty_configuration['fft_module'] == 'gfft_dummy'):
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
                raise ValueError('ERROR: unknow FFT module:' + module)

    @classmethod
    def get_codomain(cls, domain, dtype=None, zerocenter=None):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  either a shifted grid or a Fourier conjugate
            grid.

            Parameters
            ----------
            domain: RGSpace
                Space for which a codomain is to be generated
            cozerocenter : {bool, numpy.ndarray}, *optional*
                Whether or not the grid is zerocentered for each axis or not
                (default: None).

            Returns
            -------
            codomain : nifty.rg_space
                A compatible codomain.
        """
        if not isinstance(domain, RGSpace):
            raise TypeError('ERROR: domain needs to be a RGSpace')

        # parse the cozerocenter input
        if zerocenter is None:
            zerocenter = domain.zerocenter
        # if the input is something scalar, cast it to a boolean
        else:
            temp = np.empty_like(domain.zerocenter)
            temp[:] = zerocenter
            zerocenter = temp

        # calculate the initialization parameters
        distances = 1 / (np.array(domain.shape) *
                         np.array(domain.distances))
        if dtype is None:
            # create a definitely complex dtype from the dtype of domain
            one = domain.dtype.type(1)
            dtype = np.dtype(type(one + 1j))

        new_space = RGSpace(domain.shape,
                            zerocenter=zerocenter,
                            distances=distances,
                            harmonic=(not domain.harmonic),
                            dtype=dtype)
        cls.check_codomain(domain, new_space)
        return new_space

    @classmethod
    def check_codomain(cls, domain, codomain):
        if not isinstance(domain, RGSpace):
            raise TypeError('ERROR: domain is not a RGSpace')

        if codomain is None:
            return False

        if not isinstance(codomain, RGSpace):
            return False

        if not np.all(np.array(domain.shape) ==
                      np.array(codomain.shape)):
            return False

        if domain.harmonic == codomain.harmonic:
            return False

        if codomain.harmonic and not issubclass(codomain.dtype.type,
                                                np.complexfloating):
            cls.logger.warn("Codomain is harmonic but dtype is real.")

        # Check if the distances match, i.e. dist' = 1 / (num * dist)
        if not np.all(
            np.absolute(np.array(domain.shape) *
                        np.array(domain.distances) *
                        np.array(codomain.distances) - 1) <
                10**-7):
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
            # correct for forward fft.
            # naively one would set power to 0.5 here in order to
            # apply effectively a factor of 1/sqrt(N) to the field.
            # BUT: the pixel volumes of the domain and codomain are different.
            # Hence, in order to produce the same scalar product, power===1.
            val = self._transform.domain.weight(val, power=1, axes=axes)

        # Perform the transformation
        Tval = self._transform.transform(val, axes, **kwargs)

        if not self._transform.codomain.harmonic:
            # correct for inverse fft.
            # See discussion above.
            Tval = self._transform.codomain.weight(Tval, power=-1, axes=axes)

        return Tval
