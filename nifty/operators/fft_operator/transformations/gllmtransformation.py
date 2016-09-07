import numpy as np
from nifty.config import dependency_injector as gdi,\
                         about
from nifty import GLSpace, LMSpace
from slicing_transformation import SlicingTransformation
import lm_transformation_factory as ltf

libsharp = gdi.get('libsharp_wrapper_gl')


class GLLMTransformation(SlicingTransformation):

    # ---Overwritten properties and methods---

    def __init__(self, domain, codomain=None, module=None):
        if 'libsharp_wrapper_gl' not in gdi:
            raise ImportError(about._errors.cstring(
                "The module libsharp is needed but not available."))

        super(GLLMTransformation, self).__init__(domain, codomain, module)

    # ---Mandatory properties and methods---

    @staticmethod
    def get_codomain(domain):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  an instance of the :py:class:`lm_space` class.

            Parameters
            ----------
            domain: GLSpace
                Space for which a codomain is to be generated

            Returns
            -------
            codomain : LMSpace
                A compatible codomain.
        """
        if domain is None:
            raise ValueError(about._errors.cstring(
                "ERROR: cannot generate codomain for None-input"))

        if not isinstance(domain, GLSpace):
            raise TypeError(about._errors.cstring(
                "ERROR: domain needs to be a GLSpace"))

        nlat = domain.nlat
        lmax = nlat - 1
        mmax = nlat - 1
        if domain.dtype == np.dtype('float32'):
            return_dtype = np.float32
        else:
            return_dtype = np.float64

        return LMSpace(lmax=lmax, mmax=mmax, dtype=return_dtype)

    @classmethod
    def check_codomain(cls, domain, codomain):
        if not isinstance(domain, GLSpace):
            raise TypeError(about._errors.cstring(
                "ERROR: domain is not a GLSpace"))

        if not isinstance(codomain, LMSpace):
            raise TypeError(about._errors.cstring(
                "ERROR: codomain must be a LMSpace."))

        nlat = domain.nlat
        nlon = domain.nlon
        lmax = codomain.lmax
        mmax = codomain.mmax

        if lmax != mmax:
            raise ValueError('ERROR: codomain has lmax != mmax.')

        if lmax != nlat - 1:
            raise ValueError('ERROR: codomain has lmax != nlat - 1.')

        if nlon != 2 * nlat - 1:
            raise ValueError('ERROR: domain has nlon != 2 * nlat - 1.')

        return None

    def _transformation_of_slice(self, inp):
        nlat = self.domain.nlat
        nlon = self.domain.nlon
        lmax = self.codomain.lmax
        mmax = self.codomain.mmax

        if issubclass(inp.dtype.type, np.complexfloating):

            [resultReal, resultImag] = [self.libsharpMap2Alm(x,
                                                             nlat=nlat,
                                                             nlon=nlon,
                                                             lmax=lmax,
                                                             mmax=mmax)
                                        for x in (inp.real, inp.imag)]

            [resultReal, resultImag] = [ltf.buildIdx(x, lmax=lmax)
                                        for x in [resultReal, resultImag]]

            # construct correct complex dtype
            one = resultReal.dtype.type(1)
            result_dtype = np.dtype(type(one + 1j))

            result = np.empty_like(resultReal, dtype=result_dtype)
            result.real = resultReal
            result.imag = resultImag
        else:
            result = self.libsharpMap2Alm(inp, nlat=nlat, nlon=nlon, lmax=lmax,
                                          mmax=mmax)
            result = ltf.buildIdx(result, lmax=lmax)

        return result

    # ---Added properties and methods---

    def libsharpMap2Alm(self, inp, **kwargs):
        if inp.dtype == np.dtype('float32'):
            return libsharp.map2alm_f(inp, **kwargs)
        elif inp.dtype == np.dtype('float64'):
            return libsharp.map2alm(inp, **kwargs)
        else:
            about.warnings.cprint("WARNING: performing dtype conversion for "
                                  "libsharp compatibility.")
            casted_inp = inp.astype(np.dtype('float64'), copy=False)
            result = libsharp.map2alm(casted_inp, **kwargs)
            return result