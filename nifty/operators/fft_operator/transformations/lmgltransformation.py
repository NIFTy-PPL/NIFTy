import numpy as np
from nifty.config import dependency_injector as gdi
from nifty import GLSpace, LMSpace

from slicing_transformation import SlicingTransformation
import lm_transformation_factory as ltf

import logging
logger = logging.getLogger('NIFTy.LMGLTransformation')

libsharp = gdi.get('libsharp_wrapper_gl')


class LMGLTransformation(SlicingTransformation):

    # ---Overwritten properties and methods---

    def __init__(self, domain, codomain=None, module=None):
        if 'libsharp_wrapper_gl' not in gdi:
            raise ImportError(
                "The module libsharp is needed but not available.")

        super(LMGLTransformation, self).__init__(domain, codomain, module)

    # ---Mandatory properties and methods---

    @classmethod
    def get_codomain(cls, domain):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  a pixelization of the two-sphere.

            Parameters
            ----------
            domain : LMSpace
                Space for which a codomain is to be generated

            Returns
            -------
            codomain : HPSpace
                A compatible codomain.

            References
            ----------
            .. [#] M. Reinecke and D. Sverre Seljebotn, 2013,
                   "Libsharp - spherical
                   harmonic transforms revisited";
                   `arXiv:1303.4945 <http://www.arxiv.org/abs/1303.4945>`_
        """
        if not isinstance(domain, LMSpace):
            raise TypeError(
                'ERROR: domain needs to be a LMSpace')

        if domain.dtype is np.dtype('float32'):
            new_dtype = np.float32
        else:
            new_dtype = np.float64

        nlat = domain.lmax + 1
        nlon = domain.lmax * 2 + 1

        result = GLSpace(nlat=nlat, nlon=nlon, dtype=new_dtype)
        cls.check_codomain(domain, result)
        return result

    @staticmethod
    def check_codomain(domain, codomain):
        if not isinstance(domain, LMSpace):
            raise TypeError(
                'ERROR: domain is not a LMSpace')

        if not isinstance(codomain, GLSpace):
            raise TypeError(
                'ERROR: codomain must be a GLSpace.')

        nlat = codomain.nlat
        nlon = codomain.nlon
        lmax = domain.lmax
        mmax = domain.mmax

        if lmax != mmax:
            raise ValueError(
                'ERROR: domain has lmax != mmax.')

        if nlat != lmax + 1:
            raise ValueError(
                'ERROR: codomain has nlat != lmax + 1.')

        if nlon != 2 * lmax + 1:
            raise ValueError(
                'ERROR: domain has nlon != 2 * lmax + 1.')

        return None

    def _transformation_of_slice(self, inp, **kwargs):
        nlat = self.codomain.nlat
        nlon = self.codomain.nlon
        lmax = self.domain.lmax
        mmax = self.domain.mmax

        if issubclass(inp.dtype.type, np.complexfloating):
            [resultReal, resultImag] = [ltf.buildLm(x, lmax=lmax)
                                        for x in (inp.real, inp.imag)]

            [resultReal, resultImag] = [self.libsharpAlm2Map(x,
                                                             nlat=nlat,
                                                             nlon=nlon,
                                                             lmax=lmax,
                                                             mmax=mmax,
                                                             cl=False,
                                                             **kwargs)
                                        for x in [resultReal, resultImag]]

            result = self._combine_complex_result(resultReal, resultImag)

        else:
            result = ltf.buildLm(inp, lmax=lmax)
            result = self.libsharpAlm2Map(result, nlat=nlat, nlon=nlon,
                                          lmax=lmax, mmax=mmax, cl=False)

        return result

    # ---Added properties and methods---

    def libsharpAlm2Map(self, inp, **kwargs):
        if inp.dtype == np.dtype('complex64'):
            return libsharp.alm2map_f(inp, **kwargs)
        elif inp.dtype == np.dtype('complex128'):
            return libsharp.alm2map(inp, **kwargs)
        else:
            logger.debug("performing dtype conversion for libsharp "
                         "compatibility.")
            casted_inp = inp.astype(np.dtype('complex128'), copy=False)
            result = libsharp.alm2map(casted_inp, **kwargs)
            return result
