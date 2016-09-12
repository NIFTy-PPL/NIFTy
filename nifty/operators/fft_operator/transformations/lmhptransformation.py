import numpy as np
from nifty.config import dependency_injector as gdi,\
                         about
from nifty import HPSpace, LMSpace
from slicing_transformation import SlicingTransformation
import lm_transformation_factory as ltf

hp = gdi.get('healpy')


class LMHPTransformation(SlicingTransformation):

    # ---Overwritten properties and methods---

    def __init__(self, domain, codomain=None, module=None):
        if gdi.get('healpy') is None:
            raise ImportError(about._errors.cstring(
                "The module libsharp is needed but not available."))

        super(LMHPTransformation, self).__init__(domain, codomain, module)

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
            .. [#] K.M. Gorski et al., 2005, "HEALPix: A Framework for
                   High-Resolution Discretization and Fast Analysis of Data
                   Distributed on the Sphere", *ApJ* 622..759G.
        """
        if not isinstance(domain, LMSpace):
            raise TypeError(about._errors.cstring(
                'ERROR: domain needs to be a LMSpace'))

        nside = (domain.lmax + 1) // 3
        result = HPSpace(nside=nside)
        cls.check_codomain(domain, result)
        return result

    @staticmethod
    def check_codomain(domain, codomain):
        if not isinstance(domain, LMSpace):
            raise TypeError(about._errors.cstring(
                'ERROR: domain is not a LMSpace'))

        if not isinstance(codomain, HPSpace):
            raise TypeError(about._errors.cstring(
                'ERROR: codomain must be a HPSpace.'))

        nside = codomain.nside
        lmax = domain.lmax
        mmax = domain.mmax

        if lmax != mmax:
            raise ValueError(about._errors.cstring(
                'ERROR: domain has lmax != mmax.'))

        if 3*nside - 1 != lmax:
            raise ValueError(about._errors.cstring(
                'ERROR: codomain has 3*nside -1 != lmax.'))

        return None

    def _transformation_of_slice(self, inp, **kwargs):
        nside = self.codomain.nside
        lmax = self.domain.lmax
        mmax = self.domain.mmax

        if issubclass(inp.dtype.type, np.complexfloating):
            [resultReal, resultImag] = [ltf.buildLm(x, lmax=lmax)
                                        for x in (inp.real, inp.imag)]

            [resultReal, resultImag] = [hp.alm2map(x.astype(np.complex128,
                                                            copy=False),
                                                   nside,
                                                   lmax=lmax,
                                                   mmax=mmax,
                                                   pixwin=False,
                                                   fwhm=0.0,
                                                   sigma=None,
                                                   pol=True,
                                                   inplace=False,
                                                   **kwargs)
                                        for x in [resultReal, resultImag]]

            result = self._combine_complex_result(resultReal, resultImag)

        else:
            result = ltf.buildLm(inp, lmax=lmax)
            result = hp.alm2map(result.astype(np.complex128, copy=False),
                                nside, lmax=lmax, mmax=mmax, pixwin=False,
                                fwhm=0.0, sigma=None, pol=True, inplace=False)

        return result
