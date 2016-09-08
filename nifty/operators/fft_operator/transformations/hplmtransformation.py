import numpy as np
from nifty.config import dependency_injector as gdi,\
                         about
from nifty import HPSpace, LMSpace
from slicing_transformation import SlicingTransformation

import lm_transformation_factory as ltf

hp = gdi.get('healpy')


class HPLMTransformation(SlicingTransformation):

    # ---Overwritten properties and methods---

    def __init__(self, domain, codomain=None, module=None):
        if 'healpy' not in gdi:
            raise ImportError(about._errors.cstring(
                "The module healpy is needed but not available"))

        super(HPLMTransformation, self).__init__(domain, codomain, module)

    # ---Mandatory properties and methods---

    @classmethod
    def get_codomain(cls, domain):
        """
            Generates a compatible codomain to which transformations are
            reasonable, i.e.\  an instance of the :py:class:`lm_space` class.

            Parameters
            ----------
            domain: HPSpace
                Space for which a codomain is to be generated

            Returns
            -------
            codomain : LMSpace
                A compatible codomain.
        """

        if not isinstance(domain, HPSpace):
            raise TypeError(about._errors.cstring(
                "ERROR: domain needs to be a HPSpace"))

        lmax = 3 * domain.nside - 1
        mmax = lmax

        result = LMSpace(lmax=lmax, mmax=mmax, dtype=np.dtype('float64'))
        cls.check_codomain(domain, result)
        return result

    @staticmethod
    def check_codomain(domain, codomain):
        if not isinstance(domain, HPSpace):
            raise TypeError(about._errors.cstring(
                'ERROR: domain is not a HPSpace'))

        if not isinstance(codomain, LMSpace):
            raise TypeError(about._errors.cstring(
                'ERROR: codomain must be a LMSpace.'))

        nside = domain.nside
        lmax = codomain.lmax
        mmax = codomain.mmax

        if 3 * nside - 1 != lmax:
            raise ValueError(about._errors.cstring(
                'ERROR: codomain has 3*nside-1 != lmax.'))

        if lmax != mmax:
            raise ValueError(about._errors.cstring(
                'ERROR: codomain has lmax != mmax.'))

        return None

    def _transformation_of_slice(self, inp, **kwargs):
        lmax = self.codomain.lmax
        mmax = self.codomain.mmax

        if issubclass(inp.dtype.type, np.complexfloating):
            [resultReal, resultImag] = [hp.map2alm(x.astype(np.float64,
                                                            copy=False),
                                                   lmax=lmax,
                                                   mmax=mmax,
                                                   pol=True,
                                                   use_weights=False,
                                                   **kwargs)
                                        for x in (inp.real, inp.imag)]

            [resultReal, resultImag] = [ltf.buildIdx(x, lmax=lmax)
                                        for x in [resultReal, resultImag]]

            result = self._combine_complex_result(resultReal, resultImag)

        else:
            result = hp.map2alm(inp.astype(np.float64, copy=False),
                                lmax=lmax, mmax=mmax, pol=True,
                                use_weights=False)
            result = ltf.buildIdx(result, lmax=lmax)

        return result
